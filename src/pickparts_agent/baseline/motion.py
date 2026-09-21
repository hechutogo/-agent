"""Five motion stages with visual lift and release verification."""
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from ..scene.kinematics import ArmKinematics, ARM_NAMES


class PickPlace:
    def __init__(self, sim, perception, output="runs", on_stage=None):
        self.sim, self.perception = sim, perception
        self.on_stage = on_stage or (lambda stage: None)
        self.output = Path(output)
        self.kin = ArmKinematics()
        self.rest = self._arm(sim.observe())
        self._recovery_required = False

    def _arm(self, frame):
        return frame.qpos[[self.sim.joint_names.index(n) for n in ARM_NAMES]]

    def _capture(self, folder, stage):
        frame = self.sim.observe()
        Image.fromarray(frame.rgb).save(folder / f"{stage}.png")
        np.savez_compressed(folder / f"{stage}.npz", depth=frame.depth,
                            intrinsic=frame.intrinsic,
                            camera_to_base=frame.camera_to_base, qpos=frame.qpos)
        return frame

    def _move(self, position, jaw=None):
        seed = self._arm(self.sim.observe())
        pose = self.kin.forward(seed)
        if pose[2, 1] > .97:
            count = max(2, int(np.ceil(np.linalg.norm(position - pose[:3, 3]) / .008)))
            waypoints = np.linspace(pose[:3, 3], position, count + 1)[1:]
        else:
            waypoints = [position]
        for waypoint in waypoints:
            q = self.kin.solve(waypoint, seed=seed)
            self._moves.append({"position": np.asarray(waypoint).tolist(),
                                "joints": q.tolist()})
            self.sim.move_right(q, jaw=jaw, steps=6 if len(waypoints) > 1 else 35)
            seed = self._arm(self.sim.observe())
            error = np.linalg.norm(self.kin.forward(seed)[:3, 3] - waypoint)
            if error > .008:
                raise RuntimeError(f"TCP tracking error {error:.3f} m")

    def _record_failure(self, folder, result):
        # Diagnostics must not mask the original failure or prevent the latch.
        try:
            folder.mkdir(parents=True, exist_ok=True)
            self.on_stage("视觉定位")
            self._capture(folder, "error")
        except Exception:
            pass
        for name, data in (("moves.json", self._moves), ("result.json", result)):
            try:
                (folder / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception:
                pass

    def run(self, target):
        if target not in ("A", "B"):
            raise ValueError("Choose A or B")
        folder = self.output / f"{time.time_ns()}-{target}"
        result = {"success": False, "target": target, "artifacts": str(folder)}
        self._moves = []
        if self._recovery_required:
            result["message"] = (
                "Recovery required: restart the application to create a fresh Simulation "
                "before attempting another task."
            )
            self._record_failure(folder, result)
            return result
        actuation_started = False
        try:
            folder.mkdir(parents=True, exist_ok=True)
            before = self._capture(folder, "00-before")
            part = self.perception.locate(before, target)
            box = self.perception.locate(before, "box")
            p = np.asarray(part["point"]).copy()
            destination = np.asarray(box["point"]).copy()
            # RGB-D observes visible faces; grip below the measured surface.
            grasp = p.copy()
            grasp[2] -= .004
            hover = grasp.copy()
            hover[2] += .10
            above_box = destination.copy()
            above_box[2] = hover[2]
            release = destination.copy()
            release[2] += .055
            # Preflight all task poses before starting motion.
            for point in (grasp, hover, above_box, release):
                self.kin.solve(point)
            print(f"{target}: 移动", flush=True)
            self.on_stage("移动")
            # A failed command can already have changed arm or jaw targets.
            actuation_started = True
            self._move(hover, jaw=.8)
            self._capture(folder, "01-hover")
            print(f"{target}: 抓取", flush=True)
            self.on_stage("抓取")
            self._move(grasp)
            self._capture(folder, "02-open")
            self.sim.move_right(self._arm(self.sim.observe()), jaw=0., steps=25)
            self._capture(folder, "03-closed")
            print(f"{target}: 抬起", flush=True)
            self.on_stage("抬起")
            self._move(hover)
            lifted_frame = self._capture(folder, "04-lift")
            lifted = self.perception.locate(lifted_frame, target)
            lift = float(lifted["point"][2] - p[2])
            result["lift_m"] = lift
            if lift < .035:
                raise RuntimeError(f"视觉未确认抓起：高度变化 {lift:.3f} m")
            print(f"{target}: 放置", flush=True)
            self.on_stage("放置")
            self._move(above_box)
            self._move(release)
            self.sim.move_right(self._arm(self.sim.observe()), jaw=.8, steps=20)
            self._move(above_box)
            print(f"{target}: 复位", flush=True)
            self.on_stage("复位")
            self.sim.move_right(self.rest, jaw=.8)
            self.sim.hold(15)
            final_frame = self._capture(folder, "05-final")
            self.on_stage("视觉校验")
            final = self.perception.locate(final_frame, target)
            final_box = self.perception.locate(final_frame, "box")
            bbox = final_box["bbox"]
            object_bbox = final["bbox"]
            cx, cy = (object_bbox[0] + object_bbox[2]) / 2, (object_bbox[1] + object_bbox[3]) / 2
            delta = np.asarray(final["point"]) - np.asarray(final_box["point"])
            if not (bbox[0] < cx < bbox[2] and bbox[1] < cy < bbox[3]
                    and np.linalg.norm(delta[:2]) < .045 and abs(delta[2]) < .06):
                raise RuntimeError("视觉未确认零件已落入盒内")
            result.update(success=True, message=f"零件 {target} 已放入盒子。还需要什么？")
            (folder / "moves.json").write_text(json.dumps(self._moves, indent=2))
            (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        except BaseException as exc:
            if actuation_started:
                self._recovery_required = True
            result.update(success=False, message=str(exc))
            self._record_failure(folder, result)
            if not isinstance(exc, (ValueError, RuntimeError)):
                raise
        return result
