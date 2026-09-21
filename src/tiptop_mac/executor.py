"""Open-loop executor for TiPToP-style plans.

The whole trajectory is produced before execution; any motion failure aborts
immediately: no retry, no replan. A single unconditional grasp-calibration
step may translate the remaining pre-planned waypoints (grasp-relative
planning, like cuTAMP); it never branches on the task outcome.
"""
import json
from types import SimpleNamespace

import numpy as np

from pickparts_agent.agent.atoms.manipulation import (
    REST_Q, _arm_qpos, _cartesian,
)


class OpenLoopExecutor:
    def __init__(self, sim, kin, on_stage=None, *, locate=None):
        self.sim = sim
        self.kin = kin
        self.on_stage = on_stage if on_stage is not None else (lambda s: None)
        # Optional local RGB-D locator used only by fixed calibrate steps.
        self.locate = locate
        self._ctx = SimpleNamespace(sim=sim, kin=kin)

    def execute(self, plan):
        if not plan.trajectory:
            return {"success": True, "aborted": False,
                    "message": "目标已满足，无需动作。"}
        shift = np.zeros(3)
        for index, step in enumerate(plan.trajectory):
            kind = step.kind
            try:
                if kind == "move":
                    target = np.asarray(step.position, dtype=float) + shift
                    result = _cartesian(self._ctx, target, jaw=step.jaw)
                    if not result.success:
                        return self._abort(
                            f"轨迹执行中止（{result.kind}）：{result.message}")
                elif kind == "gripper":
                    self.sim.move_right(_arm_qpos(self._ctx), jaw=step.jaw)
                elif kind == "reset":
                    self.sim.move_right(REST_Q, jaw=step.jaw)
                elif kind == "calibrate":
                    measured = self._calibrate(step.label)
                    if isinstance(measured, dict):
                        return self._abort(measured["error"])
                    shift = measured
                else:
                    return self._abort(f"未知的轨迹步骤：{kind}")
            except (ValueError, RuntimeError) as exc:
                return self._abort(f"轨迹执行中止：{exc}")
            self.on_stage(
                f"开环执行 {index + 1}/{len(plan.trajectory)}：{kind}")
        return {"success": True, "aborted": False,
                "message": "开环执行完成。"}

    def _calibrate(self, label):
        if self.locate is None:
            return {"error": "抓取标定步骤缺少本地 RGB-D 定位器"}
        try:
            spec = json.loads(label)
            held, assumed = spec["held"], float(spec["assumed"])
            frame = self.sim.observe()
            located = self.locate(frame, held)
        except (ValueError, KeyError, RuntimeError) as exc:
            return {"error": f"抓取标定失败：{type(exc).__name__}"}
        tcp = self.kin.forward(_arm_qpos(self._ctx))[:3, 3]
        block = np.asarray(located["point"], dtype=float)
        xy_shift = np.array([tcp[0] - block[0], tcp[1] - block[1]])
        measured_bottom = float(tcp[2] - located["bottom_z"])
        z_shift = measured_bottom - assumed
        return np.array([xy_shift[0], xy_shift[1], z_shift])

    def _abort(self, message):
        return {"success": False, "aborted": True, "message": message}
