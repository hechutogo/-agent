"""One trajectory per attempt, with conservative sensor-only grip recovery."""
import json
from types import SimpleNamespace

import numpy as np

from pickparts_agent.agent.atoms.manipulation import REST_Q, _arm_qpos, _cartesian
from pickparts_agent.scene.kinematics import ARM_NAMES
from pickparts_agent.scene.perception import ScenePerception
from tiptop_mac.executor import OpenLoopExecutor, _MotionFailure, record_artifact
from .tamp import CartesianPathError, stable_block_overlap, validate_cartesian_path


class GripStateError(RuntimeError):
    """Grip evidence is missing/ambiguous; the parent may retry read-only."""

    error_kind = "grip_failed"


class RecoveringExecutor(OpenLoopExecutor):
    """Preserve a candidate until sensors prove holding, release or separation.

    ``reconcile(scene)`` consumes a fresh SceneGraph and never moves the robot.
    ``execute`` returns success, aborted, message, error_kind and acted. Success
    means this trajectory completed, not that the placement goal was verified;
    acted means a lift was visually verified during this call (including a
    reverified held continuation). An empty trajectory never counts as acted.

    Known 36 mm block height is used for box occlusion and conservative lift
    bounds; table contact uses the measured bottom. The box floor is 6 mm.
    Missing support geometry is not evidence of a drop. Callers must
    serialize sensor/actuator access and own retry budgets and final events.
    """

    # A 24mm block stops the commanded closed jaw near .45 rad in physics.
    # Holding still requires independent proximity and visual lift evidence.
    _CLOSED = .60
    _OPEN = .65
    _LIFT = .035
    _BLOCK_HEIGHT = .036

    def __init__(self, sim, kin, on_stage=None, *, locate=None,
                 on_event=None, recorder=None):
        super().__init__(sim, kin, on_stage, locate=locate,
                         on_event=on_event, recorder=recorder)
        self.candidate = None
        self.held = None
        self._scene = None
        self._baseline = None
        self._assumed = None
        self._empty_candidate = None
        self._uncertain = False
        self._acted = False

    def bind_sim(self, sim):
        """Rebind sensor dependencies; do not move or close either simulator."""
        locate = ScenePerception(getattr(sim, "object_specs", None)).locate
        self.sim = sim
        self._ctx = SimpleNamespace(sim=sim, kin=self.kin)
        self.locate = locate
        self.candidate = self.held = None
        self._scene = self._baseline = self._assumed = None
        self._empty_candidate = None
        self._uncertain = self._acted = False

    def _proprioception(self, qpos):
        qpos = np.asarray(qpos, dtype=float)
        indices = [self.sim.joint_names.index(n) for n in ARM_NAMES]
        jaw = float(qpos[self.sim.joint_names.index("Jaw")])
        tcp = np.asarray(self.kin.forward(qpos[indices]))[:3, 3]
        if not np.isfinite(jaw) or not np.isfinite(tcp).all():
            raise GripStateError("Invalid grip proprioception.")
        return tcp, jaw

    @classmethod
    def _geometry(cls, located):
        data = located if isinstance(located, dict) else vars(located)
        point = np.asarray(data["point"], dtype=float)
        bottom = float(data["bottom_z"])
        top = float(data.get("top_z", bottom + cls._BLOCK_HEIGHT))
        extent = np.asarray(data.get("extent", [.024, .024, cls._BLOCK_HEIGHT]),
                            dtype=float)
        if (point.shape != (3,) or extent.shape != (3,)
                or not np.isfinite(point).all() or not np.isfinite(extent).all()
                or not np.isfinite([bottom, top]).all()
                or np.any(extent <= 0) or top < bottom):
            raise GripStateError("Invalid observed grip geometry.")
        # A lower bound for conservative lift checks; only box occlusion may
        # use this inferred bottom as evidence of contact.
        physical_bottom = min(bottom, top - cls._BLOCK_HEIGHT)
        return SimpleNamespace(id=data.get("id"), point=point, bottom_z=bottom, top_z=top,
                               extent=extent, physical_bottom=physical_bottom)

    @staticmethod
    def _inside(node, support, margin=0.):
        return bool(np.all(
            np.abs(node.point[:2] - np.asarray(support.point)[:2])
            + node.extent[:2] / 2
            <= np.asarray(support.extent)[:2] / 2 + margin))

    def _support_heights(self, node):
        if self._scene is None:
            return []
        table = self._scene.table
        bounds = np.asarray(table.bounds, dtype=float)
        heights = []
        half = node.extent[:2] / 2
        if (bounds.shape == (2, 3) and np.isfinite(bounds).all()
                and np.all(node.point[:2] - half >= bounds[0, :2])
                and np.all(node.point[:2] + half <= bounds[1, :2])):
            heights.append(float(table.top_z))
        for object_id, other in self._scene.objects.items():
            if object_id == (node.id or self.candidate):
                continue
            if other.kind == "block" and stable_block_overlap(node, other):
                heights.append(float(other.top_z))
            elif other.kind == "box" and self._inside(node, other, margin=-.006):
                # The floor top is 6 mm above the base, not the visible rim.
                heights.append(float(other.bottom_z) + .006)
        return [z for z in heights if np.isfinite(z)]

    def _supported(self, node):
        if any(-.003 <= node.bottom_z - z <= .008
               for z in self._support_heights(node)):
            return True
        # Only a surrounding box explains an occluded bottom. A tilted block
        # on the table must use its observed bottom, not an upright height.
        return self._scene is not None and any(
            other.kind == "box" and self._inside(node, other, margin=-.006)
            and other.bottom_z <= node.bottom_z <= other.top_z
            and -.003 <= node.physical_bottom - (other.bottom_z + .006) <= .008
            for other in self._scene.objects.values())

    def _raised(self, node):
        heights = self._support_heights(node)
        if heights:
            return all(node.physical_bottom - z > .015 for z in heights)
        return (self._baseline is not None
                and node.point[2] - self._baseline.point[2] >= self._LIFT)

    @staticmethod
    def _near(node, tcp):
        return (np.linalg.norm(node.point[:2] - tcp[:2]) <= .035
                and abs(node.point[2] - tcp[2]) <= .050)

    def _check_empty_neighborhood(self, tcp):
        if self._scene is None:
            raise GripStateError("Empty grip requires scene evidence.")
        for object_id, observed in self._scene.objects.items():
            if object_id != self.candidate and observed.kind == "block":
                node = self._geometry(observed)
                if self._near(node, tcp) and not self._supported(node):
                    raise GripStateError("An unsupported object is near the jaw.")

    def _resolve(self, node, qpos):
        self._uncertain = True
        tcp, jaw = self._proprioception(qpos)
        near = self._near(node, tcp)
        if jaw <= self._CLOSED and near:
            # Support does not detach a clamped object. Only the first holding
            # confirmation needs a new lift; continuations retain that evidence.
            if self.held != self.candidate:
                self._verify_lift(node, qpos)
            self._uncertain = False
            return self.held
        if self._supported(node) and (jaw >= self._OPEN or not near):
            self._check_empty_neighborhood(tcp)
            # Remember the separated object, not an unrestricted "empty" bit.
            # Closed-jaw retries must recheck its visibility and support.
            self._empty_candidate = self.candidate if not near else None
            self.held = self.candidate = None
            self._baseline = self._assumed = None
            self._uncertain = False
            return None
        raise GripStateError("Grip state is ambiguous; observe again before motion.")

    def reconcile(self, scene):
        """Return the held ID/None, or raise without discarding uncertain state."""
        self._scene = scene
        self._uncertain = True
        tcp, jaw = self._proprioception(scene.qpos)
        if self.candidate is None:
            if self.held is not None:
                raise GripStateError("Held object has no known grasp candidate.")
            if jaw < self._OPEN:
                observed = scene.objects.get(self._empty_candidate)
                if observed is None:
                    raise GripStateError("Closed jaw has no visible empty-grip evidence.")
                node = self._geometry(observed)
                if self._near(node, tcp) or not self._supported(node):
                    raise GripStateError("Closed jaw is no longer verified empty.")
            self._check_empty_neighborhood(tcp)
            self._uncertain = False
            return None
        observed = scene.objects.get(self.candidate)
        if observed is None:
            self._uncertain = True
            raise GripStateError("Grasp candidate is not visible.")
        return self._resolve(self._geometry(observed), scene.qpos)

    def planning_context(self, scene, *, held):
        """Return local kinematic inputs used by TAMP path validation."""
        qpos = np.asarray(scene.qpos, dtype=float)
        indices = [self.sim.joint_names.index(name) for name in ARM_NAMES]
        arm_seed = qpos[indices]
        context = {"arm_seed": arm_seed.copy(), "grasp_offset": None}
        if held is None:
            return context
        if held != self.held or held != self.candidate:
            raise GripStateError("Planning requires a verified held object.")
        observed = scene.objects.get(held)
        if observed is None:
            raise GripStateError("Held object is unavailable for path planning.")
        node = self._geometry(observed)
        tcp = np.asarray(self.kin.forward(arm_seed), dtype=float)[:3, 3]
        assumed = max((node.top_z - node.bottom_z) / 2, .012)
        context["grasp_offset"] = self._shift(node, tcp, assumed)
        return context

    def _observe_candidate(self):
        if self.locate is None or self.candidate is None:
            raise GripStateError("Grasp verification requires a candidate and RGB-D locator.")
        frame = self.sim.observe()
        record_artifact(self.recorder, "save_rgbd", frame,
                        note="Optimized TiPToP grip observation")
        try:
            located = self.locate(frame, self.candidate)
        except ValueError as exc:
            self._uncertain = True
            raise GripStateError("Grasp candidate is not reliably visible.") from exc
        return frame, self._geometry(located)

    def _shift(self, node, tcp, assumed):
        return np.array([tcp[0] - node.point[0], tcp[1] - node.point[1],
                         tcp[2] - node.bottom_z - assumed])

    def _verify_lift(self, node, qpos):
        tcp, jaw = self._proprioception(qpos)
        if (self._baseline is None
                or node.point[2] - self._baseline.point[2] < self._LIFT
                or jaw > self._CLOSED or not self._near(node, tcp)
                or self._supported(node) or not self._raised(node)):
            raise GripStateError("Visual observation did not verify a lifted grasp.")
        self.held = self.candidate
        self._uncertain = False
        return tcp

    def _calibrate(self, label):
        spec = json.loads(label)
        if spec["held"] != self.candidate:
            raise GripStateError("Calibration does not match the grasp candidate.")
        assumed = float(spec["assumed"])
        if not np.isfinite(assumed) or assumed <= 0:
            raise ValueError("Invalid assumed grasp offset")
        frame, node = self._observe_candidate()
        tcp = self._verify_lift(node, frame.qpos)
        self._acted = True
        self._assumed = assumed
        return self._shift(node, tcp, assumed)

    def _prepare(self, plan):
        shift = np.zeros(3)
        first = plan.operators[0] if plan.operators else None
        if self.candidate is not None:
            # Use exactly one fresh observation, even after reconcile(scene).
            frame, node = self._observe_candidate()
            held = self._resolve(node, frame.qpos)
            if held is not None:
                if first is not None and first[0] == "pick":
                    raise GripStateError("Cannot replay a pick while holding an object.")
                if first is not None and first[1][0] != held:
                    raise GripStateError("Continuation targets another held object.")
                tcp, _ = self._proprioception(frame.qpos)
                self._acted = True
                planned = (self._scene.objects.get(held)
                           if self._scene is not None else None)
                assumed = (max((planned.top_z - planned.bottom_z) / 2, .012)
                           if planned is not None else self._assumed)
                if assumed is None:
                    assumed = max((node.top_z - node.bottom_z) / 2, .012)
                shift = self._shift(node, tcp, assumed)
        if first is not None and first[0] == "place" and self.held is None:
            raise GripStateError("Place-only continuation requires a verified held object.")
        return shift

    def execute(self, plan):
        self._acted = False
        span = self.recorder.start_span("execute", verification="visual_grip",
                                        steps=len(plan.trajectory))
        result = self._abort("Execution aborted.", "fatal")
        try:
            result = self._execute(plan)
            record_artifact(self.recorder, "save_state",
                            {**result, "candidate": self.candidate,
                             "held": self.held, "uncertain": self._uncertain,
                             "verification": "visual_grip"},
                            f"tiptop-optimized-execution-{getattr(span, 'span_id', 'untraced')}")
            return result
        finally:
            self.recorder.end_span(span, "ok" if result["success"] else "error",
                                   result["error_kind"], result["message"])

    def _execute(self, plan):
        shift = np.zeros(3)
        picks = iter(args[0] for kind, args in plan.operators if kind == "pick")
        released, revalidate = False, False
        for index, step in enumerate(plan.trajectory):
            label = {"move": "移动", "gripper": "夹爪", "calibrate": "抓取验证与标定",
                     "reset": "复位"}.get(step.kind, step.kind)
            self.on_stage(f"执行 {index + 1}/{len(plan.trajectory)}：{label}")
            self.on_event({"type": "activity", "kind": "action",
                           "text": f"第 {index + 1} 步：{label}"})
            self.on_event({"type": "step", "index": index, "status": "active"})
            span = self.recorder.start_span(f"atom:{step.kind}", index=index)
            try:
                if index == 0:
                    shift = self._prepare(plan)
                    if self.held is not None:
                        validate_cartesian_path(
                            self.kin, plan.trajectory, _arm_qpos(self._ctx),
                            shift)
                if step.kind == "move":
                    if revalidate:
                        validate_cartesian_path(
                            self.kin, plan.trajectory[index:],
                            _arm_qpos(self._ctx), shift, index_offset=index)
                        revalidate = False
                    if self.held is not None and not released and (
                            step.jaw is not None and step.jaw >= self._OPEN):
                        raise GripStateError("Cannot open a held grasp during motion.")
                    result = _cartesian(
                        self._ctx, np.asarray(step.position, dtype=float) + shift,
                        jaw=step.jaw)
                    if not result.success:
                        raise _MotionFailure(result.error_kind, "Motion verification failed.")
                elif step.kind == "gripper":
                    if step.jaw is not None and step.jaw <= self._CLOSED:
                        candidate = next(picks, None)
                        if candidate is None:
                            raise GripStateError("Close requires a pick operator.")
                        self.candidate = candidate
                        self.held = self._empty_candidate = None
                        self._uncertain = True
                        _, self._baseline = self._observe_candidate()
                        self._assumed = max(
                            (self._baseline.top_z - self._baseline.bottom_z) / 2, .012)
                    elif self.candidate is not None:
                        if not any(kind == "place" and args[0] == self.candidate
                                   for kind, args in plan.operators):
                            raise GripStateError("Opening a grasp requires a place operator.")
                        self._uncertain = True
                    self.sim.move_right(_arm_qpos(self._ctx), jaw=step.jaw)
                    if step.jaw is not None and step.jaw >= self._OPEN:
                        released = True
                elif step.kind == "reset":
                    if self.candidate is not None and not released:
                        raise GripStateError("Cannot reset an unresolved grasp.")
                    self.sim.move_right(REST_Q, jaw=step.jaw)
                elif step.kind == "calibrate":
                    shift = self._calibrate(step.label)
                    revalidate = True
                else:
                    raise RuntimeError("Unknown trajectory step")
            except Exception as exc:
                failed_index = (exc.step_index
                                if isinstance(exc, CartesianPathError) else index)
                kind = (exc.error_kind
                        if isinstance(exc, (GripStateError, _MotionFailure))
                        else "unreachable" if isinstance(exc, ValueError) else "fatal")
                message = f"执行未完成：{kind}（{type(exc).__name__}）。"
                if self.candidate is not None:
                    self._uncertain = True
                self.recorder.log("error", message, logger="tiptop.optimized.executor",
                                  exc_info=True)
                self._capture_failure(failed_index, span)
                self.recorder.end_span(span, "error", kind, message)
                if failed_index != index:
                    self.on_event({"type": "step", "index": index,
                                   "status": "skipped"})
                self.on_event({"type": "step", "index": failed_index,
                               "status": "failed"})
                for remaining in range(failed_index + 1, len(plan.trajectory)):
                    self.on_event({"type": "step", "index": remaining, "status": "skipped"})
                self.on_event({"type": "activity", "kind": "observation",
                               "text": message + " 后续动作已跳过，交由 ReAct 根据新观测决定恢复。"})
                return self._abort(message, kind, failed_index=failed_index)
            self.recorder.end_span(span, "ok")
            self.on_event({"type": "step", "index": index, "status": "done"})
        return {"success": True, "aborted": False, "message": "Trajectory completed.",
                "error_kind": None, "acted": bool(self._acted),
                "failed_index": None}

    def _capture_failure(self, index, span=None):
        try:
            frame = self.sim.observe()
        except Exception:
            self.recorder.log("warning", "Failure observation unavailable",
                              logger="tiptop.optimized.executor")
            return
        record_artifact(self.recorder, "save_rgbd", frame,
                        note="Optimized TiPToP failure; diagnostic only")
        record_artifact(self.recorder, "save_state",
                        {"index": index, "qpos": np.asarray(frame.qpos).tolist(),
                         "candidate": self.candidate, "held": self.held,
                         "uncertain": self._uncertain, "verification": "visual_grip"},
                        f"tiptop-optimized-failure-{getattr(span, 'span_id', index)}")

    def _abort(self, message, error_kind="fatal", failed_index=None):
        return {"success": False, "aborted": True, "message": message,
                "error_kind": error_kind, "acted": bool(self._acted),
                "failed_index": failed_index}
