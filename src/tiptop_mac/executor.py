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
from pickparts_agent.observability import NullRecorder


_STAGES = {
    "move": "移动", "gripper": "夹爪", "reset": "复位",
    "calibrate": "固定抓取标定",
}


class _MotionFailure(RuntimeError):
    def __init__(self, error_kind, message):
        super().__init__(message)
        self.error_kind = error_kind


def record_artifact(recorder, method, *args, **kwargs):
    """Artifact I/O must not change the open-loop execution result."""
    try:
        getattr(recorder, method)(*args, **kwargs)
    except Exception:
        recorder.log("warning", "TiPToP artifact capture failed",
                     logger="tiptop")


class OpenLoopExecutor:
    def __init__(self, sim, kin, on_stage=None, *, locate=None,
                 on_event=None, recorder=None):
        self.sim = sim
        self.kin = kin
        self.on_stage = on_stage if on_stage is not None else (lambda s: None)
        self.on_event = on_event if on_event is not None else (lambda e: None)
        self.recorder = recorder if recorder is not None else NullRecorder()
        # Optional local RGB-D locator used only by fixed calibrate steps.
        self.locate = locate
        self._ctx = SimpleNamespace(sim=sim, kin=kin)

    def execute(self, plan):
        span = self.recorder.start_span("execute", verification="open_loop",
                                        steps=len(plan.trajectory))
        result = self._abort("开环执行异常中止。")
        try:
            result = self._execute(plan)
            record_artifact(self.recorder, "save_state",
                            {**result, "verification": "open_loop"},
                            "tiptop-execution")
        finally:
            self.recorder.end_span(
                span, "ok" if result["success"] else "error",
                message=result["message"])
        self.on_event({"type": "finish", "success": result["success"]})
        return result

    def _execute(self, plan):
        if not plan.trajectory:
            return {"success": True, "aborted": False,
                    "message": "目标已满足，无需动作。"}
        shift = np.zeros(3)
        for index, step in enumerate(plan.trajectory):
            kind = step.kind
            stage = _STAGES.get(kind, "未知动作")
            self.on_stage(
                f"开环执行 {index + 1}/{len(plan.trajectory)}：{stage}")
            self.on_event({"type": "activity", "kind": "action",
                           "text": f"开环第 {index + 1} 步：{stage}"})
            self.on_event({"type": "step", "index": index, "status": "active"})
            span = self.recorder.start_span(f"atom:{kind}", index=index)
            try:
                if kind == "move":
                    target = np.asarray(step.position, dtype=float) + shift
                    result = _cartesian(self._ctx, target, jaw=step.jaw)
                    if not result.success:
                        raise _MotionFailure(
                            result.error_kind, "轨迹执行中止：运动未完成。")
                elif kind == "gripper":
                    self.sim.move_right(_arm_qpos(self._ctx), jaw=step.jaw)
                elif kind == "reset":
                    self.sim.move_right(REST_Q, jaw=step.jaw)
                elif kind == "calibrate":
                    measured = self._calibrate(step.label)
                    shift = measured
                else:
                    raise _MotionFailure("unknown_step", "未知的轨迹步骤。")
            except Exception as exc:
                error_kind = (exc.error_kind if isinstance(exc, _MotionFailure)
                              else type(exc).__name__)
                message = (str(exc) if isinstance(exc, _MotionFailure)
                           else f"轨迹执行中止（{type(exc).__name__}）。")
                self.recorder.log("error", message, logger="tiptop.executor",
                                  exc_info=True)
                self._capture_failure(index)
                self.recorder.end_span(span, "error", error_kind, message)
                self.on_event({"type": "step", "index": index, "status": "failed"})
                for remaining in range(index + 1, len(plan.trajectory)):
                    self.on_event({"type": "step", "index": remaining,
                                   "status": "skipped"})
                self.on_event({"type": "activity", "kind": "observation",
                               "text": message + " 已跳过后续步骤，不重试或重规划。"})
                return self._abort(message)
            self.recorder.end_span(span, "ok")
            self.on_event({"type": "step", "index": index, "status": "done"})
        return {"success": True, "aborted": False,
                "message": "开环执行完成。"}

    def _capture_failure(self, index):
        try:
            frame = self.sim.observe()
        except Exception:
            self.recorder.log("warning", "TiPToP failure observation unavailable",
                              logger="tiptop.executor")
            return
        record_artifact(self.recorder, "save_rgbd", frame,
                        note="TiPToP motion failure; diagnostic only")
        record_artifact(self.recorder, "save_state",
                        {"index": index, "status": "failed",
                         "qpos": np.asarray(frame.qpos).tolist(),
                         "verification": "open_loop"},
                        "tiptop-failure")

    def _calibrate(self, label):
        if self.locate is None:
            raise _MotionFailure(
                "calibration", "抓取标定步骤缺少本地 RGB-D 定位器。")
        spec = json.loads(label)
        held, assumed = spec["held"], float(spec["assumed"])
        frame = self.sim.observe()
        record_artifact(self.recorder, "save_rgbd", frame,
                        note="TiPToP fixed grasp calibration; not outcome verification")
        located = self.locate(frame, held)
        tcp = self.kin.forward(_arm_qpos(self._ctx))[:3, 3]
        block = np.asarray(located["point"], dtype=float)
        xy_shift = np.array([tcp[0] - block[0], tcp[1] - block[1]])
        measured_bottom = float(tcp[2] - located["bottom_z"])
        z_shift = measured_bottom - assumed
        return np.array([xy_shift[0], xy_shift[1], z_shift])

    def _abort(self, message):
        return {"success": False, "aborted": True, "message": message}
