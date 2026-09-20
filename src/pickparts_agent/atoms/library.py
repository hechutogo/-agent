"""Object-level semantic atoms. Coordinates are grounded at runtime, never LLM input."""
import functools

import numpy as np

from pickparts_agent.kinematics import ARM_NAMES
from .base import Atom, AtomResult, Check


class FindObject(Atom):
    name = "find_object"
    description = "Locate an object by RGB-D vision and update the world state."
    parameters = {
        "type": "object",
        "properties": {"target": {"type": "string", "enum": ["A", "B", "box"]}},
        "required": ["target"],
    }

    def check_pre(self, ctx, args):
        return Check(True)

    def run(self, ctx, args):
        target = args["target"]
        frame_id = ctx.state.tick()
        frame = ctx.sim.observe()
        try:
            loc = ctx.locate(frame, target)
        except Exception as exc:
            return AtomResult(False, "lost_object",
                              f"{target} 未定位（{type(exc).__name__}）")
        rec = ctx.state.apply_observation(target, loc, frame_id)
        return AtomResult(True, observed={
            "bbox": rec.bbox, "point": rec.point,
            "confidence": rec.confidence, "frame_id": frame_id})

    def verify(self, ctx, args, result):
        rec = ctx.state.objects.get(args["target"])
        if rec is None or not rec.visible or rec.confidence < 0.7:
            return Check(False, "定位结果不可用")
        return Check(True)


class VerifyState(Atom):
    name = "verify_state"
    description = "Check via vision whether an object is at the table or in the box."
    parameters = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": ["A", "B"]},
            "at": {"type": "string", "enum": ["table", "box"]},
        },
        "required": ["target", "at"],
    }

    def check_pre(self, ctx, args):
        return Check(True)

    def run(self, ctx, args):
        target, at = args["target"], args["at"]
        frame_id = ctx.state.tick()
        frame = ctx.sim.observe()
        try:
            box = ctx.locate(frame, "box")
            ctx.state.apply_observation("box", box, frame_id)
            loc = ctx.locate(frame, target)
        except Exception as exc:
            return AtomResult(False, "lost_object",
                              f"校验时丢失目标（{type(exc).__name__}）")
        rec = ctx.state.apply_observation(target, loc, frame_id)
        return AtomResult(True, observed={
            "placement": rec.placement, "expected": at})

    def verify(self, ctx, args, result):
        placement = result.observed.get("placement")
        if placement != args["at"]:
            return Check(False, f"期望在{args['at']}，实际在{placement}")
        return Check(True)


def _guard(fn):
    """Map low-level exceptions to grounded error kinds for motion atoms."""
    @functools.wraps(fn)
    def run(self, ctx, args):
        try:
            return fn(self, ctx, args)
        except ValueError as exc:
            return AtomResult(False, "unreachable", str(exc))
        except RuntimeError as exc:
            return AtomResult(False, "fatal", str(exc))
    return run


class SetGripper(Atom):
    name = "set_gripper"
    description = "Open or close the right-arm gripper."
    parameters = {
        "type": "object",
        "properties": {"open": {"type": "boolean"}},
        "required": ["open"],
    }

    def check_pre(self, ctx, args):
        return Check(True)

    @_guard
    def run(self, ctx, args):
        frame = ctx.sim.observe()
        idx = [ctx.sim.joint_names.index(n) for n in ARM_NAMES]
        q = np.asarray(frame.qpos)[idx]
        jaw = 0.8 if args["open"] else 0.0
        ctx.sim.move_right(q, jaw=jaw, steps=20)
        ctx.state.set_gripper(args["open"])
        return AtomResult(True, observed={"gripper_open": args["open"]})

    def verify(self, ctx, args, result):
        if ctx.state.gripper_open != args["open"]:
            return Check(False, "夹爪状态未到位")
        return Check(True)
