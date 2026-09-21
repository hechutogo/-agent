"""Perception and relation-verification atoms."""
import numpy as np

from ...scene.perception import table_height
from .base import Atom, AtomResult, Check


class FindObject(Atom):
    name = "find_object"
    description = "Locate an object by RGB-D vision and update the world state."
    parameters = {
        "type": "object",
        "properties": {"target": {"type": "string"}},
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
            ctx.state.mark_missing(target)
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
    description = "Verify a fresh visual relation: target in a container, on an object, or on table."
    parameters = {
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "at": {"type": "string"},
            "relation": {"type": "string", "enum": ["on", "in", "table"]},
        },
        "required": ["target", "at"],
    }

    def check_pre(self, ctx, args):
        return Check(True)

    def run(self, ctx, args):
        target, at = args["target"], args["at"]
        relation = args.get("relation", "table" if at == "table" else
                            "in" if ctx.state.is_box(at) else "on")
        if ctx.state.held_object is not None:
            return AtomResult(False, "precondition", "仍在持物，不能判定已放置")
        frame_id = ctx.state.tick()
        ctx.sim.hold(15)
        for attempt in range(2):
            frame = ctx.sim.observe()
            try:
                dest = (ctx.state.apply_observation(at, ctx.locate(frame, at), frame_id)
                        if at != "table" else None)
                rec = ctx.state.apply_observation(target, ctx.locate(frame, target), frame_id)
                break
            except Exception as exc:
                ctx.state.mark_missing(target)
                if attempt == 0 and ctx.state.gripper_open:
                    from .placement import ResetArm

                    reset = ResetArm().call(ctx, {})
                    if not reset.success:
                        return reset
                    continue
                return AtomResult(False, "lost_object",
                                  f"校验视野不完整（{type(exc).__name__}），不能断言物体仍在桌面")
        if dest is None:
            try:
                measured_height = table_height(frame, rec.point, rec.bbox)
            except (ValueError, AttributeError):
                return AtomResult(False, "lost_object", "桌面支撑面不可见，不能确认桌面接触")
            matched = abs(rec.bottom_z - measured_height) < .004
        elif relation == "on":
            delta = np.asarray(rec.point) - dest.point
            matched = (np.linalg.norm(delta[:2]) < .014
                       and .018 < delta[2] < .055
                       and abs(rec.bottom_z - dest.top_z) < .02)
        else:
            delta = np.asarray(rec.point) - dest.point
            inner_half = np.asarray(dest.extent[:2]) / 2 - .006
            footprint = np.abs(delta[:2]) + np.asarray(rec.extent[:2]) / 2
            matched = (np.all(footprint <= inner_half + .002)
                       and dest.bottom_z - .004 <= rec.bottom_z < dest.top_z - .006)
        placement = at if matched else rec.placement
        if matched:
            rec.placement = at if relation != "on" else "on:" + at
            ctx.state.update_object(rec)
        return AtomResult(True, observed={
            "placement": placement, "expected": at, "matched": bool(matched),
            "relation": relation, "frame_id": frame_id})

    def verify(self, ctx, args, result):
        placement = result.observed.get("placement")
        if not result.observed.get("matched", placement == args["at"]):
            return Check(False, f"期望在{args['at']}，实际在{placement}")
        return Check(True)
