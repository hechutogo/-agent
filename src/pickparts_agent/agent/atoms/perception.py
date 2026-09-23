"""Perception and relation-verification atoms."""
from ...scene.perception import table_height
from ...scene.support import in_box, on_block, on_table
from ...scene.verification import sample_until_stable
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
        if args["target"] == "table":
            return Check(False, "table 是支撑面，不能作为普通物体定位")
        return Check(True)

    def run(self, ctx, args):
        target = args["target"]
        frame_id = ctx.state.tick()
        frame = ctx.sim.observe()
        recorder = ctx.recorder
        if recorder is None:
            from ...observability import NullRecorder
            recorder = NullRecorder()
        try:
            with recorder.span("vision:locate", target=target):
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
        target, at, relation = (
            args.get("target"), args.get("at"), args.get("relation"))
        if target == "table":
            return Check(False, "table 不能作为待校验的可动物体")
        if relation == "table" and at != "table":
            return Check(False, "桌面关系必须使用 table 作为目的")
        if at == "table" and relation not in (None, "table"):
            return Check(False, "table 目的只能校验桌面关系")
        return Check(True)

    def run(self, ctx, args):
        target, at = args["target"], args["at"]
        relation = args.get("relation", "table" if at == "table" else
                            "in" if ctx.state.is_box(at) else "on")
        if ctx.state.held_object is not None:
            return AtomResult(False, "precondition", "仍在持物，不能判定已放置")
        from .manipulation import _check_unheld_candidate
        unheld = _check_unheld_candidate(ctx)
        if not unheld.ok:
            return AtomResult(False, "precondition", unheld.reason, unheld.facts)
        ctx.sim.hold(15)

        def sample():
            frame_id = ctx.state.tick()
            frame = ctx.sim.observe()
            try:
                dest = (ctx.state.apply_observation(at, ctx.locate(frame, at), frame_id)
                        if at != "table" else None)
                rec = ctx.state.apply_observation(target, ctx.locate(frame, target), frame_id)
                if dest is None:
                    measured_height = table_height(
                        frame, rec.point, rec.bbox, footprint=rec.extent[:2])
                    matched = on_table(rec, measured_height)
                elif relation == "on":
                    matched = on_block(rec, dest)
                else:
                    matched = in_box(rec, dest)
                return rec, frame_id, matched
            except (ValueError, AttributeError):
                ctx.state.mark_missing(target)
                raise

        def verify():
            return sample_until_stable(
                sample, lambda value: value[2], lambda: ctx.sim.hold(8),
                recoverable=(ValueError, AttributeError))

        evidence = verify()
        if (evidence.error is not None and ctx.state.gripper_open
                and ctx.state.grasp_target is None):
            # Only a confirmed open, released gripper may clear the view.
            # Transient noise is handled above without issuing any motion.
            from .placement import ResetArm
            reset = ResetArm().call(ctx, {})
            if not reset.success:
                return reset
            evidence = verify()
        if evidence.error is not None:
            return AtomResult(False, "lost_object",
                              "校验视野持续不完整，无法确认放置关系",
                              observed={"samples": evidence.samples})
        rec, frame_id, _ = evidence.value
        matched = evidence.confirmed
        placement = at if matched else rec.placement
        if matched:
            rec.placement = at if relation != "on" else "on:" + at
            ctx.state.update_object(rec)
        return AtomResult(True, observed={
            "placement": placement, "expected": at, "matched": bool(matched),
            "relation": relation, "frame_id": frame_id, "samples": evidence.samples})

    def verify(self, ctx, args, result):
        placement = result.observed.get("placement")
        if not result.observed.get("matched", placement == args["at"]):
            return Check(False, f"期望在{args['at']}，实际在{placement}")
        return Check(True)
