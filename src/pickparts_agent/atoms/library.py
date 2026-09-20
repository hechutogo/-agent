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
        ctx.sim.hold(10)  # let a released/stacked object settle before judging
        frame = ctx.sim.observe()
        try:
            box = ctx.locate(frame, "box")
            ctx.state.apply_observation("box", box, frame_id)
        except Exception as exc:
            return AtomResult(False, "lost_object",
                              f"校验时丢失盒子（{type(exc).__name__}）")
        try:
            loc = ctx.locate(frame, target)
            rec = ctx.state.apply_observation(target, loc, frame_id)
            placement = rec.placement
            occluded = False
        except Exception as exc:
            # The target is not visible (stacked/occluded inside the box).
            # Trust the last grounded fact when it already places the target
            # where expected, instead of re-grasping an achieved goal.
            prior = ctx.state.objects.get(target)
            if at == "box" and prior is not None and prior.placement == "box":
                placement = "box"
                occluded = True
            else:
                return AtomResult(False, "lost_object",
                                  f"校验时丢失目标（{type(exc).__name__}）")
        return AtomResult(True, observed={
            "placement": placement, "expected": at, "occluded": occluded})

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


REST_Q = np.array([0.0, 2.6, 2.8, 0.0, 1.57])


def _arm_qpos(ctx):
    frame = ctx.sim.observe()
    idx = [ctx.sim.joint_names.index(n) for n in ARM_NAMES]
    return np.asarray(frame.qpos)[idx]


def _cartesian(ctx, position, jaw=None):
    position = np.asarray(position, dtype=float)
    seed = _arm_qpos(ctx)
    pose = ctx.kin.forward(seed)
    dist = float(np.linalg.norm(position - pose[:3, 3]))
    if pose[2, 1] > 0.97 and dist > 0.008:
        count = max(2, int(np.ceil(dist / 0.008)))
        wps = np.linspace(pose[:3, 3], position, count + 1)[1:]
    else:
        wps = [position]
    for wp in wps:
        wp = np.asarray(wp, dtype=float)
        q = ctx.kin.solve(wp, seed=seed)
        ctx.sim.move_right(q, jaw=jaw, steps=6 if len(wps) > 1 else 35)
        seed = _arm_qpos(ctx)
        err = float(np.linalg.norm(ctx.kin.forward(seed)[:3, 3] - wp))
        if err > 0.008:
            return AtomResult(False, "verify", f"TCP tracking error {err:.3f} m")
    return AtomResult(True, observed={"tcp": [float(v) for v in wps[-1]]})


class ReachAbove(Atom):
    name = "reach_above"
    description = "Move the gripper to a hover point above an object."
    parameters = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": ["A", "B", "box"]},
            "clearance": {"type": "number"},
        },
        "required": ["target"],
    }

    def check_pre(self, ctx, args):
        rec = ctx.state.objects.get(args["target"])
        if rec is None or not rec.visible:
            return Check(False, f"尚未定位 {args['target']}")
        if ctx.state.held_object not in (None, args["target"]):
            return Check(False, "机械臂正持有其他物体")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        rec = ctx.state.objects[args["target"]]
        clearance = args.get("clearance", 0.10)
        hover = np.array(rec.point, dtype=float)
        hover[2] += clearance
        result = _cartesian(ctx, hover, jaw=0.8)
        if result.success:
            result.observed["hover"] = [float(v) for v in hover]
        return result

    def verify(self, ctx, args, result):
        target = result.observed.get("hover")
        tcp = ctx.kin.forward(_arm_qpos(ctx))[:3, 3]
        if target is None or np.linalg.norm(tcp - np.array(target)) > 0.008:
            return Check(False, "未到达目标上方")
        return Check(True)


class Grasp(Atom):
    name = "grasp"
    description = "Descend to an object and close the gripper onto it."
    parameters = {
        "type": "object",
        "properties": {"target": {"type": "string", "enum": ["A", "B"]}},
        "required": ["target"],
    }

    def check_pre(self, ctx, args):
        rec = ctx.state.objects.get(args["target"])
        if rec is None or not rec.visible:
            return Check(False, f"尚未定位 {args['target']}")
        if ctx.state.held_object is not None:
            return Check(False, "机械臂已持有物体")
        if not ctx.state.gripper_open:
            return Check(False, "夹爪未张开")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        rec = ctx.state.objects[args["target"]]
        grasp = np.array(rec.point, dtype=float)
        grasp[2] -= 0.004  # RGB-D sees the facing surface; grip just below it.
        result = _cartesian(ctx, grasp, jaw=0.8)
        if not result.success:
            return result
        ctx.sim.move_right(_arm_qpos(ctx), jaw=0.0, steps=25)
        ctx.state.set_gripper(False)
        return AtomResult(True, observed={"gripper_open": False})

    def verify(self, ctx, args, result):
        if ctx.state.gripper_open:
            return Check(False, "夹爪未闭合")
        return Check(True)


class Lift(Atom):
    name = "lift"
    description = "Lift a gripped object and visually confirm the height gain."
    parameters = {
        "type": "object",
        "properties": {"clearance": {"type": "number"}},
    }

    def check_pre(self, ctx, args):
        if ctx.state.gripper_open:
            return Check(False, "夹爪未夹持，无法抬起")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        target = None
        # Identify the gripped candidate from the most recent visible part record.
        for candidate in ("A", "B"):
            rec = ctx.state.objects.get(candidate)
            if rec is not None and rec.visible:
                target = candidate
                break
        if target is None:
            return AtomResult(False, "lost_object", "没有可抬起的零件记录")
        start_z = float(rec.point[2])
        up = np.array([rec.point[0], rec.point[1], start_z + args.get("clearance", 0.10)])
        result = _cartesian(ctx, up)
        if not result.success:
            return result
        frame = ctx.sim.observe()
        try:
            loc = ctx.locate(frame, target)
        except Exception as exc:
            return AtomResult(False, "grip_failed", f"抬起后丢失目标（{type(exc).__name__}）")
        lift_m = float(loc["point"][2] - start_z)
        if lift_m < 0.035:
            return AtomResult(False, "grip_failed",
                              f"视觉未确认抓起：高度变化 {lift_m:.3f} m")
        ctx.state.apply_observation(target, loc, ctx.state.frame_id)
        ctx.state.set_held(target)
        return AtomResult(True, observed={"lift_m": lift_m})

    def verify(self, ctx, args, result):
        if ctx.state.held_object is None or result.observed.get("lift_m", 0) < 0.035:
            return Check(False, "未确认抓起")
        return Check(True)


class CarryTo(Atom):
    name = "carry_to"
    description = "Carry a held object to a point above the destination container."
    parameters = {
        "type": "object",
        "properties": {"container": {"type": "string", "enum": ["box"]}},
        "required": ["container"],
    }

    def check_pre(self, ctx, args):
        if ctx.state.held_object is None:
            return Check(False, "未持有物体，无法搬运")
        box = ctx.state.objects.get(args["container"])
        if box is None or not box.visible:
            return Check(False, "尚未定位目标容器")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        box = np.array(ctx.state.objects[args["container"]].point, dtype=float)
        target = np.array([box[0], box[1], box[2] + 0.109])
        result = _cartesian(ctx, target)
        if result.success:
            result.observed["target_tcp"] = [float(v) for v in target]
        return result

    def verify(self, ctx, args, result):
        target = result.observed.get("target_tcp")
        tcp = ctx.kin.forward(_arm_qpos(ctx))[:3, 3]
        if target is None or np.linalg.norm(tcp - np.array(target)) > 0.008:
            return Check(False, "未到达容器上方")
        return Check(True)


class ReleaseInto(Atom):
    name = "release_into"
    description = "Open the gripper over the container so the held object drops in."
    parameters = {
        "type": "object",
        "properties": {"container": {"type": "string", "enum": ["box"]}},
        "required": ["container"],
    }

    def check_pre(self, ctx, args):
        if ctx.state.held_object is None:
            return Check(False, "未持有物体，无法释放")
        if ctx.state.objects.get(args["container"]) is None:
            return Check(False, "尚未定位目标容器")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        box = np.array(ctx.state.objects[args["container"]].point, dtype=float)
        release = np.array([box[0], box[1], box[2] + 0.055])
        result = _cartesian(ctx, release)
        if not result.success:
            return result
        ctx.sim.move_right(_arm_qpos(ctx), jaw=0.8, steps=20)
        ctx.state.set_gripper(True)
        ctx.state.set_held(None)
        up = np.array([box[0], box[1], box[2] + 0.109])
        _cartesian(ctx, up, jaw=0.8)
        return AtomResult(True, observed={"gripper_open": True, "held": None})

    def verify(self, ctx, args, result):
        if not ctx.state.gripper_open or ctx.state.held_object is not None:
            return Check(False, "物体未脱离夹爪")
        return Check(True)


class ResetArm(Atom):
    name = "reset_arm"
    description = "Return the right arm to its rest posture."
    parameters = {"type": "object"}

    def check_pre(self, ctx, args):
        return Check(True)

    @_guard
    def run(self, ctx, args):
        ctx.sim.move_right(REST_Q, jaw=0.8)
        return AtomResult(True, observed={"rest": REST_Q.tolist()})

    def verify(self, ctx, args, result):
        if np.max(np.abs(_arm_qpos(ctx) - REST_Q)) > 0.05:
            return Check(False, "右臂未回到 rest 姿态")
        return Check(True)
