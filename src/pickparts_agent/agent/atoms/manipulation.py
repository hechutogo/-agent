"""Grounded gripper and transport atoms."""
import functools

import numpy as np

from ...scene.kinematics import ARM_NAMES
from .base import Atom, AtomResult, Check
from .perception import FindObject


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


REST_Q = np.array([0.0, 2.6, 2.8, 0.0, 1.57])


def _arm_qpos(ctx):
    frame = ctx.sim.observe()
    idx = [ctx.sim.joint_names.index(n) for n in ARM_NAMES]
    return np.asarray(frame.qpos)[idx]


def _check_unheld_candidate(ctx):
    """Resolve a missed grasp from fresh sensor separation, without actuation."""
    target = ctx.state.grasp_target
    if ctx.state.held_object is not None:
        return Check(False, "仍在持物，必须通过放置原子安全释放")
    if target is None:
        return Check(True)
    facts = {"grasp_target": target}
    try:
        frame = ctx.sim.observe()
        frame_id = ctx.state.tick()
        idx = [ctx.sim.joint_names.index(n) for n in ARM_NAMES]
        joints = np.asarray(frame.qpos, dtype=float)[idx]
        tcp = ctx.kin.forward(joints)[:3, 3]
        loc = ctx.locate(frame, target)
        point = np.asarray(loc["point"], dtype=float)
        extent = np.asarray(loc["extent"], dtype=float)
        confidence = float(loc["confidence"])
        if (point.shape != (3,) or extent.shape != (3,)
                or not np.isfinite(point).all() or not np.isfinite(extent).all()
                or not np.isfinite(joints).all() or not np.isfinite(tcp).all()
                or np.any(extent <= 0) or not np.isfinite(confidence)
                or confidence < .7):
            return Check(False, "抓取候选的视觉或 TCP 观测无效，不能判定空爪", facts)
        # Visible points need not be centers. Reserve a full measured diagonal
        # plus jaw/tracking clearance, with an 80 mm minimum separation.
        required = max(.08, float(np.linalg.norm(extent)) + .04)
        distance = float(np.linalg.norm(tcp - point))
        facts.update(target_point=point.tolist(), actual_tcp=tcp.tolist(),
                     separation_m=distance, required_separation_m=required,
                     frame_id=frame_id)
        ctx.state.apply_observation(target, loc, frame_id)
    except Exception as exc:
        return Check(False, f"抓取候选状态未知（{type(exc).__name__}），不能判定空爪",
                     facts)
    if distance <= required:
        return Check(False, "抓取候选尚未与 TCP 充分分离，不能判定空爪", facts)
    with ctx.state.lock:
        ctx.state.grasp_target = None
        ctx.state.grasp_offset = None
        ctx.state.grasp_bottom_offset = None
    return Check(True, facts=facts)


def _cartesian_waypoints(kin, position, seed):
    position = np.asarray(position, dtype=float)
    pose = kin.forward(seed)
    dist = float(np.linalg.norm(position - pose[:3, 3]))
    if pose[2, 1] > 0.97 and dist > 0.008:
        count = max(2, int(np.ceil(dist / 0.008)))
        return np.linspace(pose[:3, 3], position, count + 1)[1:]
    return [position]


def _solve_path(kin, position, seed):
    for waypoint in _cartesian_waypoints(kin, position, seed):
        seed = kin.solve(waypoint, seed=seed)
    return seed


def _cartesian(ctx, position, jaw=None):
    position = np.asarray(position, dtype=float)
    seed = _arm_qpos(ctx)
    wps = _cartesian_waypoints(ctx.kin, position, seed)
    # Validate the complete interpolated segment before the first actuation.
    # Execution still resolves from actual joints to account for tracking lag.
    _solve_path(ctx.kin, position, seed)
    for index, wp in enumerate(wps):
        wp = np.asarray(wp, dtype=float)
        q = ctx.kin.solve(wp, seed=seed)
        ctx.sim.move_right(q, jaw=jaw, steps=6 if len(wps) > 1 else 35)
        seed = _arm_qpos(ctx)
        err = float(np.linalg.norm(ctx.kin.forward(seed)[:3, 3] - wp))
        if .003 < err < .015:
            ctx.sim.hold(8)
            seed = _arm_qpos(ctx)
            err = float(np.linalg.norm(ctx.kin.forward(seed)[:3, 3] - wp))
            if (.003 < err < .015 and hasattr(ctx.kin, "lower")
                    and np.max(np.abs(q - seed)) < .04):
                corrected = np.clip(q + (q - seed), ctx.kin.lower, ctx.kin.upper)
                ctx.sim.move_right(corrected, jaw=jaw, steps=12)
                seed = _arm_qpos(ctx)
                err = float(np.linalg.norm(ctx.kin.forward(seed)[:3, 3] - wp))
        if err > 0.008:
            return AtomResult(
                False, "verify", f"TCP tracking error {err:.3f} m",
                observed={"waypoint_index": index, "target_tcp": wp.tolist(),
                          "actual_tcp": ctx.kin.forward(seed)[:3, 3].tolist(),
                          "error_m": err, "commanded_joints": np.asarray(q).tolist(),
                          "actual_joints": np.asarray(seed).tolist()})
    return AtomResult(True, observed={"tcp": [float(v) for v in wps[-1]]})


class SetGripper(Atom):
    name = "set_gripper"
    description = "Open or close the right-arm gripper."
    parameters = {
        "type": "object",
        "properties": {"open": {"type": "boolean"}},
        "required": ["open"],
    }

    def check_pre(self, ctx, args):
        if args["open"] and ctx.state.held_object is not None:
            return Check(False, "持物时必须通过放置原子安全释放")
        return _check_unheld_candidate(ctx) if args["open"] else Check(True)

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


class ReachAbove(Atom):
    name = "reach_above"
    description = "Move the gripper to a hover point above an object."
    parameters = {
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "clearance": {"type": "number", "minimum": .05, "maximum": .15},
        },
        "required": ["target"],
    }

    def check_pre(self, ctx, args):
        rec = ctx.state.objects.get(args["target"])
        if rec is None or not rec.visible:
            return Check(False, f"尚未定位 {args['target']}")
        if ctx.state.held_object is not None:
            return Check(False, "机械臂正持物，请使用 carry_to 搬运")
        return _check_unheld_candidate(ctx)

    @_guard
    def run(self, ctx, args):
        rec = ctx.state.objects[args["target"]]
        clearance = args.get("clearance", 0.10)
        hover = np.array(rec.point, dtype=float)
        hover[2] += clearance
        result = _cartesian(ctx, hover, jaw=0.8)
        if result.success:
            ctx.state.set_gripper(True)
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
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
    }

    def check_pre(self, ctx, args):
        rec = ctx.state.objects.get(args["target"])
        if rec is None or not rec.visible:
            return Check(False, f"尚未定位 {args['target']}")
        if ctx.state.is_box(args["target"]):
            return Check(False, "盒子尺寸超过夹爪开度，可作为放置容器")
        if max(rec.extent[:2]) > .045:
            return Check(False, "视觉测得物体超过当前夹爪可操作尺寸")
        if ctx.state.held_object is not None:
            return Check(False, "机械臂已持有物体")
        if not ctx.state.gripper_open:
            return Check(False, "夹爪未张开")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        rec = ctx.state.objects[args["target"]]
        grasp = np.array(rec.point, dtype=float)
        grasp[2] -= 0.004
        result = _cartesian(ctx, grasp, jaw=0.8)
        if not result.success:
            return result
        q = _arm_qpos(ctx)
        # Register the candidate before issuing a close: an interrupted call
        # may already have actuated. Only Lift may confirm actual possession.
        ctx.state.set_gripper(False)
        ctx.state.grasp_target = args["target"]
        ctx.sim.move_right(q, jaw=0.0, steps=20)
        ctx.sim.hold(20)
        actual = _arm_qpos(ctx)
        tcp = ctx.kin.forward(actual)[:3, 3]
        error = float(np.linalg.norm(tcp - grasp))
        observed = {
            "gripper_open": False, "target_tcp": grasp.tolist(),
            "actual_tcp": tcp.tolist(), "error_m": error,
            "commanded_joints": np.asarray(q).tolist(),
            "actual_joints": np.asarray(actual).tolist(),
        }
        if not np.isfinite(error) or error > .008:
            return AtomResult(False, "verify", f"TCP tracking error {error:.3f} m",
                              observed=observed)
        return AtomResult(True, observed=observed)

    def verify(self, ctx, args, result):
        if ctx.state.gripper_open:
            return Check(False, "夹爪未闭合")
        return Check(True)


class Lift(Atom):
    name = "lift"
    description = "Lift a gripped object and visually confirm the height gain."
    parameters = {
        "type": "object",
        "properties": {"clearance": {"type": "number", "minimum": .05, "maximum": .15}},
    }

    def check_pre(self, ctx, args):
        if ctx.state.gripper_open:
            return Check(False, "夹爪未夹持，无法抬起")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        target = ctx.state.grasp_target
        rec = ctx.state.objects.get(target)
        if rec is None:
            return AtomResult(False, "precondition", "必须先 grasp 明确待抬起物体")
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
        ctx.state.set_held(target)
        lifted = ctx.state.apply_observation(target, loc, ctx.state.tick())
        tcp = ctx.kin.forward(_arm_qpos(ctx))[:3, 3]
        ctx.state.grasp_offset = (tcp - np.asarray(lifted.point)).tolist()
        ctx.state.grasp_bottom_offset = float(tcp[2] - lifted.bottom_z)
        return AtomResult(True, observed={"lift_m": lift_m})

    def verify(self, ctx, args, result):
        if ctx.state.held_object is None or result.observed.get("lift_m", 0) < 0.035:
            return Check(False, "未确认抓起")
        return Check(True)


class CarryTo(Atom):
    name = "carry_to"
    description = "Carry a held object above any located destination object or container."
    parameters = {
        "type": "object",
        "properties": {"container": {"type": "string"}},
        "required": ["container"],
    }

    def check_pre(self, ctx, args):
        if ctx.state.held_object is None:
            return Check(False, "未持有物体，无法搬运")
        destination = ctx.state.objects.get(args["container"])
        if destination is None or not destination.visible:
            return Check(False, "尚未定位目标容器")
        if ctx.state.held_object == args["container"]:
            return Check(False, "不能把物体放到自身")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        ctx.sim.hold(15)
        refreshed = FindObject().call(ctx, {"target": args["container"]})
        if not refreshed.success:
            return refreshed
        rec = ctx.state.objects[args["container"]]
        destination = np.array(rec.point, dtype=float)
        target = np.array([destination[0], destination[1],
                           max(.835, rec.top_z + .075)])
        offset = ctx.state.grasp_offset
        if offset is not None:
            target[:2] += np.asarray(offset[:2])
        # Motion may rotate the wrist or stop midway; old world offsets must
        # not authorize a later release until fresh hover evidence replaces them.
        ctx.state.grasp_offset = None
        ctx.state.grasp_bottom_offset = None
        result = _cartesian(ctx, target)
        if not result.success:
            return result
        result.observed["target_tcp"] = [float(v) for v in target]
        held = ctx.state.held_object
        frame = ctx.sim.observe()
        frame_id = ctx.state.tick()
        idx = [ctx.sim.joint_names.index(n) for n in ARM_NAMES]
        joints = np.asarray(frame.qpos, dtype=float)[idx]
        tcp = ctx.kin.forward(joints)[:3, 3]
        try:
            loc = ctx.locate(frame, held)
            point = np.asarray(loc["point"], dtype=float)
            bottom = float(loc["bottom_z"])
            confidence = float(loc["confidence"])
            if (point.shape != (3,) or not np.isfinite(point).all()
                    or not np.isfinite(joints).all() or not np.isfinite(tcp).all()
                    or not np.isfinite(bottom) or not np.isfinite(confidence)
                    or confidence < .7):
                raise ValueError("Invalid held-object measurement")
            shift = tcp - point
            bottom_shift = float(tcp[2] - bottom)
            result.observed.update(
                held=held, actual_tcp=tcp.tolist(), target_point=point.tolist(),
                separation_m=float(np.linalg.norm(shift)), frame_id=frame_id)
            if np.linalg.norm(shift) > .06 or not 0. <= bottom_shift <= .08:
                return AtomResult(False, "grip_failed",
                                  "搬运后目标不在 TCP 附近，禁止继续放置",
                                  observed=result.observed)
            ctx.state.apply_observation(held, loc, frame_id)
        except Exception as exc:
            return AtomResult(False, "lost_object",
                              f"搬运后持物观测不可用（{type(exc).__name__}），禁止继续放置",
                              observed=result.observed)
        ctx.state.grasp_offset = shift.tolist()
        ctx.state.grasp_bottom_offset = bottom_shift
        result.observed.update(grasp_offset=shift.tolist(),
                               grasp_bottom_offset=bottom_shift)
        return result

    def verify(self, ctx, args, result):
        target = result.observed.get("target_tcp")
        tcp = ctx.kin.forward(_arm_qpos(ctx))[:3, 3]
        if target is None or np.linalg.norm(tcp - np.array(target)) > 0.008:
            return Check(False, "未到达容器上方")
        return Check(True)
