"""Release, placement, and rest-posture atoms."""
import numpy as np

from ...scene.perception import table_drop_candidates
from .base import Atom, AtomResult, Check
from .manipulation import (
    REST_Q, _arm_qpos, _cartesian, _check_unheld_candidate, _guard, _solve_path,
)


def _check_grasp_offset(ctx):
    if ctx.state.grasp_offset is None or ctx.state.grasp_bottom_offset is None:
        return Check(False, "持物偏移未经确认，请先 carry_to 重新观测，不能直接放置")
    return Check(True)


class PlaceOnTable(Atom):
    name = "place_on_table"
    description = "Place the held object on a runtime-measured safe empty table area."
    parameters = {"type": "object"}

    def check_pre(self, ctx, args):
        held = ctx.state.held_object
        if held is None or ctx.state.gripper_open:
            return Check(False, "未持有物体，无法放到桌面")
        if ctx.state.objects.get(held) is None:
            return Check(False, "缺少持有物体的视觉尺寸")
        return _check_grasp_offset(ctx)

    @_guard
    def run(self, ctx, args):
        held = ctx.state.held_object
        rec = ctx.state.objects[held]
        footprint = np.maximum(np.asarray(rec.extent[:2], dtype=float), .024)
        frame = ctx.sim.observe()
        try:
            table_z, candidates = table_drop_candidates(
                frame, rec.point, footprint)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return AtomResult(False, "lost_object",
                              f"桌面安全落点不可用（{type(exc).__name__}）")

        offset = np.asarray(ctx.state.grasp_offset or [0., 0., 0.], dtype=float)
        bottom_offset = ctx.state.grasp_bottom_offset or max(.02, rec.extent[2] / 2)
        seed = _arm_qpos(ctx)
        selected = None
        for xy in candidates:
            release = np.array([
                xy[0] + offset[0], xy[1] + offset[1],
                table_z + bottom_offset + .004,
            ])
            hover = release.copy()
            hover[2] = max(.835, release[2] + .07)
            try:
                hover_q = _solve_path(ctx.kin, hover, seed)
                _solve_path(ctx.kin, release, hover_q)
            except ValueError:
                continue
            selected = xy, hover, release
            break
        if selected is None:
            return AtomResult(False, "unreachable", "未找到机械臂可达的桌面安全空位")

        xy, hover, release = selected
        result = _cartesian(ctx, hover)
        if not result.success:
            return result
        result = _cartesian(ctx, release)
        if not result.success:
            return result
        ctx.sim.move_right(_arm_qpos(ctx), jaw=.8, steps=25)
        ctx.state.set_gripper(True)
        result = _cartesian(ctx, hover, jaw=.8)
        if not result.success:
            return result
        ctx.sim.hold(15)
        return AtomResult(True, observed={
            "placement_xy": [float(xy[0]), float(xy[1])],
            "table_height": float(table_z),
            "gripper_open": True,
            "held": None,
        })

    def verify(self, ctx, args, result):
        return Check(ctx.state.held_object is None and ctx.state.gripper_open,
                     "物体未脱离夹爪")


class ReleaseInto(Atom):
    name = "release_into"
    description = "Open the gripper over the container so the held object drops in."
    parameters = {
        "type": "object",
        "properties": {"container": {"type": "string"}},
        "required": ["container"],
    }

    def check_pre(self, ctx, args):
        if ctx.state.held_object is None:
            return Check(False, "未持有物体，无法释放")
        if ctx.state.objects.get(args["container"]) is None:
            return Check(False, "尚未定位目标容器")
        if not ctx.state.is_box(args["container"]):
            return Check(False, "目标不是盒子，请用 place_on 放到物体上")
        return _check_grasp_offset(ctx)

    @_guard
    def run(self, ctx, args):
        box = np.array(ctx.state.objects[args["container"]].point, dtype=float)
        release = np.array([box[0], box[1], box[2] + 0.055])
        if ctx.state.grasp_offset is not None:
            release[:2] += np.asarray(ctx.state.grasp_offset[:2])
        result = _cartesian(ctx, release)
        if not result.success:
            return result
        ctx.sim.move_right(_arm_qpos(ctx), jaw=0.8, steps=20)
        ctx.state.set_gripper(True)
        ctx.state.set_held(None)
        up = release.copy()
        up[2] = max(.835, release[2] + .06)
        retreat = _cartesian(ctx, up, jaw=0.8)
        if not retreat.success:
            return retreat
        ctx.sim.hold(15)
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
        if ctx.state.held_object is not None:
            return Check(False, "仍在持物，不能张爪复位")
        return _check_unheld_candidate(ctx)

    @_guard
    def run(self, ctx, args):
        ctx.sim.move_right(REST_Q, jaw=0.8)
        ctx.state.set_gripper(True)
        return AtomResult(True, observed={"rest": REST_Q.tolist()})

    def verify(self, ctx, args, result):
        if np.max(np.abs(_arm_qpos(ctx) - REST_Q)) > 0.05:
            return Check(False, "右臂未回到 rest 姿态")
        return Check(True)


class PlaceOn(Atom):
    name = "place_on"
    description = "Place the held object on top of a located support object; then verify_state relation=on."
    parameters = {"type": "object", "properties": {"target": {"type": "string"}},
                  "required": ["target"]}

    def check_pre(self, ctx, args):
        target = args["target"]
        if target == "table":
            return Check(False, "桌面放置必须使用 place_on_table")
        rec = ctx.state.objects.get(target)
        if ctx.state.held_object is None:
            return Check(False, "未持有物体")
        if target == ctx.state.held_object:
            return Check(False, "不能叠放到自身")
        if rec is None or not rec.visible:
            return Check(False, "尚未定位支撑物体")
        if ctx.state.is_box(target):
            return Check(False, "开放盒子请用 release_into")
        return _check_grasp_offset(ctx)

    @_guard
    def run(self, ctx, args):
        support = ctx.state.objects[args["target"]]
        release = np.array(support.point, dtype=float)
        if ctx.state.grasp_offset is not None:
            release[:2] += np.asarray(ctx.state.grasp_offset[:2])
        release[2] = support.top_z + (ctx.state.grasp_bottom_offset or .02) + .004
        result = _cartesian(ctx, release)
        if not result.success:
            return result
        ctx.sim.move_right(_arm_qpos(ctx), jaw=.8, steps=25)
        ctx.state.set_gripper(True)
        ctx.state.set_held(None)
        release[2] += .07
        result = _cartesian(ctx, release, jaw=.8)
        ctx.sim.hold(15)
        return result

    def verify(self, ctx, args, result):
        return Check(ctx.state.held_object is None and ctx.state.gripper_open,
                     "尚未释放物体")
