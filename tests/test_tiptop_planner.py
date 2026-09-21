"""Tests for the CPU TAMP-lite planner."""
import numpy as np
import pytest

from tiptop_mac.perception import propose_grasps
from tiptop_mac.tamp import TAMPLite, TAMPError
from tiptop_mac.types import (
    MotionStep, ObjectNode, Predicate, SceneGraph, TableSurface,
)

from helpers import FakeKin


def make_node(object_id, kind, xy, top_z=None, bottom_z=None,
              extent_xy=.024, height=.036):
    point = np.array([xy[0], xy[1], (top_z + bottom_z) / 2], dtype=float)
    node = ObjectNode(
        id=object_id, kind=kind, label=f"对象 {object_id}",
        color=(.5, .5, .5), mask=np.zeros((4, 4), bool),
        bbox=(0, 0, 1, 1), point=point, top_z=top_z, bottom_z=bottom_z,
        extent=np.array([extent_xy, extent_xy, height]),
        cloud=point[None, :], grasps=[])
    node.grasps = propose_grasps(node)
    return node


def make_scene(nodes):
    table = TableSurface(normal=np.array([0., 0., 1.]), top_z=.745,
                         bounds=np.array([[-.30, -.67, .74], [.30, -.23, .75]]))
    return SceneGraph(table=table, objects={n.id: n for n in nodes},
                      qpos=np.zeros(6))


def default_scene():
    # A and B on the table, box as container.
    return make_scene([
        make_node("A", "block", (0.0, -.45), .781, .745),
        make_node("B", "block", (.10, -.45), .781, .745),
        make_node("box", "box", (-.15, -.45), .759, .720,
                  extent_xy=.096, height=.039),
    ])


def test_goal_already_true_gives_empty_plan():
    scene = default_scene()
    plan = TAMPLite(FakeKin()).plan(
        scene, (Predicate("on", ("A", "table")),), rationale="A 放桌上")
    assert plan.trajectory == ()
    assert plan.operators == ()


def test_plan_pick_and_place_block_on_block():
    scene = default_scene()
    plan = TAMPLite(FakeKin()).plan(
        scene, (Predicate("on", ("A", "B")),), rationale="A 叠到 B")
    assert plan.operators == (("pick", ("A",)), ("place", ("A", "B")))
    kinds = [s.kind for s in plan.trajectory]
    assert kinds.count("gripper") == 2
    assert kinds.count("calibrate") == 1
    assert kinds[-1] == "move"  # ends with vertical retreat, no stow sweep
    calibrate = next(s for s in plan.trajectory if s.kind == "calibrate")
    assert '"held": "A"' in calibrate.label
    assert plan.planning_time >= 0


def test_plan_into_box_release_height_and_footprint():
    scene = default_scene()
    plan = TAMPLite(FakeKin()).plan(
        scene, (Predicate("on", ("A", "box")),), rationale="入盒")
    box_z = scene.objects["box"].point[2]
    release = [s for s in plan.trajectory
               if s.kind == "move"
               and abs(s.position[2] - (box_z + .055)) < .002]
    assert release, "expected box release at box point z + 55mm"


def test_blocking_stack_release_height():
    scene = default_scene()
    plan = TAMPLite(FakeKin()).plan(
        scene, (Predicate("on", ("A", "B")),), rationale="")
    heights = [s.position[2] for s in plan.trajectory if s.kind == "move"]
    # B top .781 + A half-height .018 + 4mm
    assert any(abs(h - (.781 + .018 + .004)) < 1e-3 for h in heights)


def test_planner_moves_unstacked_object_to_new_support():
    # C currently stacked on A; goal moves C onto B (two operators).
    scene = make_scene([
        make_node("A", "block", (0.0, -.45), .781, .745),
        make_node("B", "block", (.10, -.45), .781, .745),
        make_node("C", "block", (0.0, -.45), .817, .781),
    ])
    plan = TAMPLite(FakeKin()).plan(
        scene, (Predicate("on", ("C", "B")),), rationale="")
    assert plan.operators == (("pick", ("C",)), ("place", ("C", "B")))


def test_holding_goal_ends_while_held_without_reset():
    scene = default_scene()
    plan = TAMPLite(FakeKin()).plan(
        scene, (Predicate("holding", ("A",)),), rationale="拿着 A")
    assert [k for k in plan.trajectory if k.kind == "reset"] == []
    assert plan.operators == (("pick", ("A",)),)


def test_ik_failure_makes_goal_infeasible():
    kin = FakeKin()
    kin.fail_solve = True
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(kin).plan(default_scene(),
                           (Predicate("on", ("A", "B")),), rationale="")


def test_small_support_footprint_rejects_placement():
    tiny = make_node("B", "block", (.10, -.45), .781, .745,
                     extent_xy=.01)
    scene = make_scene([
        make_node("A", "block", (0.0, -.45), .781, .745), tiny,
    ])
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(FakeKin()).plan(scene,
                                 (Predicate("on", ("A", "B")),), rationale="")


def test_object_without_grasp_cannot_be_picked():
    scene = default_scene()
    scene.objects["A"].grasps = []
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(FakeKin()).plan(scene,
                                 (Predicate("on", ("A", "B")),), rationale="")


def test_adjacent_object_collision_rejects_placement():
    # D sits 20mm from B's center; placing A at B must collide with D.
    scene = make_scene([
        make_node("A", "block", (0.0, -.30), .781, .745),
        make_node("B", "block", (0.0, -.50), .781, .745),
        make_node("D", "block", (0.02, -.50), .781, .745),
    ])
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(FakeKin()).plan(scene,
                                 (Predicate("on", ("A", "B")),), rationale="")


def test_circular_support_goal_is_impossible():
    # A on B and B on A cannot both hold at the same time.
    scene = make_scene([
        make_node("A", "block", (-.1, -.45), .781, .745),
        make_node("B", "block", (.10, -.45), .781, .745),
    ])
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(FakeKin()).plan(
            scene, (Predicate("on", ("A", "B")),
                    Predicate("on", ("B", "A"))), rationale="")


def test_trajectory_steps_are_frozen_values():
    plan = TAMPLite(FakeKin()).plan(
        default_scene(), (Predicate("on", ("A", "B")),), rationale="")
    assert all(isinstance(s, MotionStep) for s in plan.trajectory)
