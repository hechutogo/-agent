"""Measured-geometry regressions for ordered, single-subgoal TAMP."""
import importlib
import os

import numpy as np
import pytest

from helpers import FakeKin
from test_tiptop_planner import default_scene, make_node, make_scene
from tiptop_mac.tamp import TAMPError, _Infeasible
from tiptop_mac.types import Predicate, TableSurface


# Allows the same behavioral regressions to demonstrate the retained baseline.
planner_module = importlib.import_module(
    os.environ.get("TIPTOP_PLANNER_UNDER_TEST", "tiptop_optimized.tamp"))
TAMPLite = planner_module.TAMPLite


def on(target):
    return (Predicate("on", ("A", target)),)


def inside_box_scene():
    return make_scene([
        make_node("A", "block", (-.15, -.45), .764, .728),
        make_node("box", "box", (-.15, -.45), .759, .720,
                  extent_xy=.096, height=.039),
    ])


def release_position(plan):
    opening = next(i for i, step in enumerate(plan.trajectory)
                   if step.kind == "gripper" and step.jaw == .8)
    return plan.trajectory[opening - 1].position


def test_measured_box_support_is_not_table():
    planner = TAMPLite(FakeKin())
    scene = inside_box_scene()
    assert planner._initial_support(scene)["A"] == "box"


def test_airborne_geometry_is_not_table_support():
    scene = make_scene([
        make_node("A", "block", (0., -.45), .90, .864),
    ])
    assert TAMPLite(FakeKin())._initial_support(scene).get("A") is None


def test_box_to_table_is_an_actual_collision_free_move():
    scene = inside_box_scene()
    plan = TAMPLite(FakeKin()).plan(scene, on("table"))
    assert plan.operators == (("pick", ("A",)), ("place", ("A", "table")))
    release = release_position(plan)
    # Leave settling clearance rather than pushing the clamped block into table.
    assert release[2] == pytest.approx(.769)
    # Container rim must clear the gripper as well as the small held block.
    assert np.any(np.abs(release[:2] - [-.15, -.45]) >= .100)
    assert np.all(release[:2] >= [-.278, -.648])
    assert np.all(release[:2] <= [.278, -.252])


def test_ordered_box_table_box_uses_fresh_observations():
    planner = TAMPLite(FakeKin())
    first = planner.plan(inside_box_scene(), on("table"))
    assert first.operators == (("pick", ("A",)), ("place", ("A", "table")))
    # Independently supplied next RGB-D estimate, not symbolic plan state.
    measured = make_scene([
        make_node("A", "block", (.08, -.40), .781, .745),
        make_node("box", "box", (-.15, -.45), .759, .720,
                  extent_xy=.096, height=.039),
    ])
    second = planner.plan(measured, on("box"))
    assert second.operators == (("pick", ("A",)), ("place", ("A", "box")))
    np.testing.assert_allclose(second.trajectory[1].position, [.08, -.40, .759])
    assert planner.satisfies(inside_box_scene(), on("box"))


@pytest.mark.parametrize("target,scene_factory", [
    ("table", default_scene), ("box", inside_box_scene),
])
def test_require_action_returns_to_identical_support(target, scene_factory):
    plan = TAMPLite(FakeKin()).plan(
        scene_factory(), on(target), require_action=True, rationale="repeat")
    assert plan.operators == (("pick", ("A",)), ("place", ("A", target)))
    assert plan.trajectory
    assert plan.rationale == "repeat"


@pytest.mark.parametrize("target,scene_factory", [
    ("table", default_scene), ("box", inside_box_scene),
])
def test_ordinary_satisfied_goal_is_noop(target, scene_factory):
    plan = TAMPLite(FakeKin()).plan(scene_factory(), on(target))
    assert plan.operators == ()
    assert plan.trajectory == ()


def test_satisfied_explicitly_requires_intermediate_history():
    planner = TAMPLite(FakeKin())
    state = planner_module._State(None, (("A", "table"), ("B", "table")))
    assert planner._satisfied(state, on("table"))
    assert not planner._satisfied(state, on("table"), require_action=True)
    picked = next(s for op, s in planner._successors(default_scene(), state)
                  if op == ("pick", ("A",)))
    returned = next(s for op, s in planner._successors(default_scene(), picked)
                    if op == ("place", ("A", "table")))
    assert returned != state
    assert planner._satisfied(returned, on("table"), require_action=True)
    assert planner._satisfied(returned, (Predicate("on", ("B", "table")),))
    assert not planner._satisfied(
        returned, (Predicate("on", ("B", "table")),), require_action=True)


def test_held_continuation_places_without_opening_or_repicking():
    scene = default_scene()
    scene.objects["A"].grasps = []
    plan = TAMPLite(FakeKin()).plan(
        scene, on("box"), held="A", require_action=True)
    assert plan.operators == (("place", ("A", "box")),)
    assert [step.kind for step in plan.trajectory] == [
        "move", "move", "gripper", "move",
    ]
    assert [step.jaw for step in plan.trajectory[:2]] == [0., 0.]
    assert plan.trajectory[2].jaw == .8


def test_holding_verification_overrides_apparent_support():
    planner = TAMPLite(FakeKin())
    goal = (Predicate("holding", ("A",)),)
    assert planner.satisfies(default_scene(), goal, held="A")
    assert not planner.satisfies(default_scene(), on("table"), held="A")
    assert planner.plan(default_scene(), goal, held="A").operators == ()
    assert planner.plan(default_scene(), goal).operators == (("pick", ("A",)),)
    assert planner.observed_support(inside_box_scene()) == {"A": "box"}
    assert planner.observed_support(inside_box_scene(), held="A") == {}


def test_airborne_above_box_is_not_contained():
    scene = inside_box_scene()
    scene.objects["A"] = make_node(
        "A", "block", (-.15, -.45), .850, .814)
    planner = TAMPLite(FakeKin())
    assert planner.observed_support(scene) == {}
    assert not planner.satisfies(scene, on("box"))
    assert not planner.satisfies(scene, on("table"))


def test_table_support_requires_measured_bounds():
    scene = make_scene([
        make_node("A", "block", (.5, -.45), .781, .745),
    ])
    assert not TAMPLite(FakeKin()).satisfies(scene, on("table"))


def test_table_sampling_skips_unreachable_candidates():
    class RestrictedKin(FakeKin):
        def solve(self, position, *args, **kwargs):
            if position[0] < .05:
                raise ValueError("outside measured reachable workspace")
            return super().solve(position, *args, **kwargs)

    plan = TAMPLite(RestrictedKin()).plan(
        inside_box_scene(), on("table"), held="A")
    assert release_position(plan)[0] >= .05


def test_no_free_measured_table_space_is_infeasible():
    scene = inside_box_scene()
    narrow = TableSurface(
        normal=np.array([0., 0., 1.]), top_z=.745,
        bounds=np.array([[-.19, -.49, .74], [-.11, -.41, .75]]))
    scene = type(scene)(table=narrow, objects=scene.objects, qpos=scene.qpos)
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(FakeKin()).plan(scene, on("table"))


def test_multiple_goals_are_rejected_before_stale_geometry_can_be_used():
    with pytest.raises(TAMPError, match="single"):
        TAMPLite(FakeKin()).plan(
            default_scene(), on("box") + (Predicate("on", ("B", "table")),))


def test_supporting_block_is_not_picked_out_from_under_stack():
    scene = make_scene([
        make_node("A", "block", (0., -.45), .781, .745),
        make_node("B", "block", (0., -.45), .817, .781),
    ])
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(FakeKin()).plan(scene, (Predicate("holding", ("A",)),))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_measured_geometry_cannot_produce_verified_goal(invalid):
    scene = default_scene()
    scene.objects["A"].point[0] = invalid
    with pytest.raises(TAMPError):
        TAMPLite(FakeKin()).plan(scene, on("table"))


def test_inherited_failure_types_and_ik_checks_are_preserved():
    assert planner_module.TAMPError is TAMPError
    assert planner_module._Infeasible is _Infeasible
    kin = FakeKin()
    kin.fail_solve = True
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(kin).plan(default_scene(), on("box"))


def test_table_continuation_has_reachable_urdf_waypoints():
    from pickparts_agent.scene.kinematics import ArmKinematics

    kin = ArmKinematics()
    plan = TAMPLite(kin).plan(inside_box_scene(), on("table"), held="A")
    assert plan.operators == (("place", ("A", "table")),)
    for step in plan.trajectory:
        if step.position is not None:
            pose = kin.forward(kin.solve(step.position))
            assert np.linalg.norm(pose[:3, 3] - step.position) < .003
            assert pose[2, 1] > .97


def test_multitransfer_trajectory_is_rejected_instead_of_reusing_old_geometry():
    path = [(operator, None) for operator in [
        ("pick", ("A",)), ("place", ("A", "B")),
        ("pick", ("A",)), ("place", ("A", "box")),
    ]]
    with pytest.raises(_Infeasible):
        TAMPLite(FakeKin())._build_trajectory(default_scene(), path, on("box"))


def test_missing_grasp_cannot_be_bypassed_by_require_action():
    scene = default_scene()
    scene.objects["A"].grasps = []
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(FakeKin()).plan(scene, on("table"), require_action=True)


def test_held_other_object_cannot_be_silently_replaced():
    with pytest.raises(TAMPError):
        TAMPLite(FakeKin()).plan(default_scene(), on("box"), held="B")


def test_block_support_is_observed_before_container_containment():
    scene = inside_box_scene()
    scene.objects["B"] = make_node(
        "B", "block", (-.15, -.45), .800, .764)
    assert TAMPLite(FakeKin()).observed_support(scene) == {"A": "box", "B": "A"}


@pytest.mark.parametrize("target", ["box", "B"])
def test_measured_small_support_rejects_release(target):
    scene = default_scene()
    scene.objects[target].extent[:2] = .010
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(FakeKin()).plan(scene, on(target))


def test_adjacent_obstacle_rejects_release():
    scene = default_scene()
    scene.objects["C"] = make_node(
        "C", "block", (.12, -.45), .781, .745)
    with pytest.raises(TAMPError, match="feasible"):
        TAMPLite(FakeKin()).plan(scene, on("B"))


def test_latest_failure_context_gets_a_fully_executable_cartesian_path():
    from pickparts_agent.agent.atoms.manipulation import _cartesian_waypoints
    from pickparts_agent.scene.kinematics import ArmKinematics

    scene = make_scene([
        make_node("A", "block", (-.149534, -.238402), .866589, .831174),
        make_node("B", "block", (-.085585, -.355766), .757015, .721768),
        make_node("box", "box", (-.196549, -.324025), .760158, .724696,
                  extent_xy=.094, height=.035),
    ])
    scene = type(scene)(
        table=TableSurface(
            normal=np.array([0., 0., 1.]), top_z=.721111,
            bounds=np.array([[-.30, -.67, .72], [.30, -.205, .722]])),
        objects=scene.objects, qpos=scene.qpos)
    arm_seed = np.array([2.099993, 2.268615, 2.023880, 1.773783, .013151])
    grasp_offset = np.array([-.005955, -.006554, .017717])
    kin = ArmKinematics()

    plan = TAMPLite(kin).plan(
        scene, on("table"), held="A", arm_seed=arm_seed,
        grasp_offset=grasp_offset)

    seed = arm_seed
    for step in plan.trajectory:
        if step.kind != "move":
            continue
        target = np.asarray(step.position) + grasp_offset
        for waypoint in _cartesian_waypoints(kin, target, seed):
            seed = kin.solve(waypoint, seed=seed)


def test_failed_table_target_is_excluded_from_replan():
    planner = TAMPLite(FakeKin())
    first = planner.plan(inside_box_scene(), on("table"), held="A",
                         arm_seed=np.zeros(5), grasp_offset=np.zeros(3))
    failed_xy = release_position(first)[:2]

    second = planner.plan(
        inside_box_scene(), on("table"), held="A",
        arm_seed=np.zeros(5), grasp_offset=np.zeros(3),
        excluded_table_xy=(failed_xy,))

    assert np.linalg.norm(release_position(second)[:2] - failed_xy) >= .05


def test_only_failed_cartesian_ingress_blacklists_table_target():
    plan = TAMPLite(FakeKin()).plan(inside_box_scene(), on("table"))
    opening = next(index for index, step in enumerate(plan.trajectory)
                   if step.kind == "gripper" and step.jaw == .8)
    calibration = next(index for index, step in enumerate(plan.trajectory)
                       if step.kind == "calibrate")

    assert TAMPLite.failed_table_xy(plan, calibration) is None
    assert TAMPLite.failed_table_xy(plan, opening - 1) is not None
    assert TAMPLite.failed_table_xy(plan, opening) is None
    assert TAMPLite.failed_table_xy(plan, opening + 1) is None


def test_pick_from_stack_can_lower_hover_without_losing_lift_clearance():
    from test_tiptop_optimized_executor import CartesianKin
    class CeilingKin(CartesianKin):
        def solve(self, position, seed=None):
            if position[2] > .87:
                raise ValueError("ceiling")
            return super().solve(position, seed)

    scene = make_scene([
        make_node("A", "block", (-.06, -.27), .793, .757),
        make_node("B", "block", (-.06, -.27), .757, .721),
    ])
    # Mimic RGB-D center biased toward the visible top surfaces.
    scene.objects["A"].point[2] = .780
    plan = TAMPLite(CeilingKin()).plan(
        scene, (Predicate("holding", ("A",)),),
        arm_seed=np.array([-.06, -.27, .845, 0., 0.]))
    hover, grasp = plan.trajectory[0].position, plan.trajectory[1].position
    assert .05 <= hover[2] - grasp[2] < .10
    assert hover[2] >= scene.objects["A"].top_z + .03
    assert [s.kind for s in plan.trajectory] == [
        "move", "move", "gripper", "move", "calibrate"]


def test_tilted_block_can_replan_stack_after_falling_to_table():
    scene = default_scene()
    scene.objects["A"].extent[:2] = [.035017, .024684]
    scene.objects["B"].extent[:2] = [.025805, .024066]
    plan = TAMPLite(FakeKin()).plan(scene, on("B"))
    assert plan.operators == (("pick", ("A",)), ("place", ("A", "B")))
    np.testing.assert_allclose(release_position(plan)[:2], scene.objects["B"].point[:2])
