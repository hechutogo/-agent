"""Single-attempt execution and sensor-only grip recovery contracts."""
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from helpers import FakeSim
from pickparts_agent.agent.atoms.base import AtomResult
from pickparts_agent.observability import Recorder
from pickparts_agent.scene.kinematics import ARM_NAMES
from tiptop_mac.types import MotionStep, SceneGraph, TAMPPlan, TableSurface


class CartesianKin:
    """Deterministic joint-to-TCP mapping, independent of commanded targets."""

    def forward(self, joints):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(joints)[:3]
        return pose

    def solve(self, position, seed=None):
        return np.r_[position, 0., 0.]


def block(point=(0., -.45, .738), *, object_id="A"):
    point = np.asarray(point, dtype=float)
    return SimpleNamespace(
        id=object_id, kind="block", point=point, bottom_z=point[2] - .018,
        top_z=point[2] + .018, extent=np.array([.024, .024, .036]))


class SensorSim(FakeSim):
    def __init__(self):
        super().__init__()
        self.qpos[:5] = [0., -.45, .834, 0., 0.]
        self.qpos[-1] = .8
        self.node = block()
        self.attached = False
        self.grasp_works = True
        self.missing = False
        self.fail_close = False
        self.fail_release = False
        self.object_specs = [
            {"id": "A", "kind": "block", "label": "A", "color": [.85, .045, .03]}]

    def observe(self):
        frame = super().observe()
        frame.intrinsic = np.eye(3)
        frame.camera_to_base = np.eye(4)
        return frame

    def move_right(self, joints, jaw=None, steps=35):
        previous_jaw = self.qpos[self.joint_names.index("Jaw")]
        if jaw == 0. and previous_jaw > .5 and self.grasp_works:
            self.attached = True
            self.offset = self.arm_qpos()[:3] - self.node.point
        if jaw == .8 and self.attached:
            if self.fail_release:
                raise ValueError("release command interrupted")
            self.attached = False
            self.node = block([self.node.point[0], self.node.point[1], .738])
        super().move_right(joints, jaw, steps)
        if self.attached:
            self.node = block(self.arm_qpos()[:3] - self.offset)
        if jaw == 0. and previous_jaw > .5 and self.fail_close:
            raise ValueError("close command interrupted after actuation")

    def locate(self, frame, target):
        if self.missing:
            raise ValueError("target occluded")
        assert target == "A"
        return {**vars(self.node), "bbox": [1, 1, 3, 3]}

    def scene(self, *, objects=None):
        return SceneGraph(
            TableSurface(np.array([0., 0., 1.]), .720,
                         np.array([[-.30, -.67, .720], [.30, -.23, .720]])),
            objects if objects is not None else ({} if self.missing else {"A": self.node}),
            self.qpos.copy())


def plan(steps, operators=()):
    return TAMPPlan((), tuple(operators), tuple(steps), 0., "")


def pick_steps():
    return [
        MotionStep("move", np.array([0., -.45, .834]), .8),
        MotionStep("move", np.array([0., -.45, .734]), .8),
        MotionStep("gripper", None, 0.),
        MotionStep("move", np.array([0., -.45, .834]), 0.),
        MotionStep("calibrate", None, None, '{"held": "A", "assumed": 0.018}'),
    ]


def place_steps():
    return [
        MotionStep("move", np.array([.10, -.45, .85]), 0.),
        MotionStep("move", np.array([.10, -.45, .738]), 0.),
        MotionStep("gripper", None, .8),
        MotionStep("move", np.array([.10, -.45, .808]), .8),
    ]


@pytest.fixture
def module():
    return importlib.import_module("tiptop_optimized.executor")


@pytest.fixture
def setup(module):
    sim = SensorSim()
    executor = module.RecoveringExecutor(sim, CartesianKin(), locate=sim.locate)
    executor.reconcile(sim.scene())
    return executor, sim


@pytest.mark.parametrize("error,kind", [
    (ValueError("private IK detail"), "unreachable"),
    (RuntimeError("private backend detail"), "fatal"),
    (TypeError("private programming detail"), "fatal"),
])
def test_pregrasp_exception_aborts_once_without_open_or_reset(setup, error, kind):
    executor, sim = setup
    calls = []

    def fail(position, seed=None):
        calls.append(position)
        raise error

    executor.kin.solve = fail
    events = []
    executor.on_event = events.append
    result = executor.execute(plan(pick_steps(), [("pick", ("A",))]))
    assert result == {"success": False, "aborted": True,
                      "message": result["message"], "error_kind": kind,
                      "acted": False, "failed_index": 0}
    assert len(calls) == 1 and sim.moves == []
    assert "private" not in json.dumps([events, result])
    assert [e["status"] for e in events if e["type"] == "step"] == [
        "active", "failed", "skipped", "skipped", "skipped", "skipped"]
    assert not any(e["type"] == "finish" for e in events)


def test_cartesian_error_kind_is_preserved(module, setup, monkeypatch):
    executor, sim = setup
    monkeypatch.setattr(module, "_cartesian",
                        lambda *a, **k: AtomResult(False, "verify", "tracking"))
    result = executor.execute(plan(pick_steps(), [("pick", ("A",))]))
    assert result["error_kind"] == "verify" and not result["acted"]
    assert result["failed_index"] == 0
    assert sim.moves == []


def test_failed_grasp_visually_rejected_before_transport(setup):
    executor, sim = setup
    sim.grasp_works = False
    result = executor.execute(plan(pick_steps() + place_steps(),
                                   [("pick", ("A",)), ("place", ("A", "table"))]))
    assert result["aborted"] and result["error_kind"] == "grip_failed"
    assert result["acted"] is False
    assert len(sim.moves) == 4
    assert sim.moves[-1]["jaw"] == 0.
    assert executor.reconcile(sim.scene()) is None


def test_missed_grasps_can_replan_repeatedly_then_pick_successfully(setup):
    from tiptop_mac.perception import propose_grasps
    from tiptop_mac.types import Predicate
    from tiptop_optimized.tamp import TAMPLite

    executor, sim = setup
    planner = TAMPLite(executor.kin)
    goal = [Predicate("holding", ("A",))]
    for grasp_works in (False, False, True):
        sim.grasp_works = grasp_works
        sim.node.grasps = propose_grasps(sim.node)
        scene = sim.scene()
        held = executor.reconcile(scene)
        assert held is None
        attempt = planner.plan(scene, goal, held=held)
        assert attempt.operators == (("pick", ("A",)),)
        before = len(sim.moves)
        result = executor.execute(attempt)
        assert result["success"] is grasp_works
        assert result["acted"] is grasp_works
        assert sim.moves[before]["jaw"] == .8
        if not grasp_works:
            assert result["error_kind"] == "grip_failed"
            assert sim.qpos[-1] == 0.
            assert executor.reconcile(sim.scene()) is None
            assert executor.candidate is executor.held is None
    assert executor.reconcile(sim.scene()) == "A"


@pytest.mark.parametrize("after_empty", [False, True])
@pytest.mark.parametrize("changed", ["missing", "near_airborne", "near_supported"])
def test_closed_empty_evidence_must_remain_visible_and_separated(
        module, setup, after_empty, changed):
    executor, sim = setup
    sim.grasp_works = False
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["aborted"]
    if after_empty:
        assert executor.reconcile(sim.scene()) is None
    objects = {"A": sim.node}
    if changed == "missing":
        objects.clear()
    elif changed == "near_airborne":
        objects["B"] = block([0., -.45, .838], object_id="B")
    else:
        sim.qpos[2] = .734
    before = len(sim.moves)
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene(objects=objects))
    assert len(sim.moves) == before


def test_successful_lift_has_evidence_and_holding_persists_across_runs(setup):
    executor, sim = setup
    result = executor.execute(plan(pick_steps(), [("pick", ("A",))]))
    assert result["success"] and result["acted"] is True
    assert result["error_kind"] is None
    assert executor.reconcile(sim.scene()) == "A"
    moves = len(sim.moves)
    assert executor.execute(plan([]))["acted"] is False
    assert executor.reconcile(sim.scene()) == "A"
    assert len(sim.moves) == moves


def test_planning_context_uses_current_arm_seed_and_measured_grasp_offset(setup):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    scene = sim.scene()
    assert executor.reconcile(scene) == "A"

    context = executor.planning_context(scene, held="A")

    np.testing.assert_allclose(context["arm_seed"], sim.arm_qpos())
    tcp = executor.kin.forward(sim.arm_qpos())[:3, 3]
    expected = np.array([
        tcp[0] - sim.node.point[0],
        tcp[1] - sim.node.point[1],
        tcp[2] - sim.node.bottom_z - .018,
    ])
    np.testing.assert_allclose(context["grasp_offset"], expected)


def test_contact_limited_jaw_is_holding_with_visual_lift_evidence(setup):
    executor, sim = setup
    original = sim.move_right

    def contact_limited(joints, jaw=None, **kwargs):
        original(joints, jaw, **kwargs)
        if jaw == 0. and sim.attached:
            sim.qpos[-1] = .4523  # measured real 24mm block contact, not empty jaw

    sim.move_right = contact_limited
    result = executor.execute(plan(pick_steps(), [("pick", ("A",))]))
    assert result["success"] and result["acted"]
    assert executor.reconcile(sim.scene()) == "A"


def test_held_continuation_remeasures_shift_without_blind_open(setup):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    # A changed grasp offset must be measured again, not reused from last run.
    sim.offset[:2] = [.004, -.003]
    sim.node = block(sim.arm_qpos()[:3] - sim.offset)
    before = len(sim.moves)
    result = executor.execute(plan(place_steps(), [("place", ("A", "table"))]))
    assert result["success"]
    moves = sim.moves[before:]
    np.testing.assert_allclose(moves[0]["joints"][:3], [.104, -.453, .846])
    assert [m["jaw"] for m in moves] == [0., 0., .8, .8]
    assert executor.reconcile(sim.scene()) is None


def test_held_continuation_revalidates_latest_offset_before_motion(
        module, setup, monkeypatch):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    sim.offset[0] = .03
    sim.node = block(sim.arm_qpos()[:3] - sim.offset)
    original_solve = executor.kin.solve

    def restricted(position, seed=None):
        if position[0] > .12:
            raise ValueError("shifted route is outside workspace")
        return original_solve(position, seed)

    executor.kin.solve = restricted
    monkeypatch.setattr(
        module, "_cartesian",
        lambda *args, **kwargs: AtomResult(True, observed={"tcp": []}))

    result = executor.execute(
        plan(place_steps(), [("place", ("A", "table"))]))

    assert result["error_kind"] == "unreachable"
    assert result["failed_index"] == 0
    assert sim.moves[-1]["jaw"] == 0.


def test_fresh_grasp_revalidates_measured_transport_path_before_motion(
        module, setup, monkeypatch):
    executor, sim = setup
    original_solve, original_cartesian = executor.kin.solve, module._cartesian
    transport_calls = []

    def restricted(position, seed=None):
        if position[0] > .05:
            raise ValueError("measured transport route is outside workspace")
        return original_solve(position, seed)

    def recording_cartesian(ctx, position, jaw=None):
        if position[0] > .05:
            transport_calls.append(np.asarray(position).copy())
        return original_cartesian(ctx, position, jaw)

    executor.kin.solve = restricted
    monkeypatch.setattr(module, "_cartesian", recording_cartesian)
    result = executor.execute(plan(
        pick_steps() + place_steps(),
        [("pick", ("A",)), ("place", ("A", "table"))]))

    assert result["error_kind"] == "unreachable"
    assert result["failed_index"] == len(pick_steps())
    assert transport_calls == []
    assert executor.held == "A"


@pytest.mark.parametrize("jaw", [0., .8])
def test_missing_held_observation_is_ambiguous_not_dropped(module, setup, jaw):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    sim.missing = True
    sim.qpos[-1] = jaw
    before = len(sim.moves)
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene())
    result = executor.execute(plan(place_steps(), [("place", ("A", "table"))]))
    assert result["error_kind"] == "grip_failed"
    assert len(sim.moves) == before
    sim.missing = False
    sim.qpos[-1] = 0.
    assert executor.reconcile(sim.scene()) == "A"


def test_close_failure_keeps_candidate_even_when_command_took_effect(module, setup):
    executor, sim = setup
    sim.fail_close = True
    result = executor.execute(plan(pick_steps(), [("pick", ("A",))]))
    assert result["error_kind"] == "unreachable" and not result["acted"]
    assert len(sim.moves) == 3
    sim.missing = True
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene())
    sim.missing = False
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene())
    assert executor.candidate == "A" and executor.held is None
    before = len(sim.moves)
    result = executor.execute(plan(place_steps(), [("place", ("A", "table"))]))
    assert result["error_kind"] == "grip_failed" and not result["acted"]
    assert len(sim.moves) == before


def test_release_failure_reconciles_as_still_held(module, setup):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    sim.fail_release = True
    result = executor.execute(plan(
        [MotionStep("gripper", None, .8), MotionStep("reset", None, .8)],
        [("place", ("A", "table"))]))
    assert result["error_kind"] == "unreachable"
    assert sim.moves[-1]["jaw"] == 0.
    assert executor.reconcile(sim.scene()) == "A"


def test_release_failure_at_supported_destination_cannot_satisfy_goal(setup):
    from tiptop_mac.types import Predicate
    from tiptop_optimized.tamp import TAMPLite

    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["acted"]
    sim.fail_release = True
    result = executor.execute(plan(place_steps(), [("place", ("A", "table"))]))
    assert result["aborted"] and result["error_kind"] == "unreachable"
    assert sim.attached and sim.qpos[-1] == 0.
    assert sim.node.bottom_z == pytest.approx(.720)
    scene = sim.scene()
    held = executor.reconcile(scene)
    assert held == "A" and executor.candidate == "A"
    planner = TAMPLite(executor.kin)
    goal = [Predicate("on", ("A", "table"))]
    assert not planner.satisfies(scene, goal, held=held)
    assert planner.plan(scene, goal, held=held).operators == (
        ("place", ("A", "table")),)


@pytest.mark.parametrize("height", [.738, .758])
@pytest.mark.parametrize("reconcile_first", [False, True])
def test_verified_held_continuation_can_start_below_original_lift_height(
        setup, height, reconcile_first):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["acted"]
    sim.move_right([.10, -.45, height - .004, 0., 0.], jaw=0.)
    if reconcile_first:
        assert executor.reconcile(sim.scene()) == "A"
    before = len(sim.moves)
    result = executor.execute(plan(place_steps(), [("place", ("A", "table"))]))
    assert result["success"] and result["acted"]
    assert [m["jaw"] for m in sim.moves[before:]] == [0., 0., .8, .8]
    assert not sim.attached
    assert executor.reconcile(sim.scene()) is None


def test_release_failure_after_physical_release_can_reconcile_supported(setup):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    original = sim.move_right

    def release_then_fail(joints, jaw=None, **kwargs):
        original(joints, jaw, **kwargs)
        if jaw == .8:
            raise ValueError("failure after release")

    sim.move_right = release_then_fail
    result = executor.execute(plan([MotionStep("gripper", None, .8)],
                                   [("place", ("A", "table"))]))
    assert result["aborted"]
    assert executor.reconcile(sim.scene()) is None


def test_replanning_pick_while_held_cannot_replay_opening(setup):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    before = len(sim.moves)
    result = executor.execute(plan(pick_steps(), [("pick", ("A",))]))
    assert result["error_kind"] == "grip_failed"
    assert len(sim.moves) == before


@pytest.mark.parametrize("changed", ["far", "open", "nan", "low_gain"])
def test_calibration_requires_proximity_closed_jaw_and_height_gain(setup, changed):
    executor, sim = setup
    assert executor.execute(plan(pick_steps()[:3], [("pick", ("A",))]))["success"]
    sim.node = block([0., -.45, .838])
    sim.qpos[:3] = [0., -.45, .834]
    if changed == "far":
        sim.node = block([.15, -.45, .838])
    elif changed == "open":
        sim.qpos[-1] = .8
    elif changed == "nan":
        sim.node.point[0] = np.nan
    else:
        sim.node = block([0., -.45, .758])
        sim.qpos[2] = .754
    result = executor.execute(plan([pick_steps()[-1]]))
    assert result["error_kind"] == "grip_failed" and result["acted"] is False


def test_reconcile_cannot_promote_small_height_gain_to_verified_holding(module, setup):
    executor, sim = setup
    sim.fail_close = True
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["aborted"]
    sim.node = block([0., -.45, .758])
    sim.qpos[2] = .754
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene())
    assert executor.candidate == "A" and executor.held is None


def test_reconcile_uses_named_joints_and_never_commands_motion(setup):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    old_names, old_qpos = sim.joint_names, sim.qpos
    sim.joint_names = ["Jaw", *reversed(ARM_NAMES)]
    sim.qpos = np.array([old_qpos[old_names.index(n)] for n in sim.joint_names])
    before = len(sim.moves)
    assert executor.reconcile(sim.scene()) == "A"
    assert len(sim.moves) == before


@pytest.mark.parametrize("support_kind", ["table", "block", "box"])
@pytest.mark.parametrize("evidence", ["far", "open", "clamped"])
def test_supported_object_requires_release_or_separation_to_clear_held(
        setup, support_kind, evidence):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    support = block((0., -.45, .738), object_id="B")
    if support_kind == "box":
        support = SimpleNamespace(id="box", kind="box", point=np.array([0., -.45, .739]),
                                  bottom_z=.720, top_z=.759,
                                  extent=np.array([.096, .090, .039]))
    bottom = {"table": .720, "block": .756, "box": .726}[support_kind]
    sim.node = block([0., -.45, bottom + .018])
    sim.qpos[:3] = sim.node.point + sim.offset
    if evidence == "far":
        sim.qpos[0] = .15
    elif evidence == "open":
        sim.qpos[-1] = .8
    objects = {"A": sim.node}
    if support_kind != "table":
        objects[support.id] = support
    before = len(sim.moves)
    expected = "A" if evidence == "clamped" else None
    assert executor.reconcile(sim.scene(objects=objects)) == expected
    assert len(sim.moves) == before


def test_airborne_open_jaw_is_not_evidence_of_safe_support(module, setup):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    sim.qpos[-1] = .8
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene())


def test_initial_closed_jaw_without_candidate_requires_observation(module):
    sim = SensorSim()
    executor = module.RecoveringExecutor(sim, CartesianKin(), locate=sim.locate)
    assert executor.reconcile(sim.scene()) is None
    sim.qpos[-1] = 0.
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene())


def test_initial_open_jaw_with_airborne_object_at_tcp_is_not_assumed_empty(module):
    sim = SensorSim()
    sim.node = block([0., -.45, .838])
    executor = module.RecoveringExecutor(sim, CartesianKin(), locate=sim.locate)
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene())
    assert sim.moves == []


def test_held_gripper_open_requires_an_explicit_place_operator(setup):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    before = len(sim.moves)
    result = executor.execute(plan([MotionStep("gripper", None, .8)]))
    assert result["error_kind"] == "grip_failed"
    assert len(sim.moves) == before
    assert executor.reconcile(sim.scene()) == "A"


def test_failed_lift_motion_can_be_reconciled_without_replaying_pick(
        module, setup, monkeypatch):
    executor, sim = setup
    original = module._cartesian
    calls = []

    def interrupted(ctx, target, jaw=None):
        calls.append(target)
        result = original(ctx, target, jaw)
        return AtomResult(False, "verify", "interrupted") if len(calls) == 3 else result

    monkeypatch.setattr(module, "_cartesian", interrupted)
    result = executor.execute(plan(pick_steps(), [("pick", ("A",))]))
    assert result["error_kind"] == "verify" and result["acted"] is False
    assert len(calls) == 3
    assert executor.reconcile(sim.scene()) == "A"
    before = len(sim.moves)
    result = executor.execute(plan(place_steps(), [("place", ("A", "table"))]))
    assert result["success"] and result["acted"] is True
    assert [m["jaw"] for m in sim.moves[before:]] == [0., 0., .8, .8]


@pytest.mark.parametrize("jaw", [0., .8])
def test_box_support_uses_physical_bottom_when_only_top_surfaces_are_visible(setup, jaw):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    sim.node = SimpleNamespace(
        id="A", kind="block", point=np.array([0., -.45, .761]),
        bottom_z=.757, top_z=.762, extent=np.array([.024, .024, .005]))
    sim.qpos[:3] = [0., -.45, .757]
    sim.qpos[-1] = jaw
    box = SimpleNamespace(id="box", kind="box", point=np.array([0., -.45, .739]),
                          bottom_z=.720, top_z=.759,
                          extent=np.array([.096, .090, .039]))
    expected = "A" if jaw == 0. else None
    assert executor.reconcile(sim.scene(objects={"A": sim.node, "box": box})) == expected


def test_box_containment_alone_does_not_prove_floor_support(module, setup):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    sim.node = block([0., -.45, .763])
    sim.qpos[-1] = .8
    box = SimpleNamespace(id="box", kind="box", point=np.array([0., -.45, .739]),
                          bottom_z=.720, top_z=.759,
                          extent=np.array([.096, .090, .039]))
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene(objects={"A": sim.node, "box": box}))


def test_runtime_locator_error_is_fatal_not_a_recoverable_occlusion(setup):
    executor, sim = setup

    def fail(frame, target):
        raise RuntimeError("backend unavailable")

    executor.locate = fail
    result = executor.execute(plan(pick_steps(), [("pick", ("A",))]))
    assert result["error_kind"] == "fatal" and result["acted"] is False
    assert len(sim.moves) == 2


def test_missing_locator_at_close_retains_candidate(module, setup):
    executor, sim = setup
    executor.locate = None
    result = executor.execute(plan(pick_steps(), [("pick", ("A",))]))
    assert result["error_kind"] == "grip_failed"
    sim.missing = True
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene())
    assert len(sim.moves) == 2


def test_pick_verification_can_use_close_baseline_without_prior_reconcile(module):
    sim = SensorSim()
    executor = module.RecoveringExecutor(sim, CartesianKin(), locate=sim.locate)
    result = executor.execute(plan(pick_steps(), [("pick", ("A",))]))
    assert result["success"] and result["acted"] is True
    assert executor.reconcile(sim.scene()) == "A"


def test_real_optimized_planner_holding_then_place_only_continuation(setup):
    from tiptop_mac.perception import propose_grasps
    from tiptop_mac.types import Predicate
    from tiptop_optimized.tamp import TAMPLite

    executor, sim = setup
    sim.node.grasps = propose_grasps(sim.node)
    planner = TAMPLite(executor.kin)
    holding = planner.plan(sim.scene(), [Predicate("holding", ("A",))])
    assert executor.execute(holding)["acted"] is True
    scene = sim.scene()
    held = executor.reconcile(scene)
    continuation = planner.plan(scene, [Predicate("on", ("A", "table"))], held=held)
    assert continuation.operators == (("place", ("A", "table")),)
    before = len(sim.moves)
    assert executor.execute(continuation)["success"]
    assert sim.moves[before]["jaw"] == 0.
    scene = sim.scene()
    assert planner.satisfies(scene, continuation.goal, held=executor.reconcile(scene))


def test_bind_sim_replaces_sensor_context_and_clears_previous_candidate(setup):
    executor, sim = setup
    assert executor.execute(plan(pick_steps(), [("pick", ("A",))]))["success"]
    replacement = SensorSim()
    executor.bind_sim(replacement)
    assert executor.sim is replacement and executor._ctx.sim is replacement
    assert executor._ctx.kin is executor.kin
    assert executor.locate != sim.locate
    assert executor.reconcile(replacement.scene()) is None


def test_trace_steps_and_failures_do_not_emit_finish(module, tmp_path):
    sim, events, stages = SensorSim(), [], []
    recorder = Recorder(tmp_path)
    executor = module.RecoveringExecutor(
        sim, CartesianKin(), stages.append, locate=sim.locate,
        on_event=events.append, recorder=recorder)
    executor.reconcile(sim.scene())
    with recorder.run("command", "pick A"):
        result = executor.execute(plan(
            pick_steps() + [MotionStep("unknown", None, None),
                            MotionStep("reset", None, .8)],
            [("pick", ("A",))]))
    assert result["error_kind"] == "fatal" and result["acted"] is True
    assert len(stages) == 6
    assert not any(e["type"] == "finish" for e in events)
    trace = [json.loads(line) for path in tmp_path.glob("trace-*.jsonl")
             for line in path.read_text().splitlines()]
    starts = {e["span_id"]: e for e in trace if e["kind"] == "span_start"}
    execute = next(e for e in starts.values() if e["name"] == "execute")
    atoms = [e for e in starts.values() if e["name"].startswith("atom:")]
    assert len(atoms) == 6
    assert all(e["parent_id"] == execute["span_id"] for e in atoms)
    ended = {e["span_id"]: e for e in trace if e["kind"] == "span_end"}
    assert all(e["span_id"] in ended for e in atoms)
    assert ended[execute["span_id"]]["error_kind"] == "fatal"
    assert any(e["kind"] == "artifact" and e["type"] == "rgbd" for e in trace)


def test_attempt_state_artifacts_do_not_overwrite_each_other(module, tmp_path):
    sim, recorder = SensorSim(), Recorder(tmp_path)
    executor = module.RecoveringExecutor(sim, CartesianKin(), recorder=recorder)
    with recorder.run("command"):
        for _ in range(2):
            assert executor.execute(plan([MotionStep("unknown", None, None)]))["aborted"]
    trace = [json.loads(line) for path in tmp_path.glob("trace-*.jsonl")
             for line in path.read_text().splitlines()]
    paths = [e["path"] for e in trace if e["kind"] == "artifact" and e["type"] == "state"]
    assert len(paths) == len(set(paths)) == 4


@pytest.mark.parametrize("offset,extent,accepted", [
    (.00306, [.027909, .025305], True),
    (.00568, [.027602, .031191], True),
    (.00720, [.025438, .027421], True),
    (.00306, [.050, .050], False),
    (.018, [.027909, .025305], False),
    (.04, [.027909, .025305], False),
])
def test_noisy_stack_support_agrees_with_planner(module, offset, extent, accepted):
    from tiptop_optimized.tamp import TAMPLite
    sim = SensorSim()
    lower = block([0., -.45, .738], object_id="B")
    lower.top_z = .756510
    lower.extent[:2] = [.023897, .023799]
    upper = block([offset, -.45, .779874])
    upper.bottom_z, upper.top_z = .757599, .794
    upper.extent[:2] = extent
    scene = sim.scene(objects={"A": upper, "B": lower})
    executor = module.RecoveringExecutor(sim, CartesianKin())
    executor.candidate = executor.held = "A"
    planner = TAMPLite(executor.kin)
    assert (planner._initial_support(scene).get("A") == "B") is accepted
    if accepted:
        assert executor.reconcile(scene) is None
        assert executor.candidate is executor.held is None
    else:
        with pytest.raises(module.GripStateError):
            executor.reconcile(scene)
    assert sim.moves == []


def test_tilted_block_measured_bottom_proves_table_support(module):
    sim = SensorSim()
    sim.node.bottom_z, sim.node.top_z = .721796, .745358
    sim.node.extent = np.array([.036, .027, .023562])
    sim.qpos[2] = .76
    executor = module.RecoveringExecutor(sim, CartesianKin())
    assert executor.reconcile(sim.scene()) is None
    assert sim.moves == []


def test_short_airborne_observation_cannot_infer_table_contact(module):
    sim = SensorSim()
    sim.node = block([0., -.45, .753])
    sim.node.bottom_z, sim.node.top_z = .750, .756
    sim.node.extent[2] = .006
    sim.qpos[2] = .76
    executor = module.RecoveringExecutor(sim, CartesianKin())
    with pytest.raises(module.GripStateError):
        executor.reconcile(sim.scene())
