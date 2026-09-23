"""Self-agent grasp safety regressions; sensor doubles only, no physics."""
import numpy as np
import pytest

from helpers import FakeFrame, FakeSim, make_context
from pickparts_agent.agent.atoms import (
    CarryTo, Grasp, Lift, PlaceOn, PlaceOnTable, ReachAbove, ReleaseInto,
    ResetArm, SetGripper, VerifyState,
)


class CartesianKin:
    def solve(self, position, seed=None):
        return np.r_[position, 0., 0.]

    def forward(self, joints):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(joints)[:3]
        return pose


class SensorSim(FakeSim):
    def __init__(self):
        super().__init__()
        self.qpos[:5] = [0., -.34, .74, 0., 0.]
        self.events = []
        self.reads = 0
        self.fail_read = None
        self.close_drift = 0.
        self.close_error = None
        self.state = None

    @property
    def actors(self):
        raise AssertionError("Actor truth must not be accessed")

    def observe(self):
        self.reads += 1
        self.events.append(("observe", self.state.grasp_target))
        if self.reads == self.fail_read:
            raise RuntimeError("joint sensor unavailable")
        return FakeFrame(self.qpos.copy())

    def move_right(self, joints, jaw=None, steps=35):
        self.events.append(("move", jaw, steps, self.state.grasp_target))
        super().move_right(joints, jaw=jaw, steps=steps)
        if jaw == 0.:
            self.qpos[0] += self.close_drift
            if self.close_error is not None:
                raise self.close_error

    def hold(self, steps=1):
        self.events.append(("hold", steps))


def detection(*, z=.74, extent=None):
    return {
        "point": [0., -.34, z], "bbox": [1, 1, 3, 3],
        "confidence": .99, "extent": extent or [.024, .024, .036],
        "bottom_z": z - .018, "top_z": z + .018,
    }


def context(*, candidate=False):
    measurements = {
        "A": detection(), "B": detection(),
        "box": {
            **detection(z=.7395, extent=[.096, .09, .039]),
            "bottom_z": .72, "top_z": .759,
        },
    }
    sim = SensorSim()
    ctx = make_context(
        sim=sim, kin=CartesianKin(),
        locate=lambda frame, target: measurements[target],
    )
    sim.state = ctx.state
    for target, loc in measurements.items():
        ctx.state.apply_observation(target, loc, ctx.state.tick())
    if candidate:
        ctx.state.set_gripper(False)
        ctx.state.grasp_target = "A"
    return ctx, measurements


@pytest.mark.parametrize("drift,success", [
    (.0079, True), (.0081, False), (.02, False), (float("nan"), False),
])
def test_close_checks_actual_tcp_without_another_move(drift, success):
    ctx, _ = context()
    ctx.sim.close_drift = drift
    result = Grasp().call(ctx, {"target": "A"})
    assert result.success is success
    assert result.observed["target_tcp"] == pytest.approx([0., -.34, .736])
    assert result.observed["actual_tcp"][0] == pytest.approx(drift, nan_ok=True)
    assert result.observed["error_m"] == pytest.approx(abs(drift), nan_ok=True)
    assert len(result.observed["actual_joints"]) == 5
    assert result.observed["commanded_joints"] == pytest.approx(
        [0., -.34, .736, 0., 0.])
    if not success:
        assert result.error_kind == "verify"
    assert ctx.state.grasp_target == "A"
    assert ctx.state.held_object is None
    assert not ctx.state.gripper_open
    assert [(e[1], e[2]) for e in ctx.sim.events if e[0] == "move"] == [
        (.8, 35), (0., 20),
    ]
    assert [e for e in ctx.sim.events if e[0] == "hold"] == [("hold", 20)]


def test_preclose_joint_read_failure_does_not_register_candidate():
    ctx, _ = context()
    ctx.sim.fail_read = 3  # Path seed, reached TCP, then pre-close joints.
    result = Grasp().call(ctx, {"target": "A"})
    assert not result.success and result.error_kind == "fatal"
    assert ctx.state.grasp_target is None
    assert ctx.state.gripper_open
    assert ctx.state.held_object is None
    assert [m["jaw"] for m in ctx.sim.moves] == [.8]


def test_candidate_is_registered_at_close_dispatch_not_during_joint_read():
    ctx, _ = context()
    ctx.sim.close_error = RuntimeError("close actuated before interruption")
    result = Grasp().call(ctx, {"target": "A"})
    assert not result.success and result.error_kind == "fatal"
    assert ctx.sim.events[-2:] == [
        ("observe", None), ("move", 0., 20, "A"),
    ]
    assert ctx.state.grasp_target == "A"
    assert ctx.state.held_object is None
    assert not ctx.state.gripper_open


def test_postclose_joint_read_failure_keeps_candidate():
    ctx, _ = context()
    ctx.sim.fail_read = 4
    result = Grasp().call(ctx, {"target": "A"})
    assert not result.success and result.error_kind == "fatal"
    assert ctx.state.grasp_target == "A"
    assert not ctx.state.gripper_open


OPERATIONS = [
    (SetGripper, {"open": True}),
    (ResetArm, {}),
    (ReachAbove, {"target": "B"}),
    (VerifyState, {"target": "B", "at": "box"}),
]


@pytest.mark.parametrize("atom,args", OPERATIONS)
@pytest.mark.parametrize("evidence", [
    "near", "ambiguous_gap", "missing", "low_confidence", "nan_point",
    "missing_extent", "nan_extent", "negative_extent", "large_extent",
    "nan_tcp", "sensor_failure",
])
def test_unknown_candidate_blocks_release_and_placement(atom, args, evidence):
    ctx, measurements = context(candidate=True)
    ctx.state.objects["A"].point = [0., -.34, .2]  # Stale, apparently far away.
    ctx.sim.qpos[2] = .84
    if evidence == "near":
        measurements["A"] = detection(z=.84)
    elif evidence == "ambiguous_gap":
        measurements["A"] = detection(z=.78)
    elif evidence == "missing":
        del measurements["A"]
    elif evidence == "low_confidence":
        measurements["A"]["confidence"] = .1
    elif evidence == "nan_point":
        measurements["A"]["point"][0] = float("nan")
    elif evidence == "missing_extent":
        del measurements["A"]["extent"]
    elif evidence == "nan_extent":
        measurements["A"]["extent"][0] = float("nan")
    elif evidence == "negative_extent":
        measurements["A"]["extent"][0] = -.024
    elif evidence == "large_extent":
        measurements["A"]["extent"] = [.15, .15, .15]
    elif evidence == "nan_tcp":
        ctx.sim.qpos[0] = float("nan")
    elif evidence == "sensor_failure":
        ctx.sim.fail_read = 1
    result = atom().call(ctx, args)
    assert not result.success
    assert ctx.sim.moves == []
    assert not [e for e in ctx.sim.events if e[0] == "hold"]
    assert ctx.state.grasp_target == "A"
    assert ctx.state.held_object is None
    assert not ctx.state.gripper_open
    assert ctx.state.objects["B"].placement is None


@pytest.mark.parametrize("atom,args", OPERATIONS)
def test_fresh_separation_clears_candidate_and_allows_recovery(atom, args):
    ctx, _ = context(candidate=True)
    ctx.sim.qpos[2] = .84
    ctx.state.objects["A"].point = [0., -.34, .84]  # Stale apparent possession.
    ctx.state.grasp_offset = [.01, 0., 0.]
    ctx.state.grasp_bottom_offset = .02
    old_frame = ctx.state.objects["A"].frame_id
    result = atom().call(ctx, args)
    assert result.success
    assert ctx.state.grasp_target is None
    assert ctx.state.held_object is None
    assert ctx.state.grasp_offset is None
    assert ctx.state.grasp_bottom_offset is None
    assert ctx.state.objects["A"].frame_id > old_frame
    assert ctx.state.objects["A"].visible
    assert ctx.state.objects["A"].point == pytest.approx([0., -.34, .74])
    if atom is VerifyState:
        assert ctx.sim.moves == []
        assert not ctx.state.gripper_open  # Observation is not a jaw command.
    else:
        assert ctx.state.gripper_open


@pytest.mark.parametrize("atom,args", OPERATIONS)
def test_confirmed_held_object_is_never_cleared_by_separation(atom, args):
    ctx, _ = context(candidate=True)
    ctx.state.set_held("A")
    ctx.sim.qpos[2] = 1.
    result = atom().call(ctx, args)
    assert not result.success
    assert ctx.state.held_object == "A"
    assert ctx.state.grasp_target == "A"
    assert ctx.sim.moves == []


def test_separation_uses_joints_from_the_same_frame_as_target():
    ctx, _ = context(candidate=True)
    observe = ctx.sim.observe

    def changing_sensor():
        frame = observe()
        ctx.sim.qpos[2] = 1.
        return frame

    ctx.sim.observe = changing_sensor
    result = ResetArm().call(ctx, {})
    assert not result.success
    assert ctx.sim.moves == []
    assert ctx.state.grasp_target == "A"


def test_failed_lift_can_reset_open_and_regrasp_after_fresh_separation():
    ctx, _ = context()
    assert Grasp().call(ctx, {"target": "A"}).success
    lift = Lift().call(ctx, {})
    assert not lift.success and lift.error_kind == "grip_failed"
    assert ctx.state.grasp_target == "A"
    assert ResetArm().call(ctx, {}).success
    assert ctx.state.grasp_target is None
    assert SetGripper().call(ctx, {"open": True}).success
    assert ReachAbove().call(ctx, {"target": "A"}).success
    assert Grasp().call(ctx, {"target": "A"}).success
    assert ctx.state.grasp_target == "A"
    assert ctx.state.held_object is None


def carried_context(evidence="near"):
    ctx, measurements = context()
    ctx.state.set_gripper(False)
    ctx.state.set_held("B")
    ctx.state.grasp_target = "B"
    ctx.state.grasp_offset = [-.0058, -.0076, .02]
    ctx.state.grasp_bottom_offset = .038
    calls = []

    def locate(frame, target):
        calls.append((target, len(ctx.sim.moves)))
        if target != "B":
            return measurements[target]
        if evidence == "missing":
            raise ValueError("held target occluded")
        tcp = frame.qpos[:3]
        loc = {
            **detection(),
            "point": [tcp[0] - .0058, tcp[1] + .0022, tcp[2] - .025],
            "bottom_z": tcp[2] - .045, "top_z": tcp[2] - .009,
        }
        if evidence == "far":
            loc["point"][0] += .12
        elif evidence == "nan_point":
            loc["point"][0] = float("nan")
        elif evidence == "low_confidence":
            loc["confidence"] = .1
        elif evidence == "missing_bottom":
            del loc["bottom_z"]
        elif evidence == "nan_bottom":
            loc["bottom_z"] = float("nan")
        return loc

    ctx.locate = locate
    return ctx, calls


@pytest.mark.parametrize("atom,args,destination,release", [
    (PlaceOn, {"target": "A"}, "A", [.0058, -.3422, .807]),
    (ReleaseInto, {"container": "box"}, "box", [.0058, -.3422, .7945]),
])
def test_carry_refreshes_rotated_world_offset_before_release(
        atom, args, destination, release):
    ctx, calls = carried_context()
    result = CarryTo().call(ctx, {"container": destination})
    assert result.success
    assert calls == [(destination, 0), ("B", 1)]
    assert ctx.state.grasp_offset == pytest.approx([.0058, -.0022, .025])
    assert ctx.state.grasp_bottom_offset == pytest.approx(.045)
    assert result.observed["grasp_offset"] == pytest.approx([.0058, -.0022, .025])
    assert result.observed["grasp_bottom_offset"] == pytest.approx(.045)
    assert ctx.state.held_object == "B"
    assert len(ctx.sim.moves) == 1
    assert [e for e in ctx.sim.events if e[0] == "hold"] == [("hold", 15)]
    assert atom().call(ctx, args).success
    np.testing.assert_allclose(ctx.sim.moves[1]["joints"][:3], release)
    assert ctx.sim.moves[2]["jaw"] == .8


@pytest.mark.parametrize("evidence", [
    "missing", "far", "nan_point", "low_confidence", "missing_bottom", "nan_bottom",
])
def test_failed_hover_refresh_retains_held_but_blocks_all_placement(evidence):
    ctx, _ = carried_context(evidence)
    result = CarryTo().call(ctx, {"container": "A"})
    assert not result.success
    assert ctx.state.held_object == "B"
    assert ctx.state.grasp_target == "B"
    assert not ctx.state.gripper_open
    assert ctx.state.grasp_offset is None
    assert ctx.state.grasp_bottom_offset is None
    assert len(ctx.sim.moves) == 1
    events = list(ctx.sim.events)
    for atom, args in [
        (PlaceOn, {"target": "A"}), (ReleaseInto, {"container": "box"}),
        (PlaceOnTable, {}), (SetGripper, {"open": True}), (ResetArm, {}),
    ]:
        rejected = atom().call(ctx, args)
        assert not rejected.success
        assert ctx.sim.events == events
        assert ctx.state.held_object == "B"
        assert ctx.state.grasp_target == "B"


def test_refreshed_offset_still_gets_remaining_path_ik_preflight():
    ctx, _ = carried_context()
    assert CarryTo().call(ctx, {"container": "A"}).success
    solve = ctx.kin.solve

    def blocked_release(position, seed=None):
        if position[0] > 0. and position[2] < .82:
            raise ValueError("new offset makes descent unreachable")
        return solve(position, seed=seed)

    ctx.kin.solve = blocked_release
    before = list(ctx.sim.events)
    result = PlaceOn().call(ctx, {"target": "A"})
    assert not result.success and result.error_kind == "unreachable"
    assert len(ctx.sim.moves) == 1
    assert not [e for e in ctx.sim.events[len(before):] if e[0] == "move"]
    assert ctx.state.held_object == "B"
    assert not ctx.state.gripper_open


def test_carry_refresh_can_recover_without_clearing_held():
    ctx, _ = carried_context("missing")
    assert not CarryTo().call(ctx, {"container": "A"}).success
    working, _ = carried_context()
    ctx.locate = working.locate
    result = CarryTo().call(ctx, {"container": "A"})
    assert result.success
    assert ctx.state.held_object == "B"
    assert ctx.state.grasp_offset == pytest.approx([.0058, -.0022, .025])
    assert PlaceOn().call(ctx, {"target": "A"}).success


@pytest.mark.parametrize("failure", ["motion", "hover_sensor"])
def test_interrupted_carry_invalidates_offset_without_clearing_held(failure):
    ctx, _ = carried_context()
    if failure == "motion":
        ctx.sim.tracking_error = True
    else:
        ctx.sim.fail_read = 4  # Destination, seed, reached TCP, hover refresh.
    result = CarryTo().call(ctx, {"container": "A"})
    assert not result.success and result.error_kind == "fatal"
    assert ctx.state.held_object == "B"
    assert ctx.state.grasp_target == "B"
    assert ctx.state.grasp_offset is None
    assert ctx.state.grasp_bottom_offset is None
    before = list(ctx.sim.events)
    assert not PlaceOn().call(ctx, {"target": "A"}).success
    assert ctx.sim.events == before


def test_full_chain_with_fresh_held_rgbd_reaches_release():
    ctx, _ = carried_context()
    ctx.state.set_held(None)
    ctx.state.grasp_target = None
    ctx.state.set_gripper(True)
    assert ReachAbove().call(ctx, {"target": "B"}).success
    assert Grasp().call(ctx, {"target": "B"}).success
    assert Lift().call(ctx, {}).success
    assert ctx.state.held_object == "B"
    assert CarryTo().call(ctx, {"container": "box"}).success
    assert ReleaseInto().call(ctx, {"container": "box"}).success
    assert ctx.state.held_object is None
    assert ctx.state.grasp_target is None
    assert ctx.state.gripper_open


def test_carry_refreshes_moved_support_and_held_target():
    ctx, calls = carried_context()
    locate = ctx.locate

    def moved_support(frame, target):
        if target == "A":
            return {**detection(), "point": [.01, -.31, .74]}
        return locate(frame, target)

    ctx.locate = moved_support
    result = CarryTo().call(ctx, {"container": "A"})
    assert result.success
    assert result.observed["target_tcp"] == pytest.approx([.0042, -.3176, .835])
    assert calls == [("B", 1)]
    assert ctx.state.grasp_offset == pytest.approx([.0058, -.0022, .025])
