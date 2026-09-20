import numpy as np

from helpers import FakeKin, FakeLocate, FakeSim, make_context
from pickparts_agent.atoms.library import FindObject, SetGripper, VerifyState


def test_find_object_locates_and_updates_state():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739]})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    r = FindObject().call(ctx, {"target": "A"})
    assert r.success is True
    assert ctx.state.objects["A"].point == [-0.085, -0.34, 0.739]


def test_find_object_missing_returns_lost_object():
    ctx = make_context(FakeSim(), FakeKin(), FakeLocate({}))
    r = FindObject().call(ctx, {"target": "A"})
    assert r.success is False and r.error_kind == "lost_object"


def test_verify_state_success_when_placement_matches():
    locate = FakeLocate({"A": [0.085, -0.30, 0.760], "box": [0.085, -0.30, 0.726]})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    r = VerifyState().call(ctx, {"target": "A", "at": "box"})
    assert r.success is True and r.observed["placement"] == "box"


def test_verify_state_failure_when_placement_differs():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739], "box": [0.085, -0.30, 0.726]})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    r = VerifyState().call(ctx, {"target": "A", "at": "box"})
    assert r.success is False and r.error_kind == "verify"


def test_set_gripper_close_updates_state_and_jaw():
    sim = FakeSim()
    ctx = make_context(sim, FakeKin(), FakeLocate({}))
    r = SetGripper().call(ctx, {"open": False})
    assert r.success and ctx.state.gripper_open is False
    assert sim.moves[-1]["jaw"] == 0.0


from pickparts_agent.atoms.library import (
    CarryTo, Grasp, Lift, ReachAbove, ReleaseInto, ResetArm, REST_Q)


def prepared(ctx):
    FindObject().call(ctx, {"target": "A"})
    FindObject().call(ctx, {"target": "box"})
    ReachAbove().call(ctx, {"target": "A"})
    Grasp().call(ctx, {"target": "A"})


def test_full_motion_chain_grounds_and_holds_then_releases():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739], "box": [0.085, -0.30, 0.726]})
    sim, kin = FakeSim(), FakeKin()
    ctx = make_context(sim, kin, locate)
    prepared(ctx)
    locate.add("A", [-0.085, -0.34, 0.840])  # visually higher after lift
    lift = Lift().call(ctx, {"clearance": 0.10})
    assert lift.success and lift.observed["lift_m"] >= 0.1
    assert ctx.state.held_object == "A"
    carry = CarryTo().call(ctx, {"container": "box"})
    assert carry.success and np.allclose(kin.position[:2], [0.085, -0.30])
    release = ReleaseInto().call(ctx, {"container": "box"})
    assert release.success and ctx.state.held_object is None
    assert ctx.state.gripper_open is True


def test_reach_moves_to_hover_xy():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739], "box": [0.085, -0.30, 0.726]})
    kin = FakeKin()
    ctx = make_context(FakeSim(), kin, locate)
    FindObject().call(ctx, {"target": "A"})
    r = ReachAbove().call(ctx, {"target": "A"})
    assert r.success and np.allclose(kin.position[:2], [-0.085, -0.34])


def test_grasp_closes_gripper():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739], "box": [0.085, -0.30, 0.726]})
    sim = FakeSim()
    ctx = make_context(sim, FakeKin(), locate)
    FindObject().call(ctx, {"target": "A"})
    ReachAbove().call(ctx, {"target": "A"})
    r = Grasp().call(ctx, {"target": "A"})
    assert r.success and ctx.state.gripper_open is False
    assert sim.moves[-1]["jaw"] == 0.0


def test_lift_low_height_is_grip_failed():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739], "box": [0.085, -0.30, 0.726]})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    FindObject().call(ctx, {"target": "A"})
    ReachAbove().call(ctx, {"target": "A"})
    Grasp().call(ctx, {"target": "A"})
    # Relocated height barely moved: not actually grasped.
    locate.add("A", [-0.085, -0.34, 0.745])
    r = Lift().call(ctx, {"clearance": 0.10})
    assert r.success is False and r.error_kind == "grip_failed"


def test_reset_arm_returns_to_rest():
    sim = FakeSim()
    ctx = make_context(sim, FakeKin(), FakeLocate({}))
    r = ResetArm().call(ctx, {})
    assert r.success and np.allclose(sim.moves[-1]["joints"], REST_Q)


def test_unreachable_ik_maps_to_unreachable():
    kin = FakeKin(); kin.fail_solve = True
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739]})
    ctx = make_context(FakeSim(), kin, locate)
    FindObject().call(ctx, {"target": "A"})
    r = ReachAbove().call(ctx, {"target": "A"})
    assert r.success is False and r.error_kind == "unreachable"


def test_joint_tracking_error_maps_to_fatal():
    sim = FakeSim(); sim.tracking_error = True
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739]})
    ctx = make_context(sim, FakeKin(), locate)
    FindObject().call(ctx, {"target": "A"})
    r = ReachAbove().call(ctx, {"target": "A"})
    assert r.success is False and r.error_kind == "fatal"
