import numpy as np

from helpers import FakeKin, FakeLocate, FakeSim, make_context
from pickparts_agent.agent.atoms import FindObject, SetGripper, VerifyState


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


def test_verify_state_does_not_claim_success_from_stale_occluded_placement():
    locate = FakeLocate({"A": [0.085, -0.31, 0.745],
                         "box": [0.085, -0.30, 0.726]})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    FindObject().call(ctx, {"target": "box"})
    FindObject().call(ctx, {"target": "A"})
    ctx.state.objects["A"].placement = "box"  # An earlier successful verification.
    assert ctx.state.objects["A"].placement == "box"
    locate.fail_on.add("A")  # stacked/occluded during verification
    r = VerifyState().call(ctx, {"target": "A", "at": "box"})
    assert not r.success and r.error_kind == "lost_object"
    assert ctx.sim.moves  # cleared the camera, then retried a real observation


def test_verify_state_still_fails_when_lost_and_last_known_on_table():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739],
                         "box": [0.085, -0.30, 0.726]})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    FindObject().call(ctx, {"target": "A"})
    assert ctx.state.objects["A"].placement is None
    locate.fail_on.add("A")
    r = VerifyState().call(ctx, {"target": "A", "at": "box"})
    assert r.success is False and r.error_kind == "lost_object"


def test_set_gripper_close_updates_state_and_jaw():
    sim = FakeSim()
    ctx = make_context(sim, FakeKin(), FakeLocate({}))
    r = SetGripper().call(ctx, {"open": False})
    assert r.success and ctx.state.gripper_open is False
    assert sim.moves[-1]["jaw"] == 0.0


from pickparts_agent.agent.atoms import (
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


def test_lift_follows_last_grasp_even_when_a_was_seen_first():
    locate = FakeLocate({"A": [-.08, -.34, .739], "B": [.01, -.34, .739]})
    ctx = make_context(locate=locate)
    FindObject().call(ctx, {"target": "A"})
    FindObject().call(ctx, {"target": "B"})
    ReachAbove().call(ctx, {"target": "B"})
    Grasp().call(ctx, {"target": "B"})
    locate.add("B", [.01, -.34, .84])
    result = Lift().call(ctx, {})
    assert result.success
    assert ctx.state.held_object == "B"


def test_reach_opens_gripper_in_world_state_before_retry_grasp():
    ctx = make_context(locate=FakeLocate({"C": [-.08, -.34, .739]}))
    FindObject().call(ctx, {"target": "C"})
    ctx.state.set_gripper(False)
    assert ReachAbove().call(ctx, {"target": "C"}).success
    assert Grasp().call(ctx, {"target": "C"}).success


def test_failed_refresh_invalidates_visible_record():
    locate = FakeLocate({"A": [-.08, -.34, .739]})
    ctx = make_context(locate=locate)
    FindObject().call(ctx, {"target": "A"})
    locate.fail_on.add("A")
    assert not FindObject().call(ctx, {"target": "A"}).success
    assert not ReachAbove().check_pre(ctx, {"target": "A"}).ok


def test_stack_verification_uses_destination_and_vertical_relation():
    locate = FakeLocate({"C": [.01, -.34, .785], "B": [.01, -.34, .749]})
    ctx = make_context(locate=locate)
    result = VerifyState().call(ctx, {"target": "C", "at": "B", "relation": "on"})
    assert result.success
    locate.add("C", [.05, -.34, .785])
    result = VerifyState().call(ctx, {"target": "C", "at": "B", "relation": "on"})
    assert not result.success


def test_place_on_checks_self_target_before_moving():
    from pickparts_agent.agent.atoms import build_default_registry
    registry = build_default_registry()
    assert registry.has("place_on")
    ctx = make_context(locate=FakeLocate({"C": [.01, -.34, .749]}))
    FindObject().call(ctx, {"target": "C"})
    ctx.state.set_held("C")
    result = registry.get("place_on").call(ctx, {"target": "C"})
    assert not result.success and ctx.sim.moves == []


def test_grasp_rejects_visually_oversized_unknown_object_before_motion():
    ctx = make_context(locate=FakeLocate({"tray": [.01, -.34, .749]}))
    FindObject().call(ctx, {"target": "tray"})
    ctx.state.objects["tray"].extent = [.15, .12, .03]
    result = Grasp().call(ctx, {"target": "tray"})
    assert not result.success and ctx.sim.moves == []


def test_carry_refreshes_support_moved_during_grasp():
    locate = FakeLocate({"B": [-.08, -.34, .749]})
    kin = FakeKin()
    ctx = make_context(kin=kin, locate=locate)
    FindObject().call(ctx, {"target": "B"})
    ctx.state.set_held("A")
    locate.add("B", [.01, -.31, .739])
    result = CarryTo().call(ctx, {"container": "B"})
    assert result.success
    np.testing.assert_allclose(kin.position[:2], [.01, -.31])


def test_motion_checks_settled_tracking_without_relaxing_tolerance():
    from pickparts_agent.agent.atoms.manipulation import _cartesian
    kin, sim = FakeKin(), FakeSim()
    ctx = make_context(sim=sim, kin=kin)
    move = sim.move_right

    def delayed_move(*args, **kwargs):
        move(*args, **kwargs)
        kin.forward_offset = .006  # Slight actuator lag, over 8 mm in 3D.

    sim.move_right = delayed_move
    sim.hold = lambda steps: setattr(kin, "forward_offset", 0.)
    assert _cartesian(ctx, [0., -.34, .85]).success
    sim.hold = lambda steps: None
    assert not _cartesian(ctx, [0., -.34, .86]).success
