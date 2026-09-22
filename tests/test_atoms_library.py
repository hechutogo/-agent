import numpy as np

from helpers import FakeKin, FakeLocate, FakeSim, make_context
from pickparts_agent.agent.atoms import (
    FindObject, PlaceOn, PlaceOnTable, SetGripper, VerifyState)
from pickparts_agent.agent.state import WorldState
from pickparts_agent.scene.perception import Frame


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


def test_find_object_rejects_virtual_table_without_calling_locator():
    locate = FakeLocate({})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    result = FindObject().call(ctx, {"target": "table"})
    assert not result.success and result.error_kind == "precondition"
    assert locate.calls == []


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


def test_verify_state_rejects_table_as_movable_target_before_vision():
    locate = FakeLocate({})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    result = VerifyState().call(
        ctx, {"target": "table", "at": "table", "relation": "table"})
    assert not result.success and result.error_kind == "precondition"
    assert locate.calls == []


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


def test_set_gripper_cannot_release_held_object_without_placement_atom():
    ctx = make_context()
    ctx.state.set_gripper(False)
    ctx.state.set_held("A")
    result = SetGripper().call(ctx, {"open": True})
    assert not result.success and result.error_kind == "precondition"
    assert ctx.state.held_object == "A" and not ctx.state.gripper_open
    assert ctx.sim.moves == []


def test_place_on_rejects_virtual_table_even_if_state_is_poisoned():
    state = WorldState()
    state.apply_observation("table", {
        "bbox": [1, 1, 3, 3], "point": [0., -.4, .72],
        "confidence": 1., "top_z": .72, "bottom_z": .72,
        "extent": [.4, .4, 0.],
    }, state.tick())
    state.set_gripper(False)
    state.set_held("A")
    ctx = make_context(state=state)
    result = PlaceOn().call(ctx, {"target": "table"})
    assert not result.success and result.error_kind == "precondition"
    assert ctx.state.held_object == "A" and ctx.sim.moves == []


from pickparts_agent.agent.atoms import (
    CarryTo, Grasp, Lift, ReachAbove, ReleaseInto, ResetArm, REST_Q)


def prepared(ctx):
    FindObject().call(ctx, {"target": "A"})
    FindObject().call(ctx, {"target": "box"})
    ReachAbove().call(ctx, {"target": "A"})
    Grasp().call(ctx, {"target": "A"})


def table_frame(sim, *, visible=True):
    height, width = 120, 160
    depth = np.ones((height, width), dtype=float)
    depth[52:68, 72:88] = .96
    if not visible:
        depth[:] = np.nan
    transform = np.diag([1., 1., -1., 1.])
    transform[:3, 3] = [0., -.45, 1.72]
    intrinsic = np.array([[300., 0, width / 2],
                          [0, 300., height / 2],
                          [0, 0, 1.]])
    return Frame(np.zeros((height, width, 3), np.uint8), depth, intrinsic,
                 transform, sim.qpos.copy(), np.zeros_like(sim.qpos))


def held_over_table(*, visible=True, kin=None):
    sim, kin, locate = FakeSim(), kin or FakeKin(), FakeLocate({})
    sim.observe = lambda: table_frame(sim, visible=visible)
    state = WorldState()
    state.apply_observation("A", {
        "bbox": [72, 52, 88, 68], "point": [0., -.45, .84],
        "confidence": 1., "top_z": .856, "bottom_z": .82,
        "extent": [.024, .024, .036],
    }, state.tick())
    state.set_gripper(False)
    state.set_held("A")
    state.grasp_offset = [0., 0., .02]
    state.grasp_bottom_offset = .02
    return make_context(sim, kin, locate, state), locate


class FirstPathBlockedKin:
    def solve(self, position, seed=None):
        position = np.asarray(position, dtype=float)
        if .02 < position[0] < .04 and -.44 < position[1] < -.38:
            raise ValueError("intermediate waypoint is unreachable")
        return np.r_[position, 0., 1.57]

    def forward(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q)[:3]
        pose[2, 1] = .99
        return pose


def test_place_on_table_measures_safe_cell_without_localizing_table():
    ctx, locate = held_over_table()
    result = PlaceOnTable().call(ctx, {})
    assert result.success
    assert ctx.state.held_object is None and ctx.state.gripper_open
    assert result.observed["table_height"] == .72
    x, y = result.observed["placement_xy"]
    assert (x - .012 >= .0192 + .018 or x + .012 <= -.0256 - .018
            or y - .012 >= -.4308 + .018 or y + .012 <= -.4756 - .018)
    assert locate.calls == []


def test_place_on_table_skips_candidate_with_unreachable_intermediate_path():
    kin = FirstPathBlockedKin()
    ctx, _ = held_over_table(kin=kin)
    ctx.sim.qpos[:3] = [0., -.34, .84]
    result = PlaceOnTable().call(ctx, {})
    assert result.success
    assert np.allclose(result.observed["placement_xy"], [0., -.40])


def test_place_on_table_keeps_hold_when_support_is_not_visible():
    ctx, _ = held_over_table(visible=False)
    result = PlaceOnTable().call(ctx, {})
    assert not result.success and result.error_kind == "lost_object"
    assert ctx.state.held_object == "A" and not ctx.state.gripper_open
    assert ctx.sim.moves == []


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
