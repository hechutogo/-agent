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
