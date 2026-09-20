import numpy as np

from helpers import FakeKin, FakeLocate, FakeSim, make_context


def test_fake_sim_move_updates_qpos():
    sim = FakeSim()
    q = np.array([0.1, 2.6, 2.8, 0.0, 1.57])
    sim.move_right(q, jaw=0.0)
    assert np.allclose(sim.arm_qpos(), q)


def test_fake_kin_forward_tracks_last_solved_target():
    kin = FakeKin()
    target = np.array([-0.08, -0.34, 0.84])
    kin.solve(target)
    assert np.allclose(kin.forward(None)[:3, 3], target)


def test_fake_locate_returns_scripted_point():
    locate = FakeLocate({"A": [-0.08, -0.34, 0.74]})
    out = locate(None, "A")
    assert np.allclose(out["point"], [-0.08, -0.34, 0.74])


def test_make_context_wires_parts():
    ctx = make_context(FakeSim(), FakeKin(), FakeLocate({}), state=None)
    assert hasattr(ctx, "sim") and callable(ctx.locate)
