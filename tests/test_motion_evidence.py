import numpy as np
import pytest

from helpers import FakeSim, make_context
from test_tiptop_optimized_executor import CartesianKin


class VerticalKin(CartesianKin):
    def forward(self, joints):
        pose = super().forward(joints)
        pose[2, 1] = 1.
        return pose


def test_interior_ik_failure_is_detected_before_any_motion():
    from pickparts_agent.agent.atoms.manipulation import _cartesian
    class Restricted(VerticalKin):
        def solve(self, position, seed=None):
            if .015 < position[0] < .025:
                raise ValueError("blocked waypoint")
            return super().solve(position, seed)
    sim = FakeSim()
    sim.qpos[:5] = [0., -.45, .8, 0., 0.]
    with pytest.raises(ValueError, match="blocked waypoint"):
        _cartesian(make_context(sim=sim, kin=Restricted()), [.04, -.45, .8])
    assert sim.moves == []


def test_tracking_failure_retains_measured_error_and_waypoint():
    from pickparts_agent.agent.atoms.manipulation import _cartesian
    sim = FakeSim()
    sim.qpos[:5] = [0., -.45, .8, 0., 0.]
    original = sim.move_right
    def lagging(joints, **kwargs):
        original(joints, **kwargs)
        sim.qpos[0] -= .012
    sim.move_right = lagging
    result = _cartesian(make_context(sim=sim, kin=VerticalKin()), [.04, -.45, .8])
    assert not result.success
    assert result.observed["error_m"] == pytest.approx(.012)
    assert result.observed["waypoint_index"] == 0
    assert len(result.observed["actual_tcp"]) == 3


def test_self_grasp_close_interruption_keeps_candidate():
    from pickparts_agent.agent.atoms import FindObject, Grasp
    from helpers import FakeLocate
    sim = FakeSim()
    sim.qpos[:5] = [0., -.45, .738, 0., 0.]
    ctx = make_context(sim=sim, kin=CartesianKin(),
                       locate=FakeLocate({"A": [0., -.45, .738]}))
    FindObject().call(ctx, {"target": "A"})
    original = sim.move_right
    def interrupt(joints, jaw=None, **kwargs):
        original(joints, jaw=jaw, **kwargs)
        if jaw == 0.:
            raise RuntimeError("closed before backend interrupted")
    sim.move_right = interrupt
    result = Grasp().call(ctx, {"target": "A"})
    assert not result.success
    assert ctx.state.grasp_target == "A"
    assert not ctx.state.gripper_open
    assert ctx.state.held_object is None
