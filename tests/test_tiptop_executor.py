"""Tests for open-loop execution (TiPToP semantics: no visual feedback)."""
import inspect
import json

import numpy as np
import pytest

from tiptop_mac.executor import OpenLoopExecutor
from tiptop_mac.types import MotionStep, TAMPPlan

from helpers import FakeKin, FakeSim
from pickparts_agent.agent.atoms.manipulation import REST_Q


def make_plan(steps):
    return TAMPPlan(goal=(), operators=(), trajectory=tuple(steps),
                    planning_time=0.0, rationale="")


def sample_steps():
    return [
        MotionStep("move", np.array([0., -.45, .90]), .8),
        MotionStep("move", np.array([0., -.45, .78]), .8),
        MotionStep("gripper", None, 0.),
        MotionStep("move", np.array([.1, -.45, .90]), 0.),
        MotionStep("gripper", None, .8),
        MotionStep("reset", None, .8),
    ]


def test_executes_full_trajectory_and_reports_success():
    sim, kin = FakeSim(), FakeKin()
    result = OpenLoopExecutor(sim, kin).execute(make_plan(sample_steps()))
    assert result["success"] and not result["aborted"]
    assert sim.moves, "every step should drive the arm"
    # reset is the final move and uses REST_Q
    assert np.allclose(np.array(sim.moves[-1]["joints"]), REST_Q)
    assert sim.moves[-1]["jaw"] == .8


def test_empty_trajectory_is_success_without_motion():
    sim, kin = FakeSim(), FakeKin()
    result = OpenLoopExecutor(sim, kin).execute(make_plan([]))
    assert result["success"] and not result["aborted"]
    assert sim.moves == []


def test_tracking_failure_aborts_immediately_without_replan():
    sim, kin = FakeSim(), FakeKin()
    sim.tracking_error = True
    result = OpenLoopExecutor(sim, kin).execute(make_plan(sample_steps()))
    assert not result["success"] and result["aborted"]
    # failed first command means no recorded motion and no retry attempts
    assert sim.moves == []


def test_gripper_step_uses_current_arm_qpos():
    sim, kin = FakeSim(), FakeKin()
    OpenLoopExecutor(sim, kin).execute(
        make_plan([MotionStep("gripper", None, .35)]))
    assert len(sim.moves) == 1
    assert np.allclose(np.array(sim.moves[0]["joints"]), sim.arm_qpos())
    assert sim.moves[0]["jaw"] == .35


def test_unknown_step_kind_aborts():
    sim, kin = FakeSim(), FakeKin()
    result = OpenLoopExecutor(sim, kin).execute(
        make_plan([MotionStep("teleport", np.zeros(3), .8)]))
    assert not result["success"] and result["aborted"]


def test_no_vision_used_unless_calibrate():
    sim, kin = FakeSim(), FakeKin()

    def locate(frame, target):
        raise AssertionError("must not locate without a calibrate step")

    result = OpenLoopExecutor(sim, kin, locate=locate).execute(
        make_plan(sample_steps()))
    assert result["success"]


def test_calibrate_translates_following_move_targets():
    sim = FakeSim()
    targets = []

    class RecordingKin(FakeKin):
        def solve(self, position, seed=None):
            targets.append(np.asarray(position, float).copy())
            return super().solve(position, seed)

    kin = RecordingKin()

    def locate(frame, target):
        # Block hangs 10mm in +y and 1mm in +x below TCP; bottom offset .02.
        tcp = kin.forward(np.zeros(5))[:3, 3]
        point = np.array([tcp[0] + .01, tcp[1] + .10, tcp[2]])
        return {"point": point, "bottom_z": tcp[2] - .02}

    import json
    label = json.dumps({"held": "A", "assumed": .018})
    destination = np.array([.10, -.45, .779])
    steps = [
        MotionStep("calibrate", None, None, label=label),
        MotionStep("move", destination, .8),
    ]
    result = OpenLoopExecutor(sim, kin, locate=locate).execute(
        make_plan(steps))
    assert result["success"]
    shifted = targets[-1]
    # shift = tcp - block = (-.01, -.10, .002)
    assert np.allclose(shifted[:2], [.09, -.55], atol=1e-6)
    assert abs(shifted[2] - .781) < 1e-6


def test_calibrate_without_locator_aborts():
    sim, kin = FakeSim(), FakeKin()
    steps = [MotionStep("calibrate", None, None,
                        label='{"held": "A", "assumed": .018}')]
    result = OpenLoopExecutor(sim, kin).execute(make_plan(steps))
    assert result["aborted"]


def test_result_is_json_serializable_message():
    sim, kin = FakeSim(), FakeKin()
    result = OpenLoopExecutor(sim, kin).execute(make_plan(sample_steps()))
    json.dumps({"message": result["message"]})
