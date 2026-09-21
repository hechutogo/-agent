"""Controller lifecycle checks without physics or privileged observations."""
import json
from pathlib import Path

import numpy as np
import pytest

from pickparts_agent.baseline import motion
from pickparts_agent.scene.perception import Frame


class SensorSimulation:
    joint_names = motion.ARM_NAMES

    def __init__(self):
        self.qpos = np.zeros(5)
        self.actions = []
        self.move_error = None
        self.observe_error = None

    def observe(self):
        if self.observe_error is not None:
            raise self.observe_error
        return Frame(np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 4)),
                     np.eye(3), np.eye(4), self.qpos.copy(), np.zeros(5))

    def move_right(self, joints, jaw=None, steps=35):
        self.actions.append(("move", jaw, steps))
        self.qpos = np.asarray(joints).copy()
        if self.move_error is not None:
            raise self.move_error

    def hold(self, steps=1):
        self.actions.append(("hold", steps))


class LinearKinematics:
    """Keep orchestration and _move real, replacing only expensive robot IK."""
    def __init__(self):
        self.error = None

    def solve(self, point, seed=None):
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        return np.r_[point, 0., 0.]

    def forward(self, joints):
        pose = np.eye(4)
        pose[:3, 3] = joints[:3]
        return pose


class ScriptedPerception:
    def __init__(self):
        self.responses = iter(())

    def task(self, target, *, preflight_error=None, lift_error=None):
        initial = {"point": [-.08, -.34, .74], "bbox": [1, 1, 2, 2]}
        box = {"point": [.08, -.30, .74], "bbox": [0, 0, 4, 4]}
        lifted = {"point": [-.08, -.34, .84], "bbox": [1, 1, 2, 2]}
        final = {"point": [.08, -.30, .76], "bbox": [1, 1, 2, 2]}
        self.responses = iter([
            (target, preflight_error or initial), ("box", box),
            (target, lift_error or lifted), (target, final), ("box", box),
        ])

    def locate(self, frame, target):
        expected, response = next(self.responses)
        assert target == expected
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.fixture
def controller(monkeypatch, tmp_path):
    monkeypatch.setattr(motion, "ArmKinematics", LinearKinematics)
    return motion.PickPlace(SensorSimulation(), ScriptedPerception(), tmp_path)


def assert_rejected_without_actuation(controller, target="B"):
    previous = list(controller.sim.actions)
    controller.perception.task(target)
    result = controller.run(target)
    assert result["success"] is False
    assert result["target"] == target
    assert "recovery required" in result["message"].lower()
    assert "restart" in result["message"].lower()
    assert controller.sim.actions == previous
    folder = Path(result["artifacts"])
    assert json.loads((folder / "result.json").read_text()) == result
    assert json.loads((folder / "moves.json").read_text()) == []


def test_lift_localization_failure_blocks_next_b_and_stays_latched(controller):
    controller.perception.task("A", lift_error=RuntimeError("lift localization failed"))
    result = controller.run("A")
    assert result["success"] is False
    assert result["message"] == "lift localization failed"
    assert [a[1] for a in controller.sim.actions if a[0] == "move" and a[1] is not None] == [.8, 0.]
    assert_rejected_without_actuation(controller)
    assert_rejected_without_actuation(controller, "A")


@pytest.mark.parametrize("failure", ["perception", "ik"])
def test_preflight_failure_allows_later_task(controller, failure):
    error = ValueError("preflight failed")
    controller.perception.task("A", preflight_error=error if failure == "perception" else None)
    if failure == "ik":
        controller.kin.error = error
    result = controller.run("A")
    assert result["success"] is False
    assert result["message"] == "preflight failed"
    assert controller.sim.actions == []
    controller.perception.task("B")
    assert controller.run("B")["success"] is True
    assert controller.sim.actions


def test_successful_controller_remains_usable(controller):
    for target in ("A", "B"):
        controller.perception.task(target)
        previous = len(controller.sim.actions)
        result = controller.run(target)
        assert result["success"] is True
        assert result["lift_m"] == pytest.approx(.10)
        assert len(controller.sim.actions) > previous


def test_first_actuation_failure_also_latches(controller):
    controller.perception.task("A")
    controller.sim.move_error = RuntimeError("partial move failed")
    assert controller.run("A")["success"] is False
    assert controller.sim.actions == [("move", .8, 35)]
    controller.sim.move_error = None
    assert_rejected_without_actuation(controller)


@pytest.mark.parametrize("error_type", [LookupError, KeyboardInterrupt])
def test_unexpected_failure_is_recorded_propagated_and_latched(controller, error_type):
    error = error_type("unexpected lift failure")
    controller.perception.task("A", lift_error=error)
    with pytest.raises(error_type) as caught:
        controller.run("A")
    assert caught.value is error
    saved = list(controller.output.glob("*/result.json"))
    assert len(saved) == 1
    result = json.loads(saved[0].read_text())
    assert result["success"] is False
    assert "unexpected lift failure" in result["message"]
    assert_rejected_without_actuation(controller)


def test_error_capture_failure_does_not_mask_original_exception(controller, monkeypatch):
    error = LookupError("original localization failure")
    controller.perception.task("A", lift_error=error)
    capture = controller._capture

    def fail_error_capture(folder, stage):
        if stage == "error":
            raise OSError("camera unavailable")
        return capture(folder, stage)

    monkeypatch.setattr(controller, "_capture", fail_error_capture)
    with pytest.raises(LookupError) as caught:
        controller.run("A")
    assert caught.value is error
    result = json.loads(next(controller.output.glob("*/result.json")).read_text())
    assert result["success"] is False
    assert result["message"] == str(error)
    assert_rejected_without_actuation(controller)


def test_handled_failure_survives_error_capture_failure(controller, monkeypatch):
    controller.perception.task("A", lift_error=RuntimeError("lift failed"))
    capture = controller._capture

    def fail_error_capture(folder, stage):
        if stage == "error":
            raise OSError("camera unavailable")
        return capture(folder, stage)

    monkeypatch.setattr(controller, "_capture", fail_error_capture)
    result = controller.run("A")
    assert result["success"] is False
    assert result["message"] == "lift failed"
    assert_rejected_without_actuation(controller)


def test_artifact_failure_after_motion_latches_and_saves_result_when_possible(controller, monkeypatch):
    controller.perception.task("A")
    write_text = Path.write_text

    def fail_moves(path, *args, **kwargs):
        if path.name == "moves.json":
            raise OSError("moves artifact unavailable")
        return write_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_moves)
        with pytest.raises(OSError, match="moves artifact unavailable"):
            controller.run("A")
    saved = list(controller.output.glob("*/result.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["success"] is False
    assert_rejected_without_actuation(controller)
