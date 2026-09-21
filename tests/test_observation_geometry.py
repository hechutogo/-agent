import numpy as np
import pytest

from pickparts_agent.scene.perception import backproject, parse_target, Frame


def test_backprojection_uses_depth_and_calibration():
    k = np.array([[100., 0, 10], [0, 100., 10], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    points = backproject(np.array([[20, 10]]), np.array([2.]), k, transform)
    np.testing.assert_allclose(points, [[1.2, 2., 5.]])


@pytest.mark.parametrize("depth", [0., -1., np.nan, np.inf])
def test_invalid_depth_cannot_become_grasp_target(depth):
    with pytest.raises(ValueError):
        backproject(np.array([[0., 0.]]), np.array([depth]), np.eye(3), np.eye(4))


@pytest.mark.parametrize("text,target", [("请帮我拿零件A", "A"), ("拿 b 放盒里", "B")])
def test_instruction_selects_named_part(text, target):
    assert parse_target(text) == target


@pytest.mark.parametrize("text", ["拿零件", "拿A或者B", "拿C", "不要拿A"])
def test_ambiguous_or_negated_command_cannot_move_robot(text):
    with pytest.raises(ValueError):
        parse_target(text)


def test_frame_does_not_accept_privileged_success():
    with pytest.raises(TypeError):
        Frame(rgb=np.zeros((2, 2, 3), np.uint8), depth=np.ones((2, 2)),
              intrinsic=np.eye(3), camera_to_base=np.eye(4),
              qpos=np.zeros(12), qvel=np.zeros(12), success=True)
