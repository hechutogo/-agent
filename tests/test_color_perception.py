import numpy as np
import pytest
from pickparts_agent.scene.perception import ColorPerception, Frame


def frame(rgb, depth):
    return Frame(rgb, depth, np.array([[100., 0, 10], [0, 100., 10], [0, 0, 1]]),
                 np.eye(4), np.zeros(14), np.zeros(14))


def test_color_detection_projects_only_selected_part():
    rgb = np.zeros((30, 40, 3), dtype=np.uint8)
    rgb[8:13, 18:23] = [200, 10, 10]
    rgb[20:26, 25:31] = [10, 10, 200]
    detection = ColorPerception().locate(frame(rgb, np.full((30, 40), 2.)), "A")
    np.testing.assert_allclose(detection["point"], [.2, 0., 2.], atol=.001)


def test_visible_color_without_depth_cannot_command_motion():
    rgb = np.zeros((30, 40, 3), dtype=np.uint8)
    rgb[8:13, 18:23] = [200, 10, 10]
    with pytest.raises(ValueError):
        ColorPerception().locate(frame(rgb, np.zeros((30, 40))), "A")


def test_absent_target_reports_failure():
    with pytest.raises(ValueError):
        ColorPerception().locate(frame(np.zeros((30, 40, 3), np.uint8),
                                     np.ones((30, 40))), "B")
