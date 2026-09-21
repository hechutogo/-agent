import numpy as np
import pytest

from pickparts_agent.scene.perception import Frame


SPECS = [{"id": "part-17", "kind": "block", "label": "cyan part",
          "color": [0.05, .8, .8]}]


def sensor_frame():
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    rgb[30:50, 60:80] = [12, 204, 204]
    depth = np.full((80, 100), .75)
    return Frame(rgb, depth, np.array([[600., 0, 50], [0, 600., 40], [0, 0, 1]]),
                 np.eye(4), np.zeros(14), np.zeros(14))


def perception(specs=SPECS):
    from pickparts_agent.scene.perception import ScenePerception
    return ScenePerception(specs)


def test_dynamic_id_uses_full_image_and_returns_sensor_geometry():
    detected = perception().locate(sensor_frame(), "part-17")
    assert detected["bbox"] == [60, 30, 80, 50]
    assert detected["refinement"] == "rgbd"
    assert .7 <= detected["confidence"] <= 1.
    np.testing.assert_allclose(detected["point"], [.024375, -.000625, .75], atol=.001)
    assert detected["top_z"] == pytest.approx(.75)
    assert detected["bottom_z"] == pytest.approx(.75)
    np.testing.assert_allclose(detected["extent"], [.02375, .02375, 0], atol=.001)


def test_unknown_text_is_not_guessed_as_catalog_id():
    with pytest.raises(ValueError, match="Unknown"):
        perception().locate(sensor_frame(), "the thing on the left")


def test_two_significant_components_are_ambiguous_even_with_different_sizes():
    frame = sensor_frame()
    frame.rgb[4:18, 5:20] = [12, 204, 204]
    with pytest.raises(ValueError, match="ambiguous"):
        perception().locate(frame, "part-17")


def test_small_noise_does_not_pull_center_away_from_object():
    frame = sensor_frame()
    frame.rgb[4:6, 5:7] = [12, 204, 204]
    assert perception().locate(frame, "part-17")["bbox"] == [60, 30, 80, 50]


def test_thin_same_hue_background_is_not_a_second_block():
    frame = sensor_frame()
    frame.rgb[4:24, 5:8] = [12, 204, 204]
    assert perception().locate(frame, "part-17")["bbox"] == [60, 30, 80, 50]


def test_touching_desaturated_robot_surface_does_not_merge_with_block():
    frame = sensor_frame()
    frame.rgb[10:60, 30:60] = [120, 200, 200]
    assert perception().locate(frame, "part-17")["bbox"] == [60, 30, 80, 50]


def test_heavily_occluded_sliver_cannot_supply_grasp_center():
    frame = sensor_frame()
    frame.rgb[30:50, 60:77] = 0
    with pytest.raises(ValueError, match="occluded"):
        perception().locate(frame, "part-17")


@pytest.mark.parametrize("bad", [0, np.nan, np.inf])
def test_mostly_missing_depth_is_rejected_instead_of_biased_grasp(bad):
    frame = sensor_frame()
    frame.depth[30:47, 60:80] = bad
    with pytest.raises(ValueError, match="depth"):
        perception().locate(frame, "part-17")


def test_clipped_or_fragmented_object_is_rejected():
    frame = sensor_frame()
    frame.rgb[30:50, 60:80] = 0
    frame.rgb[30:50, :10] = [12, 204, 204]
    with pytest.raises(ValueError, match="occluded|clipped"):
        perception().locate(frame, "part-17")


@pytest.mark.parametrize("other_depth", [.8, 1.5])
def test_depth_discontinuity_cannot_join_two_surfaces_into_one_object(other_depth):
    frame = sensor_frame()
    frame.depth[30:40, 60:80] = other_depth
    with pytest.raises(ValueError, match="depth|ambiguous"):
        perception().locate(frame, "part-17")
