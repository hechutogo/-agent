"""Tests for the one-shot TiPToP scene perception on synthetic RGB-D frames."""
import colorsys

import numpy as np
import pytest

from pickparts_agent.scene.perception import Frame
from tiptop_mac.perception import (
    build_scene_graph, fit_table, propose_grasps, segment_objects,
)


SPECS = [
    {"id": "A", "kind": "block", "label": "物块 A", "color": [.86, .16, .16]},
    {"id": "box", "kind": "box", "label": "盒子", "color": [.16, .70, .25]},
]


def make_frame(block_xy=(4, 8), block_depth=.725, table_depth=.745):
    h = w = 20
    rgb = np.full((h, w, 3), [120, 90, 60], dtype=np.uint8)
    depth = np.full((h, w), table_depth, dtype=float)
    x1, y1 = block_xy
    rgb[y1:y1 + 4, x1:x1 + 4] = [220, 40, 40]
    depth[y1:y1 + 4, x1:x1 + 4] = block_depth
    intrinsic = np.array([[90., 0, 10], [0, 90., 10], [0, 0, 1]])
    return Frame(rgb=rgb, depth=depth, intrinsic=intrinsic,
                 camera_to_base=np.eye(4), qpos=np.zeros(6), qvel=np.zeros(6))


def test_segment_object_mask_cloud_and_metrics_match_pixels():
    frame = make_frame()
    nodes = segment_objects(frame, SPECS[:1])
    node = nodes["A"]
    assert node.mask.sum() == 16
    assert node.bbox == (4, 8, 8, 12)
    assert node.cloud.shape == (16, 3)
    # Identity pose: world z equals the measured depth directly.
    assert np.unique(node.cloud[:, 2]) == pytest.approx([.725])
    assert node.point[2] == pytest.approx(.725)
    assert node.extent[0] == pytest.approx(.725 * 3 / 90, abs=.003)


def test_segment_missing_object_raises():
    frame = make_frame()
    specs = [{"id": "C", "kind": "block", "label": "C",
              "color": [.15, .25, .85]}]
    with pytest.raises(ValueError, match="not visible"):
        segment_objects(frame, specs)


def test_fit_table_prefers_plane_at_object_bottom_over_floor():
    rng = np.random.default_rng(7)
    table = np.c_[rng.uniform(-.3, .3, (400, 2)), np.full(400, .745)]
    floor = np.c_[rng.uniform(-2, 2, (400, 2)), np.zeros(400)]
    surface = fit_table(np.r_[table, floor], object_bottoms=[.73])
    assert surface.normal == pytest.approx([0, 0, 1])
    assert surface.top_z == pytest.approx(.745, abs=1e-3)
    assert surface.bounds[0, 0] < surface.bounds[1, 0]


def test_fit_table_falls_back_to_best_horizontal_without_band_match():
    rng = np.random.default_rng(3)
    floor = np.c_[rng.uniform(-1, 1, (200, 2)), np.full(200, .1)]
    surface = fit_table(floor, object_bottoms=[.9])
    assert surface.top_z == pytest.approx(.1, abs=1e-3)


def test_fit_table_rejects_non_horizontal_noise():
    rng = np.random.default_rng(5)
    points = rng.uniform(-.5, .5, (300, 3))
    with pytest.raises(ValueError, match="horizontal"):
        fit_table(points, object_bottoms=[.0])


def test_block_gets_top_grasp_and_box_gets_none():
    frame = make_frame()
    nodes = segment_objects(frame, SPECS[:1])
    grasps = propose_grasps(nodes["A"])
    assert len(grasps) == 1
    assert grasps[0].position == pytest.approx(nodes["A"].point)
    box = nodes["A"]
    box.kind = "box"
    assert propose_grasps(box) == []


def test_build_scene_graph_is_one_shot_and_indexed():
    frame = make_frame()
    scene = build_scene_graph(frame, SPECS[:1])
    assert set(scene.objects) == {"A"}
    assert scene.table.top_z == pytest.approx(.745, abs=2e-3)
    assert len(scene.objects["A"].grasps) == 1
    assert scene.qpos.shape == (6,)
