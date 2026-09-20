import numpy as np

from pickparts_agent.state import ObjectRecord, WorldState


def rec(id="A", xy=(-0.085, -0.34), z=0.739, placement=None):
    return ObjectRecord(id, True, [1, 1, 4, 4],
                        [xy[0], xy[1], z], 0.9, 1, placement)


def test_snapshot_is_json_safe_and_decoupled():
    ws = WorldState()
    ws.update_object(rec())
    snap = ws.snapshot()
    assert snap["gripper_open"] is True and snap["held_object"] is None
    a = snap["objects"]["A"]
    assert a["point"] == [-0.085, -0.34, 0.739]
    snap["objects"]["A"]["point"][0] = 999
    assert ws.snapshot()["objects"]["A"]["point"][0] == -0.085


def test_apply_observation_updates_record():
    ws = WorldState()
    out = ws.apply_observation(
        "A", {"bbox": [1, 1, 4, 4], "point": np.array([-0.085, -0.34, 0.739]),
              "confidence": 0.9}, frame_id=3)
    assert out.visible and out.frame_id == 3 and out.placement == "table"


def test_placement_in_box_when_xy_and_z_band_match():
    ws = WorldState()
    ws.update_object(ObjectRecord(
        "box", True, [0, 0, 9, 9], [0.085, -0.30, 0.726], 0.9, 1, "table"))
    out = ws.apply_observation(
        "A", {"bbox": [2, 2, 6, 6], "point": np.array([0.085, -0.30, 0.760]),
              "confidence": 0.9}, frame_id=2)
    assert out.placement == "box"


def test_held_object_placement_is_gripper():
    ws = WorldState()
    ws.set_gripper(False)
    ws.set_held("A")
    out = ws.apply_observation(
        "A", {"bbox": [2, 2, 6, 6], "point": np.array([0.085, -0.30, 0.84]),
              "confidence": 0.9}, frame_id=2)
    assert out.placement == "gripper"


def test_tick_monotonic():
    ws = WorldState()
    assert ws.tick() == 1 and ws.tick() == 2
