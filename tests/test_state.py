import numpy as np

from pickparts_agent.agent.state import ObjectRecord, WorldState


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
    assert out.visible and out.frame_id == 3 and out.placement is None


def test_point_near_box_does_not_prove_containment():
    ws = WorldState()
    ws.update_object(ObjectRecord(
        "box", True, [0, 0, 9, 9], [0.085, -0.30, 0.726], 0.9, 1, "table"))
    out = ws.apply_observation(
        "A", {"bbox": [2, 2, 6, 6], "point": np.array([0.085, -0.30, 0.760]),
              "confidence": 0.9}, frame_id=2)
    assert out.placement is None


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


def test_opening_gripper_clears_held_geometry_and_stale_placement():
    ws = WorldState()
    ws.update_object(rec(placement="gripper"))
    ws.set_gripper(False)
    ws.set_held("A")
    ws.grasp_target = "A"
    ws.grasp_offset = [.01, .01, .02]
    ws.grasp_bottom_offset = .03
    ws.set_gripper(True)
    assert ws.held_object is None and ws.grasp_target is None
    assert ws.grasp_offset is None and ws.grasp_bottom_offset is None
    assert ws.objects["A"].placement is None
    assert not ws.objects["A"].visible
