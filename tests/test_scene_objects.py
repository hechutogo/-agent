"""Private generator tests may inspect truth; runtime perception never does."""
import colorsys
import copy
from types import SimpleNamespace

import numpy as np
import pytest


def layout(seed=None, objects=None):
    from pickparts_agent.scene.objects import _SceneObjects
    return _SceneObjects(seed=seed, objects=objects)


def test_seed_reproduces_layout_but_default_and_other_seeds_randomize():
    first, same, other = layout(7), layout(7), layout(8)
    assert first.specs == same.specs == other.specs
    assert first._placements == same._placements
    assert first._placements != other._placements
    assert layout()._placements != layout()._placements


def test_catalog_is_metadata_only_and_defensively_copied():
    scene = layout(3)
    specs = scene.specs
    assert [s["id"] for s in specs] == ["A", "B", "box"]
    assert [s["kind"] for s in specs] == ["block", "block", "box"]
    assert all(set(s) == {"id", "kind", "label", "color"} for s in specs)
    assert all(len(s["color"]) == 3 for s in specs)
    specs[0]["color"][0] = 0
    assert scene.specs[0]["color"][0] > .7


@pytest.mark.parametrize("seed", range(5))
def test_all_objects_fit_table_clear_neighbors_and_pass_urdf_ik(seed):
    from pickparts_agent.scene.kinematics import ArmKinematics
    scene = layout(seed)
    kin = ArmKinematics()
    occupied = []
    for spec in scene.specs:
        x, y, _ = scene._placements[spec["id"]]
        half = np.array([.012, .012] if spec["kind"] == "block" else [.048, .045])
        assert -.30 <= x - half[0] <= x + half[0] <= .30
        assert -.67 <= y - half[1] <= y + half[1] <= -.23
        for xy, extent in occupied:
            assert np.any(np.abs(np.array([x, y]) - xy) >= half + extent + .012)
        occupied.append((np.array([x, y]), half))
        for z in [.732, .752, .80, .856, .876]:
            q = kin.solve([x, y, z])
            assert np.linalg.norm(kin.forward(q)[:3, 3] - [x, y, z]) < .003
            assert kin.forward(q)[2, 1] > .97


@pytest.mark.parametrize("objects", [
    [{"id": "A", "kind": "sphere", "label": "bad", "color": [1., 0., 0.]}],
    [{"id": "A", "kind": "block", "label": "bad", "color": [float("nan"), 0., 0.]}],
    [{"id": "A", "kind": "block", "label": "bad", "color": [1., 0., 0.],
      "position": [0, 0, 0]}],
])
def test_invalid_catalog_rejected_before_creating_physics(objects):
    with pytest.raises(ValueError):
        layout(0, objects)


def test_duplicate_ids_and_indistinguishable_colors_are_rejected():
    spec = {"id": "C", "kind": "block", "label": "C", "color": [1., .1, .1]}
    with pytest.raises(ValueError):
        layout(0, [spec, copy.deepcopy(spec)])
    duplicate_color = dict(spec, id="D")
    with pytest.raises(ValueError):
        layout(0, [spec, duplicate_color])


def test_catalog_capacity_is_finite_and_rejected_before_placement():
    from pickparts_agent.scene.objects import MAX_OBJECTS
    assert 3 < MAX_OBJECTS < 30
    with pytest.raises(ValueError, match="capacity"):
        layout(0, [{} for _ in range(MAX_OBJECTS + 1)])


def assert_safe_box(scene, position, occupied, kind="box"):
    from pickparts_agent.scene.kinematics import ArmKinematics
    xy = np.array(position[:2])
    half = np.array([.048, .045] if kind == "box" else [.012, .012])
    assert np.all(xy - half >= [-.30, -.67])
    assert np.all(xy + half <= [.30, -.23])
    for spec in scene.specs:
        if spec["id"] in occupied:
            extent = [.012, .012] if spec["kind"] == "block" else [.048, .045]
            assert np.any(np.abs(xy - occupied[spec["id"]][:2]) >=
                          half + extent + .018)
    kin = ArmKinematics()
    for z in [.732, .752, .80, .856, .876]:
        pose = kin.forward(kin.solve([*xy, z]))
        assert np.linalg.norm(pose[:3, 3] - [*xy, z]) < .003
        assert pose[2, 1] > .97


@pytest.mark.parametrize("seed", range(20))
def test_default_layout_reserves_space_for_an_extra_box(seed):
    scene = layout(seed)
    before = copy.deepcopy(scene._placements)
    position = scene._sample({"kind": "box"}, scene._placements)
    assert_safe_box(scene, position, before)
    assert scene._placements == before


def test_box_sampling_finds_open_strip_between_coarse_cells():
    scene = layout(0, objects=[])
    scene._specs = [
        {"id": "left", "kind": "box", "label": "left", "color": [1., 0., 0.]},
        {"id": "right", "kind": "box", "label": "right", "color": [0., 1., 0.]},
    ]
    # The 2 mm opening around x=-.080 lies outside the old +/-4 mm jitter
    # around x=-.100 and x=-.075. The opening itself passes real URDF IK.
    occupied = {"left": [-.195, -.32, .723], "right": [.035, -.30, .723]}
    position = scene._sample({"kind": "box"}, occupied)
    assert -.081 <= position[0] <= -.079
    assert_safe_box(scene, position, occupied)


def test_box_sampling_reports_capacity_when_live_poses_fill_reach_band():
    scene = layout(0, objects=[])
    scene._specs = [
        {"id": "left", "kind": "box", "label": "left", "color": [1., 0., 0.]},
        {"id": "right", "kind": "box", "label": "right", "color": [0., 1., 0.]},
    ]
    occupied = {"left": [-.17, -.30, .723], "right": [-.025, -.30, .723]}
    before = copy.deepcopy(occupied)
    with pytest.raises(ValueError, match="Scene capacity"):
        scene._sample({"kind": "box"}, occupied)
    assert occupied == before


@pytest.mark.parametrize("count", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", [0, 6, 7, 19])
def test_sparse_recreated_inventory_reserves_an_extra_box(count, seed):
    specs = [
        {"id": str(i), "kind": "box" if i == 0 else "block", "label": str(i),
         "color": list(colorsys.hsv_to_rgb(i / 6, .9, .85))}
        for i in range(count)
    ]
    scene = layout(seed, specs)
    assert scene.specs == specs
    position = scene._sample({"kind": "box"}, scene._placements)
    assert_safe_box(scene, position, scene._placements)


def build_without_physics(scene, monkeypatch):
    import torch

    def build_actor(_scene, _spec, position):
        return SimpleNamespace(pose=SimpleNamespace(
            p=torch.tensor([position], dtype=torch.float64)))

    monkeypatch.setattr(scene, "_build_actor", build_actor)
    scene.build(None)


@pytest.mark.parametrize("kind", ["block", "box"])
def test_add_preserves_existing_bodies_and_uses_reserved_space(monkeypatch, kind):
    scene = layout(0)
    build_without_physics(scene, monkeypatch)
    before = copy.deepcopy(scene._placements)
    actors = dict(scene._actors)
    spec = scene.add(None, kind)
    assert len(scene.specs) == 4
    assert spec["kind"] == kind
    assert_safe_box(scene, scene._placements[spec["id"]], before, kind)
    for name, actor in actors.items():
        assert scene._actors[name] is actor
        assert actor.pose.p[0].tolist() == before[name]
        assert scene._placements[name] == before[name]


def test_add_rechecks_reservation_against_live_poses(monkeypatch):
    scene = layout(0)
    build_without_physics(scene, monkeypatch)
    scene._actors["A"].pose.p[0, :2] = scene._actors["A"].pose.p.new_tensor(
        scene._reserved_box[:2])
    scene._actors["B"].pose.p[0, :2] = scene._actors["B"].pose.p.new_tensor([.2, -.6])
    scene._actors["box"].pose.p[0, :2] = scene._actors["box"].pose.p.new_tensor(
        [-.2, -.6])
    occupied = {name: actor.pose.p[0].tolist() for name, actor in scene._actors.items()}
    before = copy.deepcopy(scene._placements)
    spec = scene.add(None, "box")
    assert_safe_box(scene, scene._placements[spec["id"]], occupied)
    for name, position in occupied.items():
        assert scene._actors[name].pose.p[0].tolist() == position
        assert scene._placements[name] == before[name]


def test_add_capacity_failure_leaves_inventory_and_bodies_unchanged(monkeypatch):
    scene = layout(0, objects=[])
    scene._specs = [
        {"id": "left", "kind": "box", "label": "left", "color": [1., 0., 0.]},
        {"id": "right", "kind": "box", "label": "right", "color": [0., 1., 0.]},
    ]
    scene._placements = {"left": [-.17, -.30, .723], "right": [-.025, -.30, .723]}
    build_without_physics(scene, monkeypatch)
    before, specs, actors = copy.deepcopy(scene._placements), scene.specs, dict(scene._actors)
    with pytest.raises(ValueError, match="Scene capacity"):
        scene.add(None, "box")
    assert scene.specs == specs
    assert scene._placements == before
    for name, actor in actors.items():
        assert scene._actors[name] is actor
        assert actor.pose.p[0].tolist() == before[name]


def test_full_inventory_does_not_require_reservation_and_rejects_add():
    specs = [
        {"id": str(i), "kind": "block", "label": str(i),
         "color": list(colorsys.hsv_to_rgb(i / 6, .9, .85))}
        for i in range(6)
    ]
    scene = layout(0, specs)
    assert len(scene._placements) == 6
    with pytest.raises(ValueError, match="Scene capacity"):
        scene.add(None, "box")
    assert scene.specs == specs


def test_no_reachable_cells_reports_layout_capacity(monkeypatch):
    from pickparts_agent.scene import objects as scene_objects
    monkeypatch.setattr(scene_objects, "_reachable_cells", lambda: ())
    with pytest.raises(ValueError, match="Scene capacity.*layout"):
        layout(0)


def test_reset_crowded_existing_inventory_does_not_require_another_box():
    specs = [
        {"id": str(i), "kind": "box" if i < 2 else "block", "label": str(i),
         "color": list(colorsys.hsv_to_rgb(i / 6, .9, .85))}
        for i in range(5)
    ]
    scene = layout(2, specs)
    assert scene.specs == specs and len(scene._placements) == 5


def test_blocks_leave_room_for_open_gripper_not_only_their_own_footprints():
    scene = layout(9)
    a, b = (np.array(scene._placements[name][:2]) for name in ("A", "B"))
    assert np.any(np.abs(a - b) >= .024 + .03)
