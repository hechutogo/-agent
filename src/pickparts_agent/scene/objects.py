"""Metadata catalog and private, bounded scene construction.

Only specs cross the sensor boundary. Geometry and actor poses stay here.
"""
import colorsys
from copy import deepcopy
from functools import lru_cache

import numpy as np

from .kinematics import ArmKinematics

MAX_OBJECTS = 6
_DEFAULTS = [
    {"id": "A", "kind": "block", "label": "红色物块 A", "color": [.85, .045, .03]},
    {"id": "B", "kind": "block", "label": "蓝色物块 B", "color": [.03, .12, .85]},
    {"id": "box", "kind": "box", "label": "绿色盒子", "color": [.05, .60, .12]},
]
_HALF = {"block": np.array([.012, .012]), "box": np.array([.048, .045])}
# Grasp band, intermediate lift, hover, and a two-block stacking clearance.
_HEIGHTS = (.732, .752, .80, .856, .876)


def validate_specs(objects):
    specs = deepcopy(list(_DEFAULTS if objects is None else objects))
    if len(specs) > MAX_OBJECTS:
        raise ValueError(f"Scene capacity is {MAX_OBJECTS} objects")
    ids, hues = set(), []
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {"id", "kind", "label", "color"}:
            raise ValueError("Object specs must contain only id, kind, label, color")
        if (not isinstance(spec["id"], str) or not spec["id"].strip()
                or spec["id"] in ids or spec["kind"] not in _HALF
                or not isinstance(spec["label"], str) or not spec["label"].strip()):
            raise ValueError("Invalid object identity or kind")
        try:
            color = np.asarray(spec["color"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid object color") from exc
        if (color.shape != (3,) or not np.isfinite(color).all()
                or (color < 0).any() or (color > 1).any()):
            raise ValueError("Color must contain three finite floats in [0, 1]")
        hue, saturation, value = colorsys.rgb_to_hsv(*color)
        if saturation < .55 or value < .4:
            raise ValueError("Object colors must be bright and saturated")
        if any(min(abs(hue - h), 1 - abs(hue - h)) < .075 for h in hues):
            raise ValueError("Object colors are visually ambiguous")
        ids.add(spec["id"])
        hues.append(hue)
        spec["color"] = color.tolist()
    return specs


def _reachable(kin, xy):
    try:
        seed = None
        for z in reversed(_HEIGHTS):
            seed = kin.solve([*xy, z], seed=seed)
        return True
    except ValueError:
        return False


@lru_cache(maxsize=1)
def _reachable_cells():
    kin = ArmKinematics()
    # Screen the entire table, not a hand-assumed rectangular reach region.
    return tuple((float(x), float(y))
                 for x in np.arange(-.275, .276, .025)
                 for y in np.arange(-.645, -.249, .025)
                 if _reachable(kin, (x, y)))


class _SceneObjects:
    def __init__(self, seed=None, objects=None):
        self._specs = validate_specs(objects)
        self._rng = np.random.default_rng(seed)
        self._kin = ArmKinematics()
        self._placements = {}
        self._actors = {}
        self._reserved_box = None
        # Prefer reserving an extra box, but rebuilding an existing crowded
        # inventory must not require space for a hypothetical additional box.
        ordered = sorted(self._specs, key=lambda s: s["kind"] != "box")
        reserve_extra = (len(self._specs) <= 3
                         and sum(s["kind"] == "box" for s in self._specs) < 2)
        for attempt in range(60):
            self._placements = {}
            try:
                reserved = (self._sample({"kind": "box"}, {})
                            if reserve_extra and attempt < 30 else None)
                for spec in ordered:
                    self._placements[spec["id"]] = self._sample(
                        spec, self._placements, reserved=reserved)
                self._reserved_box = reserved
                break
            except ValueError:
                continue
        else:
            raise ValueError("Scene capacity: no collision-free reachable layout")

    @property
    def specs(self):
        return deepcopy(self._specs)

    def _candidate_xy(self):
        if self._reserved_box is not None:
            yield np.array(self._reserved_box[:2])
        cells = _reachable_cells()
        order = self._rng.permutation(len(cells))
        for index in order:
            yield np.array(cells[index]) + self._rng.uniform(-.004, .004, 2)
        # Cover the gaps left by one jittered point per 25 mm cell. These
        # subcell points are candidates only: each still needs full IK checks.
        offsets = [(x, y) for x in (-.01, -.005, 0., .005, .01)
                   for y in (-.01, -.005, 0., .005, .01)]
        for index in order:
            for offset in offsets:
                yield np.array(cells[index]) + offset

    def _sample(self, spec, occupied, *, reserved=None):
        half = _HALF[spec["kind"]]
        by_id = {s["id"]: s for s in self._specs}
        for xy in self._candidate_xy():
            if (np.any(xy - half < [-.30, -.67])
                    or np.any(xy + half > [.30, -.23])):
                continue
            if (reserved is not None and
                    np.all(np.abs(xy - reserved[:2]) < half + _HALF["box"] + .018)):
                continue
            if any(np.all(np.abs(xy - np.array(pos[:2])) <
                          half + _HALF[by_id[name]["kind"]] +
                          (.03 if spec["kind"] == by_id[name]["kind"] == "block" else .018))
                   for name, pos in occupied.items()):
                continue
            if _reachable(self._kin, xy):
                return [float(xy[0]), float(xy[1]),
                        .7385 if spec["kind"] == "block" else .723]
        raise ValueError("Scene capacity: no collision-free reachable placement")

    def build(self, scene):
        for spec in self._specs:
            self._actors[spec["id"]] = self._build_actor(
                scene, spec, self._placements[spec["id"]])

    def _build_actor(self, scene, spec, position):
        import sapien
        builder = scene.create_actor_builder()
        color = [*spec["color"], 1.]
        if spec["kind"] == "block":
            builder.add_box_collision(half_size=[.012, .012, .018])
            builder.add_box_visual(half_size=[.012, .012, .018], material=color)
        else:
            for center, half in (
                ([0, 0, 0], [.045, .042, .003]),
                ([-.045, 0, .018], [.003, .042, .018]),
                ([.045, 0, .018], [.003, .042, .018]),
                ([0, -.042, .018], [.045, .003, .018]),
                ([0, .042, .018], [.045, .003, .018]),
            ):
                builder.add_box_collision(pose=sapien.Pose(center), half_size=half)
                builder.add_box_visual(pose=sapien.Pose(center), half_size=half,
                                       material=color)
        builder.initial_pose = sapien.Pose(position)
        return (builder.build(spec["id"]) if spec["kind"] == "block"
                else builder.build_static(spec["id"]))

    def add(self, scene, kind):
        if kind not in _HALF:
            raise ValueError("Object kind must be block or box")
        if len(self._specs) >= MAX_OBJECTS:
            raise ValueError(f"Scene capacity is {MAX_OBJECTS} objects")
        ids = {s["id"] for s in self._specs}
        prefix = "block" if kind == "block" else "box"
        number = 1
        while f"{prefix}_{number}" in ids:
            number += 1
        hues = [colorsys.rgb_to_hsv(*s["color"])[0] for s in self._specs]
        palette = np.arange(0, 1, 1 / 360)
        hue = max(palette, key=lambda h: min(
            (min(abs(h - old), 1 - abs(h - old)) for old in hues), default=1))
        spec = {"id": f"{prefix}_{number}", "kind": kind,
                "label": f"{'物块' if kind == 'block' else '盒子'} {number}",
                "color": list(colorsys.hsv_to_rgb(float(hue), .92, .85))}
        validate_specs([*self._specs, spec])
        # Current actor poses are read only by this private construction layer.
        occupied = {name: actor.pose.p[0].cpu().numpy().tolist()
                    for name, actor in self._actors.items()}
        position = self._sample(spec, occupied)
        actor = self._build_actor(scene, spec, position)
        self._actors[spec["id"]] = actor
        self._placements[spec["id"]] = position
        self._specs.append(spec)
        self._reserved_box = None
        return deepcopy(spec)
