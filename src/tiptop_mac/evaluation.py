"""External judge: evaluates outcomes from a fresh post-execution RGB-D frame.

The verdict never flows back to the controller, preserving one-way openness.
"""
from dataclasses import dataclass

import numpy as np

from .perception import build_scene_graph


@dataclass(frozen=True)
class Verdict:
    success: bool
    relation: str
    reason: str


def _inside_box(obj, box):
    delta = np.abs(np.asarray(obj.point[:2])
                   - np.asarray(box.point[:2]))
    box_half = np.asarray(box.extent[:2], dtype=float) / 2
    footprint = np.all(delta <= box_half - .006)
    vertical = (obj.bottom_z >= box.bottom_z - .012
                and obj.top_z <= box.top_z + .02)
    return bool(footprint and vertical)


def _on_block(obj, support):
    """Stacked blocks: XY near-coincident centers, one height-band of gap."""
    xy = float(np.linalg.norm(np.asarray(obj.point[:2])
                              - np.asarray(support.point[:2])))
    z = float(obj.point[2] - support.point[2])
    grounded = obj.bottom_z >= support.top_z - .008
    return bool(xy <= .034 and .02 <= z <= .06 and grounded)


def _on_table(obj, table):
    return bool(0 <= obj.bottom_z - table.top_z < .025)


def judge_outcome(frame, specs, goal, *, perceive=build_scene_graph):
    scene = perceive(frame, specs)
    success = True
    relation = ""
    reasons = []
    for predicate in goal:
        if predicate.name == "holding":
            if predicate.args[0] in scene.objects:
                success = False
                reasons.append(f"{predicate.args[0]} 仍在场景中，未被持有")
            else:
                relation = "holding"
            continue
        movable, target = predicate.args
        node = scene.objects.get(movable)
        if node is None:
            success = False
            reasons.append(f"场景中未观察到 {movable}")
            continue
        if target == "table":
            ok, relation = _on_table(node, scene.table), "on"
        else:
            target_node = scene.objects.get(target)
            if target_node is None:
                ok = False
            elif target_node.kind == "box":
                ok, relation = _inside_box(node, target_node), "in"
            else:
                ok, relation = _on_block(node, target_node), "on"
        if not ok:
            success = False
            reasons.append(f"{movable} 未处于目标支撑体 {target} 上")
    return Verdict(success, relation if success else "",
                   "；".join(reasons) if reasons else "目标达成")
