"""One-shot object-centric perception: CPU adaptation of the TiPToP pipeline."""
import colorsys

import numpy as np

from pickparts_agent.scene.objects import validate_specs
from pickparts_agent.scene.perception import backproject

from .types import Grasp, ObjectNode, SceneGraph, TableSurface


def _hue_mask(rgb, color):
    """Boolean HSV identity mask; parameters match ScenePerception."""
    import cv2

    target_hue = colorsys.rgb_to_hsv(float(color[0]), float(color[1]),
                                     float(color[2]))[0] * 180
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    difference = np.abs(hsv[:, :, 0].astype(float) - target_hue)
    difference = np.minimum(difference, 180 - difference)
    return ((difference < 7) & (hsv[:, :, 1] > 130)
            & (hsv[:, :, 2] > 55)).astype(np.uint8)


def segment_objects(frame, specs):
    """Segment every catalog object in one pass; reuse geometry validation."""
    import cv2
    from pickparts_agent.scene.perception import ScenePerception

    specs = validate_specs(specs)
    perceiver = ScenePerception(specs)
    nodes = {}
    for spec in specs:
        object_id = spec["id"]
        located = perceiver.locate(frame, object_id)
        mask = _hue_mask(frame.rgb, spec["color"])
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        x1, y1, x2, y2 = located["bbox"]
        matches = [
            i for i in range(1, count)
            if stats[i, 0] == x1 and stats[i, 1] == y1
            and stats[i, 0] + stats[i, 2] == x2
            and stats[i, 1] + stats[i, 3] == y2
        ]
        if len(matches) != 1:
            raise ValueError(f"{object_id}: mask and measured geometry mismatch")
        region = labels == matches[0]
        valid = region & np.isfinite(frame.depth) & (frame.depth > 0)
        ys, xs = np.nonzero(valid)
        cloud = backproject(np.c_[xs, ys], frame.depth[ys, xs],
                            frame.intrinsic, frame.camera_to_base)
        nodes[object_id] = ObjectNode(
            id=object_id, kind=spec["kind"], label=spec["label"],
            color=tuple(float(v) for v in spec["color"]), mask=region,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            point=np.asarray(located["point"], dtype=float),
            top_z=float(located["top_z"]), bottom_z=float(located["bottom_z"]),
            extent=np.asarray(located["extent"], dtype=float),
            cloud=cloud, grasps=[])
    return nodes


def fit_table(points, object_bottoms, *, seed=20260921):
    """RANSAC horizontal support; select the plane at the object bottom band."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=float)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 30:
        raise ValueError("Insufficient non-object depth for a horizontal table")
    band = float(np.median(np.asarray(object_bottoms, dtype=float)))
    min_inliers = max(15, int(.05 * len(pts)))
    best = None
    best_horizontal = None
    for _ in range(300):
        i, j, k = rng.choice(len(pts), 3, replace=False)
        triangle = pts[[i, j, k]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < np.cos(np.deg2rad(18)):
            continue
        offset = -float(normal @ triangle[0])
        inlier_mask = np.abs(pts @ normal + offset) < .004
        count = int(inlier_mask.sum())
        if count < min_inliers:
            continue
        top_z = float(np.median(pts[inlier_mask, 2]))
        candidate = (count, normal, inlier_mask)
        if best_horizontal is None or count > best_horizontal[0]:
            best_horizontal = candidate
        if abs(top_z - band) <= .03 and (best is None or count > best[0]):
            best = candidate
    chosen = best or best_horizontal
    if chosen is None:
        raise ValueError("No horizontal support plane found")
    inliers = pts[chosen[2]]
    bounds = np.percentile(inliers, [1, 99], axis=0)
    return TableSurface(normal=chosen[1],
                        top_z=float(np.median(inliers[:, 2])), bounds=bounds)


def propose_grasps(node):
    """Top 4-DoF grasp for blocks; boxes are surfaces only (TiPToP fallback style)."""
    if node.kind != "block":
        return []
    return [Grasp(position=np.asarray(node.point, dtype=float),
                  top_z=float(node.top_z),
                  width=float(max(node.extent[:2])), confidence=.9)]


def build_scene_graph(frame, specs):
    """RGB-D -> table + object nodes + grasp candidates, all before motion."""
    specs = validate_specs(specs)
    nodes = segment_objects(frame, specs)
    occupied = np.zeros(frame.depth.shape, dtype=bool)
    for node in nodes.values():
        occupied |= node.mask
    yy, xx = np.mgrid[0:frame.depth.shape[0]:2, 0:frame.depth.shape[1]:2]
    free = (~occupied[yy, xx]
            & np.isfinite(frame.depth[yy, xx])
            & (frame.depth[yy, xx] > 0))
    points = backproject(np.c_[xx[free].ravel(), yy[free].ravel()],
                         frame.depth[yy, xx][free].ravel(),
                         frame.intrinsic, frame.camera_to_base)
    bottoms = [node.bottom_z for node in nodes.values()]
    table = fit_table(points, bottoms)
    for node in nodes.values():
        node.grasps = propose_grasps(node)
    return SceneGraph(table=table, objects=nodes,
                      qpos=np.asarray(frame.qpos))
