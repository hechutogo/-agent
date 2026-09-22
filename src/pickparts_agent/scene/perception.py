"""Sensor-only geometry. No simulator objects belong in this module."""
from dataclasses import dataclass
import re

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    camera_to_base: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray


def backproject(pixels, depths, intrinsic, camera_to_base):
    pixels = np.asarray(pixels, dtype=float)
    depths = np.asarray(depths, dtype=float)
    k = np.asarray(intrinsic, dtype=float)
    transform = np.asarray(camera_to_base, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or depths.shape != (len(pixels),):
        raise ValueError("Expected Nx2 pixels and N metric depths")
    if not np.isfinite(depths).all() or (depths <= 0).any():
        raise ValueError("Depth must be positive and finite")
    if (k.shape != (3, 3) or transform.shape != (4, 4)
            or not all(np.isfinite(x).all() for x in (pixels, k, transform))):
        raise ValueError("Invalid camera calibration or pixels")
    rays = np.linalg.solve(k, np.c_[pixels, np.ones(len(pixels))].T).T
    camera_points = rays * depths[:, None]
    return camera_points @ transform[:3, :3].T + transform[:3, 3]


def _table_surface(frame, point, bbox):
    """Return the measured table height, its connected support, and scene cloud."""
    yy, xx = np.mgrid[0:frame.depth.shape[0]:2, 0:frame.depth.shape[1]:2]
    depth = frame.depth[yy, xx]
    x1, y1, x2, y2 = bbox
    valid = (np.isfinite(depth) & (depth > 0)
             & ~((xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)))
    points = backproject(np.c_[xx[valid], yy[valid]], depth[valid],
                         frame.intrinsic, frame.camera_to_base)
    near = (np.linalg.norm(points[:, :2] - np.asarray(point)[:2], axis=1) < .16)
    local = points[
        near
        & (points[:, 2] <= point[2])
        & (points[:, 2] >= .55)
        & (points[:, 2] <= .82)
    ]
    if len(local) < 80:
        raise ValueError("Insufficient visible table depth")
    bins = np.round(local[:, 2] / .003).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    height = None
    for index in np.argsort(counts)[::-1]:
        plane = local[bins == values[index]]
        if len(plane) < 80:
            break
        if np.min(np.ptp(plane[:, :2], axis=0)) >= .08:
            height = float(np.median(plane[:, 2]))
            break
    if height is None:
        raise ValueError("No broad horizontal table surface is visible")

    from scipy.ndimage import label

    plane_mask = np.zeros(valid.shape, dtype=bool)
    plane_mask[valid] = np.abs(points[:, 2] - height) <= .005
    labels, count = label(plane_mask, structure=np.ones((3, 3), dtype=int))
    components = []
    anchor = np.asarray(point)[:2]
    for component in range(1, count + 1):
        selected = labels[valid] == component
        support = points[selected, :2]
        if len(support) < 80 or np.min(np.ptp(support, axis=0)) < .08:
            continue
        distance = float(np.min(np.linalg.norm(support - anchor, axis=1)))
        components.append((distance, -len(support), support))
    if not components:
        raise ValueError("No connected table support is visible")
    _, _, support = min(components, key=lambda item: item[:2])
    return height, support, points


def table_height(frame, point, bbox, footprint=None):
    """Measure table height and optionally require full footprint support."""
    height, support, _ = _table_surface(frame, point, bbox)
    if footprint is not None:
        footprint = np.asarray(footprint, dtype=float)
        if (footprint.shape != (2,) or not np.isfinite(footprint).all()
                or (footprint <= 0).any()):
            raise ValueError("Invalid object footprint")
        from scipy.spatial import ConvexHull, QhullError

        half = footprint / 2
        corners = np.asarray(point)[:2] + np.array([
            [-half[0], -half[1]], [-half[0], half[1]],
            [half[0], -half[1]], [half[0], half[1]],
        ])
        try:
            equations = ConvexHull(support).equations
        except QhullError:
            raise ValueError("Table footprint boundary is unavailable") from None
        signed = corners @ equations[:, :2].T + equations[:, 2]
        if np.any(signed > -.006):
            raise ValueError("Object footprint is not fully supported by table")
    return height


def table_drop_candidates(frame, anchor, footprint):
    """Measure a table plane and return sensor-grounded, collision-free XY cells."""
    anchor = np.asarray(anchor, dtype=float)
    footprint = np.asarray(footprint, dtype=float)
    if (anchor.shape != (3,) or footprint.shape != (2,)
            or not np.isfinite(anchor).all() or not np.isfinite(footprint).all()
            or (footprint <= 0).any()):
        raise ValueError("Invalid table placement geometry")
    if (frame.depth.ndim != 2 or frame.rgb.shape[:2] != frame.depth.shape):
        raise ValueError("Invalid RGB-D frame for table placement")

    table_z, support, points = _table_surface(
        frame, anchor, [0, 0, 0, 0])
    obstacles = points[points[:, 2] > table_z + .008, :2]

    from scipy.spatial import cKDTree

    support_tree = cKDTree(support)
    obstacle_tree = cKDTree(obstacles) if len(obstacles) else None
    half = footprint / 2
    edge_margin = half + .012
    low, high = np.percentile(support, [1, 99], axis=0)
    low, high = low + edge_margin, high - edge_margin
    if np.any(low >= high):
        raise ValueError("No table area can support the held object")

    spacing = .025
    xs = np.arange(np.ceil(low[0] / spacing) * spacing,
                   high[0] + 1e-9, spacing)
    ys = np.arange(np.ceil(low[1] / spacing) * spacing,
                   high[1] + 1e-9, spacing)
    offsets = np.array([
        [0., 0.],
        [-edge_margin[0], -edge_margin[1]],
        [-edge_margin[0], edge_margin[1]],
        [edge_margin[0], -edge_margin[1]],
        [edge_margin[0], edge_margin[1]],
    ])
    clearance = half + .018
    obstacle_radius = float(np.linalg.norm(clearance))
    candidates = []
    for x in xs:
        for y in ys:
            candidate = np.array([x, y])
            distances, _ = support_tree.query(candidate + offsets)
            if np.max(distances) > .012:
                continue
            if obstacle_tree is not None:
                nearby = obstacle_tree.query_ball_point(candidate, obstacle_radius)
                if nearby:
                    delta = np.abs(obstacles[nearby] - candidate)
                    if np.any(np.all(delta < clearance, axis=1)):
                        continue
            candidates.append(candidate)
    if not candidates:
        raise ValueError("No safe empty table area is visible")
    candidates.sort(key=lambda xy: (
        float(np.linalg.norm(xy - anchor[:2])), float(xy[0]), float(xy[1])))
    return table_z, np.asarray(candidates)


def parse_target(text):
    if any(word in text.lower() for word in ("不要", "别", "不拿", "停止", "取消", "don't", "not")):
        raise ValueError("Negated command requires clarification")
    targets = re.findall(r"(?<![A-Za-z])([AB])(?![A-Za-z])", text.upper())
    if len(set(targets)) != 1:
        raise ValueError("请明确选择零件 A 或 B")
    return targets[0]


class ColorPerception:
    """Explicit deterministic baseline for the red/blue/green demo scene."""
    def locate(self, frame, target):
        import cv2
        rgb = frame.rgb.astype(float)
        r, g, b = rgb.transpose(2, 0, 1)
        if target == "A":
            mask = (r > 65) & (r > g * 1.8) & (r > b * 1.8)
        elif target == "B":
            mask = (b > 65) & (b > r * 1.8) & (b > g * 1.35)
        elif target == "box":
            mask = (g > 65) & (g > r * 1.6) & (g > b * 1.3)
        else:
            raise ValueError("Unknown visual target")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        if count < 2:
            raise ValueError(f"{target} is not visible")
        label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = labels == label
        if mask.sum() < 12:
            raise ValueError(f"{target} has insufficient visible pixels")
        valid = mask & np.isfinite(frame.depth) & (frame.depth > 0)
        ys, xs = np.nonzero(valid)
        if len(xs) < 8:
            raise ValueError(f"{target} has insufficient valid depth")
        points = backproject(np.c_[xs, ys], frame.depth[ys, xs],
                             frame.intrinsic, frame.camera_to_base)
        center = np.median(points, axis=0)
        # The camera-facing side has more pixels than the far side. Center the
        # visible geometric extent instead of letting that face bias XY.
        center[:2] = np.mean(np.percentile(points[:, :2], [1, 99], axis=0), axis=0)
        x, y, w, h, _ = stats[label]
        return {"bbox": [int(x), int(y), int(x + w), int(y + h)],
                "point": center, "confidence": 1.}


class ScenePerception:
    """Catalog hue identity plus RGB-D measurements, with no scene reference.

    ``point.z`` is the median visible surface, retaining the baseline grasp
    convention. Extents/bottom_z describe observed surfaces, not hidden geometry.
    """
    def __init__(self, specs):
        import colorsys
        from .objects import validate_specs
        self._specs = {s["id"]: s for s in validate_specs(specs)}
        self._hues = {name: colorsys.rgb_to_hsv(*s["color"])[0] * 180
                      for name, s in self._specs.items()}

    def locate(self, frame, target):
        import cv2
        if target not in self._specs:
            raise ValueError(f"Unknown visual target: {target}")
        if (frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3
                or frame.depth.shape != frame.rgb.shape[:2]):
            raise ValueError("Invalid RGB-D frame shape")
        hsv = cv2.cvtColor(frame.rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
        difference = np.abs(hsv[:, :, 0].astype(float) - self._hues[target])
        difference = np.minimum(difference, 180 - difference)
        # Catalog materials are saturated. Pale robot panels can share their
        # hue and even touch the target in image space; exclude them before
        # connected components so their geometry cannot swallow a real block.
        mask = (difference < 7) & (hsv[:, :, 1] > 130) & (hsv[:, :, 2] > 55)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 12]
        kind = self._specs[target]["kind"]
        if kind == "block":
            plausible = []
            for i in candidates:
                valid = (labels == i) & np.isfinite(frame.depth) & (frame.depth > 0)
                # Missing depth is not evidence that a competing object is noise.
                if valid.sum() < .8 * stats[i, cv2.CC_STAT_AREA]:
                    plausible.append(i)
                    continue
                yy, xx = np.nonzero(valid)
                if np.ptp(np.percentile(frame.depth[yy, xx], [2, 98])) > .075:
                    plausible.append(i)
                    continue
                cloud = backproject(np.c_[xx, yy], frame.depth[yy, xx],
                                    frame.intrinsic, frame.camera_to_base)
                extent = np.ptp(np.percentile(cloud, [1, 99], axis=0), axis=0)
                # Known block dimensions permit rejecting narrow robot highlights,
                # without a workspace crop or any privileged object position.
                if np.min(extent[:2]) >= .012 and np.max(extent) <= .06:
                    plausible.append(i)
            candidates = plausible
        if not candidates:
            raise ValueError(f"{target} is not visible or is occluded")
        candidates.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        label = candidates[0]
        area = int(stats[label, cv2.CC_STAT_AREA])
        if any(stats[i, cv2.CC_STAT_AREA] >= max(12, area * .12)
               for i in candidates[1:]):
            raise ValueError(f"{target} is ambiguous or fragmented by occlusion")
        mask = labels == label
        x, y, w, h, _ = stats[label]
        height, width = frame.depth.shape
        if x == 0 or y == 0 or x + w == width or y + h == height:
            raise ValueError(f"{target} is clipped or occluded")
        valid = mask & np.isfinite(frame.depth) & (frame.depth > 0)
        if valid.sum() < max(8, .8 * area):
            raise ValueError(f"{target} has insufficient valid depth coverage")
        ys, xs = np.nonzero(valid)
        depths = frame.depth[ys, xs]
        if np.ptp(np.percentile(depths, [2, 98])) > (.075 if kind == "block" else .15):
            raise ValueError(f"{target} has ambiguous depth surfaces")
        # Split image neighbors at depth jumps. A broad depth range alone can
        # accidentally merge two same-color surfaces into one grasp target.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        indices = np.full(frame.depth.shape, -1, dtype=int)
        indices[ys, xs] = np.arange(len(xs))
        edges = []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            yy, xx = ys + dy, xs + dx
            inside = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            source = np.flatnonzero(inside)
            dest = indices[yy[inside], xx[inside]]
            connected = (dest >= 0) & (np.abs(depths[source] - depths[dest]) < .012)
            edges.append((source[connected], dest[connected]))
        rows = np.concatenate([edge[0] for edge in edges])
        cols = np.concatenate([edge[1] for edge in edges])
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)),
                           shape=(len(xs), len(xs)))
        _, regions = connected_components(graph, directed=False)
        sizes = np.sort(np.bincount(regions))
        if len(sizes) > 1 and sizes[-2] >= max(8, .12 * len(xs)):
            raise ValueError(f"{target} has ambiguous disconnected depth surfaces")
        dominant = regions == np.argmax(np.bincount(regions))
        xs, ys, depths = xs[dominant], ys[dominant], depths[dominant]
        points = backproject(np.c_[xs, ys], depths, frame.intrinsic, frame.camera_to_base)
        low, high = np.percentile(points, [1, 99], axis=0)
        top_z = float(np.percentile(points[:, 2], 95))
        bottom_z = float(low[2])
        center = np.median(points, axis=0)
        center[:2] = (low[:2] + high[:2]) / 2
        # Equalize the camera-facing side's sampling bias using the top plane.
        # Only use it when it spans a useful area, not a thin wall/rim or edge.
        top = points[points[:, 2] >= top_z - .003]
        if kind == "block":
            if np.min(high[:2] - low[:2]) < .012:
                raise ValueError(f"{target} is occluded: insufficient visible extent")
            if len(top) >= 12:
                top_low, top_high = np.percentile(top[:, :2], [1, 99], axis=0)
                if np.min(top_high - top_low) >= .016:
                    center[:2] = (top_low + top_high) / 2
        return {"bbox": [int(x), int(y), int(x + w), int(y + h)],
                "point": center, "confidence": float(min(1., valid.sum() / area)),
                "refinement": "rgbd", "top_z": top_z, "bottom_z": bottom_z,
                "extent": (high - low).tolist()}
