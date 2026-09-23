"""Shared measured placement relations for both closed-loop agents."""
import numpy as np


def stable_block_overlap(node, support, *, center=None):
    """Central contact with at least half the upper footprint overlapping."""
    center = np.asarray(node.point[:2] if center is None else center)
    base = np.asarray(support.point[:2])
    half = np.asarray(node.extent[:2]) / 2
    base_half = np.asarray(support.extent[:2]) / 2
    overlap = np.maximum(
        0., np.minimum(center + half, base + base_half)
        - np.maximum(center - half, base - base_half))
    return bool(np.all(np.abs(center - base) <= base_half - .002)
                and np.prod(overlap) >= .50 * np.prod(2 * half))


def on_block(node, support):
    return bool(-.003 <= node.bottom_z - support.top_z <= .008
                and stable_block_overlap(node, support))


def in_box(node, box):
    """Visible containment below the rim, not proof of interior floor contact."""
    footprint = (np.abs(np.asarray(node.point[:2]) - box.point[:2])
                 + np.asarray(node.extent[:2]) / 2)
    return bool(np.all(footprint <= np.asarray(box.extent[:2]) / 2 - .003)
                and box.bottom_z - .003 <= node.bottom_z < box.top_z)


def on_table(node, height):
    """Measured contact; a hidden bottom cannot imply table support."""
    return bool(-.003 <= node.bottom_z - height < .004)
