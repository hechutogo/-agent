import numpy as np
import pytest

from helpers import make_context
from pickparts_agent.agent.atoms import VerifyState
from pickparts_agent.scene.perception import Frame


def detection(x=0., bottom=.726, kind="block"):
    extent = [.096, .09, .039] if kind == "box" else [.024, .024, .036]
    return {"point": [x, 0., bottom + extent[2] / 2],
            "bbox": [44, 44, 56, 56], "confidence": .99,
            "bottom_z": bottom, "top_z": bottom + extent[2], "extent": extent}


def context_with_measurements(part):
    observations = {"A": part, "box": detection(bottom=.72, kind="box")}
    ctx = make_context(locate=lambda frame, target: observations[target])
    transform = np.diag([1., 1., -1., 1.])
    transform[2, 3] = 1.72
    frame = Frame(np.zeros((100, 100, 3), np.uint8), np.ones((100, 100)),
                  np.array([[500., 0, 50], [0, 500., 50], [0, 0, 1.]]),
                  transform, ctx.sim.qpos, np.zeros_like(ctx.sim.qpos))
    ctx.sim.observe = lambda: frame
    return ctx


@pytest.mark.parametrize("x,bottom,expected", [
    (0., .726, True), (.035, .726, False), (0., .759, False),
])
def test_box_requires_footprint_containment_and_bottom_below_rim(x, bottom, expected):
    ctx = context_with_measurements(detection(x=x, bottom=bottom))
    result = VerifyState().call(ctx, {"target": "A", "at": "box", "relation": "in"})
    assert result.success is expected


@pytest.mark.parametrize("bottom,expected", [(.72, True), (.756, False), (.726, False)])
def test_table_requires_measured_contact_not_default_placement(bottom, expected):
    # Other supports have not been observed; that cannot imply table contact.
    ctx = context_with_measurements(detection(bottom=bottom))
    result = VerifyState().call(ctx, {"target": "A", "at": "table", "relation": "table"})
    assert result.success is expected


def test_table_verification_rejects_object_hanging_over_support_edge():
    part = detection(x=.195, bottom=.72)
    ctx = context_with_measurements(part)
    depth = np.full((100, 100), np.nan)
    depth[10:90, 10:90] = 1.
    transform = np.diag([1., 1., -1., 1.])
    transform[2, 3] = 1.72
    frame = Frame(np.zeros((100, 100, 3), np.uint8), depth,
                  np.array([[200., 0, 50], [0, 200., 50], [0, 0, 1.]]),
                  transform, ctx.sim.qpos, np.zeros_like(ctx.sim.qpos))
    ctx.sim.observe = lambda: frame
    result = VerifyState().call(
        ctx, {"target": "A", "at": "table", "relation": "table"})
    assert not result.success
