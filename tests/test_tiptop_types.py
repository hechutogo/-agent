"""Structure tests for the object-centric TiPToP scene types."""
import dataclasses

import numpy as np
import pytest

from tiptop_mac.types import (
    Grasp, MotionStep, ObjectNode, Predicate, SceneGraph, TAMPPlan, TableSurface,
)


def make_node(object_id="A", kind="block"):
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:3, 1:3] = True
    return ObjectNode(
        id=object_id, kind=kind, label=f"物块 {object_id}",
        color=(0.8, 0.1, 0.1), mask=mask, bbox=(1, 1, 3, 3),
        point=np.array([0.0, -0.4, 0.76]), top_z=0.78, bottom_z=0.744,
        extent=np.array([0.024, 0.024, 0.036]),
        cloud=np.array([[0.0, -0.4, 0.76]]), grasps=[],
    )


def test_object_node_is_mutable_and_scenegraph_indexes_by_id():
    node = make_node()
    table = TableSurface(normal=np.array([0.0, 0.0, 1.0]), top_z=0.745,
                         bounds=np.array([[-.30, -.67, .74], [.30, -.23, .75]]))
    scene = SceneGraph(table=table, objects={"A": node}, qpos=np.zeros(6))
    assert scene.objects["A"].point.shape == (3,)
    assert scene.table.top_z == pytest.approx(.745)
    # Object nodes are updated in place during feasibility checks.
    node.top_z = 0.79
    assert scene.objects["A"].top_z == 0.79


def test_folded_value_types_are_frozen():
    grasp = Grasp(position=np.array([0., -.4, .76]), top_z=.78, width=.024,
                  confidence=.9)
    predicate = Predicate("on", ("A", "B"))
    step = MotionStep("move", np.array([0., -.4, .86]), None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        grasp.width = .03
    with pytest.raises(dataclasses.FrozenInstanceError):
        predicate.name = "holding"
    with pytest.raises(dataclasses.FrozenInstanceError):
        step.jaw = .8


def test_tamp_plan_carries_full_trajectory_and_metadata():
    plan = TAMPPlan(
        goal=(Predicate("on", ("A", "B")),),
        operators=(("pick", ("A",)), ("place", ("A", "B"))),
        trajectory=(MotionStep("gripper", None, 0.0),),
        planning_time=0.12, rationale="A 放到 B 上",
    )
    assert len(plan.trajectory) == 1
    assert plan.operators[0][0] == "pick"
    assert plan.planning_time == pytest.approx(.12)
