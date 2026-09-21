"""Tests for the external post-execution judge."""
import numpy as np

from tiptop_mac.evaluation import Verdict, judge_outcome
from tiptop_mac.perception import propose_grasps
from tiptop_mac.types import ObjectNode, Predicate, SceneGraph, TableSurface

from helpers import FakeFrame


SPECS = [
    {"id": "A", "kind": "block", "label": "A", "color": [.8, .1, .1]},
    {"id": "B", "kind": "block", "label": "B", "color": [.1, .1, .8]},
    {"id": "box", "kind": "box", "label": "box", "color": [.1, .8, .1]},
]


def make_node(object_id, kind, xy, top_z, bottom_z,
              extent_xy=.024, height=.036):
    point = np.array([xy[0], xy[1], (top_z + bottom_z) / 2], float)
    node = ObjectNode(
        id=object_id, kind=kind, label=object_id, color=(.5, .5, .5),
        mask=np.zeros((4, 4), bool), bbox=(0, 0, 1, 1), point=point,
        top_z=top_z, bottom_z=bottom_z,
        extent=np.array([extent_xy, extent_xy, height]),
        cloud=point[None, :], grasps=[])
    node.grasps = propose_grasps(node)
    return node


def make_scene(nodes):
    return SceneGraph(
        table=TableSurface(np.array([0., 0., 1.]), .745,
                           np.array([[-.3, -.67, .74], [.3, -.23, .75]])),
        objects={n.id: n for n in nodes}, qpos=np.zeros(6))


def judge(scene, *predicates):
    frame = FakeFrame(np.zeros(6))
    return judge_outcome(frame, SPECS, predicates,
                         perceive=lambda f, s: scene)


def test_judge_stack_on_block_success():
    scene = make_scene([
        make_node("A", "block", (.10, -.45), .817, .781),
        make_node("B", "block", (.10, -.45), .781, .745),
        make_node("box", "box", (-.15, -.45), .759, .720,
                  extent_xy=.096, height=.039),
    ])
    verdict = judge(scene, Predicate("on", ("A", "B")))
    assert isinstance(verdict, Verdict)
    assert verdict.success and verdict.relation == "on"


def test_judge_inside_box_success():
    scene = make_scene([
        make_node("A", "block", (-.15, -.45), .756, .722),
        make_node("B", "block", (.10, -.45), .781, .745),
        make_node("box", "box", (-.15, -.45), .759, .720,
                  extent_xy=.096, height=.039),
    ])
    verdict = judge(scene, Predicate("on", ("A", "box")))
    assert verdict.success and verdict.relation == "in"


def test_judge_fails_when_object_still_on_table():
    scene = make_scene([
        make_node("A", "block", (0., -.45), .781, .745),
        make_node("B", "block", (.10, -.45), .781, .745),
        make_node("box", "box", (-.15, -.45), .759, .720,
                  extent_xy=.096, height=.039),
    ])
    verdict = judge(scene, Predicate("on", ("A", "B")))
    assert not verdict.success and verdict.relation == ""


def test_judge_holding_when_object_absent():
    scene = make_scene([
        make_node("B", "block", (.10, -.45), .781, .745),
    ])
    verdict = judge(scene, Predicate("holding", ("A",)))
    assert verdict.success and verdict.relation == "holding"


def test_judge_fails_holding_when_present():
    scene = make_scene([
        make_node("A", "block", (0., -.45), .781, .745),
    ])
    verdict = judge(scene, Predicate("holding", ("A",)))
    assert not verdict.success


def test_judge_all_predicates_must_hold():
    scene = make_scene([
        make_node("A", "block", (.10, -.45), .817, .781),
        make_node("B", "block", (.10, -.45), .781, .745),
    ])
    verdict = judge(scene, Predicate("on", ("A", "B")),
                    Predicate("on", ("B", "table")))
    assert verdict.success
