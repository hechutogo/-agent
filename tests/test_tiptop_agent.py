"""Tests for the TiPToP-style one-shot agent assembly."""
from types import SimpleNamespace

import numpy as np

from tiptop_mac.agent import TiPToPAgent, build_tiptop_agent
from tiptop_mac.executor import OpenLoopExecutor
from tiptop_mac.grounding import Grounder
from tiptop_mac.tamp import TAMPLite
from tiptop_mac.types import ObjectNode, SceneGraph, TableSurface
from tiptop_mac.perception import propose_grasps

from helpers import FakeChat, FakeKin, FakeSim


SPECS = [
    {"id": "A", "kind": "block", "label": "对象 A", "color": [.8, .1, .1]},
    {"id": "B", "kind": "block", "label": "对象 B", "color": [.1, .1, .8]},
    {"id": "box", "kind": "box", "label": "盒子", "color": [.1, .8, .1]},
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


def make_scene():
    return SceneGraph(
        table=TableSurface(np.array([0., 0., 1.]), .745,
                           np.array([[-.3, -.67, .74], [.3, -.23, .75]])),
        objects={
            "A": make_node("A", "block", (0., -.45), .781, .745),
            "B": make_node("B", "block", (.10, -.45), .781, .745),
            "box": make_node("box", "box", (-.15, -.45), .759, .720,
                             extent_xy=.096, height=.039),
        },
        qpos=np.zeros(6))


def make_agent(scene=None, chat=None):
    scene = make_scene() if scene is None else scene
    sim = FakeSim()
    sim.object_specs = SPECS
    observe_calls = {"n": 0}
    original_observe = sim.observe

    def count_observe():
        observe_calls["n"] += 1
        return original_observe()

    sim.observe = count_observe
    perceive_calls = {"n": 0}

    def perceive(frame, specs):
        perceive_calls["n"] += 1
        return scene

    def locate(frame, target):
        node = scene.objects[target]
        return {"point": node.point, "bottom_z": node.bottom_z}

    agent = TiPToPAgent(
        sim, Grounder(chat), TAMPLite(FakeKin()),
        OpenLoopExecutor(sim, FakeKin(), locate=locate),
        perceive=perceive)
    return agent, sim, observe_calls, perceive_calls


def test_run_plans_and_executes_full_task():
    chat = FakeChat([{"goal": [{"predicate": "on", "args": ["A", "B"]}],
                     "rationale": "A 叠到 B"}])
    agent, sim, observe_calls, perceive_calls = make_agent(chat=chat)
    result = agent.run("把 A 叠到 B 上")
    assert result["success"] and not result["aborted"]
    assert result["rationale"] == "A 叠到 B"
    assert sim.moves  # open-loop execution happened
    assert perceive_calls["n"] == 1  # single one-shot scene snapshot


def test_satisfied_goal_is_noop_success():
    chat = FakeChat([{"goal": [{"predicate": "on", "args": ["A", "table"]}],
                     "rationale": "A 在桌面"}])
    agent, sim, _, _ = make_agent(chat=chat)
    result = agent.run("把 A 放桌上")
    assert result["success"]
    assert "无需动作" in result["message"]
    assert sim.moves == []


def test_grounding_failure_does_not_execute():
    chat = FakeChat([{"goal": [{"predicate": "fly", "args": ["A"]}],
                     "rationale": ""}])
    agent, sim, _, _ = make_agent(chat=chat)
    result = agent.run("让 A 飞")
    assert not result["success"] and not result["aborted"]
    assert sim.moves == []


def test_infeasible_plan_does_not_execute():
    scene = make_scene()
    scene.objects["A"].grasps = []
    chat = FakeChat([{"goal": [{"predicate": "on", "args": ["A", "B"]}],
                     "rationale": ""}])
    agent, sim, _, _ = make_agent(scene=scene, chat=chat)
    result = agent.run("把 A 叠到 B 上")
    assert not result["success"]
    assert sim.moves == []


def test_execution_abort_is_reported():
    chat = FakeChat([{"goal": [{"predicate": "on", "args": ["A", "B"]}],
                     "rationale": ""}])
    agent, sim, _, _ = make_agent(chat=chat)
    sim.tracking_error = True
    result = agent.run("把 A 叠到 B 上")
    assert not result["success"] and result["aborted"]


def test_build_tiptop_agent_factory():
    sim = FakeSim()
    sim.object_specs = SPECS
    endpoint = SimpleNamespace(
        client=lambda: object(), model="some-model",
        base_url="https://example.com/v1")
    agent = build_tiptop_agent(sim, endpoint)
    assert isinstance(agent, TiPToPAgent)
