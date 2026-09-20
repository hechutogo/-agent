import json

from helpers import FakeChat
from pickparts_agent.atoms import build_default_registry
from pickparts_agent.llm import LLMError
from pickparts_agent.planner import Planner
from pickparts_agent.state import WorldState


def planner(chat):
    return Planner(chat, build_default_registry())


def valid_steps():
    return [
        {"atom": "find_object", "args": {"target": "A"}},
        {"atom": "reach_above", "args": {"target": "A"}},
        {"atom": "grasp", "args": {"target": "A"}},
        {"atom": "lift", "args": {"clearance": 0.1}},
        {"atom": "carry_to", "args": {"container": "box"}},
        {"atom": "release_into", "args": {"container": "box"}},
        {"atom": "verify_state", "args": {"target": "A", "at": "box"}},
        {"atom": "reset_arm", "args": {}},
    ]


def test_feasible_plan_returns_validated_steps():
    chat = FakeChat([{"feasible": True, "blockers": [], "steps": valid_steps(),
                      "rationale": "all visible"}])
    plan = planner(chat).analyze("把 A 放进盒子", WorldState())
    assert plan.feasible and len(plan.steps) == 8
    assert "find_object" in chat.calls[0]["system"]


def test_infeasible_returns_blockers_and_no_steps():
    chat = FakeChat([{"feasible": False,
                      "blockers": [{"need": "目标可见", "reason": "没有零件 C"}],
                      "steps": [], "rationale": ""}])
    plan = planner(chat).analyze("把 C 放进盒子", WorldState())
    assert not plan.feasible and plan.steps == []
    assert plan.blockers[0]["need"] == "目标可见"


def test_bad_json_returns_infeasible_plan():
    plan = planner(FakeChat([LLMError("x")])).analyze("do it", WorldState())
    assert not plan.feasible and plan.steps == []


def test_unknown_atom_in_plan_fails_closed():
    steps = valid_steps() + [{"atom": "explode", "args": {}}]
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("goal", WorldState())
    assert not plan.feasible and plan.steps == []
