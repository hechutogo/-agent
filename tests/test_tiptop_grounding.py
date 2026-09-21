"""Tests for grounding natural language into TiPToP goal predicates."""
import pytest

from tiptop_mac.grounding import Goal, Grounder, GroundingError
from tiptop_mac.types import Predicate

from helpers import FakeChat


SPECS = [
    {"id": "A", "kind": "block", "label": "红色物块 A", "color": [.86, .16, .16]},
    {"id": "B", "kind": "block", "label": "蓝色物块 B", "color": [.16, .25, .80]},
    {"id": "box", "kind": "box", "label": "绿色盒子", "color": [.16, .70, .25]},
]


def response(predicates, rationale="done"):
    return {"goal": [{"predicate": p, "args": list(a)}
                     for p, a in predicates], "rationale": rationale}


def test_ground_block_on_block():
    chat = FakeChat([response([("on", ("A", "B"))], "A 叠到 B")])
    goal = Grounder(chat).ground("把 A 放到 B 上", SPECS)
    assert isinstance(goal, Goal)
    assert goal.predicates == (Predicate("on", ("A", "B")),)
    assert goal.rationale == "A 叠到 B"
    assert chat.calls[0]["user"]


def test_ground_into_box_uses_on_predicate():
    chat = FakeChat([response([("on", ("A", "box"))])])
    goal = Grounder(chat).ground("把 A 放进盒子", SPECS)
    assert goal.predicates[0].args == ("A", "box")


def test_ground_holding():
    chat = FakeChat([response([("holding", ("B",))])])
    goal = Grounder(chat).ground("拿着 B", SPECS)
    assert goal.predicates == (Predicate("holding", ("B",)),)


def test_ground_table_support():
    chat = FakeChat([response([("on", ("A", "table"))])])
    goal = Grounder(chat).ground("把 A 放到桌上", SPECS)
    assert goal.predicates[0].args[1] == "table"


@pytest.mark.parametrize("payload", [
    response([("on", ("A", "C"))]),           # unknown support
    response([("on", ("box", "A"))]),          # movable is a box
    response([("on", ("A", "A"))]),            # self placement
    response([("holding", ("box",))]),         # holding a box
    response([("holding", ("A", "B"))]),       # holding wrong arity
    response([("near", ("A", "B"))]),          # unsupported predicate
    {"goal": "on(A,B)", "rationale": ""},      # malformed shape
    {"rationale": "x"},                         # missing goal
    {"goal": [], "rationale": ""},             # empty goal
])
def test_invalid_responses_are_rejected(payload):
    chat = FakeChat([payload])
    with pytest.raises(GroundingError):
        Grounder(chat).ground("任务", SPECS)


def test_llm_failure_becomes_grounding_error():
    from pickparts_agent.agent.llm import LLMError

    chat = FakeChat([LLMError("down")])
    with pytest.raises(GroundingError):
        Grounder(chat).ground("任务", SPECS)
