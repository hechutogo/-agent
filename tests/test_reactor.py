import json
from types import SimpleNamespace

import pytest

from helpers import FakeChat
from pickparts_agent.agent.atoms import build_default_registry
from pickparts_agent.agent.atoms.base import AtomResult
from pickparts_agent.agent.llm import LLMError
from pickparts_agent.agent.reactor import Reactor
from pickparts_agent.agent.state import WorldState


def reactor(chat):
    return Reactor(chat, build_default_registry())


def failed(kind):
    return AtomResult(False, kind, "blocked")


def test_retry_returns_refresh_steps():
    chat = FakeChat([{"thought": "落点偏", "decision": "retry", "reason": "刷新",
                      "steps": [{"atom": "find_object", "args": {"target": "A"}}]}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("transient"), 1)
    assert d.kind == "retry" and d.steps[0]["atom"] == "find_object"


def test_unknown_decision_kind_falls_back_to_replan():
    chat = FakeChat([{"thought": "", "decision": "fly", "steps": []}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("transient"), 1)
    assert d.kind == "replan"


def test_bad_step_falls_back_to_replan():
    chat = FakeChat([{"thought": "", "decision": "replace",
                      "steps": [{"atom": "bogus", "args": {}}]}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("verify"), 1)
    assert d.kind == "replan"


def test_fatal_forces_abort_even_if_model_retries():
    chat = FakeChat([{"thought": "", "decision": "retry", "steps": []}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("fatal"), 1)
    assert d.kind == "abort"


def test_llm_error_fallback_matches_error_kind():
    transient = reactor(FakeChat([LLMError("x")])).decide(
        "g", [], 0, WorldState(), failed("transient"), 1)
    fatal = reactor(FakeChat([LLMError("x")])).decide(
        "g", [], 0, WorldState(), failed("fatal"), 1)
    assert transient.kind == "replan" and fatal.kind == "abort"


def test_ask_user_keeps_question():
    chat = FakeChat([{"thought": "需要确认", "decision": "ask_user",
                      "reason": "r", "question": "先处理哪个？", "steps": []}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("precondition"), 1)
    assert d.kind == "ask_user" and d.question == "先处理哪个？"


@pytest.mark.parametrize("kind", ["retry", "replace", "replan", "ask_user", "abort"])
def test_fatal_aborts_without_consulting_model(kind):
    chat = FakeChat([{"decision": kind, "steps": []}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("fatal"), 1)
    assert d.kind == "abort" and not d.steps
    assert chat.calls == []


@pytest.mark.parametrize("payload", [
    {"decision": "replace", "steps": []},
    {"decision": "replace", "steps": None},
    {"decision": "retry", "steps": {}},
    {"decision": "retry", "steps": False},
    {"decision": "retry", "rationale": []},
    {"decision": ["retry"]},
    {"decision": "replan", "steps": [
        {"atom": "reset_arm", "args": {}}]},
])
def test_malformed_recovery_fails_closed(payload):
    d = reactor(FakeChat([payload])).decide(
        "g", [], 0, WorldState(), failed("transient"), 1)
    assert d.kind == "replan" and not d.steps


def test_public_rationale_maps_to_legacy_decision_thought():
    d = reactor(FakeChat([{
        "decision": "retry", "steps": [], "rationale": "Refresh stale vision.",
        "reasoning_content": "PRIVATE_REASONING",
    }])).decide("g", [], 0, WorldState(), failed("transient"), 1)
    assert d.thought == "Refresh stale vision."


def test_latest_placement_and_age_reach_reactor_without_rewriting_as_table():
    snapshot = {"frame_id": 20, "catalog": {}, "objects": {
        "A": {"id": "A", "visible": False, "frame_id": 19, "placement": "box"}}}
    verify = {"atom": "verify_state", "args": {"target": "A", "at": "box"}}
    steps = [{"atom": "reach_above", "args": {"target": "A"}}, verify]
    chat = FakeChat([{"decision": "replace", "steps": [verify],
                      "rationale": "Verify the last observed placement."}])
    d = reactor(chat).decide(
        "A into box", steps, 0, SimpleNamespace(snapshot=lambda: snapshot),
        failed("lost_object"), 1)
    assert d.kind == "replace" and d.steps == [verify]
    assert json.loads(chat.calls[0]["user"])["world"] == snapshot
