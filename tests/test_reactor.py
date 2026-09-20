from helpers import FakeChat
from pickparts_agent.atoms import build_default_registry
from pickparts_agent.atoms.base import AtomResult
from pickparts_agent.llm import LLMError
from pickparts_agent.reactor import Reactor
from pickparts_agent.state import WorldState


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
