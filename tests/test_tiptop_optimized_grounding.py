"""Ordered language contracts: temporal goals must not collapse into a set."""
import pytest

from helpers import FakeChat
from tiptop_mac.grounding import GroundingError


SPECS = [{"id": "A", "kind": "block", "label": "红块", "color": [1, 0, 0]},
         {"id": "box", "kind": "box", "label": "盒子", "color": [0, 0, 1]}]


def step(support, **extra):
    return {"instruction": f"A 放到 {support}",
            "goal": [{"predicate": "on", "args": ["A", support]}], **extra}


def ground(data):
    from tiptop_optimized.grounding import OrderedGrounder
    return OrderedGrounder(FakeChat([data])).ground("先放桌上再放回盒里", SPECS)


def test_retains_order_and_duplicate_goals():
    tasks = ground({"subtasks": [step("table"), step("box"), step("table")]})
    assert [t.goal.predicates[0].args for t in tasks] == [
        ("A", "table"), ("A", "box"), ("A", "table")]


def test_explicit_repeat_retains_action_requirement():
    tasks = ground({"subtasks": [step("box", require_action=True)]})
    assert tasks[0].require_action is True


@pytest.mark.parametrize("data", [
    {"subtasks": []},
    {"subtasks": [step("box")] * 17},
    {"subtasks": [step("missing")]},
    {"subtasks": [step("box", require_action="false")]},
    {"subtasks": [step("box", goal=[
        {"predicate": "on", "args": ["A", "table"]},
        {"predicate": "on", "args": ["A", "box"]}])]},
])
def test_invalid_or_unbounded_decomposition_rejected(data):
    with pytest.raises(GroundingError):
        ground(data)


def test_reactor_rejects_arbitrary_actions_and_coordinates():
    from tiptop_optimized.grounding import RecoveryReactor
    reactor = RecoveryReactor(FakeChat([{"action": "move", "position": [1, 2, 3]}]))
    decision = reactor.decide({"error_kind": "verify"})
    assert decision.action == "abort"


def test_reactor_network_failure_uses_bounded_reobservation_fallback():
    from tiptop_optimized.grounding import RecoveryReactor
    reactor = RecoveryReactor(FakeChat([TimeoutError("key=secret")]))
    decision = reactor.decide({"error_kind": "verify"})
    assert decision.action == "reobserve"
    assert "secret" not in decision.reason
