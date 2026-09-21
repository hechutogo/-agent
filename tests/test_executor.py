from helpers import FakeChat, FakeKin, FakeLocate, FakeSim
from pickparts_agent.agent.atoms import FindObject
from pickparts_agent.agent.atoms.base import Atom, AtomResult, Check
from pickparts_agent.agent.atoms.registry import AtomRegistry
from pickparts_agent.agent.planner import Plan, Planner
from pickparts_agent.agent.reactor import Reactor
from pickparts_agent.agent.executor import ReActExecutor
from pickparts_agent.agent.state import WorldState


class FlakyAtom(Atom):
    name = "flaky"
    description = "controllable"
    parameters = {"type": "object"}

    def __init__(self, fail_times=0, kind="transient"):
        self.remaining, self.kind, self.calls = fail_times, kind, 0

    def check_pre(self, ctx, args):
        return Check(True)

    def run(self, ctx, args):
        self.calls += 1
        if self.calls <= self.remaining:
            return AtomResult(False, self.kind, "fail")
        return AtomResult(True, message="ok")

    def verify(self, ctx, args, result):
        return Check(True)


def build(flaky, chat, *, ask_user=None):
    registry = AtomRegistry([flaky, FindObject()])
    planner = Planner(chat, registry)
    reactor = Reactor(chat, registry)
    locate = FakeLocate({"A": [-0.08, -0.34, 0.74]})
    return ReActExecutor(FakeSim(), locate, FakeKin(), registry,
                         WorldState(), planner, reactor, ask_user=ask_user), registry


def test_happy_path_emits_plan_steps_and_finish():
    events = []
    executor, _ = build(FlakyAtom(0), FakeChat())
    executor.on_event = events.append
    plan = Plan(True, [], [{"atom": "flaky", "args": {}}], "")
    result = executor.execute(plan, "goal")
    types = [e["type"] for e in events if e["type"] != "activity"]
    assert result["success"] is True
    assert types == ["plan", "step", "step", "finish"]


def test_retry_refreshes_then_succeeds():
    events = []
    flaky = FlakyAtom(1)  # first call fails
    chat = FakeChat([
        {"thought": "刷新", "decision": "retry", "reason": "重新定位",
         "steps": [{"atom": "find_object", "args": {"target": "A"}}]},
    ])
    executor, _ = build(flaky, chat)
    executor.on_event = events.append
    plan = Plan(True, [], [{"atom": "flaky", "args": {}}], "")
    result = executor.execute(plan, "goal")
    assert result["success"] and flaky.calls == 2
    assert any(e["type"] == "react" and e["decision"] == "retry" for e in events)


def test_replace_swaps_in_alt_atom():
    class AltAtom(Atom):
        name = "alt"
        description = "alt"
        parameters = {"type": "object"}
        ran = 0

        def check_pre(self, ctx, args):
            return Check(True)

        def run(self, ctx, args):
            AltAtom.ran += 1
            return AtomResult(True, message="alt-ok")

        def verify(self, ctx, args, result):
            return Check(True)

    alt = AltAtom()
    flaky = FlakyAtom(1)
    chat = FakeChat([
        {"thought": "换方案", "decision": "replace", "reason": "用 alt",
         "steps": [{"atom": "alt", "args": {}}]},
    ])
    registry = AtomRegistry([flaky, alt, FindObject()])
    locate = FakeLocate({"A": [-0.08, -0.34, 0.74]})
    executor = ReActExecutor(
        FakeSim(), locate, FakeKin(), registry, WorldState(),
        Planner(chat, registry), Reactor(chat, registry))
    result = executor.execute(
        Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert result["success"] and flaky.calls == 1 and AltAtom.ran == 1


def test_replan_succeeds_with_new_plan():
    flaky = FlakyAtom(1)
    chat = FakeChat([
        {"thought": "重规划", "decision": "replan", "steps": []},
        {"feasible": True, "blockers": [],
         "steps": [{"atom": "flaky", "args": {}}], "rationale": ""},
    ])
    executor, _ = build(flaky, chat)
    result = executor.execute(
        Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert result["success"] and flaky.calls == 2


def retry_chat():
    return FakeChat([
        {"thought": "", "decision": "retry", "steps": []},
        {"thought": "", "decision": "retry", "steps": []},
        {"thought": "", "decision": "retry", "steps": []},
        {"thought": "", "decision": "retry", "steps": []},
    ])


def test_react_budget_exhausted():
    flaky = FlakyAtom(99)  # always fails
    executor, _ = build(flaky, retry_chat())
    result = executor.execute(
        Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert result["success"] is False and "预算" in result["message"]
    assert flaky.calls == 5  # 4 retries allowed, 5th failure terminates


def test_fatal_aborts_without_retry():
    flaky = FlakyAtom(99, kind="fatal")
    chat = FakeChat([{"thought": "", "decision": "retry", "steps": []}])
    executor, _ = build(flaky, chat)
    result = executor.execute(
        Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert result["success"] is False
    assert flaky.calls == 1  # reactor override forces abort immediately
    assert chat.calls == []
    assert result["recovery_required"]


def test_execution_emits_action_summaries_and_does_not_accept_empty_plan():
    events = []
    executor, _ = build(FlakyAtom(0), FakeChat())
    executor.on_event = events.append
    result = executor.execute(Plan(True, [], [], ""), "goal")
    assert not result["success"]
    executor.execute(Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert any(e["type"] == "activity" and e["kind"] == "action" for e in events)


def test_llm_budget_exhausted_on_repeated_replan():
    flaky = FlakyAtom(99)
    cycle = [
        {"thought": "", "decision": "replan", "steps": []},
        {"feasible": True, "blockers": [],
         "steps": [{"atom": "flaky", "args": {}}], "rationale": ""},
    ]
    # 3 full cycles (6 responses) then a 4th reactor response (7th) -> LLM=9.
    chat = FakeChat(cycle * 3 + [cycle[0]])
    executor, _ = build(flaky, chat)
    result = executor.execute(
        Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert result["success"] is False
    assert "LLM" in result["message"] and flaky.calls == 4


def test_recovery_cannot_replace_goal_verification_with_localization():
    verifier = FlakyAtom(99, kind="verify")
    verifier.name = "verify_state"
    chat = FakeChat([{
        "thought": "刷新观测", "decision": "replace",
        "steps": [{"atom": "find_object", "args": {"target": "A"}}],
    }])
    executor, _ = build(verifier, chat)
    result = executor.execute(Plan(True, [], [{
        "atom": "verify_state", "args": {"target": "A", "at": "box"},
    }], ""), "把 A 放进盒子")
    assert not result["success"]
    assert verifier.calls == 2  # Original goal is checked again after recovery.
