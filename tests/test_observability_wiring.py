import json

import pytest

from helpers import FakeChat, FakeLocate, make_context
from pickparts_agent.observability import Recorder


def _names(tmp_path):
    path = next(tmp_path.glob("trace-*.jsonl"))
    return [json.loads(line).get("name")
            for line in path.read_text().splitlines()]


def test_orchestrator_opens_plan_execute_spans(tmp_path):
    rec = Recorder(tmp_path)

    class PlannerStub:
        def analyze(self, goal, state):
            return type("Plan", (), {
                "feasible": True, "blockers": [], "steps": [], "rationale": ""})()

    class ExecutorStub:
        def execute(self, plan, goal):
            return {"success": True, "message": "ok"}

    from pickparts_agent.agent.orchestrator import Orchestrator
    from pickparts_agent.agent.state import WorldState
    orch = Orchestrator(WorldState(), None, PlannerStub(), None,
                        ExecutorStub(), recorder=rec)
    with rec.run("command", "g"):
        orch.turn("g")
    names = _names(tmp_path)
    assert "plan" in names and "execute" in names


def test_executor_opens_atom_spans(tmp_path):
    from test_executor import FlakyAtom, build
    from pickparts_agent.agent.planner import Plan
    rec = Recorder(tmp_path)
    executor, _ = build(FlakyAtom(0), FakeChat())
    executor.recorder = rec
    with rec.run("command", "g"):
        executor.execute(Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert "atom:flaky" in _names(tmp_path)


def test_executor_opens_react_span_on_failure(tmp_path):
    from test_executor import FlakyAtom, build
    from pickparts_agent.agent.planner import Plan
    rec = Recorder(tmp_path)
    chat = FakeChat([{"rationale": "再试", "decision": "retry",
                      "steps": [], "reason": ""} for _ in range(5)])
    executor, _ = build(FlakyAtom(99), chat)
    executor.recorder = rec
    with rec.run("command", "g"):
        executor.execute(Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert "react" in _names(tmp_path)


def _recorded_span(tmp_path, name):
    events = [json.loads(line) for line in next(
        tmp_path.glob("trace-*.jsonl")).read_text().splitlines()]
    start = next(event for event in events
                 if event.get("kind") == "span_start"
                 and event.get("name") == name)
    return next(event for event in events
                if event.get("kind") == "span_end"
                and event.get("span_id") == start["span_id"])


def test_executor_closes_atom_span_when_atom_raises(tmp_path):
    from test_executor import FlakyAtom, build
    from pickparts_agent.agent.planner import Plan

    class ExplodingAtom(FlakyAtom):
        def call(self, ctx, args):
            raise RuntimeError("atom exploded")

    rec = Recorder(tmp_path)
    executor, _ = build(ExplodingAtom(), FakeChat())
    executor.recorder = rec
    with pytest.raises(RuntimeError):
        with rec.run("command", "g"):
            executor.execute(
                Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")

    end = _recorded_span(tmp_path, "atom:flaky")
    assert end["status"] == "error" and end["error_kind"] == "RuntimeError"


def test_executor_closes_final_verify_span_when_verifier_raises(tmp_path):
    from test_executor import FlakyAtom, build
    from pickparts_agent.agent.atoms.base import AtomResult
    from pickparts_agent.agent.planner import Plan

    class ExplodingFinalVerifier(FlakyAtom):
        name = "verify_state"

        def call(self, ctx, args):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("verification exploded")
            return AtomResult(True, message="ok")

    rec = Recorder(tmp_path)
    executor, _ = build(ExplodingFinalVerifier(), FakeChat())
    executor.recorder = rec
    with pytest.raises(ValueError):
        with rec.run("command", "g"):
            executor.execute(Plan(True, [], [{
                "atom": "verify_state",
                "args": {"target": "A", "at": "box"},
            }], ""), "goal")

    end = _recorded_span(tmp_path, "final_verify")
    assert end["status"] == "error" and end["error_kind"] == "ValueError"


def test_find_object_tolerates_missing_recorder():
    from pickparts_agent.agent.atoms import FindObject
    ctx = make_context(locate=FakeLocate({"A": [0, 0, 0]}))
    assert ctx.recorder is None
    result = FindObject().call(ctx, {"target": "A"})
    assert result.success
