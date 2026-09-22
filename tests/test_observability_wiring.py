import json

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


def test_find_object_tolerates_missing_recorder():
    from pickparts_agent.agent.atoms import FindObject
    ctx = make_context(locate=FakeLocate({"A": [0, 0, 0]}))
    assert ctx.recorder is None
    result = FindObject().call(ctx, {"target": "A"})
    assert result.success
