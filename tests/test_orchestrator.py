from pickparts_agent.agent.orchestrator import Orchestrator
from pickparts_agent.agent.planner import Plan


class StubPlanner:
    def __init__(self, plan):
        self.plan = plan
        self.seen = []

    def analyze(self, goal, state):
        self.seen.append(state)
        return self.plan


class StubExecutor:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def execute(self, plan, goal):
        self.calls += 1
        return dict(self.result)


def make(plan, result):
    planner = StubPlanner(plan)
    executor = StubExecutor(result)
    return Orchestrator(object(), object(), planner, object(), executor), planner, executor


def test_infeasible_returns_blocker_and_makes_zero_moves():
    plan = Plan(False, [{"need": "目标可见", "reason": "没有零件 C"}], [], "")
    orch, _, executor = make(plan, {"success": True, "message": "should-not-run"})
    r = orch.turn("把 C 放进盒子")
    assert r["success"] is False and "没有零件 C" in r["message"]
    assert executor.calls == 0


def test_success_asks_followup():
    plan = Plan(True, [], [{"atom": "x", "args": {}}], "")
    orch, _, _ = make(plan, {"success": True, "message": "done"})
    r = orch.turn("goal")
    assert r["success"] and "还需要我做什么？" in r["message"]


def test_followup_not_duplicated():
    plan = Plan(True, [], [], "")
    orch, _, _ = make(plan, {"success": True, "message": "done 还需要我做什么？"})
    r = orch.turn("goal")
    assert r["message"].count("还需要我做什么？") == 1


def test_two_turns_share_same_world_state():
    plan = Plan(True, [], [], "")
    planner = StubPlanner(plan)
    executor = StubExecutor({"success": True, "message": "ok"})
    from pickparts_agent.agent.state import WorldState
    state = WorldState()
    orch = Orchestrator(state, object(), planner, object(), executor)
    orch.turn("第一个任务")
    orch.turn("第二个任务")
    assert planner.seen[0] is state and planner.seen[1] is state
    assert len(orch.history) == 2
