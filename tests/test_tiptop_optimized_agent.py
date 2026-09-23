"""Orchestration tests with a deterministic motion boundary and real planner."""
from collections import deque
from types import SimpleNamespace

from helpers import FakeChat, FakeKin
from test_tiptop_planner import default_scene


def build(steps, outcomes=(), *, initial="table", require_action=False):
    from tiptop_optimized.agent import OptimizedTiPToPAgent
    from tiptop_optimized.grounding import OrderedGrounder, RecoveryReactor
    from tiptop_optimized.tamp import TAMPLite
    scene = default_scene()
    def place(support):
        node = scene.objects["A"]
        if support == "table":
            xy, bottom = [0., -.35], .745
        else:
            rec = scene.objects[support]
            xy, bottom = rec.point[:2], rec.bottom_z + .004
        node.point[:] = [*xy, bottom + .018]
        node.bottom_z, node.top_z = bottom, bottom + .036
    place(initial)
    sim = SimpleNamespace(
        object_specs=[{"id": "A", "kind": "block"},
                      {"id": "box", "kind": "box"}], observe=lambda: scene)
    results = deque(outcomes)
    class MotionBoundary:
        held = None
        candidate = None

        def reconcile(self, observed):
            return self.held

        def planning_context(self, observed, *, held):
            return {"arm_seed": [0., 0., 0., 0., 0.],
                    "grasp_offset": [0., 0., 0.] if held else None}

        def execute(self, plan):
            outcome = results.popleft() if results else "ok"
            if outcome in ("ok", "retreat_failed"):
                place(plan.goal[0].args[1])
                self.held = self.candidate = None
            elif outcome == "unreachable":
                self.held = self.candidate = plan.goal[0].args[0]
            succeeded = outcome in ("ok", "false_success")
            return {"success": succeeded, "aborted": not succeeded,
                    "message": outcome, "acted": bool(plan.trajectory),
                    "error_kind": "" if succeeded else
                    "verify" if outcome in ("miss", "retreat_failed") else
                    "unreachable" if outcome == "unreachable" else "fatal",
                    "failed_index": 5 if outcome == "unreachable" else None}
    grounder = OrderedGrounder(FakeChat([{"subtasks": [
        {"instruction": f"A 放到 {s}", "goal": [
            {"predicate": "on", "args": ["A", s]}],
         "require_action": require_action} for s in steps]}]))
    reactor = RecoveryReactor(FakeChat([
        {"action": "replan", "reason": "重新观测后规划当前子任务"}] * 16))
    events = []
    agent = OptimizedTiPToPAgent(
        sim, grounder, TAMPLite(FakeKin()), MotionBoundary(),
        reactor=reactor, perceive=lambda frame, specs: frame,
        on_event=events.append)
    return agent, events, scene


def test_round_trip_uses_two_verified_subtasks_and_two_plans():
    agent, events, scene = build(["table", "box"], initial="box")
    result = agent.run("先桌面再放回盒子")
    assert result["success"] and result["verification"] == "visual"
    assert result["completed_subtasks"] == result["total_subtasks"] == 2
    plans = [e for e in events if e["type"] == "plan"]
    assert len(plans) == 2 and all(e["steps"] for e in plans)
    assert [s["status"] for s in [e for e in events if e["type"] == "subtasks"][-1]["steps"]] == ["done", "done"]


def test_failure_replans_current_subtask_without_replaying_completed_prefix():
    agent, events, _ = build(["table", "box"], ["ok", "miss", "ok"], initial="box")
    result = agent.run("桌面再回盒子")
    assert result["success"] and result["attempts"] == 3
    assert [e["goal"] for e in events if e["type"] == "plan"] == [
        "A 放到 table", "A 放到 box", "A 放到 box"]
    assert len([e for e in events if e["type"] == "react"]) == 1
    assert len([e for e in events if e["type"] == "finish"]) == 1


def test_motion_success_does_not_override_failed_visual_postcondition():
    agent, events, _ = build(["box"], ["false_success"] * 5)
    result = agent.run("入盒")
    assert not result["success"] and result["attempts"] == 5
    assert result["completed_subtasks"] == 0
    assert len([e for e in events if e["type"] == "react"]) == 4
    assert "达到 5 次尝试上限" in result["message"]


def test_fatal_failure_is_not_retried():
    agent, events, _ = build(["box"], ["fatal"])
    result = agent.run("入盒")
    assert not result["success"] and result["aborted"]
    assert result["attempts"] == 1
    assert not [e for e in events if e["type"] == "react"]


def test_release_succeeded_but_retreat_failed_can_finish_after_visual_confirmation():
    agent, events, _ = build(["box"], ["retreat_failed"])
    result = agent.run("入盒")
    assert result["success"] and result["attempts"] == 1
    assert not [e for e in events if e["type"] == "react"]


def test_ordinary_already_satisfied_subtask_does_not_move():
    agent, events, _ = build(["table"])
    result = agent.run("放到桌面")
    assert result["success"]
    assert [e for e in events if e["type"] == "plan"][0]["steps"] == []


def test_explicit_repeat_requires_executed_intermediate_action():
    agent, events, _ = build(["box"], initial="box", require_action=True)
    result = agent.run("再拿起放回盒子")
    assert result["success"]
    assert [e for e in events if e["type"] == "plan"][0]["steps"]


def test_trace_has_subtask_attempt_and_visual_evidence(tmp_path):
    import json
    from pickparts_agent.observability import Recorder
    agent, _, _ = build(["box"])
    recorder = agent.recorder = Recorder(tmp_path)
    with recorder.run("test", "入盒", agent="tiptop_optimized"):
        assert agent.run("入盒")["success"]
    records = [json.loads(line) for p in tmp_path.rglob("*.jsonl")
               for line in p.read_text().splitlines()]
    names = [e.get("name") for e in records if e["kind"] == "span_start"]
    assert "subtask:1" in names and "attempt:1" in names and "verify" in names


def test_trace_retains_separate_plans_for_all_attempts(tmp_path):
    import json
    from pickparts_agent.observability import Recorder
    agent, _, _ = build(["box"], ["miss", "ok"])
    recorder = agent.recorder = Recorder(tmp_path)
    with recorder.run("test", "入盒", agent="tiptop_optimized"):
        assert agent.run("入盒")["success"]
    records = [json.loads(line) for p in tmp_path.rglob("*.jsonl")
               for line in p.read_text().splitlines()]
    paths = [e["path"] for e in records if e["kind"] == "artifact"
             and "optimized-plan" in e["path"]]
    assert len(paths) == len(set(paths)) == 2


def test_planning_failure_while_inheriting_grip_requires_recovery():
    from tiptop_mac.tamp import TAMPError
    agent, _, _ = build(["box"])
    agent.executor.candidate = agent.executor.held = "A"
    agent.executor.reconcile = lambda scene: "A"
    def impossible(*args, **kwargs):
        raise TAMPError("infeasible target while held")
    agent.planner.plan = impossible
    result = agent.run("把手中的 A 放到盒子")
    assert not result["success"]
    assert result["aborted"] and result["recovery_required"]


def test_unreachable_table_target_is_excluded_from_current_subtask_replan():
    agent, events, _ = build(["table"], ["unreachable", "ok"], initial="box")
    calls = []
    original = agent.planner.plan

    def recording_plan(*args, **kwargs):
        calls.append(tuple(tuple(x) for x in kwargs.get("excluded_table_xy", ())))
        return original(*args, **kwargs)

    agent.planner.plan = recording_plan
    result = agent.run("把 A 放到桌面")

    assert result["success"] and result["attempts"] == 2
    assert calls[0] == ()
    assert len(calls[1]) == 1
    plans = [event for event in events if event["type"] == "plan"]
    assert len(plans) == 2
    first_release = plans[0]["steps"][6]["args"]["position"][:2]
    second_release = plans[1]["steps"][1]["args"]["position"][:2]
    assert sum((a - b) ** 2 for a, b in zip(first_release, second_release)) >= .05 ** 2


def test_visually_failed_table_target_is_excluded_from_replan():
    agent, events, _ = build(
        ["table"], ["false_success", "ok"], initial="box")
    calls = []
    original = agent.planner.plan

    def recording_plan(*args, **kwargs):
        calls.append(tuple(tuple(x) for x in kwargs.get("excluded_table_xy", ())))
        return original(*args, **kwargs)

    agent.planner.plan = recording_plan
    result = agent.run("把 A 放到桌面")

    assert result["success"] and result["attempts"] == 2
    assert calls[0] == () and len(calls[1]) == 1
    plans = [event for event in events if event["type"] == "plan"]
    first_release = plans[0]["steps"][6]["args"]["position"][:2]
    second_release = plans[1]["steps"][6]["args"]["position"][:2]
    assert sum((a - b) ** 2 for a, b in zip(first_release, second_release)) >= .05 ** 2


def test_failed_target_survives_ambiguous_post_failure_observation():
    from tiptop_optimized.grounding import RecoveryDecision
    agent, _, _ = build(["table"], ["unreachable", "ok"], initial="box")
    calls, contexts = [], []
    original_plan, original_observe = agent.planner.plan, agent._observe
    verify_calls = 0

    def recording_plan(*args, **kwargs):
        calls.append(tuple(tuple(x) for x in kwargs.get("excluded_table_xy", ())))
        return original_plan(*args, **kwargs)

    def ambiguous_checkpoint(phase):
        nonlocal verify_calls
        if phase == "verify" and verify_calls < 3:
            verify_calls += 1
            raise ValueError("occlusion outlasts local sampling budget")
        return original_observe(phase)

    agent.planner.plan = recording_plan
    agent._observe = ambiguous_checkpoint
    def decide(context):
        contexts.append(context)
        return RecoveryDecision("reobserve", "重新观测")
    agent.reactor.decide = decide
    result = agent.run("把 A 放到桌面")

    assert result["success"] and result["attempts"] == 2
    assert calls[0] == () and len(calls[1]) == 1
    assert contexts[0]["failures"][0]["phase"] == "execute"
    assert contexts[0]["failures"][0]["observation_error"]["phase"] == "verify"
    assert contexts[0]["grip_state"] == "unknown"


def test_recovery_gets_latest_scene_and_unknown_grip_after_successful_motion():
    from copy import deepcopy
    from tiptop_optimized.executor import GripStateError
    from tiptop_optimized.grounding import RecoveryDecision
    agent, events, _ = build(["box"])
    agent.perceive = lambda frame, specs: deepcopy(frame)
    original = agent.executor.reconcile
    calls, contexts = [], []

    def reconcile(scene):
        calls.append(scene)
        if 2 <= len(calls) <= 4:
            raise GripStateError("Grasp candidate is not visible.")
        return original(scene)

    def decide(context):
        contexts.append(context)
        return RecoveryDecision("reobserve", "重新读取持物证据")

    agent.executor.reconcile = reconcile
    agent.reactor.decide = decide
    result = agent.run("把 A 放进盒子")
    assert result["success"] and result["attempts"] == 2
    assert contexts[0]["support"]["A"] == "box"
    assert contexts[0]["grip_state"] == "unknown"
    assert contexts[0]["error_kind"] == "grip_unknown"
    assert contexts[0]["failures"][0]["phase"] == "verify"
    assert "Grasp candidate is not visible" in contexts[0]["failures"][0]["message"]
    assert len([e for e in events if e["type"] == "plan" and e["steps"]]) == 1
