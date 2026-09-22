"""Plan + ReAct execution: monitor each atom and react to failures within budgets."""
import copy

from .atoms.base import AtomContext

MAX_REACT_ITERS = 4
MAX_ATOM_CALLS = 25
MAX_LLM_CALLS = 8

_SEMANTIC_ARGS = ("target", "container", "at", "relation", "open")

_STAGE = {
    "find_object": "视觉定位", "reach_above": "移动", "grasp": "抓取",
    "lift": "抬起", "carry_to": "移动", "release_into": "放置",
    "verify_state": "视觉校验", "reset_arm": "复位", "set_gripper": "抓取",
    "place_on": "叠放", "place_on_table": "桌面放置",
}


class ReActExecutor:
    def __init__(self, sim, locate, kin, registry, state, planner, reactor,
                 on_stage=None, on_event=None, ask_user=None, recorder=None):
        self.sim, self.locate, self.kin = sim, locate, kin
        self.registry, self.state = registry, state
        self.planner, self.reactor = planner, reactor
        self.on_stage = on_stage or (lambda s: None)
        self.on_event = on_event or (lambda e: None)
        self.ask_user = ask_user
        if recorder is None:
            from ..observability import NullRecorder
            recorder = NullRecorder()
        self.recorder = recorder

    def _semantic_args(self, args):
        return {k: args[k] for k in _SEMANTIC_ARGS if k in args}

    def _capture_failure(self):
        try:
            frame = self.sim.observe()
            self.recorder.save_rgbd(frame, note="failure scene")
            self.recorder.save_state(self.state.snapshot(), "state-after")
        except Exception:
            self.recorder.log("warning", "failure artifact capture failed",
                              logger="executor")

    def _call_atom(self, ctx, name, args, *, span_name=None):
        span = self.recorder.start_span(
            span_name or f"atom:{name}", **self._semantic_args(args))
        try:
            result = self.registry.get(name).call(ctx, args)
        except Exception as exc:
            self.recorder.end_span(
                span, "error", type(exc).__name__, str(exc))
            raise
        self.recorder.end_span(
            span, "ok" if result.success else "error",
            result.error_kind, result.message)
        return result

    def _emit(self, event):
        self.on_event(event)

    def _plan_event(self, goal, steps, statuses, pointer):
        self._emit({
            "type": "plan", "goal": goal, "current": pointer, "status": "running",
            "steps": [
                {"index": i, "atom": steps[i]["atom"],
                 "args": steps[i].get("args", {}), "status": statuses[i]}
                for i in range(len(steps))],
        })

    def _fail(self, message):
        self._emit({"type": "finish", "success": False})
        return {"success": False, "message": message}

    def execute(self, plan, goal):
        steps = copy.deepcopy(plan.steps)
        # Recovery may change the route, but cannot remove the original goals.
        # If a source is moved several times, its last requested relation wins.
        goals = {call["args"]["target"]: copy.deepcopy(call["args"])
                 for call in steps if call["atom"] == "verify_state"}
        if not steps:
            return self._fail("没有可执行步骤，无法确认任务完成。")
        statuses = ["pending"] * len(steps)
        pointer = 0
        atom_calls, react_iters, llm_calls = 0, 0, 1
        ctx = AtomContext(self.sim, self.locate, self.kin, self.state,
                          self.on_stage, self.recorder.trace, self.recorder)
        self._plan_event(goal, steps, statuses, pointer)
        while pointer < len(steps):
            atom_calls += 1
            if atom_calls > MAX_ATOM_CALLS:
                return self._fail("动作预算耗尽，请重置或调整需求。")
            call = steps[pointer]
            self.on_stage(_STAGE.get(call["atom"], "执行"))
            self._emit({"type": "activity", "kind": "action",
                        "text": f"第 {pointer + 1} 步：{_STAGE.get(call['atom'], call['atom'])}"
                                + " · " + "，".join(f"{k}={v}" for k, v in call.get("args", {}).items())})
            self._emit({"type": "step", "index": pointer, "status": "active"})
            result = self._call_atom(
                ctx, call["atom"], call.get("args", {}))
            if result.success:
                statuses[pointer] = "done"
                self._emit({"type": "step", "index": pointer, "status": "done"})
                pointer += 1
                continue
            self._capture_failure()
            statuses[pointer] = "failed"
            self._emit({"type": "step", "index": pointer, "status": "failed"})
            self._emit({"type": "activity", "kind": "observation",
                        "text": f"本步未完成：{result.message}"})
            if result.error_kind == "fatal":
                payload = self._fail("关节跟踪异常，已停止后续运动，请重置场景后继续。")
                payload["recovery_required"] = True
                return payload
            react_iters += 1
            if react_iters > MAX_REACT_ITERS:
                return self._fail("反思预算耗尽，仍未完成，请重置后重试。")
            llm_calls += 1
            if llm_calls > MAX_LLM_CALLS:
                return self._fail("LLM 调用预算耗尽，请简化或重置。")
            with self.recorder.span("react", attempt=react_iters):
                decision = self.reactor.decide(
                    goal, steps, pointer, self.state, result, react_iters)
            self._emit({"type": "react", "attempt": react_iters,
                        "thought": decision.thought, "decision": decision.kind,
                        "detail": decision.reason})
            self._emit({"type": "activity", "kind": "recovery",
                        "text": f"恢复方案：{decision.thought}\n{decision.reason}"})
            if decision.kind == "replan":
                llm_calls += 1
                if llm_calls > MAX_LLM_CALLS:
                    return self._fail("LLM 调用预算耗尽，请简化或重置。")
            pointer, steps, statuses = self._apply(
                decision, goal, steps, statuses, pointer, ctx)
            if isinstance(pointer, dict):  # failure payload
                return self._fail(pointer["message"])
        for args in goals.values():
            atom_calls += 1
            if atom_calls > MAX_ATOM_CALLS:
                return self._fail("动作预算耗尽，尚未完成最终目标校验。")
            self.on_stage("视觉校验")
            self._emit({"type": "activity", "kind": "observation",
                        "text": f"最终确认：{args['target']} → {args['at']}，重新读取视觉观测。"})
            result = self._call_atom(
                ctx, "verify_state", args, span_name="final_verify")
            if not result.success:
                payload = self._fail("最终目标未获视觉确认：" + result.message)
                if result.error_kind == "fatal":
                    payload["recovery_required"] = True
                return payload
        self._emit({"type": "finish", "success": True})
        return {"success": True, "message": f"已完成：{goal}"}

    def _apply(self, decision, goal, steps, statuses, pointer, ctx):
        kind = decision.kind
        if kind == "abort":
            return {"message": decision.reason or "执行被安全中止。"}, steps, statuses
        if kind == "ask_user":
            if self.ask_user is None:
                return {"message": decision.question or "需要你确认后才能继续。"}, steps, statuses
            self.ask_user(decision.question or "请确认如何继续。")
            return pointer, steps, statuses
        if kind == "retry":
            inserted = decision.steps or []
            steps = steps[:pointer] + list(inserted) + steps[pointer:]
            statuses = statuses[:pointer] + ["pending"] * len(inserted) + statuses[pointer:]
            self._plan_event(goal, steps, statuses, pointer)
            return pointer, steps, statuses
        if kind == "replace":
            new = decision.steps or []
            steps = steps[:pointer] + list(new) + steps[pointer + 1:]
            statuses = statuses[:pointer] + ["pending"] * len(new) + statuses[pointer + 1:]
            self._plan_event(goal, steps, statuses, pointer)
            return pointer, steps, statuses
        if kind == "replan":
            return self._replan(goal, steps, statuses, pointer)
        return {"message": "未知反思决策。"}, steps, statuses

    def _replan(self, goal, steps, statuses, pointer):
        plan = self.planner.analyze(goal, self.state)
        if not plan.feasible:
            return {"message": "重规划不可行：" + "; ".join(
                b.get("reason", "") for b in plan.blockers)}, steps, statuses
        steps = steps[:pointer] + [dict(c) for c in plan.steps]
        statuses = statuses[:pointer] + ["pending"] * len(plan.steps)
        self._plan_event(goal, steps, statuses, pointer)
        return pointer, steps, statuses
