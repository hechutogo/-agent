"""Plan + ReAct execution: monitor each atom and react to failures within budgets."""
from .atoms.base import AtomContext

MAX_REACT_ITERS = 4
MAX_ATOM_CALLS = 25
MAX_LLM_CALLS = 8

_STAGE = {
    "find_object": "视觉定位", "reach_above": "移动", "grasp": "抓取",
    "lift": "抬起", "carry_to": "移动", "release_into": "放置",
    "verify_state": "视觉校验", "reset_arm": "复位", "set_gripper": "抓取",
}


class ReActExecutor:
    def __init__(self, sim, locate, kin, registry, state, planner, reactor,
                 on_stage=None, on_event=None, ask_user=None):
        self.sim, self.locate, self.kin = sim, locate, kin
        self.registry, self.state = registry, state
        self.planner, self.reactor = planner, reactor
        self.on_stage = on_stage or (lambda s: None)
        self.on_event = on_event or (lambda e: None)
        self.ask_user = ask_user

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
        steps = [dict(call) for call in plan.steps]
        statuses = ["pending"] * len(steps)
        pointer = 0
        atom_calls, react_iters, llm_calls = 0, 0, 1
        ctx = AtomContext(self.sim, self.locate, self.kin, self.state,
                          self.on_stage, lambda t: None)
        self._plan_event(goal, steps, statuses, pointer)
        while pointer < len(steps):
            atom_calls += 1
            if atom_calls > MAX_ATOM_CALLS:
                return self._fail("动作预算耗尽，请重置或调整需求。")
            call = steps[pointer]
            self.on_stage(_STAGE.get(call["atom"], "执行"))
            self._emit({"type": "step", "index": pointer, "status": "active"})
            result = self.registry.get(call["atom"]).call(ctx, call.get("args", {}))
            if result.success:
                statuses[pointer] = "done"
                self._emit({"type": "step", "index": pointer, "status": "done"})
                pointer += 1
                continue
            statuses[pointer] = "failed"
            self._emit({"type": "step", "index": pointer, "status": "failed"})
            react_iters += 1
            if react_iters > MAX_REACT_ITERS:
                return self._fail("反思次数耗尽，仍未完成，请重置后重试。")
            llm_calls += 1
            if llm_calls > MAX_LLM_CALLS:
                return self._fail("LLM 调用预算耗尽，请简化或重置。")
            decision = self.reactor.decide(
                goal, steps, pointer, self.state, result, react_iters)
            self._emit({"type": "react", "attempt": react_iters,
                        "thought": decision.thought, "decision": decision.kind,
                        "detail": decision.reason})
            pointer, steps, statuses = self._apply(
                decision, goal, steps, statuses, pointer, ctx)
            if isinstance(pointer, dict):  # failure payload
                return self._fail(pointer["message"])
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
