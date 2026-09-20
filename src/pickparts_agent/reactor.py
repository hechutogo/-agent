"""ReAct reflection: choose retry/replace/replan/ask/abort after an atom fails."""
import json

from .llm import LLMError

_KINDS = {"retry", "replace", "replan", "ask_user", "abort"}

SYSTEM_PROMPT = """你在监控机器人执行。给定失败原子、错误类型、观测事实和当前计划，\
选择下一步：retry（可在 steps 中先插入刷新动作）、replace（用 steps 替换当前原子）、\
replan（重规划剩余任务）、ask_user（向用户提问）、abort（停止并上报）。\
fatal 安全错误必须 abort，不得重试。严禁输出坐标。只输出 JSON：
{"thought":"...","decision":"retry","steps":[],"reason":"...","question":"..."}"""


class Decision:
    def __init__(self, kind, thought="", reason="", steps=None, question=None):
        self.kind = kind
        self.thought = thought
        self.reason = reason
        self.steps = steps
        self.question = question


class Reactor:
    def __init__(self, chat, registry):
        self.chat = chat
        self.registry = registry

    def _fallback(self, error_kind):
        return Decision("abort" if error_kind == "fatal" else "replan",
                        reason="反思器不可用，采用安全默认策略")

    def decide(self, goal, steps, pointer, state, failed, attempt):
        user = json.dumps({
            "goal": goal, "attempt": attempt,
            "failed_atom": steps[pointer] if steps and pointer < len(steps) else None,
            "error_kind": failed.error_kind, "message": failed.message,
            "observed": failed.observed, "remaining": steps[pointer:],
            "world": state.snapshot(),
        }, ensure_ascii=False)
        try:
            data = self.chat.json(SYSTEM_PROMPT, user)
            kind = data.get("decision")
            if kind not in _KINDS:
                raise ValueError("bad decision kind")
            new_steps = data.get("steps") or []
            if not isinstance(new_steps, list):
                raise ValueError("steps must be a list")
            for call in new_steps:
                self.registry.validate_call(call)
        except (LLMError, ValueError):
            return self._fallback(failed.error_kind)
        # Safety override: never retry/replace through a fatal error.
        if failed.error_kind == "fatal" and kind in ("retry", "replace"):
            return self._fallback("fatal")
        return Decision(kind, data.get("thought", ""), data.get("reason", ""),
                        new_steps, data.get("question"))
