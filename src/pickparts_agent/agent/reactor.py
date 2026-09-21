"""ReAct reflection: choose retry/replace/replan/ask/abort after an atom fails."""
import json

from .llm import LLMError

_KINDS = {"retry", "replace", "replan", "ask_user", "abort"}

SYSTEM_PROMPT = """你在监控机器人执行。给定失败原子、错误类型、观测事实和当前计划，
选择下一步：retry（先执行 steps，再重试当前原子）、
replace（用非空 steps 替换当前失败原子，其后的 remaining 原子仍会执行）、
replan（重规划剩余任务，steps 必须为空）、ask_user（向用户提问，steps 为空）、
abort（停止并上报，steps 为空）。不能用空 replace 跳过失败并制造成功。
steps 只能使用实时目录中的原子，需要刷新定位时用 find_object，不要臆造原子。
fatal 安全错误必须立即 abort，不能恢复。严禁输出或推断坐标。

诊断时比较 world.frame_id 和各 objects 观测的 frame_id，旧观测不等于当前事实。
world.catalog 仅为 ID -> {{id,kind,label,color}} 语义元数据，不是物体位置真值。
根据元数据解析同义描述，使用动态 ID；未知对象可通过任意视觉描述 find_object。
lost_object 只表示位置未知或遮挡，绝不是物体回到 table 的证据。
若最近一次可信 placement 已满足目标，且没有更新的相反观测，则 replace 失败移动
为 verify_state 校验源对象和目的；不要从目的处重新抓取。仍须校验，不可直接宣告成功。
verify_state 使用 target=源、at=目的 ID 或 table，relation 可选 on/in/table。
不得将旧 placement=table 覆盖更新的放置记录。视觉不确定时刷新或询问。

grip_failed 表示未确认持有，不能直接 retry lift 或 carry_to。
先 reset_arm、set_gripper {{open:true}}、find_object 源，再 reach_above 源、grasp 源。
如果失败原子为 lift，这些前置 steps 后 retry lift，steps 不要再包含 lift。
如果失败原子为 carry_to，前置 steps 还需 lift 及 find_object 目的，再 retry carry_to。
失败原子为 grasp 时前置 steps 截止 reach_above，再 retry grasp，不要重复 grasp。
只有持有源且目的已定位才能 carry_to {{container:目的ID}}，
再 place_on {{target:目的ID}} 放置当前持有源到 block 上，或 release_into 到 box 内。
任何恢复都须与剩余计划连贯；replace 不会删除其后步骤，不连贯则 replan。
rationale 仅给面向用户的简短诊断和决策摘要，不输出内部思维链、
逐步推理、reasoning_content 或隐藏分析。

可用原子：
{catalog}

只输出 JSON：
{{"rationale":"...","decision":"retry",\
"steps":[{{"atom":"find_object","args":{{"target":"A"}}}}],\
"reason":"...","question":"..."}}"""


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
        if failed.error_kind == "fatal":
            return Decision("abort", reason="检测到安全错误，已中止执行", steps=[])
        user = json.dumps({
            "goal": goal, "attempt": attempt,
            "failed_atom": steps[pointer] if steps and pointer < len(steps) else None,
            "error_kind": failed.error_kind, "message": failed.message,
            "observed": failed.observed, "remaining": steps[pointer:],
            "world": state.snapshot(),
        }, ensure_ascii=False)
        try:
            system = SYSTEM_PROMPT.format(catalog=self.registry.catalog_for_prompt())
            data = self.chat.json(system, user)
            if not isinstance(data, dict):
                raise ValueError("Decision must be an object")
            kind = data.get("decision")
            if not isinstance(kind, str) or kind not in _KINDS:
                raise ValueError("bad decision kind")
            new_steps = data.get("steps", [])
            if not isinstance(new_steps, list):
                raise ValueError("steps must be a list")
            if kind == "replace" and not new_steps:
                raise ValueError("Replacement steps must not be empty")
            if kind in ("replan", "ask_user", "abort") and new_steps:
                raise ValueError("Non-executing decisions must not supply steps")
            rationale = data.get("rationale", data.get("thought", ""))
            reason, question = data.get("reason", ""), data.get("question")
            if (not isinstance(rationale, str) or not isinstance(reason, str)
                    or (question is not None and not isinstance(question, str))):
                raise ValueError("Decision summaries must be strings")
            for call in new_steps:
                self.registry.validate_call(call)
        except (LLMError, TypeError, ValueError):
            return self._fallback(failed.error_kind)
        # Keep the existing executor/UI field, but expose only the public summary.
        return Decision(kind, rationale, reason, new_steps, question)
