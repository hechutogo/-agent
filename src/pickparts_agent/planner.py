"""Feasibility assessment and decomposition into a flat atom workflow."""
import json

from .llm import LLMError

SYSTEM_PROMPT = """你是机器人任务规划者。先判断可行性与前置条件，再拆解任务。\
只能使用下面目录中的原子。目标通过语义 id（A/B/box）引用，\
A、B、box 是已知场景对象，即使当前世界状态尚未观测到也应判为可行，\
执行时 find_object 会完成定位；只有引用未知对象或当前能力无法满足时才判不可行。\
严禁输出或推断坐标。\
每个改变世界状态的动作后必须有对应校验（如放置后 verify_state）。\
计划必须是扁平有序步骤，不得包含循环或嵌套结构。只输出规定 JSON。

可用原子：
{catalog}

输出 JSON：
{{"feasible":true,"blockers":[{{"need":"...","reason":"..."}}],
"steps":[{{"atom":"...","args":{{}}}}],"rationale":"..."}}"""


class Plan:
    def __init__(self, feasible, blockers, steps, rationale):
        self.feasible = bool(feasible)
        self.blockers = blockers or []
        self.steps = steps or []
        self.rationale = rationale or ""


class Planner:
    def __init__(self, chat, registry):
        self.chat = chat
        self.registry = registry

    def analyze(self, goal, state):
        system = SYSTEM_PROMPT.format(catalog=self.registry.catalog_for_prompt())
        user = json.dumps({"goal": goal, "world": state.snapshot()},
                          ensure_ascii=False)
        try:
            data = self.chat.json(system, user)
            feasible = bool(data.get("feasible"))
            steps = data.get("steps", [])
            if not isinstance(steps, list):
                raise ValueError("steps must be a list")
            for call in steps:
                self.registry.validate_call(call)
        except (LLMError, ValueError):
            return Plan(False, [{"need": "重新描述",
                                 "reason": "规划器未能理解或返回了非法计划"}], [], "")
        blockers = data.get("blockers", [])
        if not isinstance(blockers, list):
            blockers = []
        return Plan(feasible, blockers, steps if feasible else [],
                    data.get("rationale", ""))
