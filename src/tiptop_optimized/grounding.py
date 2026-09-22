"""Ordered semantic subgoals and bounded, coordinate-free recovery decisions."""
from dataclasses import dataclass
import json

from tiptop_mac.grounding import Goal, GroundingError
from tiptop_mac.types import Predicate


SYSTEM_PROMPT = """你是 TiPToP 优化版的任务拆解器。根据实时 catalog 解析对象指代，
把用户要求拆成扁平、有序的子任务。必须保留“先…再…”的顺序和重复动作，不能仅输出
最终状态。初始状态描述不是动作。例如“A 已在盒内，先把 A 放到桌上再放回盒里”
输出两个子任务 on(A,table)、on(A,box)，不得合并，不得因为最终等于初始而省略。
每个子任务只有一个谓词：on(block,support) 或 holding(block)。
support 是 table 或 catalog 内的 block/box；只能使用目录 ID，禁止坐标。
普通目标已满足可无动作；明确要求“再做一次/拿起再放回”时 require_action=true，
它要求本段实际抓起（中间状态）后才允许完成。“先拿起再放下”也可拆成 holding、on。
最多16个子任务，不能添加用户没要求的目标；拒绝无法表达的任务。
严格输出 JSON：
{"subtasks":[{"instruction":"面向用户的子任务说明",
"goal":[{"predicate":"on","args":["A","table"]}],"require_action":false}],
"rationale":"简短公开摘要"}。"""


@dataclass(frozen=True)
class Subtask:
    instruction: str
    goal: Goal
    require_action: bool = False


class OrderedGrounder:
    def __init__(self, chat):
        self.chat = chat

    def ground(self, instruction, specs):
        try:
            data = self.chat.json(SYSTEM_PROMPT, json.dumps(
                {"instruction": instruction, "catalog": specs}, ensure_ascii=False))
            items = data["subtasks"]
            if not isinstance(items, list) or not 1 <= len(items) <= 16:
                raise GroundingError("需要 1–16 个有序子任务")
            by_id = {item["id"]: item for item in specs}
            tasks = []
            for item in items:
                text = item["instruction"]
                atoms = item["goal"]
                required = item.get("require_action", False)
                if (not isinstance(text, str) or not text.strip()
                        or type(required) is not bool
                        or not isinstance(atoms, list) or len(atoms) != 1):
                    raise GroundingError("每个子任务必须包含一个目标谓词")
                name, args = atoms[0]["predicate"], atoms[0]["args"]
                if (name not in ("on", "holding")
                        or not isinstance(args, list)
                        or len(args) != (2 if name == "on" else 1)
                        or not all(isinstance(a, str) and a.strip() for a in args)):
                    raise GroundingError("子任务谓词无效")
                args = tuple(a.strip() for a in args)
                if by_id.get(args[0], {}).get("kind") != "block":
                    raise GroundingError("只能移动目录中的 block")
                if name == "on" and (
                    args[0] == args[1] or
                    (args[1] != "table" and
                     by_id.get(args[1], {}).get("kind") not in ("block", "box"))
                ):
                    raise GroundingError("放置目标无效")
                rationale = data.get("rationale", "")
                if not isinstance(rationale, str):
                    raise GroundingError("rationale 必须是字符串")
                tasks.append(Subtask(text.strip(),
                                     Goal((Predicate(name, args),), rationale),
                                     required))
            return tuple(tasks)
        except GroundingError:
            raise
        except Exception:
            raise GroundingError("未能拆解为有序子任务") from None


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str


RECOVERY_PROMPT = """你负责机器人当前子任务的 ReAct 恢复。输入是执行错误和最新观测摘要，
completed_subtasks 是已完成前缀，禁止重放或修改目标。只输出公开决策摘要，不输出思维链。
允许 action: replan（根据新场景重规划当前目标），reobserve（再次只读观测后再规划），
abort（无法安全继续）。无法确认持物状态时必须 reobserve 或 abort。
grip_state=empty 表示已确认空爪，holding 表示已确认持物，unknown 表示证据不足；
held=null 本身不能区分空爪与未知。grip_unknown 是观测状态不确定，不代表抓取失败，
请结合 failures 中的 phase、具体原因和最新 support 判断，不能凭空认定动作失败。
禁止坐标、关节值、任意代码或新任务。相同失败反复出现可 abort。
严格 JSON: {"action":"replan","reason":"简短的依据与处理说明"}。"""


class RecoveryReactor:
    def __init__(self, chat):
        self.chat = chat

    def decide(self, context):
        try:
            data = self.chat.json(RECOVERY_PROMPT,
                                  json.dumps(context, ensure_ascii=False),
                                  label="react")
        except Exception:
            return RecoveryDecision("reobserve", "恢复决策暂不可用，先重新观测当前子任务。")
        action, reason = data.get("action"), data.get("reason", "")
        if action not in ("replan", "reobserve", "abort") or not isinstance(reason, str):
            return RecoveryDecision("abort", "恢复决策格式无效，停止执行。")
        return RecoveryDecision(action, reason)
