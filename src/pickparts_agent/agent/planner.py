"""Feasibility assessment and decomposition into a flat atom workflow."""
import json

from .llm import LLMError

SYSTEM_PROMPT = """你是机器人任务规划者。先评估能力和运行时前置条件，再拆解任务。
只能使用下面实时目录中的原子，严格遵守参数。严禁输出或推断坐标。
world.catalog 是语义元数据字典：ID -> {{id,kind:'block'|'box',label,color}}；
根据 label、color、kind 解析用户的同义描述并使用对应 ID，不能只识别 A/B/box。
A、B、box 仍是默认已知对象。catalog 不是视觉事实，world.objects 才是视觉观测，
需结合 frame_id 判断观测新旧。没有观测不等于物体不存在。
未注册对象也可用任意视觉描述作为 find_object.target，不能仅因未知 ID 一律拒绝。
先定位并由运行时前置条件确认可见、可达、可抓取；能力确实不支持、指代冲突或有
明确安全障碍才给出具体 blockers，不得伪造可行的空计划。

搬放到对象时必须从 find_object 源对象、find_object 目的对象开始，二者均须在
grasp 前定位。顺序为 reach_above 源、grasp 源、lift、carry_to 目的、放置、
verify_state、reset_arm。
carry_to 的参数仍叫 container，但可指任意目的 ID，包括 block 或 box。
place_on 的参数是 target（目的 ID），放置当前 held_object，不是抓取 target。
box 用 release_into，block 上堆叠用 place_on。
table 是运行时测量的虚拟支撑面，不是对象，严禁 find_object/carry_to/place_on(table)。
放到桌面必须使用 find_object 源、reach_above、grasp、lift、place_on_table、
verify_state {{target:源,at:"table",relation:"table"}}、reset_arm。
held_object 非空时严禁 set_gripper {{open:true}} 直接释放，必须使用对应放置原子。
verify_state 使用 target=源、at=目的 ID 或 table；
relation 可省略，明确填写时 block 上为 on，box 内为 in，桌面为 table。
例如 A 叠到 B：find_object A、find_object B、reach_above A、grasp A、lift、
carry_to {{container:"B"}}、place_on {{target:"B"}}、
verify_state {{target:"A",at:"B",relation:"on"}}、reset_arm。
每次放置后立即校验对应源和目的，不得校验错对象或重复抓取已达成目标的物体。
重规划时若已持有源对象，定位目的后从 carry_to 继续，不要再次 grasp。
若最近放置已达成目标，输出 verify_state 而不是空计划。
计划必须是扁平有序步骤，不得包含循环或嵌套结构。
rationale 必须简短说明面向用户的任务理解和决策摘要，不要输出内部思维链、
逐步推理、reasoning_content 或隐藏分析。只输出规定 JSON，feasible 必须是 JSON 布尔值。
feasible=true 时 blockers 为空且 steps 非空；false 时写明 blockers 且 steps 为空。

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
        world = state.snapshot()
        user = json.dumps({"goal": goal, "world": world}, ensure_ascii=False)
        try:
            data = self.chat.json(system, user, label="plan")
            if not isinstance(data, dict) or type(data.get("feasible")) is not bool:
                raise ValueError("feasible must be a JSON boolean")
            feasible = data["feasible"]
            steps = data.get("steps", [])
            if not isinstance(steps, list):
                raise ValueError("steps must be a list")
            blockers = data.get("blockers", [])
            if not isinstance(blockers, list) or any(
                not isinstance(b, dict)
                or not isinstance(b.get("need"), str)
                or not isinstance(b.get("reason"), str) for b in blockers
            ):
                raise ValueError("blockers must contain need/reason strings")
            rationale = data.get("rationale", "")
            if not isinstance(rationale, str):
                raise ValueError("rationale must be a public summary string")
            if feasible and (not steps or blockers):
                raise ValueError("Feasible plans need steps and no blockers")
            for call in steps:
                self.registry.validate_call(call)
            if feasible:
                self._validate_sequence(steps, world)
        except (LLMError, TypeError, ValueError):
            return Plan(False, [{"need": "重新描述",
                                 "reason": "规划器未能理解或返回了非法计划"}], [], "")
        if not feasible and not blockers:
            blockers = [{"need": "确认任务前置条件",
                         "reason": rationale or "规划器未确认任务可行"}]
        return Plan(feasible, blockers, steps if feasible else [], rationale)

    def _validate_sequence(self, steps, world):
        """Check semantic ordering, leaving physical feasibility to the atoms."""
        located = set()
        reached = carried = None
        held = world.get("held_object")
        lifted = held is not None
        gripper_open = world.get("gripper_open", True)
        pending = None
        catalog = world.get("catalog", {})
        for index, call in enumerate(steps):
            name, args = call["atom"], call.get("args", {})
            for key in ("target", "container", "at"):
                if key in args and not args[key].strip():
                    raise ValueError("Object references must not be blank")
            if (name in ("find_object", "reach_above", "grasp", "place_on")
                    and args.get("target") == "table"):
                raise ValueError("Table is not an object target")
            if (name in ("carry_to", "release_into")
                    and args.get("container") == "table"):
                raise ValueError("Table needs place_on_table")
            if pending is not None and name != "verify_state":
                raise ValueError("Placement must be followed by verification")
            if name == "find_object":
                located.add(args["target"])
            elif name == "reach_above":
                reached = args["target"]
                if reached not in located:
                    raise ValueError("Locate the source before reaching")
                gripper_open = True
            elif name == "grasp":
                target = args["target"]
                if (held is not None or not gripper_open
                        or target not in located or reached != target):
                    raise ValueError("Locate and reach before grasping with an empty hand")
                # Find the destination for this grasp, not a later task's placement.
                for future in steps[index + 1:]:
                    if future["atom"] == "grasp":
                        break
                    if future["atom"] in ("place_on", "release_into"):
                        key = "target" if future["atom"] == "place_on" else "container"
                        destination = future.get("args", {})[key]
                        if destination == target or destination not in located:
                            raise ValueError("Locate a distinct destination before grasp")
                        break
                held, lifted, carried = target, False, None
                gripper_open = False
            elif name == "set_gripper":
                gripper_open = args["open"]
                if gripper_open:
                    if held is not None:
                        raise ValueError("Held objects require a placement atom")
                    held, carried, lifted = None, None, False
            elif name == "lift":
                if held is None:
                    raise ValueError("Grasp before lifting")
                lifted = True
            elif name == "carry_to":
                destination = args["container"]
                if not held or not lifted or destination not in located:
                    raise ValueError("Lift source and locate destination before carrying")
                carried = destination
            elif name in ("place_on", "release_into"):
                destination = args["target" if name == "place_on" else "container"]
                if held is None or carried != destination or held == destination:
                    raise ValueError("Carry the held source to its destination before placement")
                kind = catalog.get(destination, {}).get("kind")
                relation = ("in" if name == "release_into" or kind == "box"
                            else "on" if kind == "block" else None)
                pending = (held, destination, relation)
                held, carried, lifted = None, None, False
                gripper_open = True
            elif name == "place_on_table":
                if held is None or not lifted:
                    raise ValueError("Lift the held source before table placement")
                pending = (held, "table", "table")
                held, carried, lifted = None, None, False
                gripper_open = True
            elif name == "verify_state":
                relation = args.get("relation")
                if relation is not None and relation not in ("on", "in", "table"):
                    raise ValueError("Unknown placement relation")
                if relation == "table" and args["at"] != "table":
                    raise ValueError("Table relation needs table destination")
                if args["at"] == "table" and relation not in (None, "table"):
                    raise ValueError("Table destination needs table relation")
                if pending is not None:
                    source, destination, expected_relation = pending
                    if args["target"] != source or args["at"] != destination:
                        raise ValueError("Verification must match the placement")
                    if relation and expected_relation and relation != expected_relation:
                        raise ValueError("Verification relation conflicts with destination kind")
                    pending = None
        if pending is not None:
            raise ValueError("Unverified placement")
