"""Ground natural language into TiPToP goal predicates (replaces Gemini step)."""
from dataclasses import dataclass
import json

from .types import Predicate


SYSTEM_PROMPT = """你是 TiPToP 的语言接地器。把用户指令翻译成目标状态谓词。
catalog 是实时对象目录：每项 {id,kind:'block'|'box',label,color}，根据 label、
颜色和 kind 解析同义指代，只能引用目录中存在的 id；另外允许支撑体 "table"。
TiPToP 谓词只有 on 和 holding：
- "放到/放进/放到...上/放入" 都输出 on(movable,support)，入盒与叠放同为 on；
- "拿着/握住" 输出 holding(movable)。
movable 必须是 kind=block 的对象；support 可以是 table、box 或其他 block，
且不能等于 movable。严格输出 JSON，不要输出推理过程：
{"goal":[{"predicate":"on","args":["A","B"]}],"rationale":"一句话面向用户的说明"}
rationale 只能是简短公开摘要。goal 必须非空。"""


class GroundingError(ValueError):
    pass


@dataclass(frozen=True)
class Goal:
    predicates: tuple[Predicate, ...]
    rationale: str


class Grounder:
    def __init__(self, chat):
        self.chat = chat

    def ground(self, instruction, specs):
        user = json.dumps({"instruction": instruction, "catalog": specs},
                          ensure_ascii=False)
        try:
            data = self.chat.json(SYSTEM_PROMPT, user)
            atoms = data["goal"]
            if not isinstance(atoms, list) or not atoms:
                raise GroundingError("goal must be a nonempty list")
            by_id = {item["id"]: item for item in specs}
            predicates, seen = [], set()
            for atom in atoms:
                if not isinstance(atom, dict):
                    raise GroundingError("goal items must be objects")
                name = atom.get("predicate")
                args = atom.get("args")
                if name not in ("on", "holding"):
                    raise GroundingError(f"unsupported predicate: {name}")
                if (not isinstance(args, list) or not args
                        or not all(isinstance(x, str) and x.strip() for x in args)):
                    raise GroundingError("predicate args must be nonempty strings")
                args = tuple(x.strip() for x in args)
                if name == "on":
                    if len(args) != 2:
                        raise GroundingError("on requires movable and support")
                    movable, support = args
                    if by_id.get(movable, {}).get("kind") != "block":
                        raise GroundingError("on movable must be a block")
                    if support != "table" and support not in by_id:
                        raise GroundingError(f"unknown support: {support}")
                    if movable == support:
                        raise GroundingError("cannot place an object on itself")
                else:
                    if len(args) != 1:
                        raise GroundingError("holding requires one block")
                    if by_id.get(args[0], {}).get("kind") != "block":
                        raise GroundingError("holding target must be a block")
                key = (name, args)
                if key not in seen:
                    seen.add(key)
                    predicates.append(Predicate(name, args))
            rationale = data.get("rationale", "")
            if not isinstance(rationale, str):
                raise GroundingError("rationale must be a string")
            return Goal(tuple(predicates), rationale)
        except GroundingError:
            raise
        except Exception:
            raise GroundingError("未能把指令接地为目标谓词") from None
