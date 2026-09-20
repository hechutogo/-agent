# 柔性原子 Agent（Flexible Atom + ReAct）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把单体硬编码五段式改造成「可行性评估 → 扁平原子工作流 → ReAct 执行/反思 → 主动追问」的柔性具身 Agent。

**Architecture:** 对象级语义原子（LLM 只传 A/B/box 语义引用，坐标由感知 + 深度反投影 + URDF IK 运行时接地）；轻量自研编排器（OpenAI 兼容 chat + 严格 JSON）；规划 + ReAct 混合控制，有界预算。

**Tech Stack:** Python 3.11、ManiSkill 3.0.1 / SAPIEN 3.0.3、numpy 2.4.6、scipy 1.17.1、openai 3.16.2、FastAPI、pytest 9.1.1。

**Spec:** `docs/superpowers/specs/2026-09-20-flexible-atom-react-agent-design.md`

## Global Constraints

- 非特权观测：只用 `observe()` 的 RGB-D 与本体状态；禁止读取 `actor.pose` 等仿真真值。
- LLM 只传语义引用（A/B/box、table/box），计划与反思中禁止出现坐标。
- 计划始终为扁平有序列表，不使用循环容器嵌套。
- macOS：首个 Vulkan 设备在主线程创建；worker 仅复用已初始化渲染；保留渲染锚。
- 不新增本地模型与新第三方运行时依赖；不改动 `simulation.py` / `kinematics.py` / `perception.py` 的对外接口。
- 预算默认值：MAX_REACT_ITERS=4、MAX_ATOM_CALLS=25、MAX_LLM_CALLS=8。
- 每个任务结束于可独立运行的测试与一次提交；提交信息使用 `feat:`/`refactor:`/`test:` 前缀。

## File Structure

| 文件 | 责任 |
|---|---|
| `src/pickparts_agent/state.py` | `ObjectRecord` + `WorldState` 黑板（含 placement 观测推断） |
| `src/pickparts_agent/atoms/__init__.py` | 导出 + `build_default_registry()` |
| `src/pickparts_agent/atoms/base.py` | `Check/AtomContext/AtomResult/Atom` 契约 |
| `src/pickparts_agent/atoms/registry.py` | `AtomRegistry` 校验与目录文本 |
| `src/pickparts_agent/atoms/library.py` | 9 个语义原子 + 笛卡尔运动助手 |
| `src/pickparts_agent/llm.py` | `LLMError` + `JSONChat`（坏 JSON 修复） |
| `src/pickparts_agent/planner.py` | `Plan/Planner` 可行性与拆解 |
| `src/pickparts_agent/reactor.py` | `Decision/Reactor` 卡点反思 |
| `src/pickparts_agent/executor.py` | `RunView` + `ReActExecutor` |
| `src/pickparts_agent/orchestrator.py` | `Orchestrator` + `build_orchestrator` |
| `src/pickparts_agent/cloud.py` | 保留 Endpoint/CloudPerception/transcribe；退役旧类；移除日志插桩 |
| `src/pickparts_agent/motion.py` | 收口为固定计划走同一执行器 |
| `src/pickparts_agent/web.py` | RobotBackend 接 Orchestrator；state 增 workflow/react |
| `src/pickparts_agent/app.py` | CLI 改用 Orchestrator |
| `static/index.html|app.js|style.css` | 动态工作流泳道 + ReAct 泳道 |
| `tests/helpers.py` | 集中测试假件（FakeSim/FakeKin/FakeLocate/FakeChat） |

约定导入根：测试以 `from pickparts_agent...` 与 `from helpers import ...`（pytest 根为 tests，pyproject 已设 pythonpath=src）。为确保 `helpers` 可导入，新增 `tests/conftest.py`（若不存在）并在其中 `import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))`。

---

## 阶段一：状态黑板与原子层

### Task 1: WorldState 世界状态黑板

**Files:**
- Create: `src/pickparts_agent/state.py`
- Create: `tests/test_state.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `ObjectRecord(id,visible,bbox,point,confidence,frame_id,placement=None)`；`WorldState()` 方法 `tick()->int`、`update_object(rec)`、`set_gripper(open_)`、`set_held(id|None)`、`object_point(id)->np.ndarray|None`、`apply_observation(target,locate_result,frame_id)->ObjectRecord`、`snapshot()->dict`。

- [ ] **Step 1: 写 conftest 以便导入 helpers**

```python
# tests/conftest.py
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_state.py
import numpy as np

from pickparts_agent.state import ObjectRecord, WorldState


def rec(id="A", xy=(-0.085, -0.34), z=0.739, placement=None):
    return ObjectRecord(id, True, [1, 1, 4, 4],
                        [xy[0], xy[1], z], 0.9, 1, placement)


def test_snapshot_is_json_safe_and_decoupled():
    ws = WorldState()
    ws.update_object(rec())
    snap = ws.snapshot()
    assert snap["gripper_open"] is True and snap["held_object"] is None
    a = snap["objects"]["A"]
    assert a["point"] == [-0.085, -0.34, 0.739]
    snap["objects"]["A"]["point"][0] = 999
    assert ws.snapshot()["objects"]["A"]["point"][0] == -0.085


def test_apply_observation_updates_record():
    ws = WorldState()
    out = ws.apply_observation(
        "A", {"bbox": [1, 1, 4, 4], "point": np.array([-0.085, -0.34, 0.739]),
              "confidence": 0.9}, frame_id=3)
    assert out.visible and out.frame_id == 3 and out.placement == "table"


def test_placement_in_box_when_xy_and_z_band_match():
    ws = WorldState()
    ws.update_object(ObjectRecord(
        "box", True, [0, 0, 9, 9], [0.085, -0.30, 0.726], 0.9, 1, "table"))
    out = ws.apply_observation(
        "A", {"bbox": [2, 2, 6, 6], "point": np.array([0.085, -0.30, 0.760]),
              "confidence": 0.9}, frame_id=2)
    assert out.placement == "box"


def test_held_object_placement_is_gripper():
    ws = WorldState()
    ws.set_gripper(False)
    ws.set_held("A")
    out = ws.apply_observation(
        "A", {"bbox": [2, 2, 6, 6], "point": np.array([0.085, -0.30, 0.84]),
              "confidence": 0.9}, frame_id=2)
    assert out.placement == "gripper"


def test_tick_monotonic():
    ws = WorldState()
    assert ws.tick() == 1 and ws.tick() == 2
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/test_state.py -v`
Expected: FAIL（`ModuleNotFoundError: pickparts_agent.state`）

- [ ] **Step 4: 最小实现**

```python
# src/pickparts_agent/state.py
"""Non-privileged world-state blackboard. Facts come only from observations."""
import copy
import threading

import numpy as np


class ObjectRecord:
    def __init__(self, id, visible, bbox, point, confidence, frame_id,
                 placement=None):
        self.id = id
        self.visible = bool(visible)
        self.bbox = list(bbox)
        self.point = [float(v) for v in point]
        self.confidence = float(confidence)
        self.frame_id = int(frame_id)
        self.placement = placement

    def to_dict(self):
        return {"id": self.id, "visible": self.visible, "bbox": self.bbox,
                "point": list(self.point), "confidence": self.confidence,
                "frame_id": self.frame_id, "placement": self.placement}


class WorldState:
    def __init__(self):
        self.lock = threading.RLock()
        self.objects = {}
        self.gripper_open = True
        self.held_object = None
        self.qpos = None
        self.frame_id = 0
        self._tick = 0

    def tick(self):
        with self.lock:
            self._tick += 1
            self.frame_id = self._tick
            return self._tick

    def update_object(self, record):
        with self.lock:
            self.objects[record.id] = record

    def set_gripper(self, open_):
        with self.lock:
            self.gripper_open = bool(open_)

    def set_held(self, object_id):
        with self.lock:
            self.held_object = object_id

    def object_point(self, object_id):
        with self.lock:
            rec = self.objects.get(object_id)
            return np.array(rec.point) if rec is not None else None

    def _infer_placement(self, target_id, point):
        if target_id == "box":
            return "table"
        if self.held_object == target_id:
            return "gripper"
        box = self.objects.get("box")
        if box is not None:
            xy = np.hypot(point[0] - box.point[0], point[1] - box.point[1])
            if xy < 0.045 and box.point[2] - 0.01 <= point[2] <= box.point[2] + 0.07:
                return "box"
        return "table"

    def apply_observation(self, target, locate_result, frame_id):
        with self.lock:
            point = np.asarray(locate_result["point"], dtype=float)
            placement = self._infer_placement(target, point)
            rec = ObjectRecord(
                target, True, list(locate_result["bbox"]), point,
                float(locate_result.get("confidence", 0.9)), frame_id, placement)
            self.objects[target] = rec
            self.frame_id = int(frame_id)
            return copy.copy(rec)

    def snapshot(self):
        with self.lock:
            return {
                "frame_id": self.frame_id,
                "gripper_open": self.gripper_open,
                "held_object": self.held_object,
                "objects": {k: v.to_dict() for k, v in self.objects.items()},
            }
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/test_state.py -v`
Expected: PASS（5 个测试）

- [ ] **Step 6: 提交**

```bash
git add tests/conftest.py tests/test_state.py src/pickparts_agent/state.py
git commit -m "feat: add non-privileged world-state blackboard"
```

---

### Task 2: 原子契约基类

**Files:**
- Create: `src/pickparts_agent/atoms/base.py`
- Create: `tests/test_atom_base.py`

**Interfaces:**
- Produces: `Check(ok,reason="",facts=None)`、`AtomContext(sim,locate,kin,state,on_stage,on_trace)`、`AtomResult(success,error_kind=None,message="",observed=None)`、`Atom`（类属性 name/description/parameters；方法 check_pre/run/verify/call）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_atom_base.py
from pickparts_agent.atoms.base import Atom, AtomContext, AtomResult, Check


class DemoAtom(Atom):
    name = "demo"
    description = "demo atom"
    parameters = {"type": "object"}

    def __init__(self, pre=True, post=True):
        self.pre, self.post = pre, post

    def check_pre(self, ctx, args):
        return Check(self.pre, "pre-failed")

    def run(self, ctx, args):
        return AtomResult(True, message="ran", observed={"v": 1})

    def verify(self, ctx, args, result):
        return Check(self.post, "post-failed")


CTX = AtomContext(None, None, None, None, lambda s: None, lambda t: None)


def test_precondition_failure_short_circuits():
    r = DemoAtom(pre=False).call(CTX, {})
    assert r.success is False and r.error_kind == "precondition"
    assert r.message == "pre-failed"


def test_postcondition_failure_maps_to_verify():
    r = DemoAtom(post=False).call(CTX, {})
    assert r.success is False and r.error_kind == "verify"


def test_happy_path():
    r = DemoAtom().call(CTX, {})
    assert r.success is True and r.observed == {"v": 1}
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_atom_base.py -v`
Expected: FAIL（无 `atoms.base`）

- [ ] **Step 3: 实现**

```python
# src/pickparts_agent/atoms/base.py
"""Atom contract: precondition -> grounded run -> postcondition verify."""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Check:
    ok: bool
    reason: str = ""
    facts: dict = field(default_factory=dict)


@dataclass
class AtomContext:
    sim: object
    locate: Callable
    kin: object
    state: object
    on_stage: Callable[[str], None]
    on_trace: Callable[[dict], None]


@dataclass
class AtomResult:
    success: bool
    error_kind: str | None = None
    message: str = ""
    observed: dict = field(default_factory=dict)


class Atom:
    name = ""
    description = ""
    parameters = {}

    def check_pre(self, ctx, args):
        raise NotImplementedError

    def run(self, ctx, args):
        raise NotImplementedError

    def verify(self, ctx, args, result):
        raise NotImplementedError

    def call(self, ctx, args):
        pre = self.check_pre(ctx, args)
        if not pre.ok:
            return AtomResult(False, "precondition", pre.reason, pre.facts)
        out = self.run(ctx, args)
        if out.success:
            post = self.verify(ctx, args, out)
            if not post.ok:
                return AtomResult(False, "verify", post.reason, post.facts)
        return out
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_atom_base.py -v`
Expected: PASS（3 个）

- [ ] **Step 5: 提交**

```bash
git add src/pickparts_agent/atoms/base.py tests/test_atom_base.py
git commit -m "feat: add atom contract base classes"
```

### Task 3: AtomRegistry 校验与目录

**Files:**
- Create: `src/pickparts_agent/atoms/registry.py`
- Create: `tests/test_registry.py`

**Interfaces:**
- Consumes: `Atom`（Task 2）。
- Produces: `AtomRegistry(atoms=None)`：`register(atom)`、`get(name)->Atom`、`has(name)->bool`、`validate_call(call)`（非法抛 `ValueError`）、`catalog_for_prompt()->str`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_registry.py
import pytest

from pickparts_agent.atoms.base import Atom
from pickparts_agent.atoms.registry import AtomRegistry


class GreetAtom(Atom):
    name = "greet"
    description = "Say hello to a named object."
    parameters = {
        "type": "object",
        "properties": {"target": {"type": "string", "enum": ["A", "B"]}},
        "required": ["target"],
    }

    def check_pre(self, ctx, args): ...
    def run(self, ctx, args): ...
    def verify(self, ctx, args, result): ...


def reg():
    return AtomRegistry([GreetAtom()])


def test_validate_accepts_well_formed_call():
    reg().validate_call({"atom": "greet", "args": {"target": "A"}})


@pytest.mark.parametrize("call", [
    {"atom": "missing", "args": {}},
    {"atom": "greet"},
    {"atom": "greet", "args": {"target": "Z"}},
    {"atom": "greet", "args": {}},
    {"atom": "greet", "args": {"target": "A", "point": [1, 2, 3]}},
    "greet",
])
def test_validate_rejects_unknown_atom_bad_enum_or_extra_args(call):
    with pytest.raises(ValueError):
        reg().validate_call(call)


def test_get_and_has():
    r = reg()
    assert r.has("greet") and not r.has("nope")
    assert isinstance(r.get("greet"), GreetAtom)
    with pytest.raises(KeyError):
        r.get("nope")


def test_catalog_mentions_name_and_param():
    text = reg().catalog_for_prompt()
    assert "greet" in text and "target" in text
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL（无 registry 模块）

- [ ] **Step 3: 实现**

```python
# src/pickparts_agent/atoms/registry.py
"""Registry with hand-written JSON-schema-ish validation (no new dependency)."""


class AtomRegistry:
    def __init__(self, atoms=None):
        self.atoms = {}
        for atom in atoms or []:
            self.register(atom)

    def register(self, atom):
        if not getattr(atom, "name", ""):
            raise ValueError("Atom must define a name")
        self.atoms[atom.name] = atom

    def has(self, name):
        return name in self.atoms

    def get(self, name):
        return self.atoms[name]

    def validate_call(self, call):
        if not isinstance(call, dict):
            raise ValueError("Atom call must be an object")
        if set(call) - {"atom", "args"}:
            raise ValueError("Atom call accepts only atom and args")
        name = call.get("atom")
        if not isinstance(name, str) or name not in self.atoms:
            raise ValueError(f"Unknown atom: {name!r}")
        args = call.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("Atom args must be an object")
        schema = self.atoms[name].parameters or {}
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in args:
                raise ValueError(f"Missing required arg: {key}")
        for key, value in args.items():
            spec = props.get(key)
            if spec is None:
                raise ValueError(f"Unknown arg: {key}")
            expected = spec.get("type")
            if expected == "string" and not isinstance(value, str):
                raise ValueError(f"Arg {key} must be a string")
            if expected == "number" and type(value) not in (int, float):
                raise ValueError(f"Arg {key} must be a number")
            if expected == "boolean" and not isinstance(value, bool):
                raise ValueError(f"Arg {key} must be a boolean")
            if "enum" in spec and value not in spec["enum"]:
                raise ValueError(f"Arg {key} must be one of {spec['enum']}")

    def catalog_for_prompt(self):
        lines = []
        for atom in self.atoms.values():
            lines.append(f"- {atom.name}: {atom.description}")
            props = (atom.parameters or {}).get("properties", {})
            if props:
                fields = []
                for key, spec in props.items():
                    enum = spec.get("enum")
                    fields.append(key + (" ∈ " + "/".join(map(str, enum))
                                        if enum else ""))
                lines.append("    args: " + ", ".join(fields))
        return "\n".join(lines)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_registry.py -v`
Expected: PASS（4 个测试函数）

- [ ] **Step 5: 提交**

```bash
git add src/pickparts_agent/atoms/registry.py tests/test_registry.py
git commit -m "feat: add atom registry with validation and catalog"
```

---

### Task 4: 测试假件集中地

**Files:**
- Create: `tests/helpers.py`

**Interfaces:**
- Produces: `FakeKin`、`FakeSim`、`FakeFrame`、`FakeLocate`、`FakeChat`、`make_context(...)`。供 Task 5 起所有原子/规划/执行测试复用。

- [ ] **Step 1: 写一个自检测试（验证假件行为）**

```python
# tests/test_helpers.py
import numpy as np

from helpers import FakeKin, FakeLocate, FakeSim, make_context


def test_fake_sim_move_updates_qpos():
    sim = FakeSim()
    q = np.array([0.1, 2.6, 2.8, 0.0, 1.57])
    sim.move_right(q, jaw=0.0)
    assert np.allclose(sim.arm_qpos(), q)


def test_fake_kin_forward_tracks_last_solved_target():
    kin = FakeKin()
    target = np.array([-0.08, -0.34, 0.84])
    kin.solve(target)
    assert np.allclose(kin.forward(None)[:3, 3], target)


def test_fake_locate_returns_scripted_point():
    locate = FakeLocate({"A": [-0.08, -0.34, 0.74]})
    out = locate(None, "A")
    assert np.allclose(out["point"], [-0.08, -0.34, 0.74])


def test_make_context_wires_parts():
    ctx = make_context(FakeSim(), FakeKin(), FakeLocate({}), state=None)
    assert hasattr(ctx, "sim") and callable(ctx.locate)
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_helpers.py -v`
Expected: FAIL（无 helpers）

- [ ] **Step 3: 实现假件**

```python
# tests/helpers.py
"""Deterministic test doubles. No physics, network, or real cloud."""
import numpy as np

from pickparts_agent.atoms.base import AtomContext
from pickparts_agent.kinematics import ARM_NAMES
from pickparts_agent.state import WorldState


class FakeFrame:
    def __init__(self, qpos):
        self.qpos = np.asarray(qpos, dtype=float)
        self.rgb = np.zeros((4, 4, 3), np.uint8)
        self.depth = np.ones((4, 4))


class FakeSim:
    def __init__(self):
        self.joint_names = list(ARM_NAMES) + ["Jaw"]
        self.qpos = np.zeros(len(self.joint_names))
        # rest-like default
        defaults = {"Pitch": 2.6, "Elbow": 2.8, "Wrist_Roll": 1.57}
        for i, n in enumerate(ARM_NAMES):
            self.qpos[i] = defaults.get(n, 0.0)
        self.moves = []
        self.tracking_error = False

    def arm_qpos(self):
        idx = [self.joint_names.index(n) for n in ARM_NAMES]
        return self.qpos[idx]

    def observe(self):
        return FakeFrame(self.qpos)

    def hold(self, steps=1):
        return None

    def move_right(self, joints, jaw=None, steps=35):
        joints = np.asarray(joints, dtype=float)
        if self.tracking_error:
            raise RuntimeError("Joint tracking error 0.500 rad")
        idx = [self.joint_names.index(n) for n in ARM_NAMES]
        self.qpos[idx] = joints
        if jaw is not None:
            self.qpos[self.joint_names.index("Jaw")] = jaw
        self.moves.append({"joints": joints.tolist(), "jaw": jaw})

    def close(self):
        pass


class FakeKin:
    def __init__(self):
        self.q = np.array([0.0, 2.6, 2.8, 0.0, 1.57])
        self.position = np.array([0.0, -0.34, 0.84])
        self.fail_solve = False
        self.forward_offset = 0.0

    def solve(self, position, seed=None):
        if self.fail_solve:
            raise ValueError("IK unreachable: position error 0.2000 m")
        self.position = np.asarray(position, float)
        return self.q

    def forward(self, q):
        pose = np.eye(4)
        pose[:3, 3] = self.position + self.forward_offset
        pose[2, 1] = 0.99
        return pose


class FakeLocate:
    def __init__(self, points=None, bbox=None, confidence=0.9):
        self.points = {k: np.asarray(v, float) for k, v in (points or {}).items()}
        self.bbox = bbox or [1, 1, 4, 4]
        self.confidence = confidence
        self.calls = []
        self.fail_on = set()

    def add(self, target, point):
        self.points[target] = np.asarray(point, float)

    def __call__(self, frame, target):
        self.calls.append(target)
        if target in self.fail_on or target not in self.points:
            raise ValueError(f"{target} is not visible")
        return {"bbox": self.bbox, "point": self.points[target],
                "confidence": self.confidence}


class FakeChat:
    """Returns scripted dict responses; can fail on chosen calls."""
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def add(self, payload):
        self.responses.append(payload)

    def json(self, system, user):
        self.calls.append({"system": system, "user": user})
        if not self.responses:
            raise RuntimeError("FakeChat exhausted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return dict(item)


def make_context(sim=None, kin=None, locate=None, state=None,
                 stages=None, traces=None):
    state = state if state is not None else WorldState()
    locate = locate if locate is not None else FakeLocate({})
    return AtomContext(
        sim if sim is not None else FakeSim(),
        locate,
        kin if kin is not None else FakeKin(),
        state,
        (lambda s: stages.append(s)) if stages is not None else (lambda s: None),
        (lambda t: traces.append(t)) if traces is not None else (lambda t: None),
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_helpers.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: 提交**

```bash
git add tests/helpers.py tests/test_helpers.py
git commit -m "test: add shared deterministic test doubles"
```

### Task 5: 感知 / 校验 / 夹爪原子

**Files:**
- Create: `src/pickparts_agent/atoms/library.py`
- Create: `tests/test_atoms_library.py`

**Interfaces:**
- Consumes: `Atom/AtomResult/Check`（Task 2）、`WorldState`（Task 1）、ARM_NAMES。
- Produces（本任务）：`FindObject`、`VerifyState`、`SetGripper`（类名，Task 7 注册其实例）。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_atoms_library.py
import numpy as np

from helpers import FakeKin, FakeLocate, FakeSim, make_context
from pickparts_agent.atoms.library import FindObject, SetGripper, VerifyState


def test_find_object_locates_and_updates_state():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739]})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    r = FindObject().call(ctx, {"target": "A"})
    assert r.success is True
    assert ctx.state.objects["A"].point == [-0.085, -0.34, 0.739]


def test_find_object_missing_returns_lost_object():
    ctx = make_context(FakeSim(), FakeKin(), FakeLocate({}))
    r = FindObject().call(ctx, {"target": "A"})
    assert r.success is False and r.error_kind == "lost_object"


def test_verify_state_success_when_placement_matches():
    locate = FakeLocate({"A": [0.085, -0.30, 0.760], "box": [0.085, -0.30, 0.726]})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    r = VerifyState().call(ctx, {"target": "A", "at": "box"})
    assert r.success is True and r.observed["placement"] == "box"


def test_verify_state_failure_when_placement_differs():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739], "box": [0.085, -0.30, 0.726]})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    r = VerifyState().call(ctx, {"target": "A", "at": "box"})
    assert r.success is False and r.error_kind == "verify"


def test_set_gripper_close_updates_state_and_jaw():
    sim = FakeSim()
    ctx = make_context(sim, FakeKin(), FakeLocate({}))
    r = SetGripper().call(ctx, {"open": False})
    assert r.success and ctx.state.gripper_open is False
    assert sim.moves[-1]["jaw"] == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_atoms_library.py -v`
Expected: FAIL（无 library 模块）

- [ ] **Step 3: 实现**

```python
# src/pickparts_agent/atoms/library.py
"""Object-level semantic atoms. Coordinates are grounded at runtime, never LLM input."""
import functools

import numpy as np

from pickparts_agent.kinematics import ARM_NAMES
from .base import Atom, AtomResult, Check


class FindObject(Atom):
    name = "find_object"
    description = "Locate an object by RGB-D vision and update the world state."
    parameters = {
        "type": "object",
        "properties": {"target": {"type": "string", "enum": ["A", "B", "box"]}},
        "required": ["target"],
    }

    def check_pre(self, ctx, args):
        return Check(True)

    def run(self, ctx, args):
        target = args["target"]
        frame_id = ctx.state.tick()
        frame = ctx.sim.observe()
        try:
            loc = ctx.locate(frame, target)
        except Exception as exc:
            return AtomResult(False, "lost_object",
                              f"{target} 未定位（{type(exc).__name__}）")
        rec = ctx.state.apply_observation(target, loc, frame_id)
        return AtomResult(True, observed={
            "bbox": rec.bbox, "point": rec.point,
            "confidence": rec.confidence, "frame_id": frame_id})

    def verify(self, ctx, args, result):
        rec = ctx.state.objects.get(args["target"])
        if rec is None or not rec.visible or rec.confidence < 0.7:
            return Check(False, "定位结果不可用")
        return Check(True)


class VerifyState(Atom):
    name = "verify_state"
    description = "Check via vision whether an object is at the table or in the box."
    parameters = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": ["A", "B"]},
            "at": {"type": "string", "enum": ["table", "box"]},
        },
        "required": ["target", "at"],
    }

    def check_pre(self, ctx, args):
        return Check(True)

    def run(self, ctx, args):
        target, at = args["target"], args["at"]
        frame_id = ctx.state.tick()
        frame = ctx.sim.observe()
        try:
            box = ctx.locate(frame, "box")
            ctx.state.apply_observation("box", box, frame_id)
            loc = ctx.locate(frame, target)
        except Exception as exc:
            return AtomResult(False, "lost_object",
                              f"校验时丢失目标（{type(exc).__name__}）")
        rec = ctx.state.apply_observation(target, loc, frame_id)
        return AtomResult(True, observed={
            "placement": rec.placement, "expected": at})

    def verify(self, ctx, args, result):
        placement = result.observed.get("placement")
        if placement != args["at"]:
            return Check(False, f"期望在{args['at']}，实际在{placement}")
        return Check(True)


def _guard(fn):
    """Map low-level exceptions to grounded error kinds for motion atoms."""
    @functools.wraps(fn)
    def run(self, ctx, args):
        try:
            return fn(self, ctx, args)
        except ValueError as exc:
            return AtomResult(False, "unreachable", str(exc))
        except RuntimeError as exc:
            return AtomResult(False, "fatal", str(exc))
    return run


class SetGripper(Atom):
    name = "set_gripper"
    description = "Open or close the right-arm gripper."
    parameters = {
        "type": "object",
        "properties": {"open": {"type": "boolean"}},
        "required": ["open"],
    }

    def check_pre(self, ctx, args):
        return Check(True)

    @_guard
    def run(self, ctx, args):
        frame = ctx.sim.observe()
        idx = [ctx.sim.joint_names.index(n) for n in ARM_NAMES]
        q = np.asarray(frame.qpos)[idx]
        jaw = 0.8 if args["open"] else 0.0
        ctx.sim.move_right(q, jaw=jaw, steps=20)
        ctx.state.set_gripper(args["open"])
        return AtomResult(True, observed={"gripper_open": args["open"]})

    def verify(self, ctx, args, result):
        if ctx.state.gripper_open != args["open"]:
            return Check(False, "夹爪状态未到位")
        return Check(True)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_atoms_library.py -v`
Expected: PASS（5 个）

- [ ] **Step 5: 提交**

```bash
git add src/pickparts_agent/atoms/library.py tests/test_atoms_library.py
git commit -m "feat: add find/verify/set-gripper semantic atoms"
```

### Task 6: 运动原子（reach/grasp/lift/carry/release/reset）

**Files:**
- Modify: `src/pickparts_agent/atoms/library.py`（在末尾追加助手与 6 个类）
- Modify: `tests/test_atoms_library.py`（追加用例）

**Interfaces:**
- Produces：`ReachAbove`、`Grasp`、`Lift`、`CarryTo`、`ReleaseInto`、`ResetArm`、常量 `REST_Q`、模块助手 `_cartesian(ctx,position,jaw)`。

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 tests/test_atoms_library.py
from pickparts_agent.atoms.library import (
    CarryTo, Grasp, Lift, ReachAbove, ReleaseInto, ResetArm, REST_Q)


def prepared(ctx):
    FindObject().call(ctx, {"target": "A"})
    FindObject().call(ctx, {"target": "box"})
    ReachAbove().call(ctx, {"target": "A"})
    Grasp().call(ctx, {"target": "A"})


def test_full_motion_chain_grounds_and_holds_then_releases():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739], "box": [0.085, -0.30, 0.726]})
    sim, kin = FakeSim(), FakeKin()
    ctx = make_context(sim, kin, locate)
    prepared(ctx)
    locate.add("A", [-0.085, -0.34, 0.840])  # visually higher after lift
    lift = Lift().call(ctx, {"clearance": 0.10})
    assert lift.success and lift.observed["lift_m"] >= 0.1
    assert ctx.state.held_object == "A"
    carry = CarryTo().call(ctx, {"container": "box"})
    assert carry.success and np.allclose(kin.position[:2], [0.085, -0.30])
    release = ReleaseInto().call(ctx, {"container": "box"})
    assert release.success and ctx.state.held_object is None
    assert ctx.state.gripper_open is True


def test_reach_moves_to_hover_xy():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739], "box": [0.085, -0.30, 0.726]})
    kin = FakeKin()
    ctx = make_context(FakeSim(), kin, locate)
    FindObject().call(ctx, {"target": "A"})
    r = ReachAbove().call(ctx, {"target": "A"})
    assert r.success and np.allclose(kin.position[:2], [-0.085, -0.34])


def test_grasp_closes_gripper():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739], "box": [0.085, -0.30, 0.726]})
    sim = FakeSim()
    ctx = make_context(sim, FakeKin(), locate)
    FindObject().call(ctx, {"target": "A"})
    ReachAbove().call(ctx, {"target": "A"})
    r = Grasp().call(ctx, {"target": "A"})
    assert r.success and ctx.state.gripper_open is False
    assert sim.moves[-1]["jaw"] == 0.0


def test_lift_low_height_is_grip_failed():
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739], "box": [0.085, -0.30, 0.726]})
    ctx = make_context(FakeSim(), FakeKin(), locate)
    FindObject().call(ctx, {"target": "A"})
    ReachAbove().call(ctx, {"target": "A"})
    Grasp().call(ctx, {"target": "A"})
    # Relocated height barely moved: not actually grasped.
    locate.add("A", [-0.085, -0.34, 0.745])
    r = Lift().call(ctx, {"clearance": 0.10})
    assert r.success is False and r.error_kind == "grip_failed"


def test_reset_arm_returns_to_rest():
    sim = FakeSim()
    ctx = make_context(sim, FakeKin(), FakeLocate({}))
    r = ResetArm().call(ctx, {})
    assert r.success and np.allclose(sim.moves[-1]["joints"], REST_Q)


def test_unreachable_ik_maps_to_unreachable():
    kin = FakeKin(); kin.fail_solve = True
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739]})
    ctx = make_context(FakeSim(), kin, locate)
    FindObject().call(ctx, {"target": "A"})
    r = ReachAbove().call(ctx, {"target": "A"})
    assert r.success is False and r.error_kind == "unreachable"


def test_joint_tracking_error_maps_to_fatal():
    sim = FakeSim(); sim.tracking_error = True
    locate = FakeLocate({"A": [-0.085, -0.34, 0.739]})
    ctx = make_context(sim, FakeKin(), locate)
    FindObject().call(ctx, {"target": "A"})
    r = ReachAbove().call(ctx, {"target": "A"})
    assert r.success is False and r.error_kind == "fatal"
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_atoms_library.py -v`
Expected: FAIL（`ImportError: ... ReachAbove`）

- [ ] **Step 3: 在 library.py 末尾追加实现**

```python
# 追加到 src/pickparts_agent/atoms/library.py
REST_Q = np.array([0.0, 2.6, 2.8, 0.0, 1.57])


def _arm_qpos(ctx):
    frame = ctx.sim.observe()
    idx = [ctx.sim.joint_names.index(n) for n in ARM_NAMES]
    return np.asarray(frame.qpos)[idx]


def _cartesian(ctx, position, jaw=None):
    position = np.asarray(position, dtype=float)
    seed = _arm_qpos(ctx)
    pose = ctx.kin.forward(seed)
    dist = float(np.linalg.norm(position - pose[:3, 3]))
    if pose[2, 1] > 0.97 and dist > 0.008:
        count = max(2, int(np.ceil(dist / 0.008)))
        wps = np.linspace(pose[:3, 3], position, count + 1)[1:]
    else:
        wps = [position]
    for wp in wps:
        wp = np.asarray(wp, dtype=float)
        q = ctx.kin.solve(wp, seed=seed)
        ctx.sim.move_right(q, jaw=jaw, steps=6 if len(wps) > 1 else 35)
        seed = _arm_qpos(ctx)
        err = float(np.linalg.norm(ctx.kin.forward(seed)[:3, 3] - wp))
        if err > 0.008:
            return AtomResult(False, "verify", f"TCP tracking error {err:.3f} m")
    return AtomResult(True, observed={"tcp": [float(v) for v in wps[-1]]})


class ReachAbove(Atom):
    name = "reach_above"
    description = "Move the gripper to a hover point above an object."
    parameters = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": ["A", "B", "box"]},
            "clearance": {"type": "number"},
        },
        "required": ["target"],
    }

    def check_pre(self, ctx, args):
        rec = ctx.state.objects.get(args["target"])
        if rec is None or not rec.visible:
            return Check(False, f"尚未定位 {args['target']}")
        if ctx.state.held_object not in (None, args["target"]):
            return Check(False, "机械臂正持有其他物体")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        rec = ctx.state.objects[args["target"]]
        clearance = args.get("clearance", 0.10)
        hover = np.array(rec.point, dtype=float)
        hover[2] += clearance
        result = _cartesian(ctx, hover, jaw=0.8)
        if result.success:
            result.observed["hover"] = [float(v) for v in hover]
        return result

    def verify(self, ctx, args, result):
        target = result.observed.get("hover")
        frame = ctx.sim.observe()
        tcp = ctx.kin.forward(_arm_qpos(ctx))[:3, 3]
        if target is None or np.linalg.norm(tcp - np.array(target)) > 0.008:
            return Check(False, "未到达目标上方")
        return Check(True)


class Grasp(Atom):
    name = "grasp"
    description = "Descend to an object and close the gripper onto it."
    parameters = {
        "type": "object",
        "properties": {"target": {"type": "string", "enum": ["A", "B"]}},
        "required": ["target"],
    }

    def check_pre(self, ctx, args):
        rec = ctx.state.objects.get(args["target"])
        if rec is None or not rec.visible:
            return Check(False, f"尚未定位 {args['target']}")
        if ctx.state.held_object is not None:
            return Check(False, "机械臂已持有物体")
        if not ctx.state.gripper_open:
            return Check(False, "夹爪未张开")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        rec = ctx.state.objects[args["target"]]
        grasp = np.array(rec.point, dtype=float)
        grasp[2] -= 0.004  # RGB-D sees the facing surface; grip just below it.
        result = _cartesian(ctx, grasp, jaw=0.8)
        if not result.success:
            return result
        ctx.sim.move_right(_arm_qpos(ctx), jaw=0.0, steps=25)
        ctx.state.set_gripper(False)
        return AtomResult(True, observed={"gripper_open": False})

    def verify(self, ctx, args, result):
        if ctx.state.gripper_open:
            return Check(False, "夹爪未闭合")
        return Check(True)


class Lift(Atom):
    name = "lift"
    description = "Lift a gripped object and visually confirm the height gain."
    parameters = {
        "type": "object",
        "properties": {"clearance": {"type": "number"}},
    }

    def check_pre(self, ctx, args):
        if ctx.state.gripper_open:
            return Check(False, "夹爪未夹持，无法抬起")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        target = None
        # Identify the gripped candidate from the most recent visible part record.
        for candidate in ("A", "B"):
            rec = ctx.state.objects.get(candidate)
            if rec is not None and rec.visible:
                target = candidate
                break
        if target is None:
            return AtomResult(False, "lost_object", "没有可抬起的零件记录")
        start_z = float(rec.point[2])
        up = np.array([rec.point[0], rec.point[1], start_z + args.get("clearance", 0.10)])
        result = _cartesian(ctx, up)
        if not result.success:
            return result
        frame = ctx.sim.observe()
        try:
            loc = ctx.locate(frame, target)
        except Exception as exc:
            return AtomResult(False, "grip_failed", f"抬起后丢失目标（{type(exc).__name__}）")
        lift_m = float(loc["point"][2] - start_z)
        if lift_m < 0.035:
            return AtomResult(False, "grip_failed",
                              f"视觉未确认抓起：高度变化 {lift_m:.3f} m")
        ctx.state.apply_observation(target, loc, ctx.state.frame_id)
        ctx.state.set_held(target)
        return AtomResult(True, observed={"lift_m": lift_m})

    def verify(self, ctx, args, result):
        if ctx.state.held_object is None or result.observed.get("lift_m", 0) < 0.035:
            return Check(False, "未确认抓起")
        return Check(True)


class CarryTo(Atom):
    name = "carry_to"
    description = "Carry a held object to a point above the destination container."
    parameters = {
        "type": "object",
        "properties": {"container": {"type": "string", "enum": ["box"]}},
        "required": ["container"],
    }

    def check_pre(self, ctx, args):
        if ctx.state.held_object is None:
            return Check(False, "未持有物体，无法搬运")
        box = ctx.state.objects.get(args["container"])
        if box is None or not box.visible:
            return Check(False, "尚未定位目标容器")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        box = np.array(ctx.state.objects[args["container"]].point, dtype=float)
        target = np.array([box[0], box[1], box[2] + 0.109])
        result = _cartesian(ctx, target)
        if result.success:
            result.observed["target_tcp"] = [float(v) for v in target]
        return result

    def verify(self, ctx, args, result):
        target = result.observed.get("target_tcp")
        tcp = ctx.kin.forward(_arm_qpos(ctx))[:3, 3]
        if target is None or np.linalg.norm(tcp - np.array(target)) > 0.008:
            return Check(False, "未到达容器上方")
        return Check(True)


class ReleaseInto(Atom):
    name = "release_into"
    description = "Open the gripper over the container so the held object drops in."
    parameters = {
        "type": "object",
        "properties": {"container": {"type": "string", "enum": ["box"]}},
        "required": ["container"],
    }

    def check_pre(self, ctx, args):
        if ctx.state.held_object is None:
            return Check(False, "未持有物体，无法释放")
        if ctx.state.objects.get(args["container"]) is None:
            return Check(False, "尚未定位目标容器")
        return Check(True)

    @_guard
    def run(self, ctx, args):
        box = np.array(ctx.state.objects[args["container"]].point, dtype=float)
        release = np.array([box[0], box[1], box[2] + 0.055])
        result = _cartesian(ctx, release)
        if not result.success:
            return result
        ctx.sim.move_right(_arm_qpos(ctx), jaw=0.8, steps=20)
        ctx.state.set_gripper(True)
        ctx.state.set_held(None)
        up = np.array([box[0], box[1], box[2] + 0.109])
        _cartesian(ctx, up, jaw=0.8)
        return AtomResult(True, observed={"gripper_open": True, "held": None})

    def verify(self, ctx, args, result):
        if not ctx.state.gripper_open or ctx.state.held_object is not None:
            return Check(False, "物体未脱离夹爪")
        return Check(True)


class ResetArm(Atom):
    name = "reset_arm"
    description = "Return the right arm to its rest posture."
    parameters = {"type": "object"}

    def check_pre(self, ctx, args):
        return Check(True)

    @_guard
    def run(self, ctx, args):
        ctx.sim.move_right(REST_Q, jaw=0.8)
        return AtomResult(True, observed={"rest": REST_Q.tolist()})

    def verify(self, ctx, args, result):
        if np.max(np.abs(_arm_qpos(ctx) - REST_Q)) > 0.05:
            return Check(False, "右臂未回到 rest 姿态")
        return Check(True)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_atoms_library.py -v`
Expected: PASS（Task 5 的 5 个 + 本任务 8 个）

- [ ] **Step 5: 提交**

```bash
git add src/pickparts_agent/atoms/library.py tests/test_atoms_library.py
git commit -m "feat: add grounded motion atoms"
```

### Task 7: 默认原子注册表与包导出

**Files:**
- Create: `src/pickparts_agent/atoms/__init__.py`
- Create: `tests/test_atom_registry_default.py`

**Interfaces:**
- Produces: `build_default_registry()->AtomRegistry`；包级导出所有原子类。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_atom_registry_default.py
from pickparts_agent.atoms import build_default_registry

EXPECTED = {"find_object", "reach_above", "grasp", "lift", "carry_to",
            "release_into", "verify_state", "reset_arm", "set_gripper"}


def test_default_registry_registers_all_nine_atoms():
    registry = build_default_registry()
    assert set(registry.atoms) == EXPECTED


def test_catalog_is_non_empty_text():
    assert "find_object" in build_default_registry().catalog_for_prompt()
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_atom_registry_default.py -v`
Expected: FAIL（无 build_default_registry）

- [ ] **Step 3: 实现**

```python
# src/pickparts_agent/atoms/__init__.py
"""Semantic atom package."""
from .base import Atom, AtomContext, AtomResult, Check
from .library import (
    CarryTo, FindObject, Grasp, Lift, ReachAbove, ReleaseInto, ResetArm,
    SetGripper, VerifyState, REST_Q,
)
from .registry import AtomRegistry


def build_default_registry():
    return AtomRegistry([
        FindObject(), ReachAbove(), Grasp(), Lift(), CarryTo(),
        ReleaseInto(), VerifyState(), ResetArm(), SetGripper(),
    ])


__all__ = [
    "Atom", "AtomContext", "AtomResult", "Check", "AtomRegistry",
    "CarryTo", "FindObject", "Grasp", "Lift", "ReachAbove",
    "ReleaseInto", "ResetArm", "SetGripper", "VerifyState", "REST_Q",
    "build_default_registry",
]
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_atom_registry_default.py tests/test_atoms_library.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/pickparts_agent/atoms/__init__.py tests/test_atom_registry_default.py
git commit -m "feat: wire default atom registry"
```

---

## 阶段二：结构化云端调用、规划器与反思器

### Task 8: JSONChat 严格 JSON 调用

**Files:**
- Create: `src/pickparts_agent/llm.py`
- Create: `tests/test_llm_json.py`

**Interfaces:**
- Produces: `LLMError(RuntimeError)`、`JSONChat(client,model,max_repair=1)` 方法 `json(system,user)->dict`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_llm_json.py
import httpx
import pytest

from pickparts_agent.llm import JSONChat, LLMError


def client_with(payloads):
    it = iter(payloads)
    def handle(request):
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": next(it)}}]})
    from openai import OpenAI
    return OpenAI(api_key="k", base_url="https://cloud.invalid/v1",
                  http_client=httpx.Client(transport=httpx.MockTransport(handle)),
                  max_retries=0)


def test_json_returns_parsed_object():
    chat = JSONChat(client_with(['{"a": 1}']), "m")
    assert chat.json("sys", "user") == {"a": 1}


def test_bad_json_triggers_one_repair_then_succeeds():
    chat = JSONChat(client_with(['not-json', '{"a": 2}']), "m")
    assert chat.json("sys", "user") == {"a": 2}


def test_bad_json_after_repair_raises_llm_error():
    chat = JSONChat(client_with(['bad', 'still-bad']), "m")
    with pytest.raises(LLMError):
        chat.json("sys", "user")


def test_network_error_passes_through_openai_exception():
    from openai import APITimeoutError
    def raise_once(request):
        raise APITimeoutError(request=request)
    from openai import OpenAI
    c = OpenAI(api_key="k", base_url="https://cloud.invalid/v1",
               http_client=httpx.Client(transport=httpx.MockTransport(raise_once)),
               max_retries=0)
    with pytest.raises(APITimeoutError):
        JSONChat(c, "m").json("sys", "user")
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_llm_json.py -v`
Expected: FAIL（无 llm 模块）

- [ ] **Step 3: 实现**

```python
# src/pickparts_agent/llm.py
"""OpenAI-compatible chat with strict JSON output and a single repair retry."""
import json


class LLMError(RuntimeError):
    pass


class JSONChat:
    def __init__(self, client, model, max_repair=1):
        self.client = client
        self.model = model
        self.max_repair = max_repair

    def _complete(self, system, user):
        response = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return response.choices[0].message.content

    def json(self, system, user):
        raw = self._complete(system, user)
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except (TypeError, json.JSONDecodeError):
            pass
        for _ in range(self.max_repair):
            repaired = self._complete(
                system, user + "\n只输出合法 JSON，不要解释或 Markdown 代码块。")
            try:
                data = json.loads(repaired)
                if isinstance(data, dict):
                    return data
            except (TypeError, json.JSONDecodeError):
                continue
        raise LLMError("LLM did not return valid JSON")
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_llm_json.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: 提交**

```bash
git add src/pickparts_agent/llm.py tests/test_llm_json.py
git commit -m "feat: add strict-JSON cloud chat helper"
```

### Task 9: Planner 可行性评估与任务拆解

**Files:**
- Create: `src/pickparts_agent/planner.py`
- Create: `tests/test_planner.py`

**Interfaces:**
- Consumes: `JSONChat`（.json）、`AtomRegistry`（validate_call/catalog_for_prompt）、`WorldState.snapshot`。
- Produces: `Plan(feasible,blockers,steps,rationale)`；`Planner(chat,registry)` 方法 `analyze(goal,state)->Plan`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_planner.py
import json

from helpers import FakeChat
from pickparts_agent.atoms import build_default_registry
from pickparts_agent.llm import LLMError
from pickparts_agent.planner import Planner
from pickparts_agent.state import WorldState


def planner(chat):
    return Planner(chat, build_default_registry())


def valid_steps():
    return [
        {"atom": "find_object", "args": {"target": "A"}},
        {"atom": "reach_above", "args": {"target": "A"}},
        {"atom": "grasp", "args": {"target": "A"}},
        {"atom": "lift", "args": {"clearance": 0.1}},
        {"atom": "carry_to", "args": {"container": "box"}},
        {"atom": "release_into", "args": {"container": "box"}},
        {"atom": "verify_state", "args": {"target": "A", "at": "box"}},
        {"atom": "reset_arm", "args": {}},
    ]


def test_feasible_plan_returns_validated_steps():
    chat = FakeChat([{"feasible": True, "blockers": [], "steps": valid_steps(),
                      "rationale": "all visible"}])
    plan = planner(chat).analyze("把 A 放进盒子", WorldState())
    assert plan.feasible and len(plan.steps) == 8
    assert "find_object" in chat.calls[0]["system"]


def test_infeasible_returns_blockers_and_no_steps():
    chat = FakeChat([{"feasible": False,
                      "blockers": [{"need": "目标可见", "reason": "没有零件 C"}],
                      "steps": [], "rationale": ""}])
    plan = planner(chat).analyze("把 C 放进盒子", WorldState())
    assert not plan.feasible and plan.steps == []
    assert plan.blockers[0]["need"] == "目标可见"


def test_bad_json_returns_infeasible_plan():
    plan = planner(FakeChat([LLMError("x")])).analyze("do it", WorldState())
    assert not plan.feasible and plan.steps == []


def test_unknown_atom_in_plan_fails_closed():
    steps = valid_steps() + [{"atom": "explode", "args": {}}]
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("goal", WorldState())
    assert not plan.feasible and plan.steps == []
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL（无 planner 模块）

- [ ] **Step 3: 实现**

```python
# src/pickparts_agent/planner.py
"""Feasibility assessment and decomposition into a flat atom workflow."""
import json

from .llm import LLMError

SYSTEM_PROMPT = """你是机器人任务规划者。先判断可行性与前置条件，再拆解任务。\
只能使用下面目录中的原子，以及当前世界状态中存在的对象。\
严禁输出或推断坐标；目标通过语义 id（A/B/box）引用。\
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_planner.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: 提交**

```bash
git add src/pickparts_agent/planner.py tests/test_planner.py
git commit -m "feat: add feasibility planner"
```

---

### Task 10: Reactor 卡点反思决策

**Files:**
- Create: `src/pickparts_agent/reactor.py`
- Create: `tests/test_reactor.py`

**Interfaces:**
- Consumes: `JSONChat`、`AtomRegistry`、`AtomResult`。
- Produces: `Decision(kind,thought,reason,steps=None,question=None)`；`Reactor(chat,registry)` 方法 `decide(goal,steps,pointer,state,failed,attempt)->Decision`。kind ∈ retry/replace/replan/ask_user/abort。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_reactor.py
from helpers import FakeChat
from pickparts_agent.atoms import build_default_registry
from pickparts_agent.atoms.base import AtomResult
from pickparts_agent.llm import LLMError
from pickparts_agent.reactor import Reactor
from pickparts_agent.state import WorldState


def reactor(chat):
    return Reactor(chat, build_default_registry())


def failed(kind):
    return AtomResult(False, kind, "blocked")


def test_retry_returns_refresh_steps():
    chat = FakeChat([{"thought": "落点偏", "decision": "retry", "reason": "刷新",
                      "steps": [{"atom": "find_object", "args": {"target": "A"}}]}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("transient"), 1)
    assert d.kind == "retry" and d.steps[0]["atom"] == "find_object"


def test_unknown_decision_kind_falls_back_to_replan():
    chat = FakeChat([{"thought": "", "decision": "fly", "steps": []}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("transient"), 1)
    assert d.kind == "replan"


def test_bad_step_falls_back_to_replan():
    chat = FakeChat([{"thought": "", "decision": "replace",
                      "steps": [{"atom": "bogus", "args": {}}]}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("verify"), 1)
    assert d.kind == "replan"


def test_fatal_forces_abort_even_if_model_retries():
    chat = FakeChat([{"thought": "", "decision": "retry", "steps": []}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("fatal"), 1)
    assert d.kind == "abort"


def test_llm_error_fallback_matches_error_kind():
    transient = reactor(FakeChat([LLMError("x")])).decide(
        "g", [], 0, WorldState(), failed("transient"), 1)
    fatal = reactor(FakeChat([LLMError("x")])).decide(
        "g", [], 0, WorldState(), failed("fatal"), 1)
    assert transient.kind == "replan" and fatal.kind == "abort"


def test_ask_user_keeps_question():
    chat = FakeChat([{"thought": "需要确认", "decision": "ask_user",
                      "reason": "r", "question": "先处理哪个？", "steps": []}])
    d = reactor(chat).decide("g", [], 0, WorldState(), failed("precondition"), 1)
    assert d.kind == "ask_user" and d.question == "先处理哪个？"
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_reactor.py -v`
Expected: FAIL（无 reactor 模块）

- [ ] **Step 3: 实现**

```python
# src/pickparts_agent/reactor.py
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_reactor.py -v`
Expected: PASS（6 个）

- [ ] **Step 5: 提交**

```bash
git add src/pickparts_agent/reactor.py tests/test_reactor.py
git commit -m "feat: add react reactor with safety override"
```

## 阶段三：ReAct 执行器与编排总装

### Task 11: ReActExecutor 成功路径

**Files:**
- Create: `src/pickparts_agent/executor.py`
- Create: `tests/test_executor.py`

**Interfaces:**
- Consumes: `AtomRegistry`、`Planner.analyze->Plan`、`Reactor.decide->Decision`、`AtomContext`、`ArmKinematics`。
- Produces: `ReActExecutor(sim,locate,kin,registry,state,planner,reactor,on_stage=None,on_event=None,ask_user=None)`；方法 `execute(plan,goal)->dict`。事件回调形状见实现。

- [ ] **Step 1: 写失败测试（含一个可控原子）**

```python
# tests/test_executor.py
from helpers import FakeChat, FakeKin, FakeLocate, FakeSim
from pickparts_agent.atoms import FindObject, build_default_registry
from pickparts_agent.atoms.base import Atom, AtomResult
from pickparts_agent.atoms.registry import AtomRegistry
from pickparts_agent.planner import Plan
from pickparts_agent.reactor import Reactor
from pickparts_agent.planner import Planner
from pickparts_agent.executor import ReActExecutor
from pickparts_agent.state import WorldState


class FlakyAtom(Atom):
    name = "flaky"
    description = "controllable"
    parameters = {"type": "object"}

    def __init__(self, fail_times=0, kind="transient"):
        self.remaining, self.kind, self.calls = fail_times, kind, 0

    def check_pre(self, ctx, args):
        from pickparts_agent.atoms.base import Check
        return Check(True)

    def run(self, ctx, args):
        self.calls += 1
        if self.calls <= self.remaining:
            return AtomResult(False, self.kind, "fail")
        return AtomResult(True, message="ok")

    def verify(self, ctx, args, result):
        from pickparts_agent.atoms.base import Check
        return Check(True)


def build(flaky, chat, *, ask_user=None):
    registry = AtomRegistry([flaky, FindObject()])
    planner = Planner(chat, registry)
    reactor = Reactor(chat, registry)
    locate = FakeLocate({"A": [-0.08, -0.34, 0.74]})
    return ReActExecutor(FakeSim(), locate, FakeKin(), registry,
                         WorldState(), planner, reactor, ask_user=ask_user), registry


def test_happy_path_emits_plan_steps_and_finish():
    events = []
    executor, _ = build(FlakyAtom(0), FakeChat())
    plan = Plan(True, [], [{"atom": "flaky", "args": {}}], "")
    result = executor.execute(plan, "goal")
    types = [e["type"] for e in events]
    assert result["success"] is True
    assert types == ["plan", "step", "step", "finish"]
```

注意：上面把 `on_event=events.append` 注入。为使测试成立，`build` 默认不传事件；在测试内改为：

```python
    executor, _ = build(FlakyAtom(0), FakeChat())
    executor.on_event = events.append
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_executor.py::test_happy_path_emits_plan_steps_and_finish -v`
Expected: FAIL（无 executor 模块）

- [ ] **Step 3: 实现执行器（先支持成功路径，卡点分支 Task 12 补全）**

```python
# src/pickparts_agent/executor.py
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
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_executor.py::test_happy_path_emits_plan_steps_and_finish -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/pickparts_agent/executor.py tests/test_executor.py
git commit -m "feat: add react executor core loop"
```

---

### Task 12: 卡点 retry / replace / replan

**Files:**
- Modify: `tests/test_executor.py`

**Interfaces:**
- Consumes: `Reactor/Planner`（FakeChat 脚本）；本任务只加测试，执行器实现已在 Task 11 覆盖这些分支。

- [ ] **Step 1: 追加失败/恢复测试**

```python
# 追加到 tests/test_executor.py
def test_retry_refreshes_then_succeeds():
    events = []
    flaky = FlakyAtom(1)  # first call fails
    chat = FakeChat([
        {"thought": "刷新", "decision": "retry", "reason": "重新定位",
         "steps": [{"atom": "find_object", "args": {"target": "A"}}]},
    ])
    executor, _ = build(flaky, chat)
    executor.on_event = events.append
    plan = Plan(True, [], [{"atom": "flaky", "args": {}}], "")
    result = executor.execute(plan, "goal")
    assert result["success"] and flaky.calls == 2
    assert any(e["type"] == "react" and e["decision"] == "retry" for e in events)


def test_replace_swaps_in_alt_atom():
    from pickparts_agent.atoms.base import Check

    class AltAtom(Atom):
        name = "alt"
        description = "alt"
        parameters = {"type": "object"}
        ran = 0

        def check_pre(self, ctx, args):
            return Check(True)

        def run(self, ctx, args):
            AltAtom.ran += 1
            return AtomResult(True, message="alt-ok")

        def verify(self, ctx, args, result):
            return Check(True)

    alt = AltAtom()
    flaky = FlakyAtom(1)
    chat = FakeChat([
        {"thought": "换方案", "decision": "replace", "reason": "用 alt",
         "steps": [{"atom": "alt", "args": {}}]},
    ])
    registry = AtomRegistry([flaky, alt, FindObject()])
    locate = FakeLocate({"A": [-0.08, -0.34, 0.74]})
    executor = ReActExecutor(
        FakeSim(), locate, FakeKin(), registry, WorldState(),
        Planner(chat, registry), Reactor(chat, registry))
    result = executor.execute(
        Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert result["success"] and flaky.calls == 1 and AltAtom.ran == 1


def test_replan_succeeds_with_new_plan():
    flaky = FlakyAtom(1)
    chat = FakeChat([
        {"thought": "重规划", "decision": "replan", "steps": []},
        {"feasible": True, "blockers": [],
         "steps": [{"atom": "flaky", "args": {}}], "rationale": ""},
    ])
    executor, _ = build(flaky, chat)
    result = executor.execute(
        Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert result["success"] and flaky.calls == 2
```

- [ ] **Step 2: 运行确认通过**

Run: `pytest tests/test_executor.py -v`
Expected: PASS（Task 11 的 1 个 + 本任务 3 个）

- [ ] **Step 3: 提交**

```bash
git add tests/test_executor.py
git commit -m "test: cover retry/replace/replan recovery"
```

### Task 13: 预算上限与 fatal 安全

**Files:**
- Modify: `src/pickparts_agent/executor.py`（在 react 事件后增加 replan 的 LLM 计数）
- Modify: `tests/test_executor.py`

**Interfaces:**
- 变更：execute 对「即将发起的重规划」也计入 MAX_LLM_CALLS。

- [ ] **Step 1: 修改 executor：重规划前计数**

在 `self._emit({"type": "react", ...})` 之后、`self._apply(...)` 之前插入：

```python
            if decision.kind == "replan":
                llm_calls += 1
                if llm_calls > MAX_LLM_CALLS:
                    return self._fail("LLM 调用预算耗尽，请简化或重置。")
```

- [ ] **Step 2: 追加测试**

```python
# 追加到 tests/test_executor.py
def retry_chat():
    return FakeChat([
        {"thought": "", "decision": "retry", "steps": []},
        {"thought": "", "decision": "retry", "steps": []},
        {"thought": "", "decision": "retry", "steps": []},
        {"thought": "", "decision": "retry", "steps": []},
    ])


def test_react_budget_exhausted():
    flaky = FlakyAtom(99)  # always fails
    executor, _ = build(flaky, retry_chat())
    result = executor.execute(
        Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert result["success"] is False and "预算" in result["message"]
    assert flaky.calls == 5  # 4 retries allowed, 5th failure terminates


def test_fatal_aborts_without_retry():
    flaky = FlakyAtom(99, kind="fatal")
    chat = FakeChat([{"thought": "", "decision": "retry", "steps": []}])
    executor, _ = build(flaky, chat)
    result = executor.execute(
        Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert result["success"] is False
    assert flaky.calls == 1  # reactor override forces abort immediately


def test_llm_budget_exhausted_on_repeated_replan():
    flaky = FlakyAtom(99)
    cycle = [
        {"thought": "", "decision": "replan", "steps": []},
        {"feasible": True, "blockers": [],
         "steps": [{"atom": "flaky", "args": {}}], "rationale": ""},
    ]
    # 3 full cycles (6 responses) then a 4th reactor response (7th) -> LLM=9.
    chat = FakeChat(cycle * 3 + [cycle[0]])
    executor, _ = build(flaky, chat)
    result = executor.execute(
        Plan(True, [], [{"atom": "flaky", "args": {}}], ""), "goal")
    assert result["success"] is False
    assert "LLM" in result["message"] and flaky.calls == 4
```

- [ ] **Step 3: 运行确认通过**

Run: `pytest tests/test_executor.py -v`
Expected: PASS（共 8 个测试）

- [ ] **Step 4: 提交**

```bash
git add src/pickparts_agent/executor.py tests/test_executor.py
git commit -m "feat: enforce react/atom/llm budgets and fatal abort"
```

---

### Task 14: Orchestrator 六步总装与多轮状态

**Files:**
- Create: `src/pickparts_agent/orchestrator.py`
- Create: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: Planner/Reactor/ReActExecutor、WorldState、Endpoint。
- Produces: `Orchestrator(state,registry,planner,reactor,executor)` 方法 `turn(text)->dict`；`build_orchestrator(sim,locate,llm_endpoint,on_stage=None,on_event=None,ask_user=None)`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_orchestrator.py
from pickparts_agent.orchestrator import Orchestrator
from pickparts_agent.planner import Plan


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
    from pickparts_agent.state import WorldState
    state = WorldState()
    orch = Orchestrator(state, object(), planner, object(), executor)
    orch.turn("第一个任务")
    orch.turn("第二个任务")
    assert planner.seen[0] is state and planner.seen[1] is state
    assert len(orch.history) == 2
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL（无 orchestrator 模块）

- [ ] **Step 3: 实现**

```python
# src/pickparts_agent/orchestrator.py
"""Six-step embodied loop: assess -> plan -> execute/react -> follow-up."""
from .atoms import build_default_registry
from .executor import ReActExecutor
from .kinematics import ArmKinematics
from .llm import JSONChat
from .planner import Planner
from .reactor import Reactor
from .state import WorldState


def render_blockers(blockers):
    if not blockers:
        return "这个需求当前无法完成，请换一种说法。"
    parts = []
    for b in blockers:
        need, reason = b.get("need", ""), b.get("reason", "")
        parts.append(f"{need}（{reason}）" if reason else need)
    return "无法完成：" + "；".join(parts)


class Orchestrator:
    def __init__(self, state, registry, planner, reactor, executor):
        self.state = state
        self.registry = registry
        self.planner = planner
        self.reactor = reactor
        self.executor = executor
        self.history = []

    def turn(self, text):
        plan = self.planner.analyze(text, self.state)
        if not plan.feasible:
            return {"success": False, "message": render_blockers(plan.blockers)}
        result = self.executor.execute(plan, goal=text)
        if result.get("success") and "还需要" not in result.get("message", ""):
            result["message"] += " 还需要我做什么？"
        self.history.append({"goal": text, **result})
        return result


def build_orchestrator(sim, locate, llm_endpoint,
                       on_stage=None, on_event=None, ask_user=None):
    state = WorldState()
    registry = build_default_registry()
    chat = JSONChat(llm_endpoint.client(), llm_endpoint.model)
    planner = Planner(chat, registry)
    reactor = Reactor(chat, registry)
    executor = ReActExecutor(
        sim, locate, ArmKinematics(), registry, state, planner, reactor,
        on_stage=on_stage, on_event=on_event, ask_user=ask_user)
    return Orchestrator(state, registry, planner, reactor, executor)
```

- [ ] **Step 4: 运行确认通过**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS（4 个）

- [ ] **Step 5: 提交**

```bash
git add src/pickparts_agent/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add six-step orchestrator with persistent state"
```

## 阶段四：接线与前端

### Task 15: web.py 接入 Orchestrator 并暴露 workflow/react

**Files:**
- Modify: `src/pickparts_agent/web.py`
- Modify: `tests/test_web.py`（假 Backend 增加可选第三参）

**Interfaces:**
- Consumes: `build_orchestrator(sim,locate,llm_endpoint,on_stage,on_event)`。
- Produces: Console 新增 `view(event)`；factory 调用变为三参 `(publish, stage, view)`；state 新增 `workflow`、`react`。

- [ ] **Step 1: 修改 RobotBackend**

将 `RobotBackend` 整体替换为：

```python
class RobotBackend:
    def __init__(self, publish, stage, view=None):
        from dotenv import load_dotenv
        from .cloud import Endpoint, CloudPerception
        from .orchestrator import build_orchestrator
        from .simulation import Simulation

        load_dotenv(ROOT / ".env")
        llm, vision = Endpoint.from_env("LLM"), Endpoint.from_env("VLM")
        self.client = vision.client()
        self.sim = None
        try:
            self.sim = Simulation()
            self.sim.on_frame = publish
            locate = CloudPerception(self.client, vision.model).locate
            self.orchestrator = build_orchestrator(
                self.sim, locate, llm,
                on_stage=stage, on_event=view or (lambda e: None))
            self.models = {"llm": llm.model, "vision": vision.model}
            publish(self.sim.observe())
        except Exception:
            self.close()
            raise

    def turn(self, text):
        result = self.orchestrator.turn(text)
        result["recovery_required"] = False
        self.sim.on_frame(self.sim.observe())
        return result

    def close(self):
        if self.sim is not None:
            self.sim.close()
        self.client.close()
```

- [ ] **Step 2: Console 初始 state 增加字段并实现 view**

state 初始化改为含 `"workflow": None, "react": []`。新增方法：

```python
    def view(self, event):
        kind = event["type"]
        with self.lock:
            if kind == "plan":
                self.state["workflow"] = {
                    "goal": event["goal"], "current": event["current"],
                    "status": "running", "steps": event["steps"]}
            elif kind == "step":
                wf = self.state.get("workflow")
                if wf:
                    for step in wf["steps"]:
                        if step["index"] == event["index"]:
                            step["status"] = event["status"]
            elif kind == "react":
                self.state.setdefault("react", []).append({
                    "attempt": event["attempt"], "thought": event["thought"],
                    "decision": event["decision"], "detail": event["detail"]})
            elif kind == "finish":
                wf = self.state.get("workflow")
                if wf:
                    wf["status"] = "done" if event["success"] else "failed"
```

- [ ] **Step 3: 修改 factory 调用传第三参；reset 时清空 view**

worker 中 reset 分支的 backend 构造改为 `backend = self.factory(self.publish, self.stage, self.view)`，
并在同分支的 `self.state.update(...)` 中加入 `workflow=None, react=[]`。

- [ ] **Step 4: 更新测试假 Backend 签名并断言新字段**

把 `tests/test_web.py` 中 `Backend.__init__(self, publish, stage)` 改为
`def __init__(self, publish, stage, view=None)`。在首个测试 ready 后加：

```python
        initial = client.get("/api/state").json()
        assert initial["workflow"] is None and initial["react"] == []
```

- [ ] **Step 5: 运行测试**

Run: `pytest tests/test_web.py -v`
Expected: PASS（2 个测试）

- [ ] **Step 6: 提交**

```bash
git add src/pickparts_agent/web.py tests/test_web.py
git commit -m "refactor: wire orchestrator into web console"
```

---

### Task 16: 前端动态工作流泳道 + ReAct 泳道

**Files:**
- Modify: `src/pickparts_agent/static/index.html`
- Modify: `src/pickparts_agent/static/app.js`
- Modify: `src/pickparts_agent/static/style.css`

**Interfaces:**
- 消费 state.workflow（{goal,current,status,steps[]}）与 state.react[]。

- [ ] **Step 1: index.html 替换固定五段**

把 `<div class="steps" id="steps"> ... </div>` 整块替换为：

```html
          <div id="workflow-lane" class="workflow-lane"></div>
          <div id="react-lane" class="react-lane"></div>
```

- [ ] **Step 2: app.js 顶部移除固定 phases**

删除第 2 行 `const phases = [...]`，在其后新增原子中文标签：

```javascript
const ATOM_LABEL = {
  find_object: "定位", reach_above: "移动到上方", grasp: "抓取",
  lift: "抬起", carry_to: "搬运", release_into: "放入",
  verify_state: "校验", reset_arm: "复位", set_gripper: "夹爪",
};
```

- [ ] **Step 3: app.js 新增两个渲染函数**

```javascript
function renderWorkflow(wf) {
  const lane = $("workflow-lane");
  lane.replaceChildren();
  if (!wf) {
    lane.textContent = "等待可执行的工作流…";
    return;
  }
  for (const step of wf.steps) {
    const node = document.createElement("div");
    node.className = `wf-node ${step.status}`;
    const idx = document.createElement("b");
    idx.textContent = String(step.index + 1).padStart(2, "0");
    const label = document.createElement("span");
    label.textContent = ATOM_LABEL[step.atom] || step.atom;
    node.append(idx, label);
    lane.append(node);
  }
}

function renderReact(entries) {
  const lane = $("react-lane");
  lane.replaceChildren();
  for (const entry of entries || []) {
    const item = document.createElement("div");
    item.className = "react-item";
    const tag = document.createElement("span");
    tag.className = "react-tag";
    tag.textContent = `第 ${entry.attempt} 次 · ${entry.decision}`;
    const text = document.createElement("p");
    text.textContent = `${entry.thought} ${entry.detail}`;
    item.append(tag, text);
    lane.append(item);
  }
}
```

- [ ] **Step 4: 在 render() 中接线并删除旧 data-stage 逻辑**

删除 render() 中 `const phaseIndex ...` 到对应 `forEach(data-stage)` 整块；替换为：

```javascript
  renderWorkflow(state.workflow);
  renderReact(state.react);
```

- [ ] **Step 5: style.css 追加样式**

```css
.workflow-lane{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0;min-height:20px;color:#9aa193;font-size:11px}
.wf-node{display:flex;align-items:center;gap:7px;border:1px solid #e5e8e0;background:#f8f9f6;border-radius:6px;padding:8px 10px;color:#92998c}
.wf-node b{font:10px ui-monospace,monospace;color:#b1b8a9}
.wf-node.active{border-color:#4e8568;background:#edf4eb;color:var(--green)}
.wf-node.done{background:#eaf1e5;border-color:#e0e9d8;color:var(--green)}
.wf-node.done b{color:#729366}
.wf-node.failed{background:#fbf1e6;border-color:#e8cdb4;color:#ae7450}
.react-lane{display:flex;flex-direction:column;gap:8px;margin:6px 0 0}
.react-item{border-left:2px solid #8ca382;background:#f4f7f0;border-radius:0 6px 6px 0;padding:8px 10px}
.react-tag{font:9px ui-monospace,monospace;color:#56714d}
.react-item p{margin:5px 0 0;font-size:11px;line-height:1.6;color:#677062}
```

- [ ] **Step 6: 手工验证（服务运行时）**

Run: `pytest tests/test_web.py -v`，再在浏览器 http://127.0.0.1:8765 发送「把零件 A 放进盒子」，确认泳道节点依次点亮。

- [ ] **Step 7: 提交**

```bash
git add src/pickparts_agent/static/
git commit -m "feat: render dynamic workflow and react swimlanes"
```

### Task 17: 退役 cloud.py 旧 Agent/工具并移除日志插桩

**Files:**
- Modify: `src/pickparts_agent/cloud.py`
- Modify: `tests/test_cloud.py`

**Interfaces:**
- 保留：`Endpoint`（含 client/qwen_config）、`CloudPerception`、`transcribe`、`ConfigurationError/CloudError/PerceptionError`、`Detection`。
- 删除：`PickAndPlaceTool`、`_BoundedAssistant`、`_explicit_target`、`CloudAgent`。

- [ ] **Step 1: 删除 locate 中的临时日志插桩**

在 `CloudPerception.locate` 中，删除 `_raw = response.choices[0].message.content` 之后紧接的整段
`try: import tempfile, os; with open(... "vlm_raw.log") ... except Exception: pass`，
保留 `data = json.loads(_raw)`。

- [ ] **Step 2: 删除四个旧类**

删除 `PickAndPlaceTool`、`_BoundedAssistant`、`_explicit_target`、`CloudAgent` 四个类/函数定义。

- [ ] **Step 3: 清理不再使用的导入**

删除顶部 `import copy`、`import re`、`from qwen_agent.agents import Assistant`、
`from qwen_agent.llm.schema import Message`、`from qwen_agent.tools import BaseTool`；
将 `from .perception import Frame, backproject, parse_target` 改为
`from .perception import Frame, backproject`。
保留 `Endpoint.qwen_config`（其 qwen 导入位于方法体内，不影响）。

- [ ] **Step 4: 更新 test_cloud.py 删除依赖旧类的用例**

删除辅助函数 `qwen_remote`、`tool_delta`、`make_agent`，以及测试函数：
`test_tool_budget_blocks_repetition_and_resets_per_turn`、
`test_tool_rejects_unapproved_target_or_coordinates`、
`test_tool_catches_action_failure_and_consumes_budget`、
`test_tool_does_not_coerce_invalid_action_results_to_success`、
`test_qwen_history_keeps_only_final_sequence_and_reports_action_truth`、
`test_qwen_repeated_tool_requests_never_repeat_action`、
`test_ambiguous_or_negated_commands_do_not_call_cloud_or_action`、
`test_cloud_without_tool_cannot_claim_success`、
`test_success_followup_is_not_duplicated`、
`test_multiple_calls_in_one_qwen_response_execute_only_once`、
`test_explicit_commands_authorize_only_the_named_part`。
保留所有 CloudPerception / Endpoint / transcribe / CLI / save_frame 用例。

- [ ] **Step 5: 运行确认通过**

Run: `pytest tests/test_cloud.py -v`
Expected: PASS（剩余用例全绿）

- [ ] **Step 6: 提交**

```bash
git add src/pickparts_agent/cloud.py tests/test_cloud.py
git commit -m "refactor: retire hardcoded agent and vlm log instrumentation"
```

---

### Task 18: app.py CLI 改用 Orchestrator

**Files:**
- Modify: `src/pickparts_agent/app.py`

**Interfaces:**
- Consumes: `build_orchestrator(sim,locate,llm_endpoint)`。

- [ ] **Step 1: 修正云端段导入**

把 main 中云端分支的
`from .cloud import CloudAgent, CloudPerception, Endpoint, transcribe`
改为
`from .cloud import CloudPerception, Endpoint, transcribe`。

- [ ] **Step 2: 替换云端 Agent 构造与交互循环**

删除 sim 创建之后的
`controller = PickPlace(sim, perception, output=args.output)`、
`agent = CloudAgent(controller.run, llm.qwen_config())` 及其下旧的 Qwen 循环，
替换为：

```python
            from .orchestrator import build_orchestrator

            orchestrator = build_orchestrator(sim, perception.locate, llm)
            print("Cloud flexible Agent mode. 用自然语言描述需求；quit 退出。")
            turn = 0
            while True:
                if command is None:
                    try:
                        text = input("> ")
                    except EOFError:
                        return 0
                    if text.strip().lower() in ("quit", "exit"):
                        return 0
                else:
                    text = command
                result = orchestrator.turn(text)
                print(result["message"])
                save_frame(sim.observe(), args.output / f"turn-{turn:03d}")
                turn += 1
                if command is not None:
                    return 0 if result["success"] else 1
```

`--demo` 分支（PickPlace + ColorPerception）与 `--smoke` 保持不变。

- [ ] **Step 3: 验证 CLI 帮助不引入重依赖且可导入**

Run: `python -m pickparts_agent.app --help`
Expected: 正常退出，列出全部参数。

- [ ] **Step 4: 提交**

```bash
git add src/pickparts_agent/app.py
git commit -m "refactor: drive CLI via flexible orchestrator"
```

---

### Task 19: motion.py 范围说明（保留为固定基线）

> **用户裁决（2026-09-20）：** 选择「保留为固定基线」，**不执行** Spec 11.2 的收口（Step 2 不做）。
> 这是相对 Spec 11.2 的有意偏差：以最小改动避免重写 artifact/recovery 契约与两个基线测试的回归。
> 执行方式：当前会话内逐任务执行（executing-plans）。

**Files:**
- 无代码改动（仅范围裁决；如需收口见下）。

- [ ] **Step 1: 确认保留**

`PickPlace`（motion.py）保留为 `--demo` 的**确定性固定基线**（文档已标注 NOT cloud Agent validation）；
柔性 Agent（orchestrator/executor）不再引用它。这样避免重写其 artifact/recovery 契约导致 test_pick_place、
test_motion_recovery 大量回归，符合最小改动。

- [ ] **Step 2（可选，需用户明确同意）: 收口为单一逻辑源**

若坚持 spec 第 11.2 节收口：将 `PickPlace.run` 重写为构造固定计划并交给 `ReActExecutor`
（注入确定性 Reactor：decide 恒返回带 find_object 刷新的 retry），并同步重写 test_pick_place、
test_motion_recovery 的断言。此项风险与工作量较大，默认**不执行**。

- [ ] **Step 3: 确认无新增引用**

Run: `pytest tests/test_pick_place.py tests/test_motion_recovery.py -v`
Expected: PASS（保持现状）

---

## 阶段五：文档与端到端验证

### Task 20: 更新技术文档并完成验收取证

**Files:**
- Modify: `docs/技术文档.md`（4.3 Agent 编排、第 5 节系统架构相关段落）

- [ ] **Step 1: 更新文档描述新链路**

将技术文档中描述「CloudAgent + 单体 pick_and_place + 固定五段」的段落，更新为：
WorldState 黑板、9 个语义原子、Planner 可行性拆解、ReActExecutor（retry/replace/replan/ask/abort）、
Orchestrator 六步与主动追问；保留非特权观测与预算（4/25/8）说明。

- [ ] **Step 2: 全量测试**

Run: `pytest -q`
Expected: 全部 PASS，无回归。

- [ ] **Step 3: 连续多任务 + 卡点恢复取证**

启动 web（`./web.sh` 或 `pickparts-web`），依次：
1. 发送「把零件 A 放进盒子」→ 成功且主动追问；
2. 不重置，发送「把零件 B 放进盒子」→ 成功，工作流正确反映 A 已在盒内；
3. （在另一次重置后）制造一次定位抖动（重置后立即发送，必要时重复）观察 ReAct 泳道出现
   thought→decision→observation 并在预算内恢复。
保存页面截图到 `docs/evidence/`（如 flex-ab.png、flex-react.png）。

- [ ] **Step 4: 提交**

```bash
git add docs/技术文档.md docs/evidence/
git commit -m "docs: document flexible atom react agent and capture evidence"
```
