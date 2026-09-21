# 柔性原子 Agent 架构设计（Flexible Atom + ReAct）

- 日期：2026-09-20
- 状态：已评审通过，待编写实施计划
- 适用项目：`pickparts-xlerobot-agent`（ManiSkill 3 / SAPIEN 3.0 / XLeRobot v0.3）
- 作者：具身 Agent 架构设计

---

## 1. 背景与问题

当前系统能够完成「把零件 A/B 放进盒子」，但只能执行**预设的固定动作**：

1. 物理原子能力只有关节空间的 `Simulation.move_right()`、`hold()`、`observe()`。
2. `PickPlace.run()` 把「移动 → 抓取 → 抬起 → 放置 → 复位」硬编码为一个整体技能。
3. 暴露给 LLM 的只有单体工具 `pick_and_place(target)`。
4. `_explicit_target()` 在调用 LLM 之前，就用正则把指令限制为「A/B 放盒子」。

因此 LLM 没有可拼装的零件、没有前置条件判断、没有失败后重新选择动作的闭环。系统无法处理非预设指令、连续多任务或执行卡点后的自主恢复。

## 2. 目标与非目标

### 2.1 目标

对应用户期望的六步流程：

1. 用户提出需求，Agent 评估是否可完成（前置条件检查）。
2. 对可实现需求进行拆解，得到可拼装的扁平工作流。
3. 通过组装原子功能实现完整工作流（语义接地，不向 LLM 暴露坐标）。
4. 执行操作。
5. 执行中监控；顺利则返回结果；出现卡点则获取结果并反思（ReAct），重新选择动作，最终完成需求。
6. 需求完成后主动询问是否有新需求；有则继续，且复用已建立的世界状态。

### 2.2 非目标（本期不做）

- 不向 LLM 开放裸 3D 坐标参数（不做纯笛卡尔工具集）。
- 不新增第二个容器、堆叠等高复杂度场景；本期在现有单桌 + A/B + 绿盒场景验证非预设链路与连续多任务。
- 不在本地部署任何大模型；规划与反思全部走云端 API。
- 不改变非特权观测原则：不读取 `actor.pose` 等仿真真值。
- 不使用循环容器嵌套；计划始终是扁平有序列表。

### 2.3 已确认的关键决策

| 决策点 | 选择 |
|---|---|
| 原子粒度 | 对象级语义原子（LLM 只传语义引用，坐标运行时接地） |
| 编排框架 | 原生 OpenAI 兼容 chat 接口 + 轻量自研结构化编排器 |
| 控制范式 | 规划 + ReAct 混合（先出计划，执行中局部反思/重规划） |
| 验证范围 | 现有场景 + 连续多任务 + 卡点恢复 |

---

## 3. 总体架构

```text
用户目标
  │
  ▼
┌───────────────────────────────────────────────────────┐
│ Orchestrator（进程内常驻，跨多轮保留状态）                │
│                                                         │
│  ① Planner.analyze   读 WorldState 快照 + Atom 目录       │
│       feasible? blockers? → 扁平 AtomCall 列表            │
│                                                         │
│  ② ReActExecutor    逐原子执行                           │
│       前置检查 → 执行(感知+深度反投影+IK 接地) → 后置视觉校验 │
│            │ 失败/卡点                                   │
│            ▼                                            │
│  ③ Reactor.decide  retry / replace / replan / ask / abort│
│       （有界次数 + 原子/LLM 预算）                        │
│                                                         │
│  ④ 成功 → 主动追问；WorldState/对话保留，迎接下一目标       │
└───────────────────────────────────────────────────────┘
  │ 仅通过窄边界
  ▼
Simulation.observe / move_right（不改） ＋ Perception.locate ＋ Kinematics
```

设计原则：

- **计划是扁平有序列表**。ReAct 只在该列表上做「重试 / 替换当前 / 重排尾部」，不引入嵌套循环容器。
- **LLM 只接触语义引用**（A/B/box、table/box 等），坐标在原子内部由「感知 + 深度反投影 + IK」运行时接地。
- **非特权观测不变**：世界事实只来自 `observe()` 的 RGB-D 与本体状态。
- **单一逻辑源**：固定基线与云端 Agent 最终共用同一套原子与执行器，避免两套动作逻辑漂移。

---

## 4. 文件改动总览

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/pickparts_agent/state.py` | 新增 | `WorldState` 世界状态黑板 |
| `src/pickparts_agent/llm.py` | 新增 | chat → 严格 JSON 的统一调用 |
| `src/pickparts_agent/atoms/__init__.py` | 新增 | 包导出 |
| `src/pickparts_agent/atoms/base.py` | 新增 | `Atom` / `AtomContext` / `AtomResult` / `Check` |
| `src/pickparts_agent/atoms/library.py` | 新增 | 9 个对象级语义原子 |
| `src/pickparts_agent/atoms/registry.py` | 新增 | `AtomRegistry` 注册 / 校验 / 目录文本 |
| `src/pickparts_agent/planner.py` | 新增 | `Planner.analyze` 可行性与拆解 |
| `src/pickparts_agent/reactor.py` | 新增 | `Reactor.decide` 卡点反思 |
| `src/pickparts_agent/executor.py` | 新增 | `ReActExecutor` 执行 + 监控 + 预算 |
| `src/pickparts_agent/orchestrator.py` | 新增 | `Orchestrator` 六步总装 |
| `src/pickparts_agent/simulation.py` | 不改 | 保持窄边界 |
| `src/pickparts_agent/kinematics.py` | 不改 | 原子复用 `solve/forward` |
| `src/pickparts_agent/perception.py` | 不改 | `ColorPerception` 基线保留 |
| `src/pickparts_agent/cloud.py` | 修改 | 保留 `Endpoint/transcribe`；退役旧 Agent/工具/正则类 |
| `src/pickparts_agent/motion.py` | 修改（收口） | 固定技能改为走同一执行器 |
| `src/pickparts_agent/web.py` | 修改 | 接 Orchestrator；state 增 `workflow`/`react` |
| `src/pickparts_agent/app.py` | 修改 | CLI 交互循环改用 Orchestrator |
| `src/pickparts_agent/static/index.html` | 修改 | 两条泳道容器 |
| `src/pickparts_agent/static/app.js` | 修改 | 动态工作流 + ReAct 泳道渲染 |
| `src/pickparts_agent/static/style.css` | 修改 | 泳道样式 |
| `tests/` | 新增 + 修改 | 见第 12 节 |

---

## 5. 核心数据结构

### 5.1 `state.py` — 世界状态黑板

仅承载来自非特权观测的事实，并提供线程安全快照。

```python
from dataclasses import dataclass, field
import threading
import numpy as np


@dataclass
class ObjectRecord:
    id: str                         # "A" | "B" | "box"
    visible: bool
    bbox: list[int]                 # 最近一次像素框 [x1,y1,x2,y2]
    point: list[float]              # 基座系 xyz（JSON 安全的 list）
    confidence: float
    frame_id: int
    placement: str | None = None    # "table" | "box" | "gripper"（观测推断）


class WorldState:
    def __init__(self):
        self.lock = threading.RLock()
        self.objects: dict[str, ObjectRecord] = {}
        self.gripper_open = True
        self.held_object: str | None = None
        self.qpos: list[float] | None = None
        self.frame_id = 0

    def update_object(self, record: ObjectRecord) -> None: ...
    def set_gripper(self, open_: bool) -> None: ...
    def set_held(self, object_id: str | None) -> None: ...

    def apply_observation(self, target: str, locate_result: dict,
                          frame_id: int) -> ObjectRecord:
        """用 locate 返回的 bbox/point 更新对象，并推断 placement。"""

    def snapshot(self) -> dict:
        """深拷贝、JSON 安全；结构固定，供 Planner/前端使用。"""
```

`placement` 推断只用观测几何：

- 点的 XY 是否落在绿盒 bbox 内、Z 是否在盒口高度带内 → `box`。
- 对象是否随夹爪同步运动、且夹爪闭合 → `gripper`。
- 其余落在桌面高度带 → `table`。

快照结构：

```json
{
  "frame_id": 12,
  "gripper_open": false,
  "held_object": "A",
  "objects": {
    "A": {"visible": true, "bbox": [247,252,272,284],
          "point": [-0.084,-0.338,0.732], "confidence": 0.9,
          "frame_id": 12, "placement": "gripper"}
  }
}
```

### 5.2 `llm.py` — 统一 chat → JSON

```python
import json
from openai import OpenAI


class LLMError(RuntimeError):
    pass


class JSONChat:
    def __init__(self, client: OpenAI, model: str, *, max_repair=1):
        self.client = client
        self.model = model
        self.max_repair = max_repair

    def json(self, system: str, user_payload: str) -> dict:
        """发起一次结构化调用；坏 JSON 时追加一次「只输出 JSON」修复。

        - 成功：返回解析后的 dict。
        - 仍失败：抛 LLMError，由上层 fail-closed。
        - 网络/超时：透传 openai 异常类型，便于执行器区分。
        """
```

要点：`temperature=0`；不使用 function-calling（兼容性最稳）；通过提示强制纯 JSON；修复重试只做一次，避免无限调用。

### 5.3 `atoms/base.py` — 原子契约

```python
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Check:
    ok: bool
    reason: str = ""
    facts: dict = field(default_factory=dict)


@dataclass
class AtomContext:
    sim: object                                   # Simulation
    locate: Callable[[object, str], dict]         # CloudPerception/ColorPerception.locate
    kin: object                                   # ArmKinematics
    state: object                                 # WorldState
    on_stage: Callable[[str], None]
    on_trace: Callable[[dict], None]


@dataclass
class AtomResult:
    success: bool
    error_kind: str | None   # transient | precondition | unreachable
                             # lost_object | grip_failed | verify | fatal
    message: str
    observed: dict = field(default_factory=dict)


class Atom:
    name: str
    description: str
    parameters: dict         # JSON Schema（仅语义参数）

    def check_pre(self, ctx: AtomContext, args: dict) -> Check: ...
    def run(self, ctx: AtomContext, args: dict) -> AtomResult: ...
    def verify(self, ctx: AtomContext, args: dict,
               result: AtomResult) -> Check: ...

    def call(self, ctx: AtomContext, args: dict) -> AtomResult:
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

错误分类语义：

- `transient`：定位偶发抖动、IK 近失，重试通常可恢复。
- `precondition`：前置不满足（如未定位就抓取）。
- `unreachable`：IK 多起点均失败。
- `lost_object`：执行后视野内找不到目标。
- `grip_failed`：闭合后无夹持阻力 / 抬起高度不足。
- `verify`：后置视觉校验不通过。
- `fatal`：关节跟踪误差超阈值等安全问题，立即停臂，不盲目重试。

### 5.4 `atoms/registry.py`

```python
class AtomRegistry:
    def __init__(self, atoms: list[Atom]): ...
    def get(self, name: str) -> Atom: ...
    def validate_call(self, call: dict) -> None:
        """校验：是 dict 且仅含 atom/args；atom 存在；args 满足 schema。
        手写必需字段、类型、枚举检查，不引入新依赖。非法即 ValueError。"""
    def catalog_for_prompt(self) -> str:
        """逐原子输出 name / description / 参数 schema，供 Planner/Reactor。"""
```

`AtomCall` 标准形态：

```json
{"atom": "grasp", "args": {"target": "A"}}
```

---

## 6. 原子动作库（9 个语义原子）

| 原子名 | 参数 | 前置条件 | 后置校验 | 复用现有代码 |
|---|---|---|---|---|
| `find_object` | target ∈ A/B/box | 相机可用 | visible 且置信度 ≥ 阈值 | `CloudPerception.locate` |
| `reach_above` | target, clearance=0.10 | 对象已定位；空手或持有该物 | TCP 到位误差 ≤ 8 mm | `kin.solve` + `Simulation.move_right` |
| `grasp` | target | 位于上方；夹爪张开 | 夹爪闭合且有夹持阻力 | `move_right(..., jaw=0)` |
| `lift` | clearance=0.10 | 已夹持 | 目标随夹爪 Δz ≥ 0.035 m | `move_right` + 视觉比对 |
| `carry_to` | container=box | 正持有物体 | 到达容器上方 | `kin.solve` + `move_right` |
| `release_into` | container=box | 位于容器上方 | 夹爪张开；物体已脱离 | `move_right(..., jaw=0.8)` |
| `verify_state` | target, at ∈ table/box | — | placement 与期望一致 | 视觉定位 + placement 推断 |
| `reset_arm` | — | — | 右臂回到 rest 姿态 | `move_right(rest)` |
| `set_gripper` | open: bool | — | jaw 角度到位 | `move_right(..., jaw=)` |

每个原子的 `run` 内部完成接地闭环，例如 `reach_above`：

1. 从 `WorldState` 读取目标最近 `point`；若无则要求先 `find_object`（前置失败）。
2. 计算 hover 点：`hover = point.copy(); hover[2] += clearance`。
3. `kin.solve(hover, seed=当前臂角)` 得到关节解；不可达 → `unreachable`。
4. `sim.move_right(q, jaw=当前 jaw)`；随后比对 `kin.forward(新 qpos)` 与 hover：误差 > 8 mm 归为 `verify`（末端未到位）；若 `move_right` 报告关节跟踪误差 > 0.12 rad，则归为 `fatal`。
5. 成功返回 `AtomResult(True, observed={"tcp":[...], "frame_id":...})`。

### 6.1 单任务示例计划

「把零件 A 放进盒子」：

```json
[
  {"atom": "find_object",  "args": {"target": "A"}},
  {"atom": "find_object",  "args": {"target": "box"}},
  {"atom": "reach_above",  "args": {"target": "A"}},
  {"atom": "grasp",        "args": {"target": "A"}},
  {"atom": "lift",         "args": {"clearance": 0.10}},
  {"atom": "carry_to",     "args": {"container": "box"}},
  {"atom": "release_into", "args": {"container": "box"}},
  {"atom": "verify_state", "args": {"target": "A", "at": "box"}},
  {"atom": "reset_arm",    "args": {}}
]
```

### 6.2 连续多任务示例

「把 A 和 B 都放进盒子」生成 A 全段（结尾 `reset_arm`）+ B 全段。第二段规划时 `WorldState` 已记录 A 在盒内，无需全量重置。

---

## 7. Planner（流程 ①②③）

`planner.py`：

```python
class Plan:
    def __init__(self, feasible, blockers, steps, rationale): ...


class Planner:
    def __init__(self, chat: JSONChat, registry: AtomRegistry): ...

    def analyze(self, goal: str, state: WorldState) -> Plan:
        user_payload = self._render(goal, state.snapshot(),
                                    self.registry.catalog_for_prompt())
        data = self.chat.json(SYSTEM_PROMPT, user_payload)
        steps = data.get("steps", [])
        for call in steps:
            self.registry.validate_call(call)   # fail-closed
        return Plan(data["feasible"], data.get("blockers", []),
                    steps if data["feasible"] else [],
                    data.get("rationale", ""))
```

系统提示要点（SYSTEM_PROMPT）：

- 你是机器人任务规划者；先判断可行性与前置条件，再拆解。
- 只能使用「原子目录」中给出的原子和当前世界状态中存在的对象。
- 严禁输出或推断坐标；目标通过语义 id 引用。
- 每个改变世界状态的动作后必须有对应校验（如放置后 `verify_state`）。
- 计划必须是扁平有序步骤，不得包含循环或嵌套结构。
- 只输出规定 JSON，不要解释性散文。

输出契约：

```json
{
  "feasible": true,
  "blockers": [
    {"need": "目标可见", "reason": "当前视野内没有零件 C"}
  ],
  "steps": [],
  "rationale": "A、box 均可见且机械臂空闲，可执行抓放。"
}
```

行为约束：

- `feasible=false` → Orchestrator 不执行任何动作，将 blockers 转成用户可理解的提问或澄清选项。
- JSON 解析失败 → `JSONChat` 修复一次；仍失败 → 返回「规划失败，请换一种说法」。
- 出现未知原子/未知对象/非法枚举 → `validate_call` 抛错，按规划失败处理。

需识别的前置条件：对象是否存在、是否可见、机械臂是否已持有他物、目标容器是否存在、场景是否处于恢复锁。

---

## 8. ReAct Executor 与 Reactor（流程 ④⑤）

### 8.1 执行算法

`executor.py` 伪代码：

```text
pointer = 0
atom_calls = 0
react_iters = 0
while pointer < len(steps):
    call = steps[pointer]
    atom = registry.get(call["atom"])
    atom_calls += 1
    if atom_calls > MAX_ATOM_CALLS: return failed("动作预算耗尽")
    result = atom.call(ctx, call["args"])
    trace(action=call, observation=result)
    update_state_from(result)
    if result.success:
        mark(pointer, "done"); pointer += 1; continue

    # ── 卡点：进入有界 ReAct ──
    react_iters += 1
    if react_iters > MAX_REACT_ITERS: return failed("反思次数耗尽")
    decision = reactor.decide(goal, steps, pointer, state, result, react_iters)
    switch decision.kind:
        retry:
            steps = insert_refresh_if_needed(steps, pointer, decision)
            mark(pointer, "pending")          # pointer 不前进，重跑当前
        replace:
            steps = steps[:pointer] + decision.steps + steps[pointer+1:]
            mark(pointer, "pending")
        replan:
            tail = planner.analyze(remaining_goal, state)
            steps = steps[:pointer] + tail.steps
        ask_user:
            answer = await_user(decision.question)  # web 卡片 / CLI 提问
            state.record_user_answer(answer)
        abort:
            stop_arm(); return failed(decision.reason)
return success()
```

### 8.2 Reactor 决策契约

`reactor.py`：

```python
class Decision:
    def __init__(self, kind, thought, reason, steps=None, question=None): ...


class Reactor:
    def __init__(self, chat: JSONChat, registry: AtomRegistry): ...
    def decide(self, goal, steps, pointer, state, failed_result,
               attempt) -> Decision: ...
```

输入：失败原子、`error_kind`、观测事实、当前尝试次数、剩余计划、世界快照。
输出严格 JSON：

```json
{
  "thought": "抓取时落点可能偏了，先重新定位再下降。",
  "decision": "retry",
  "steps": [
    {"atom": "find_object", "args": {"target": "A"}}
  ],
  "reason": "瞬时定位偏差，刷新后重试成本最低。"
}
```

### 8.3 错误类型到默认策略

| error_kind | 默认策略 |
|---|---|
| transient | 先插入 `find_object` 刷新，再 retry |
| lost_object / grip_failed | retry 一次；仍失败则 replan |
| unreachable | replan（更换路径/顺序） |
| precondition | replan 或 ask_user |
| verify | retry 一次；仍失败则 replan |
| fatal | 立即停臂，ask_user 或 abort，不盲目重试 |

该映射同时写入 Reactor 提示，并在执行器内兜底：即使模型给出越界决策（如对 fatal 选择重试），执行器也强制安全策略。

### 8.4 预算与安全

| 预算 | 默认值 | 触发后行为 |
|---|---|---|
| MAX_REACT_ITERS | 4 | 明确失败，建议重置或人工介入 |
| MAX_ATOM_CALLS | 25 | 明确失败，防止动作空转 |
| MAX_LLM_CALLS | 8（Planner + Reactor 合计） | 明确失败，避免无限云调用 |

安全：

- `fatal` 立即停止下发动作，保留恢复锁。
- ReAct 可自主恢复的 `transient` 不再要求重启程序。
- 所有轨迹落盘：`runs/web/<ts>/plan.json`、`react.jsonl`、`result.json` 与各阶段图像。

---

## 9. Orchestrator（六步总装）

`orchestrator.py`：

```python
class Orchestrator:
    def __init__(self, sim, locate, llm: Endpoint, vlm_client):
        self.state = WorldState()          # 跨多轮常驻
        self.history = []
        self.registry = build_default_registry()
        chat = JSONChat(llm.client(), llm.model)
        self.planner = Planner(chat, self.registry)
        self.reactor = Reactor(chat, self.registry)
        self.executor = ReActExecutor(
            sim, locate, self.registry, self.state,
            self.planner, self.reactor,
            on_trace=self._trace)

    def turn(self, text: str) -> dict:
        plan = self.planner.analyze(text, self.state)       # ①②
        if not plan.feasible:                               # 不可行：零动作
            return {"success": False,
                    "message": render_blockers(plan.blockers)}
        result = self.executor.execute(plan, goal=text)     # ③④⑤
        if result.success:
            result["message"] = f"{result['message']} 还需要我做什么？"  # ⑥
        self.history.append({"goal": text, **result})
        return result
```

多轮语义：

- `WorldState` 与 `history` 在进程内常驻，仅 `/api/reset` 清空。
- 处理完 A 后，B 的规划可见「A 已在盒内」，避免重复处理。
- 成功即主动追问并保持 `ready`；用户继续表述即进入下一轮。

---

## 10. Web 与前端改造

### 10.1 `web.py`

- `RobotBackend.__init__` 构造 `Orchestrator`（`Simulation`、`locate`、LLM 端点保持不变）。
- `state` 在现有字段上新增：

```json
{
  "workflow": {
    "goal": "把零件 A 放进盒子",
    "current": 3,
    "status": "running",
    "steps": [
      {"index": 0, "atom": "find_object", "args": {"target": "A"}, "status": "done"},
      {"index": 1, "atom": "find_object", "args": {"target": "box"}, "status": "done"},
      {"index": 2, "atom": "reach_above", "args": {"target": "A"}, "status": "active"}
    ]
  },
  "react": [
    {"attempt": 1, "thought": "落点可能偏了", "decision": "retry",
     "detail": "先重新定位 A"}
  ]
}
```

`workflow.status` 取值：`planning | running | blocked | done | failed`。

### 10.2 前端两条泳道

将 `app.js` 顶部的固定 `phases` 与对应五段指示灯逻辑替换：

1. **工作流泳道**：按 `workflow.steps` 渲染 `pending/active/done/failed`；多目标时按 A 段、B 段物理分区换行（沿用多行泳道偏好）。
2. **ReAct 泳道**：时间线展示 `thought → decision → observation` 与第几次尝试，仅在出现卡点时出现。

不变项：消息流渲染、首帧逻辑、`showDialog`、重置确认。
新增：`ask_user` 决策复用 `showDialog` 弹卡片，用户回答后写回，不中断会话。
`index.html` 增加两个泳道容器；`style.css` 增加步骤节点与 ReAct 时间线样式。

---

## 11. cloud.py 与 motion.py 收口

### 11.1 cloud.py

- 保留：`Endpoint`、`transcribe`、`CloudPerception`、`ConfigurationError/CloudError/PerceptionError`。
- 退役（新链路绿灯后删除，过渡期先不引用）：`PickAndPlaceTool`、`_BoundedAssistant`、`_explicit_target`、`CloudAgent`。

### 11.2 motion.py

将 `PickPlace.run` 收口为：构造与五段式等价的固定计划，交给同一 `ReActExecutor`（基线模式使用确定性反思策略，不调用 LLM）。使 `--demo` 基线与云端 Agent 共用同一套原子与校验，消除两套动作逻辑漂移。

降风险安排：阶段一到阶段三不改动 motion.py；阶段四接线绿灯后再收口。

---

## 12. 测试计划（TDD，使用假件）

测试假件：

- `FakeChat`（原 FakeLLM）：按脚本依次返回 Planner / Reactor 的 JSON，可在第 N 次返回坏 JSON。
- `FakeSim`：记录 move_right 调用，可让指定步骤返回跟踪误差。
- `FakeLocate`：可配置返回值或在第 N 次抛错。

测试文件：

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_state.py` | 快照 JSON 安全；placement 推断；并发更新 |
| `tests/test_atoms.py` | 前置/后置；接地调用顺序；各 error_kind 分类 |
| `tests/test_registry.py` | schema 校验；未知原子/参数/枚举拒绝 |
| `tests/test_planner.py` | 可行计划；不可行 blockers；坏 JSON 修复后 fail-closed；未知对象拒绝 |
| `tests/test_reactor.py` | 各错误类型的决策；越界决策被安全兜底 |
| `tests/test_executor.py` | 成功路径；retry/replace/replan 成功；预算耗尽即失败；fatal 后不再下发动作 |
| `tests/test_orchestrator.py` | 不可行零动作并提问；成功主动追问；第二轮复用持久状态 |
| `tests/test_web.py` | `/api/state` 含 workflow/react；origin/reset 旧用例保留 |
| `tests/test_cloud.py`（修改） | 移除 CloudAgent/_explicit_target 用例；保留 Endpoint/transcribe |

现有 simulation / kinematics / perception 测试保持不回归。

---

## 13. 分阶段落地

| 阶段 | 内容 | 可交付验证 |
|---|---|---|
| 一 | `state.py` + `atoms/*`（base/registry/library） | 原子与状态单测通过；旧链路不变 |
| 二 | `llm.py` + `planner.py` + `reactor.py` | FakeChat 下规划/反思测试通过 |
| 三 | `executor.py` + `orchestrator.py` | 失败注入、预算、多轮测试通过 |
| 四 | `web.py`/`app.py` 接线 + 前端两泳道；删除 cloud 旧类；收口 motion.py | 端到端可交互，无回归 |
| 五 | 更新技术文档 4.3/5 节；连续 A→B + 卡点恢复取证；全量 pytest | 验收证据齐全 |

每阶段独立可测、可回退；不在单阶段内混合无关改动。

---

## 14. 验收标准

1. 非预设指令：对可实现需求能生成扁平计划并执行；对不可实现需求返回 blockers 且**零物理动作**。
2. 语义接地：计划与反思中不出现坐标；坐标全部来自感知 + 深度反投影 + IK。
3. 卡点恢复：注入一次瞬时失败，Agent 能在预算内通过 retry/replace/replan 恢复并完成；fatal 时立即停臂。
4. 连续多任务：完成 A 后不重置即可处理 B；世界状态正确反映 A 已在盒内。
5. 主动追问：每次成功后输出「还需要我做什么？」并保持 ready。
6. 可观测：`workflow` 与 `react` 在前端实时呈现，证据落盘。
7. 质量：全量 pytest 通过；非特权观测约束保持；无新增本地模型。

---

## 15. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| VLM/relay 不稳定导致定位抖动 | 误判卡点 | `find_object` 刷新 + transient 重试；端点已切 DeepSeek |
| ReAct 不收敛 | 动作/Token 空转 | 三重预算 + 每阶段明确失败上报 |
| LLM 输出非法计划 | 执行越界 | registry 强校验 + fail-closed |
| 收口 motion 引入回归 | 基线受影响 | 放在阶段四绿灯后，且共用执行器有测试覆盖 |
| 子线程 Vulkan 崩溃 | 服务退出 | 沿用主线程渲染锚方案，Orchestrator 在 worker 中仅复用已初始化渲染 |

---

## 16. 后续演进（非本期）

- 双层原子：在语义层下补充笛卡尔内核原子，支持更细的接触 rich 操作。
- 更丰富场景：第二容器、堆叠、遮挡避让。
- 基于轨迹的策略学习：将成功/失败的原子轨迹沉淀为可复用技能或微调数据。
