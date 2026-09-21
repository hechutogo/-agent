# TiPToP-Mac 对照实现设计

> 日期：2026-09-21
> 目的：在同一仓库内新增独立包 `src/tiptop_mac/`，按 TiPToP 的模块化
> “一次性感知 → 目标谓词 → 全局 TAMP 规划 → 开环执行”架构实现一个可在
> Mac/CPU 运行的对照系统，与现有 ReAct Agent 在同场景、同任务、同种子下比较。
> 现有 Agent 不做任何替换或行为修改。

## 1. 参考依据

- 官方仓库：<https://github.com/tiptop-robot/tiptop>
- 对照版本：`d8f5afdaa94a7432220c3042f9f80be5ab45aae8`（2026-09-21 浅克隆只读参考）
- 官方流水线：

```text
RGB(+深度)
  -> Gemini：开放词汇检测 bbox + 目标谓词(on/holding/near)
  -> SAM2：像素分割
  -> M2T2：点云抓取候选
  -> RANSAC 桌面 + 每对象凸包/点云
  -> cuTAMP：GPU 并行任务与运动搜索，产出完整关节轨迹
  -> 开环执行（阻抗控制，无执行中视觉复核/重规划）
```

## 2. Mac/CPU 适配映射

| TiPToP 组件 | Mac 适配 | 说明 |
|---|---|---|
| 相机驱动 | 复用 `scene.simulation.Simulation.observe()` | 非特权 RGB-D `Frame` |
| FoundationStereo | 直接使用 ManiSkill 米制深度 | 单目固定机位 |
| Gemini 检测+谓词 | LLM 仅做指代消解与谓词接地 | 对象检测由目录颜色感知完成 |
| SAM2 分割 | 目录 HSV 颜色分割（同 `ScenePerception` 色相逻辑） | 确定性、CPU |
| M2T2 抓取 | 点云顶部抓取启发式（4-DoF） | 对应 TiPToP 自身的 4-DoF fallback |
| RANSAC 桌面 | CPU RANSAC 平面拟合 | 以对象底部高度选桌面而非地面 |
| cuTAMP | CPU TAMP-lite：符号 BFS + IK/几何可行性 | 完整轨迹在运动前确定 |
| cuRobo 运动 | 复用 SciPy IK + 笛卡尔插值运动后端 | 与现 Agent 同一执行后端 |
| 开环执行 | 顺序执行完整轨迹，失败即中止 | 无 verify/retry/replan |

为保证公平，**低层执行后端直接复用**
`pickparts_agent.agent.atoms.manipulation._cartesian`（同样的 IK、插值、跟踪
误差阈值）；两系统的差异只体现在感知组织、规划与失败恢复架构上。

## 3. 与现有架构的受控差异

| 维度 | 现有 Agent | tiptop_mac |
|---|---|---|
| 感知时机 | 原子按需 `find_object` | 规划前一次性构建全场景 |
| 场景表示 | `WorldState` 增量黑板 | 对象中心 `SceneGraph`：桌面立方体 + 对象节点 + 抓取集 |
| LLM 输出 | 扁平原子计划（含 verify 步骤） | 目标谓词 `on/holding` |
| 规划 | LLM 序列 + 语义校验 | TAMP-lite 符号 BFS + IK/支撑/碰撞可行性 |
| 执行 | 每原子 verify + ReAct（retry/replace/replan/ask_user/abort） | 开环执行；跟踪失败中止 |
| 最终判定 | 控制回路内的视觉门 | 仅供评测的外部裁判，不反馈控制 |

谓词保持 TiPToP 语义：入盒与叠放同为 `on(x, y)`；放置模式由支撑体几何类型
推导（box → 释放入盒；block → 表面叠放；table → 桌面）。TiPToP 原版受凸包
限制不支持叠放与 inside/on 区分；本适配保留其谓词集合，几何层允许两种放置，
并在对比文档中明确这一点。

## 4. 模块结构

```text
src/tiptop_mac/
  __init__.py
  types.py       # Grasp / ObjectNode / TableSurface / SceneGraph / Predicate / MotionStep / TAMPPlan
  perception.py  # segment_objects / fit_table / propose_grasps / build_scene_graph
  grounding.py   # Grounder：LLM -> 目标谓词（严格校验）
  tamp.py        # TAMPLite：初始关系推导、符号 BFS、几何可行性、轨迹生成
  executor.py    # OpenLoopExecutor：无 locate、无重规划，失败中止
  agent.py       # TiPToPAgent.run + build_tiptop_agent 组装
  evaluation.py  # judge_outcome 外部裁判
  benchmark.py   # 同场景双系统评测：JSON + Markdown 报告
  cli.py         # python -m tiptop_mac.cli {run,benchmark}
```

依赖约束：除 `executor.py` 复用低层运动函数与 `agent.py` 组装外，
`tiptop_mac` 不导入现有 Agent 的 planner/executor/reactor/orchestrator。

## 5. 关键数据结构

```python
@dataclass(frozen=True)
class Grasp:
    position: np.ndarray   # 抓取点 XYZ
    top_z: float
    width: float           # 物体 XY 最大尺寸（夹爪信息）
    confidence: float

@dataclass
class ObjectNode:
    id: str; kind: str; label: str; color: tuple[float, float, float]
    mask: np.ndarray       # bool HxW
    bbox: tuple[int, int, int, int]
    point: np.ndarray      # 中心 XYZ
    top_z: float; bottom_z: float
    extent: np.ndarray     # 观测 XYZ 尺寸
    cloud: np.ndarray      # Nx3
    grasps: list[Grasp]

@dataclass(frozen=True)
class TableSurface:
    normal: np.ndarray     # 单位法向量（朝上）
    top_z: float
    bounds: np.ndarray     # 2x3 观测区域 min/max

@dataclass(frozen=True)
class SceneGraph:
    table: TableSurface
    objects: dict[str, ObjectNode]
    qpos: np.ndarray

@dataclass(frozen=True)
class Predicate:
    name: str              # "on" | "holding"
    args: tuple[str, ...]

@dataclass(frozen=True)
class MotionStep:
    kind: str              # "move" | "gripper"
    position: np.ndarray | None
    jaw: float | None

@dataclass(frozen=True)
class TAMPPlan:
    goal: tuple[Predicate, ...]
    operators: tuple[tuple[str, tuple], ...]
    trajectory: tuple[MotionStep, ...]
    planning_time: float
    rationale: str
```

## 6. 感知与几何约束

- 每对象：HSV 色相差 < 7、饱和度 > 130、亮度 > 55；最大连通域；有效深度
  覆盖 ≥ 80%；拒绝贴边与多连通域（参数与 `ScenePerception` 一致）。
- 点云由 `backproject` 计算；中心 XY 取 1/99 分位中点，Z 取中位；
  `top_z` 取 95 分位，`bottom_z` 取最低分位。
- 桌面 RANSAC：非对象像素点云中随机三点拟面，内阈 4 mm，要求法向量与 Z
  夹角 < 18°，且桌面高度与对象底部中位差 ≤ 30 mm（避免选到地面）；
  固定随机种子保证可复现。
- 抓取：仅 block 生成顶部抓取；box 不生成抓取，只作为支撑面。

## 7. TAMP-lite 规划

- 初始关系：block 底部与桌面高差 ≤ 12 mm 且 XY 在桌面范围内 → on(table)；
  若落在另一对象顶部区域 → on(对象)。
- 算子：`pick(block)`（手空、存在支撑、有抓取、hover/grasp IK 可行）；
  `place(block, support)`（support ∈ table/box/block，排除自身与其承载的对象）。
- 搜索：符号 BFS（支撑映射 + 持物状态），最多 500 个展开状态；目标状态
  匹配全部谓词（holding 目标时不要求手空）。
- 几何可行性：放置高度按支撑类型计算（block：support.top_z + 持物半高
  + 4 mm；box：box 点 z + 55 mm；table：保持原 XY）；XY 落点 footprint
  不越界、不与同层对象重叠；所有运动点 IK 预检。
- 轨迹（完整、运动前确定）：

```text
pick:  hover(jaw=.8) -> grasp(jaw=.8) -> close(0) -> lift hover
carry: above destination (max(.835, top+.075))
place(block): release -> open(.8) -> retreat +.07
place(box):   release(top+.055) -> open(.8) -> retreat
finish: REST_Q, jaw=.8
```

## 8. 执行器

- 输入 `TAMPPlan`，顺序执行；`move` 走复用的 `_cartesian`，`gripper` 走
  `sim.move_right(当前右臂角, jaw)`。
- 任一 move 跟踪失败：返回 `success=False, aborted=True`，立即停止；
  不调用 locate，不做 retry/replan。
- 执行器构造不接收 locate，以类型边界保证开环。

## 9. 评测口径

- 任务：`把 A 放进 box`、`把 A 叠到 B 上`、`把 B 叠到 A 上`。
- 每任务在相同种子集合上为两个系统各建全新 `Simulation`。
- 指标：外部裁判成功率、规划耗时、运动步数、ReAct 恢复次数（现 Agent）、
  是否中止（tiptop_mac）。
- 外部裁判 `judge_outcome`：执行后新 RGB-D 观测，复用 `ScenePerception`
  与关系几何阈值（同 `VerifyState`，但结果不回控）。
- 报告：`runs/benchmark/<时间戳>/{results.json,report.md}`。

## 10. 验收标准

1. 新模块全部具备先失败后通过的单元测试；
2. 真实仿真端到端：seeds (0, A→B on)、(3, B→A on)、(7, A→box in) 通过；
3. benchmark 可在两系统上运行（现系统需云端配置，缺失时显式跳过）；
4. 现有 356 项测试不回归，全量测试通过；
5. README 与技术文档补充对照包说明与对比结论。
