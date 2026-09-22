# tiptop_mac 与官方 TiPToP 源码对照（溯源）

本文件用于核对 `src/tiptop_mac/` 确实基于官方 TiPToP 实现，而非另起炉灶。

- 官方仓库：<https://github.com/tiptop-robot/tiptop>
- 对照 commit：`d8f5afdaa94a7432220c3042f9f80be5ab45aae8`（2026-08-03，`Add changelog and document experimental features (#37)`）
- 论文：arXiv:2603.09971，*TiPToP: A Modular Open-Vocabulary Planning System for Robotic Manipulation*
- 本地参考副本（gitignore，不入库）：`.cache/tiptop-reference`，可用
  `git clone https://github.com/tiptop-robot/tiptop .cache/tiptop-reference` 重建并 `git checkout d8f5afd`。

## 1. 官方真实流水线（来自源码，非论文转述）

入口
[tiptop/tiptop_run.py::async_entrypoint](https://github.com/tiptop-robot/tiptop/blob/d8f5afdaa94a7432220c3042f9f80be5ab45aae8/tiptop/tiptop_run.py)：

```text
到拍摄位、张爪、读取指令
  → 采集一帧 RGB-D
  → 感知（一次性）
       gemini.detect_and_translate   : bboxes + predicates(name,args)
       sam2.sam2_segment_objects     : 每个对象的 mask
       foundation_stereo / 硬件深度  : 深度图
       depth_to_xyz + 点云
       m2t2.generate_grasps          : 6-DoF 抓取
       process_scene_geometry        : RANSAC 桌面、凸包/RecGen 形状、KDTree 抓取关联
       create_tamp_environment       : movables/surfaces + 目标状态(On/Holding/HandEmpty/Near)
  → 规划（全局、一次性）
       planning.run_planning = cutamp.run_cutamp（粒子优化 + 符号搜索）
       curobo IKSolver / MotionGen 给出关节轨迹
  → 执行（开环）
       execute_plan.execute_cutamp_plan：顺序执行 trajectory/gripper，
       任一步失败即抛 ExecutionFailure，不重试、不重规划
```

## 2. 逐文件 / 逐函数映射

| 阶段 | 官方文件 :: 函数 | tiptop_mac :: 函数 | 一致性 |
| --- | --- | --- | --- |
| 主编排 | `tiptop_run.py::async_entrypoint` | [agent.py::TiPToPAgent.run](../src/tiptop_mac/agent.py) | **一致**：感知→接地→TAMP→开环的顺序与“一次性”边界相同 |
| 语言接地/谓词 | `perception/gemini.py::detect_and_translate`（输出 `predicates[{name,args}]`） | [grounding.py::Grounder.ground](../src/tiptop_mac/grounding.py) | **语义一致**：同为 `on/holding` 谓词；模型 Gemini→OpenAI 兼容 LLM |
| 目标对象检测 | 同上（输出 `bboxes`） | [perception.py::segment_objects](../src/tiptop_mac/perception.py) | 职责一致，算法替换 |
| 分割 | `perception/sam2.py::sam2_segment_objects` | `perception.py::_hue_mask` | **替换**：SAM2→目录 HSV 颜色分割 |
| 深度 | `perception/foundation_stereo.py` | `Simulation.observe()` 的 ManiSkill metric depth | **替换**：双目立体→仿真器深度 |
| 反投影/点云 | `perception/utils.py::depth_to_xyz` | `pickparts_agent.scene.perception.backproject` | **替换**（等价几何） |
| 抓取提议 | `perception/m2t2.py::generate_grasps`（6-DoF，多候选） | [perception.py::propose_grasps](../src/tiptop_mac/perception.py) | **简化**：顶部单个 4-DoF；box 不生成抓取 |
| 抓取关联 | `tiptop_run.py::process_scene_geometry`（KDTree + 接触点近邻） | 按对象 id 直接挂载 | **简化** |
| 桌面拟合 | `process_scene_geometry::segment_table_with_ransac` | [perception.py::fit_table](../src/tiptop_mac/perception.py) | **一致**：RANSAC 水平面 + 内点阈值 |
| 物体形状 | 凸包 / `perception/shape_completion.py`(RecGen) | `types.ObjectNode.extent`（三维 bbox） | **简化**：凸包/补全→包围盒 |
| TAMP 环境/目标 | `tiptop_run.py::create_tamp_environment`（`On/Holding/HandEmpty/Near`，movables/surfaces） | [tamp.py](../src/tiptop_mac/tamp.py) `_satisfied/_successors` | **谓词语义一致**；`Near` 未实现；隐含 `HandEmpty` |
| 任务+运动规划 | `planning.py::run_planning`（`cutamp.run_cutamp`） | `tamp.py::TAMPLite.plan`（符号 BFS，500 状态上限） | **架构一致、算法替换**：GPU 粒子优化→CPU 符号 BFS |
| IK / 运动生成 | `curobo` `IKSolver` / `MotionGen` | `scene/kinematics.py::ArmKinematics`(SciPy) + `agent/atoms/manipulation._cartesian` | **替换**：GPU→CPU |
| 开环执行 | [execute_plan.py::execute_cutamp_plan](https://github.com/tiptop-robot/tiptop/blob/d8f5afdaa94a7432220c3042f9f80be5ab45aae8/tiptop/execute_plan.py) | [executor.py::OpenLoopExecutor.execute](../src/tiptop_mac/executor.py) | **高度一致**：trajectory/gripper 顺序执行，失败即 `ExecutionFailure`/abort，不 retry/replan |
| 机器人平台 | `ur5/`（UR5 真机 + Robotiq 夹爪） | XLeRobot（ManiSkill 3 仿真） | **平台不同** |
| 录制 / 可视化 / 评测 | `recording.py`、`viz_utils.py`、rerun | [evaluation.py::judge_outcome](../src/tiptop_mac/evaluation.py)、[benchmark.py](../src/tiptop_mac/benchmark.py) | 目的一致（外部、事后评测），实现不同 |

## 3. 必须如实说明的差异（不是逐行移植）

官方 TiPToP 是一套**真机 + CUDA** 系统：cuTAMP、cuRobo、FoundationStereo、SAM2、M2T2
均为 GPU/CUDA 实现，且面向 UR5 实机，在 Mac/CPU 上完全无法运行。因此本包是
**控制结构忠实、组件逐一替换**的再实现，保留的是：

- 一次性感知、规划后不再看图的边界；
- 目标用谓词（`on/holding`，并隐含 `handempty`）描述；
- 规划是全局一次性完成的 TAMP；
- 开环执行、失败即中止、不重试不重规划。

被替换/简化的部分：

1. 真机 → ManiSkill 仿真；全部 CUDA 模块 → CPU 对应模块；
2. M2T2 6-DoF 多抓取 → 物块顶部单个 4-DoF 启发式；
3. 凸包/RecGen 形状补全 → 三维包围盒；
4. cuTAMP 的粒子连续优化 + 符号搜索 → 纯符号 BFS（500 状态上限）；
5. `near` 谓词未实现；
6. **新增了官方没有的 `calibrate` 抓取标定步**：实测夹爪闭合会把物块物理拖动约 1cm，
   纯按 TCP 预排的放置点会失准。该步在 lift 后无条件执行一次，仅平移后续预定 waypoint
   （对应官方“放置相对所选抓取位姿计算”的 grasp-relative 思想），不依据结果做任何分支，
   因此不改变 TiPToP 的开环语义。

若需要与官方**算法层**（cuTAMP 连续优化、M2T2 抓取）逐一对齐，则必须在带 NVIDIA GPU
与 CUDA 的环境中直接运行官方仓库，这超出 Mac/CPU 的范围。
