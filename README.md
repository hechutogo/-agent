# PickParts：XLeRobot + ManiSkill 3 具身 Agent

基于 ManiSkill 3 / SAPIEN 的 XLeRobot v0.3 桌面操作系统。底盘和左臂锁定，
右臂根据自然语言计划执行动态物体定位、抓取、入盒和物块叠放。Agent 只使用
RGB-D 与机器人本体状态，不读取物体 pose、接触、奖励或仿真成功标志。

## 当前能力

- 默认场景包含红色物块 `A`、蓝色物块 `B` 和绿色盒子 `box`；
- 可随机添加 `block_N` 和 `box_N`，当前最多 6 个对象；
- 每次重置保留对象目录，但重新生成不重叠、桌面内、右臂可达的位置；
- 支持“把 A 放进 box”和“把 A 放到 B 上”等语义任务；
- Web 控制台可在自研 Agent、TiPToP 初版和 TiPToP 优化版之间切换；
- 支持实时执行事件、持久化日志、分层 Trace 与关联 RGB-D/LLM 产物。

详细设计见 [技术文档](docs/技术文档.md)。

## 三套 Agent

| Agent | 入口包 | 执行模式 | 完成判定 |
|---|---|---|---|
| 自研 Agent | `src/pickparts_agent/agent` | 扁平原子计划 + ReAct 恢复 | 新 RGB-D 视觉复核 |
| TiPToP 初版 | `src/tiptop_mac` | 一次性感知、全局 TAMP、开环执行 | 轨迹执行完成，不声明视觉成功 |
| TiPToP 优化版 | `src/tiptop_optimized` | 有序子任务、逐段 TAMP、ReAct 恢复 | 每段通过新 RGB-D 视觉复核 |

## 仓库结构

```text
src/pickparts_agent/
  agent/                  # Planner、Executor、Reactor、状态和原子操作
    atoms/                # 感知/校验、抓取/运动、放置原子
  scene/                  # ManiSkill 场景、动态对象、RGB-D、IK、机器人资产
  services/               # 云端 LLM/VLM/ASR 接入
  interfaces/             # CLI、FastAPI Web 和静态页面
  baseline/               # 无云端依赖的固定抓放基线
  app.py                  # python -m pickparts_agent.app 薄入口
  web.py                  # python -m pickparts_agent.web 薄入口
  runtime.py              # macOS Vulkan/MoltenVK 配置
src/tiptop_mac/           # TiPToP 风格对照包：一次性感知→全局 TAMP→开环执行
src/tiptop_optimized/     # 优化版：有序子任务→逐段 TAMP→视觉验证/ReAct
tests/                    # 单元、接口和真实仿真闭环测试
docs/                     # 技术文档、设计和验收证据
```

依赖方向：

```text
interfaces -> agent / services / scene / baseline
agent      -> scene
services   -> scene
baseline   -> scene
tiptop_mac -> pickparts_agent.scene / agent 低层运动（不改其行为）
scene      -> runtime
```

## TiPToP Mac/CPU 对照系统

[tiptop_mac](src/tiptop_mac/) 是与现有 ReAct Agent 对照的独立实现，复刻 TiPToP
流水线但去除 CUDA 依赖：一次性 RGB-D（HSV 分割 + RANSAC 桌面 + 顶部抓取启发式）
→ 语言接地为目标谓词 → CPU TAMP-lite（符号 BFS + IK/几何预检）→ 整条轨迹开环执行，
失败即中止，不做执行中复核或重规划。低层仿真、RGB-D、SciPy IK 与运动安全与主系统共用。

- 抓取闭合会把物体物理拖动约 1cm；因此规划在 lift 后插入一次**无条件、不可分支的
  抓取标定**（量物体相对 TCP 的偏移并平移后续预定点，对应 cuTAMP 的 grasp-relative
  规划），它不依据任务结果做任何决策，与 ReAct 的条件式重试有本质区别。
- 端到端验收固定同场景同种子：seed 0 `A→B`、seed 3 `B→A`、seed 7 `A→box`。

```bash
.venv/bin/python -m tiptop_mac.cli run --seed 0 "把 A 叠到 B 上"
.venv/bin/python -m tiptop_mac.cli benchmark
# 报告输出 runs/benchmark/<timestamp>/{results.json,report.md}
```

缺少云端配置时 benchmark 对相应系统显式标记 skipped，不会崩溃。

## 安装与运行

```bash
git clone https://github.com/hechutogo/-agent.git
cd ./-agent
bash setup.sh
```

Web 控制台：

```bash
./web.sh
# 打开 http://127.0.0.1:8765
```

网页顶部可选择 **自研 Agent**、**TiPToP 初版** 或 **TiPToP 优化版**，然后在右侧输入指令。切换保留当前
仿真场景，重新开始所选 Agent 的会话；运行中及需要恢复时不能切换。随机重置会保留
已选择的 Agent。自研 Agent 支持视觉验证与 ReAct 纠偏；TiPToP 展示完整预规划轨迹，
执行成功仅表示开环轨迹完成，不代表已通过视觉结果验证。

优化版保留指令的中间步骤。例如 A 已在盒内，输入“先把 A 放到桌面，再把 A 放回盒子”，
会按顺序执行两个子任务，每段重新读取 RGB-D、规划一次、执行一次并验证结果。
明确要求“重新拿起再放回”时，还会检查实际抓起这一中间动作；普通目标已经满足时仍可不移动。
可恢复失败进入 ReAct：重新观测并规划当前未完成子任务，默认每段最多 5 次尝试，
已验证的前序子任务不会重放。持物时继续放置；持物状态不明时先重新观测；
致命运动错误立即停止。控制台展示子任务进度与纠偏摘要，Trace 展示每段的尝试和视觉证据。
初版 `src/tiptop_mac` 的规划及开环行为保持不变，便于对照。
版本差异、恢复边界及实测记录见 [TiPToP 优化版说明](docs/tiptop-optimized.md)。

点击当前任务的 **查看 Trace**，或打开 [运行历史](http://127.0.0.1:8765/history)，
可按日期、Agent、状态与关键字查找记录，查看分层耗时、关联日志和 RGB-D/LLM 产物。
当前选中的运行会自动刷新，可直接分享本地 `/history?run=<run_id>` 链接。
日志默认保留 7 天，配置项为 `PICKPARTS_LOG_DIR`、`PICKPARTS_LOG_RETENTION_DAYS`
和 `PICKPARTS_LOG_LEVEL`。

自然语言 CLI：

```bash
bash run.sh --view
```

无需云端 Key 的本地基线：

```bash
bash run.sh --demo A B --view
bash run.sh --smoke
```

云端模式读取 `.env` 中的 `LLM_*`、`VLM_*` 和可选 `ASR_*`。Agent 主链路
使用项目内轻量 Planner/ReAct 编排和 OpenAI 兼容 API；Qwen-Agent 兼容工具
保留在服务层，但不作为默认编排器。

完整测试：

```bash
.venv/bin/python -m pytest -q
```

机器人资产来自 [Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot)，
固定 revision 与修改记录见 [第三方来源](third_party/README.md)。当前不包含移动
底盘导航、双臂协同或实机控制。

## 文档

- [具身 Agent 设计文档](docs/具身Agent设计文档.md)：完整架构、三引擎边界和演进路线。
- [技术文档](docs/技术文档.md)：系统架构、原子操作、Agent、Web 与 Trace。
- [TiPToP 初版源码对照](docs/tiptop-mac-源码对照.md)：官方实现映射与 Mac/CPU 替换边界。
- [TiPToP 优化版](docs/tiptop-optimized.md)：子任务、恢复机制和验收记录。
- [技术选型](docs/技术选型方案.md) 与 [开源具身 Agent 调研](docs/开源具身Agent调研.md)。

## 第三方来源

机器人资产及修改记录见 [third_party/README.md](third_party/README.md)，
对应许可证见 [third_party/XLeRobot-LICENSE](third_party/XLeRobot-LICENSE)。
