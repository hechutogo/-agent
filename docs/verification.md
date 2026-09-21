# 本机验证记录

日期：2026-09-20；Apple Silicon / Apple M5 Pro；Python 3.11.15；
ManiSkill 3.0.1、SAPIEN 3.0.3、MoltenVK 1.4.2。

| 检查 | 实际结果 |
|---|---|
| `bash setup.sh` | 退出码 0；完成 editable 安装、运行库 SHA-256 校验、真实 RGB-D 渲染 |
| `bash run.sh --demo B` | 退出码 0；B 经五段动作入盒 |
| `bash run.sh --demo A B --output runs/acceptance-ab` | 退出码 0；同一场景中 A 后 B 均通过视觉检查 |
| `.venv/bin/python -m pytest -q` | 最终修复后 123 passed，15.33 秒；包括 seed=3 的真实 A 抓取 |
| 安装器修复后复跑 | 使用已校验的缓存安装运行库，退出码 0；中断下载/损坏缓存路径有专项测试 |
| 窗口检查 | `enable_viewer()`、40 个物理/渲染步、`close()`，退出码 0 |
| 窗口内连续抓放 | viewer 开启时 A→B 全部成功，保存于 `runs/viewer-ab/`，退出码 0 |
| 窗口视角 | 从实际窗口 Color buffer 抓图，确认机器人和桌面可见；修正了初始 yaw 朝向 |
| 最终 `bash run.sh` | 默认窗口入口实际运行，A/B 均成功，窗口保留供查看 |
| 云端 Agent/ASR | 单元测试通过；未配置真实服务凭据，端到端未验证 |
| 麦克风 | 可选录音入口已实现，硬件录音未验证 |

连续任务原始结果：

- A：`runs/acceptance-ab/1789879219721955000-A/result.json`，
  RGB-D 测得抬升 **0.097095 m**。
- B：`runs/acceptance-ab/1789879221899578000-B/result.json`，
  RGB-D 测得抬升 **0.090884 m**。
- 两个目录均保存抓取前、悬停、张开/闭合夹爪、抬起、最终状态的 PNG；
  对应 NPZ 包含米制深度、标定与本体关节状态。
- 随代码保留的截图：[A 抬起](evidence/a-lift.png)、
  [B 抬起](evidence/b-lift.png)、[连续任务最终状态](evidence/ab-final.png)、
  [实际窗口画面](evidence/viewer.png)、[场景全貌](evidence/overview.png)。
- 最终 A/B 在同一盒内投影范围，B 可能堆叠在 A 上；
  本版本未承诺分格摆放、平铺、或完全低于盒沿。

这些结果来自 OpenCV RGB-D 基线，证明了视觉驱动的物理抓放链路。
不代表云模型已经实际选对像素框，也不代表随机场景的统计成功率。
控制不使用 actor pose、接触事件或环境奖励；判据不使用仿真真值。

已知上游日志：pytest 有 861 条 warning，主要来自 ManiSkill/NumPy 的
`__array_wrap__` 弃用提示；另外有 Vulkan loader 回退、DashScope
Assistants API 弃用提示。SAPIEN 提示缺少 validation layer；真实渲染成功。
未修改依赖包源码或全局屏蔽警告。

默认 `run.sh` 为窗口演示；传 `--demo A B` 为无头演示，
传 `--view` 为云端文本模式。窗口设置保存在项目 `.runtime/imgui.ini`。
Qwen 层限制每轮逻辑调用次数；底层 OpenAI SDK 可能自行重试 HTTP 请求。
物理动作工具每轮最多执行一次，不因 SDK 重试而重复执行。

审查后增加了动作失败锁定：操作开始后任一步出错，后续任务均不能发出
新动作，避免在未知夹持状态下打开夹爪。恢复方式为退出后重新启动仿真；
动作前定位/IK 检查失败则仍可重试。回归测试还覆盖诊断保存失败和异常中断。

整体审查和修复复核已完成，无未处理的严重/重要问题。
最终默认入口的结果文件为 `runs/1789880002603652000-A/result.json`
及 `runs/1789880009451511000-B/result.json`，二者 `success=true`。
