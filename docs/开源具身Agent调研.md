# 开源具身 Agent 调研与选型建议

> 调研日期：2026-09-21
> 调研范围：桌面操作、视觉语言动作模型、LLM/VLM 规划型 Agent，以及相关
> 仿真、数据与训练基础设施。
> 主要依据：项目官方 GitHub、官方文档、项目主页和论文。仓库活跃度、模型版本
> 与许可证可能继续变化，正式商用前需再次核验。

## 1. 执行摘要

“开源具身 Agent”不是单一技术类别。当前公开项目主要分为三类：

1. **模块化规划 Agent**：LLM/VLM 负责理解、规划或生成约束，传统感知、运动
   规划器和控制器负责接地执行。代表项目为 TiPToP、VoxPoser、Code as
   Policies。
2. **学习型策略或 VLA**：模型直接从图像、语言和本体状态预测动作或动作块。
   代表项目为 GR00T N1.7、openpi π0.5、SmolVLA、OpenVLA-OFT、Octo。
3. **基础设施**：提供仿真、数据、训练、硬件接口或评测，本身不是完整 Agent。
   代表项目为 ManiSkill、LeRobot、RoboTwin、Isaac Lab、Habitat 和
   Open X-Embodiment。

对当前 `ManiSkill 3 + XLeRobot + RGB-D + 原子 ReAct` 项目，结论是：

- **不建议整体替换现有 Agent。** 当前系统已经具备非特权感知、显式状态、
  可审计原子、失败恢复和最终视觉复核，而主流 VLA 通常不包含这些系统能力。
- **最值得参考的新项目是 TiPToP。** 它验证了“开放词汇感知 + 显式任务与运动
  规划 + 执行”的模块化路线，但其 CUDA/cuTAMP/DROID 技术栈不能直接搬到 Mac。
- **最值得引入的生态是 LeRobotDataset。** 先统一采集和导出数据，再决定训练
  SmolVLA、OpenVLA-OFT、π0.5 或其他策略。
- **最合理的中期架构是混合式。** 保留当前 Planner/Reactor/Verifier，把 VLA
  作为某些 learned atom 的低层执行器，而不是让端到端模型直接取代安全闭环。
- **大模型宜与当前 Mac 控制端解耦。** OpenPI、GR00T 和 OpenVLA-OFT 宜运行在
  Linux/NVIDIA GPU 服务上；SmolVLA 可以在 Mac/CPU 上试验，但高效训练仍建议
  使用 GPU。

## 2. 评估口径

### 2.1 什么算 Agent

本文将完整具身 Agent 定义为能够围绕目标持续完成：

```text
感知 -> 规划 -> 执行 -> 结果验证 -> 失败恢复
```

因此：

- 只有视觉到动作映射的 VLA，是**策略模型**，不自动等于完整 Agent；
- 只有任务和传感器的仿真器，是**环境基础设施**；
- 只有轨迹、数据格式或训练脚本的项目，是**数据/训练工具链**；
- 论文公开但没有可用代码和许可证，不列为可直接采用的开源实现。

### 2.2 对比维度

每个项目按以下维度评价：

- 开放程度：代码、权重、训练数据和训练脚本是否公开；
- 许可证：代码与模型权重是否使用同一种许可证；
- 感知与动作：RGB/RGB-D/状态输入，关节或末端动作输出；
- 闭环能力：是否包含计划、验证、重试和重规划；
- embodiment 适配：接入新机器人需要多少数据与控制开发；
- 算力和平台：macOS、Linux、CUDA、显存要求；
- 工程成熟度：安装、测试、部署服务、维护状态；
- 当前项目适配度：是否适合 XLeRobot 和 ManiSkill 3。

## 3. 总览对比

| 项目 | 类型 | 核心输出 | 训练需求 | 完整闭环 | 当前项目适配度 |
|---|---|---|---|---|---|
| TiPToP | 模块化 TAMP Agent | 关节轨迹与夹爪命令 | 零机器人训练数据 | 否，规划后开环执行 | 中，适合借鉴对象中心 TAMP |
| VoxPoser | LLM/VLM 三维规划 | 6-DoF 轨迹 | 无需策略训练 | 部分 | 中，适合借鉴几何接地 |
| Code as Policies | 代码生成 Agent | 可执行 Python policy | 无需策略训练 | 弱 | 高，适合借鉴受限工具接口 |
| Inner Monologue | 闭环规划方法 | 技能序列 | 无需额外训练 | 是 | 高，但无官方代码 |
| OpenPI π0.5 | 通用 VLA | 连续动作块 | 新机器人通常需微调 | 否 | 中，适合未来 learned atom |
| GR00T N1.7 | 跨 embodiment VLA | 连续动作块 | 新 embodiment 通常需后训练 | 否 | 中低，依赖 NVIDIA 栈 |
| SmolVLA | 小型 VLA | 连续动作块 | 少量示范微调 | 否 | 中高，适合低成本试验 |
| OpenVLA-OFT | 优化 VLA | 连续动作块 | 需要任务/机器人数据 | 否 | 中，适合高频/双臂研究 |
| OpenVLA | 通用 VLA 基线 | 离散动作 token | 新机器人通常需微调 | 否 | 中低，模型较重 |
| Octo | 通用机器人策略 | 扩散动作块 | 少量目标域数据 | 否 | 中低，维护较弱 |
| RoboAgent | 多技能模仿策略 | 动作块 | 需要多技能示范 | 否 | 低，Franka 栈较固定 |
| VIMA | 多模态提示策略 | 参数化动作 | 依赖 VIMA 数据 | 否 | 低，主要是模拟基准 |

“当前项目适配度”是本文工程判断，不是项目方给出的官方结论。

## 4. 模块化规划型具身 Agent

### 4.1 TiPToP

- 官方代码：[tiptop-robot/tiptop](https://github.com/tiptop-robot/tiptop)
- 项目页：[tiptop-robot.github.io](https://tiptop-robot.github.io/)
- 文档：[tiptop-robot.readthedocs.io](https://tiptop-robot.readthedocs.io/en/latest/)
- 论文：[arXiv:2603.09971](https://arxiv.org/abs/2603.09971)
- 许可证：TiPToP 主代码 MIT；cuRobo、cuTAMP、M2T2、FoundationStereo 及
  相关检查点受 NVIDIA 非商业研究/评估条款约束。

**定位**

TiPToP 是 2026 年发布的开放词汇模块化操作系统。输入为图像和自然语言，
输出机器人关节轨迹与夹爪动作。系统把感知、Task and Motion Planning 和执行
明确分开，不依赖目标机器人示范数据。

**架构**

```text
双目 RGB
  -> 深度、检测、分割、抓取候选
  -> 对象中心三维场景表示
  -> 语言任务解析
  -> GPU 并行 cuTAMP 搜索
  -> 关节阻抗控制
```

官方实现面向 DROID/Franka，并提供 Isaac Lab 仿真、UR5e 和 WidowX 扩展说明。

**优点**

- 组件边界清晰，失败可以定位到感知、规划或控制层；
- 零示范数据即可执行开放词汇任务；
- 显式物理约束和搜索比纯 LLM 计划更容易验证；
- 与当前项目“语义规划、运行时接地”的方向接近；
- 支持多步 pick-and-place 和可见障碍物重排。

**缺点**

- 依赖 CUDA、cuRobo/cuTAMP 和多个 GPU 感知服务；
- 官方硬件与 XLeRobot 运动学、夹爪和相机配置不同；
- 使用 Gemini API，仍有云模型依赖；
- 2026 年的新项目，生态和长期稳定性尚需观察；
- 完整轨迹规划后开环执行，不包含执行中视觉验证或重规划；
- 单固定视角不能感知完全遮挡目标；
- 当前不支持物体叠放，凸包表示也难区分开放容器的 inside/on 关系；
- 不能在当前 Apple Silicon 环境中低成本原样运行。

**对当前项目的价值**

优先借鉴其对象中心场景表示和可行性搜索，而不是替换现有 ManiSkill 环境。
它不能直接覆盖当前叠放、执行中恢复和最终视觉验证要求。未来增加复杂障碍或
多步整理时，可将 TAMP 服务作为新的 Planner 或 planned atom 后端。

### 4.2 VoxPoser

- 官方代码：[huangwl18/VoxPoser](https://github.com/huangwl18/VoxPoser)
- 项目页：[voxposer.github.io](https://voxposer.github.io/)
- 论文：[arXiv:2307.05973](https://arxiv.org/abs/2307.05973)
- 许可证：MIT。

**定位与架构**

VoxPoser 使用 LLM 生成调用 VLM 的程序，将语言中的 affordance 和约束组合为
三维 value map，再由模型规划器生成 6-DoF 末端轨迹。语言程序可缓存，视觉
变化后重新计算 value map，因此具有一定扰动恢复能力。

**优点**

- 开放词汇和零样本能力强；
- RGB-D 到三维轨迹的接地过程具有解释性；
- 不要求训练一个新的端到端动作模型；
- 对“靠近、避开、保持某方向”等连续空间关系表达强。

**缺点**

- 公布代码主要是 RLBench demo，不是完整产品框架；
- 公开模拟实现读取环境 object mask，属于特权观测；
- 论文中的真实感知链没有完整发布；
- 主要围绕 Franka 和 6-DoF 末端轨迹，移植到 XLeRobot 仍需适配；
- LLM 生成代码和值函数带来执行安全和数值稳定性风险。

**对当前项目的价值**

适合在现有 `place_on`、`carry_to` 之外增加复杂空间关系原子。应保留当前原则：
模型只输出语义约束，坐标和值图在受控运行时计算。

### 4.3 Code as Policies

- 官方代码：[google-research/code_as_policies](https://github.com/google-research/google-research/tree/master/code_as_policies)
- 项目页：[code-as-policies.github.io](https://code-as-policies.github.io/)
- 论文：[arXiv:2209.07753](https://arxiv.org/abs/2209.07753)
- 许可证：Google Research 仓库 Apache-2.0。

**定位与架构**

Code as Policies 让代码模型根据自然语言生成 Python policy。生成代码可以调用
对象检测、几何函数、`pick/place`、轨迹和控制 API，也可递归生成未定义函数。

**优点**

- 技能 API 可组合，任务表达力高；
- 控制流和几何计算可读、可调试；
- 跨机器人时主要替换底层 API；
- 与当前 AtomCall 思路高度一致。

**缺点**

- 原始方案执行模型生成的 Python，攻击面和故障面较大；
- 正确性依赖提示样例和既有技能；
- 缺少强类型 schema、权限隔离和系统化执行后复核；
- 官方实现更接近论文复现代码，维护频率较低。

**对当前项目的价值**

当前做法可以视为更保守的 CaP：LLM 不能生成任意 Python，只能生成
`AtomRegistry` 校验后的扁平调用。这个限制应继续保留。

### 4.4 Inner Monologue

- 项目页：[innermonologue.github.io](https://innermonologue.github.io/)
- 论文：[arXiv:2207.05608](https://arxiv.org/abs/2207.05608)
- 开放状态：未发现官方完整代码仓库或软件许可证。

Inner Monologue 将动作成功检测、对象列表、场景描述和人工反馈持续加入 LLM
上下文，使高层计划从单次生成变成 `plan -> act -> observe -> replan` 闭环。

**优点**

- 明确解决长任务中动作失败后继续盲目执行的问题；
- LLM 仍只选择技能，低层控制保持独立；
- 支持主动询问用户；
- 与当前 Planner/Executor/Reactor 架构高度同构。

**缺点**

- 没有可直接复用的官方框架；
- 部分实验依赖 oracle 或人工反馈；
- 最终能力受技能库和成功检测器上限约束。

**判断**

这是当前 ReAct 机制的重要思想参考，但不能作为可安装的开源依赖。

### 4.5 ReKep

- 官方代码：[huangwl18/ReKep](https://github.com/huangwl18/ReKep)
- 项目页：[rekep-robot.github.io](https://rekep-robot.github.io/)
- 论文：[arXiv:2409.01652](https://arxiv.org/abs/2409.01652)
- 开放状态：代码可见，但检索时仓库没有明确 LICENSE，严格意义上应视为
  source-available。

ReKep 使用 DINOv2、SAM 和 VLM 生成关系关键点约束，再通过分层优化求解
SE(3) 子目标和路径。

**优点**

- 对精细空间关系、障碍和动态扰动的表达强；
- 关键点约束比固定 pick/place 原子更灵活；
- 单臂和双臂实验都具有参考价值。

**缺点**

- 公开模拟代码使用关键点、mask、SDF 等仿真信息；
- 真实感知、跟踪和 SDF 管线没有完整公开；
- 依赖 NVIDIA GPU、Isaac 和 GPT-4o；
- 许可证不明确，不适合作为直接产品依赖。

### 4.6 不应误判为可直接采用的开源 Agent

- **MALMM**：论文和 prompts 已公开，但项目页仍标注代码 “coming soon”，
  因此不能视为已交付的开源实现。
- **Inner Monologue**：项目思想和实验公开，但没有官方可安装代码。
- **ReKep**：源代码可见，但缺少明确软件许可证。

## 5. 开源 VLA 与学习型策略

### 5.1 OpenPI：π0、π0-FAST、π0.5

- 官方代码：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- π0 论文：[arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- π0.5 论文：[arXiv:2504.16054](https://arxiv.org/abs/2504.16054)
- FAST：[官方说明](https://www.pi.website/research/fast)
- 代码许可证：Apache-2.0。

**架构**

- π0/π0.5：PaliGemma VLM + action expert，通过 flow matching 输出连续动作块；
- π0-FAST：使用 FAST 动作 tokenizer，自回归生成压缩动作 token；
- 当前仓库支持 JAX，并提供 π0/π0.5 的 PyTorch 实现；
- 提供 WebSocket 远程策略服务和 DROID、ALOHA、UR5、LIBERO 示例。

官方仓库当前列出 π0、π0-FAST 和 π0.5。虽然项目方已经发布后续研究模型，
不能在 openpi README 明确支持前将其计为该仓库已开放能力。

**资源要求**

| 模式 | 官方最低显存说明 |
|---|---:|
| 推理 | 大于 8 GB |
| LoRA 微调 | 大于 22.5 GB |
| 全量微调 | 大于 70 GB |

官方主要支持 Ubuntu 22.04 和 NVIDIA GPU，不支持当前 macOS 作为正式运行环境。

**优点**

- 预训练数据规模大，连续动作和动作块适合真实操作；
- π0.5 兼顾语言跟随和开放场景迁移；
- 数据转换、微调和远程推理链路较完整；
- 可将 GPU 推理服务与机器人控制端解耦。

**缺点**

- 新机器人通常仍需要示范数据和动作/状态适配；
- 10k+ 小时完整预训练数据没有全部公开，无法完全复现预训练；
- PyTorch 功能仍少于 JAX 路径；
- 项目方明确说明迁移到任意新机器人不保证成功；
- 检查点再分发前应单独确认权重条款，而不能只看代码许可证。

**当前项目适配**

适合作为远端 learned atom 服务，不适合在 Mac 本机直接替换现有执行器。

### 5.2 NVIDIA Isaac GR00T N1.7

- 官方代码：[NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- 产品页：[NVIDIA Isaac GR00T](https://developer.nvidia.com/isaac/gr00t)
- 论文：[arXiv:2503.14734](https://arxiv.org/abs/2503.14734)
- 代码许可证：Apache-2.0；
- 模型许可证：[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)。

截至调研日，官方主分支为 **GR00T N1.7 3B**，N1.5/N1.6 是历史分支。N1.7
使用 Cosmos-Reason2-2B / Qwen3-VL 架构视觉语言骨干和 flow-matching DiT，
通过 relative EEF action 统一不同机器人与人类动作表示，输出连续动作块。

**优点**

- 数据、训练、评测、Policy API、ONNX/TensorRT 部署链完整；
- 覆盖 DROID、LIBERO、SimplerEnv、SO100 和人形机器人；
- 与 LeRobot 数据格式连接紧密；
- 官方给出桌面 GPU、DGX Spark、Jetson Thor/Orin 路径；
- 当前是 GA 版本，工程稳定性信号强。

**缺点**

- 推理要求 16 GB 以上 NVIDIA GPU，微调建议 40 GB 以上；
- Python 3.12、CUDA 和 TensorRT 栈与当前 Mac/Python 3.11 不兼容；
- 新 embodiment 需要 modality 配置、数据转换和后训练；
- 依赖 gated Cosmos 权重；
- 模型权重不是简单 Apache-2.0，需要遵守 NVIDIA 模型条款。

**当前项目适配**

只有在增加 Linux/NVIDIA 推理服务器、转向人形或需要 TensorRT 边缘部署时，
优先级才高于 SmolVLA/openpi。

### 5.3 LeRobot 与 SmolVLA

- 官方代码：[huggingface/lerobot](https://github.com/huggingface/lerobot)
- 文档：[huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot/)
- SmolVLA：[官方介绍](https://huggingface.co/blog/smolvla)
- SmolVLA 论文：[arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
- 许可证：LeRobot 代码和 SmolVLA 权重为 Apache-2.0；Hub 上其他数据集与模型
  许可证需分别核查。

LeRobot 本身是机器人学习工具链，不是单个 Agent。它提供统一 Robot 接口、
LeRobotDataset、采集/回放/训练/评测 CLI，以及 ACT、Diffusion、π0、
π0-FAST、π0.5、GR00T N1.7、SmolVLA、XVLA、EO-1、MolmoAct2、
WALL-OSS、EVO1 等策略实现。

SmolVLA 是约 450M 参数的小型 VLA，使用多视角 RGB、语言和状态，结合
flow-matching action expert 输出连续动作块，并支持异步推理。

**优点**

- PyTorch 原生，数据、硬件、训练和部署接口统一；
- 低成本机器人与第三方硬件插件生态较完整；
- LeRobotDataset 使用视频/图像、Parquet 和元数据统一轨迹；
- SmolVLA 体量显著小于 3B/7B 模型；
- 官方支持 Apple Silicon MPS，适合低成本原型和数据处理；
- 项目维护活跃。

**缺点**

- XLeRobot 尚未原生合入 LeRobot；XLeRobot 官方仓库提供临时适配和 VLA
  工作流，但目前仍需要复制或补丁式集成；
- 新版 LeRobot 的 Python/PyTorch 版本与当前 Python 3.11 环境存在冲突；
- SmolVLA 仍需要高质量目标任务示范，不能把小模型等同于零样本通用操作；
- Mac 可用于部分推理/训练实验，但大型 VLA 训练仍主要依赖 NVIDIA GPU。

**当前项目适配**

这是近期最值得接入的外部生态。建议先在独立环境中实现
`ManiSkill/XLeRobot trajectory -> LeRobotDataset` 导出，不要直接把 LeRobot
依赖装进现有仿真环境。

### 5.4 OpenVLA 与 OpenVLA-OFT

- OpenVLA：[代码](https://github.com/openvla/openvla)、
  [项目页](https://openvla.github.io/)、[论文](https://arxiv.org/abs/2406.09246)
- OpenVLA-OFT：[代码](https://github.com/moojink/openvla-oft)、
  [项目页](https://openvla-oft.github.io/)、[论文](https://arxiv.org/abs/2502.19645)
- 许可证：代码为 MIT；权重继承 OpenVLA/Llama 2 等基础模型条款。

OpenVLA 是约 7B 参数的 VLA，使用 DINOv2 + SigLIP 视觉编码器和 Llama 2，
把连续动作离散为 token 自回归输出。它在约 970K 条 Open X-Embodiment 轨迹
上训练。

OpenVLA-OFT 保留 7B 主干，但增加连续动作头、动作分块、并行解码、多相机和
proprioception；OFT+ 还可使用 FiLM。官方报告相对原始 OpenVLA 获得更快推理
和更高控制频率。

**优点**

- OpenVLA 是成熟且广泛使用的开放 VLA 基线；
- 完整提供权重、推理、LoRA、全量微调和 RLDS 数据管线；
- OFT 更适合高频动作、双臂和多相机；
- 支持通过独立 REST 服务把 GPU 推理与机器人端分离。

**缺点**

- 原始 OpenVLA 逐 token 动作慢，官方建议约 5-10 Hz 数据；
- OFT 推理仍约需 16-18 GB 显存；
- LoRA 通常需要约 27 GB 以上显存，全量训练成本更高；
- XLeRobot 不在现成 embodiment 中，需要采集数据、归一化动作和微调；
- 代码 MIT 不代表模型权重没有 Llama 2 条款。

**当前项目适配**

若后续升级到双臂连续控制，OpenVLA-OFT 比原始 OpenVLA 更值得做对比实验。

### 5.5 Octo

- 官方代码：[octo-models/octo](https://github.com/octo-models/octo)
- 项目页：[octo-models.github.io](https://octo-models.github.io/)
- 论文：[arXiv:2405.12213](https://arxiv.org/abs/2405.12213)
- 许可证：MIT。

Octo 1.5 是 JAX Transformer diffusion policy，提供 Octo-Small（27M）和
Octo-Base（93M），使用约 800K 条 Open X-Embodiment 轨迹训练。它支持多 RGB
相机、语言或目标图像，并输出连续动作块。

**优点**

- 模型小，结构模块化；
- observation/action head 可替换，适应不同动作维度；
- 单张消费级 GPU 即可进行较轻量实验；
- 是跨 embodiment 通用策略的重要公开基线。

**缺点**

- JAX/CUDA 环境与当前工程不一致；
- 语言语义先验弱于采用大型 VLM 的方案；
- 官方主仓更新主要停留在 2024 年；
- 对 XLeRobot 仍需数据和动作适配。

### 5.6 RoboAgent

- 官方代码：[robopen/roboagent](https://github.com/robopen/roboagent)
- 项目页：[robopen.github.io](https://robopen.github.io/)
- 论文：[arXiv:2309.01918](https://arxiv.org/abs/2309.01918)
- 许可证：代码和 RoboSet 为 MIT。

RoboAgent 的核心是 MT-ACT：语义增强、语言条件和 action chunking 组成的
多任务模仿策略。官方使用约 7,500 条轨迹学习 12 类技能，并在 Franka +
Robotiq 的厨房场景中评测。

**优点**

- 关注低数据条件下的多技能泛化；
- 动作块有利于平滑控制；
- 数据集、训练代码和 checkpoint 相对完整。

**缺点**

- 它不是在线 LLM 规划或 ReAct Agent；
- 传感器、动作维度和硬件栈较固定；
- 迁移 XLeRobot 需要重新采集多视角数据和训练；
- 维护状态更接近论文代码快照。

### 5.7 VIMA

- 官方代码：[vimalabs/VIMA](https://github.com/vimalabs/VIMA)
- 基准：[vimalabs/VimaBench](https://github.com/vimalabs/VimaBench)
- 项目页：[vimalabs.github.io](https://vimalabs.github.io/)
- 论文：[arXiv:2210.03094](https://arxiv.org/abs/2210.03094)
- 许可证：模型和基准 MIT，公开轨迹数据为 CC BY 4.0。

VIMA 使用交错文字和图像/视频的多模态 prompt，通过 T5 prompt encoder 和
因果 Transformer 输出参数化动作。VIMA-Bench 提供 17 类程序化桌面任务和
多级泛化评测。

**优点**

- 数据、模型、benchmark 和多规模 checkpoint 较完整；
- 适合评测视觉引用、示例模仿和组合泛化；
- 模型规模从 2M 到 200M，实验成本可控。

**缺点**

- 主要是 VIMA-Bench/PyBullet 模拟策略；
- 依赖对象级 token 和参数化 pick-place 动作；
- 不包含显式计划、执行验证和 ReAct；
- 官方仓库长期没有明显更新。

## 6. 基础设施与 Agent 的边界

### 6.1 ManiSkill 3

- 官方代码：[mani-skill/ManiSkill](https://github.com/haosulab/ManiSkill)
- 文档：[maniskill.readthedocs.io](https://maniskill.readthedocs.io/en/latest/)
- 论文：[arXiv:2410.00425](https://arxiv.org/abs/2410.00425)
- 代码许可证：Apache-2.0；ManiSkill 官方资产为 CC BY-NC 4.0，外部资产另按
  来源条款。

ManiSkill 是仿真、任务、数据生成和策略训练框架，不是 Agent。它支持 RGB-D、
点云、多种机器人、操作任务、RL/IL/VLA 基线和自定义 URDF。Linux/NVIDIA
环境能使用 GPU 并行；macOS 只能使用受限的 CPU 仿真路径与 Vulkan/MoltenVK。

**当前判断：继续作为主仿真平台。** 项目已经完成 XLeRobot、相机和非特权
观测适配，迁移到其他模拟器不会直接提高 Agent 能力。

### 6.2 LeRobot / LeRobotDataset

LeRobot 是训练、数据和硬件工具链，而不是完整 Agent。它最有价值的部分是：

- 统一真实机器人和相机接口；
- Parquet + MP4/图像的轨迹格式；
- Hub 数据分发；
- ACT、Diffusion、SmolVLA、π0、GR00T 等统一训练入口。

建议把它作为独立数据层引入，而不是替换当前 Planner/ReActor。

### 6.3 RoboTwin 2.0

- 官方代码：[RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)
- 文档：[robotwin-platform.github.io](https://robotwin-platform.github.io/doc/)
- 论文：[arXiv:2506.18088](https://arxiv.org/abs/2506.18088)
- 许可证：代码 MIT；资产和第三方依赖需分别核查。

RoboTwin 是双臂操作数据生成和评测平台，提供大量双臂任务、物体和专家轨迹，
并支持 LeRobot 格式。它不是部署时 Agent。

**优点**：双臂任务、自动专家轨迹、随机化和评测覆盖强。
**缺点**：主路径依赖 Linux/NVIDIA/CUDA/CuRobo；XLeRobot 不是现成
embodiment；不适合当前 Mac 主环境。
**建议**：未来在独立 Linux GPU 机器上作为双臂泛化 benchmark。

### 6.4 NVIDIA Isaac Lab

- 官方代码：[isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)
- 文档：[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- 许可证：核心 BSD-3-Clause，Isaac Sim、Omniverse、资产和部分组件另有条款。

Isaac Lab 是建立在 Isaac Sim 上的 GPU 机器人学习框架，不是完整 Agent。它在
高保真传感器、大规模并行 RL/IL、域随机化、合成数据和人形机器人方面强，但
不支持当前 macOS 主机作为正式训练环境。

### 6.5 Habitat

- 官方代码：[Habitat-Lab](https://github.com/facebookresearch/habitat-lab)、
  [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- 项目页：[aihabitat.org](https://aihabitat.org/)
- 许可证：代码 MIT；场景数据许可证独立。

Habitat 主要面向室内导航、移动操作和家庭任务。大规模室内场景与导航评测成熟，
但桌面精细抓放不是其核心。截至调研日，Meta 已停止对 v0.3.4 之后版本的官方
主动开发和维护。只有项目扩展到房间级语义导航时才值得引入。

### 6.6 Open X-Embodiment

- 官方代码：[google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment)
- 项目页：[robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/)
- 代码许可证：Apache-2.0；各贡献数据集许可证和引用要求独立。

OXE 是跨机器人真实轨迹集合和 RLDS 数据接口，不是 Agent 或仿真器。它适合
预训练和数据规范研究，但异构机器人、相机和动作空间使直接迁移成本很高。
仓库另提供 RT-1-X JAX checkpoint；其输入只有单路工作区 RGB 和任务文本，
不接收深度或腕部相机。

### 6.7 RLBench

- 官方代码：[stepjam/RLBench](https://github.com/stepjam/RLBench)
- 论文：[arXiv:1909.12271](https://arxiv.org/abs/1909.12271)
- 许可证：非商业、内部或学术研究许可，不是宽松开源许可证。

RLBench 提供大量桌面任务和运动规划演示，是 VoxPoser 等工作的常用评测环境。
但它依赖 PyRep/CoppeliaSim 的旧 Linux 技术栈，许可证也不适合直接商业采用。

## 7. 开放程度与许可证

| 项目 | 代码 | 权重 | 数据 | 主要许可证注意点 |
|---|---|---|---|---|
| TiPToP | 开放 | 依赖外部模型 | 无需机器人示范 | 主代码 MIT，关键 NVIDIA 依赖/权重限非商业研究与评估 |
| VoxPoser | 开放 demo | 使用外部模型 | 无 | MIT，真实感知未完整公开 |
| Code as Policies | 开放 demo | 不适用 | 无 | Apache-2.0 |
| Inner Monologue | 未开放完整实现 | 不适用 | 未完整开放 | 无软件许可证 |
| ReKep | 源码可见 | 使用外部模型 | 无 | 未发现 LICENSE |
| openpi | 开放 | π0/FAST/π0.5 开放 | 部分 | 代码 Apache-2.0，权重条款需复核 |
| GR00T N1.7 | 开放 | 开放 | 部分 | 代码 Apache-2.0，权重 NVIDIA OML |
| LeRobot/SmolVLA | 开放 | 开放 | 开放集合 | Apache-2.0；Hub 单项另行核查 |
| OpenVLA/OFT | 开放 | 开放 | OXE/部分数据 | 代码 MIT，权重继承 Llama 2 条款 |
| Octo | 开放 | 开放 | OXE | MIT；数据集许可证独立 |
| RoboAgent | 开放 | 开放 | RoboSet 开放 | MIT |
| VIMA | 开放 | 开放 | 开放 | 代码 MIT，数据 CC BY 4.0 |

“代码可见”不自动等于具备开放源代码许可；代码许可证也不自动覆盖模型权重、
数据集、机器人资产和第三方模型。

## 8. 选型建议

### 8.1 近期：保持当前模块化 Agent

继续使用：

```text
ManiSkill 3 + XLeRobot + RGB-D
  -> WorldState
  -> Planner
  -> Atomic Executor
  -> Reactor
  -> fresh visual verifier
```

原因：

- 当前任务只有少量几何明确的抓取、入盒和叠放；
- 不需要为简单任务承担 VLA 训练数据和 GPU 成本；
- 当前执行过程可解释、可测试，可针对每个原子设置安全边界；
- Mac 上可直接运行，已完成真实仿真闭环；
- 非特权观测要求已经在接口和测试中固化。

近期应优先借鉴：

1. TiPToP 的对象中心表示和显式可行性规划；
2. VoxPoser/ReKep 的关系约束表达；
3. Code as Policies 的工具组合思想；
4. Inner Monologue 的环境反馈协议。

### 8.2 中期：把 VLA 作为 learned atom

推荐架构：

```text
用户任务
  -> 当前 Planner/Reactor
  -> 选择原子
       -> 解析式原子：find/reach/place/verify
       -> learned atom：VLA policy server
  -> 本地安全检查
  -> XLeRobot 控制
  -> RGB-D 结果验证
```

VLA 不直接拥有全局任务成功判定。它只在明确的局部阶段输出一段动作，例如：

- `grasp_open_vocabulary_object`
- `insert_into_tight_container`
- `reorient_object`
- `recover_failed_grasp`

这样既能使用学习策略处理难以手写的接触动作，又保留任务级可解释性、预算和
安全停止机制。

接入任何 VLA 前都必须增加观测、时序和动作映射。当前系统只有单路 workspace
RGB-D、右臂 5 关节加夹爪的仿真控制，没有腕部相机或 XLeRobot 实机执行接口；
不能把常见的 7-DoF EEF delta 动作直接发送给现有控制器。

### 8.3 数据路线

优先实现独立的数据转换：

```text
ManiSkill Frame + qpos + action + language
  -> episode recorder
  -> LeRobotDataset
  -> 独立 Linux/NVIDIA 训练环境
  -> policy server
  -> 当前 Mac/XLeRobot 客户端
```

建议记录：

- 工作区和腕部 RGB；
- 深度或点云派生数据；
- qpos/qvel；
- 末端位姿和夹爪状态；
- 实际发送动作；
- 任务文本、原子阶段和成功标签；
- 失败类型与恢复决策。

### 8.4 候选优先级

| 优先级 | 项目 | 建议 |
|---:|---|---|
| 1 | LeRobotDataset | 先统一数据，不改变当前控制链 |
| 2 | TiPToP | 借鉴场景表示、TAMP 和失败分析 |
| 3 | SmolVLA | 在独立环境做低成本 learned atom PoC |
| 4 | OpenVLA-OFT | 评估双臂、多相机和高频动作 |
| 5 | openpi π0.5 | 有足够数据和 GPU 后评估连续控制 |
| 6 | GR00T N1.7 | 转向 NVIDIA/人形/边缘部署时采用 |
| 7 | Octo/OpenVLA/VIMA/RoboAgent | 作为研究基线和对照实验 |

## 9. 建议的验证指标

不能只比较论文中的平均成功率。接入当前项目时应统一测量：

| 维度 | 指标 |
|---|---|
| 成功率 | 每任务、每对象、每 seed 的成功比例 |
| 泛化 | 新位置、新颜色、新对象、新语言表达 |
| 闭环恢复 | 定位丢失、空抓、碰撞、目标移动后的恢复率 |
| 安全 | 越界动作、IK 失败、碰撞、错误继续执行次数 |
| 实时性 | 感知、模型推理、动作块执行的 P50/P95 延迟 |
| 数据效率 | 达到目标成功率所需示范条数与采集时间 |
| 资源 | 峰值显存、内存、网络吞吐和模型服务成本 |
| 可解释性 | 是否能定位失败原子、观测和恢复原因 |
| 可复现性 | 代码、权重、数据、许可证和环境是否齐全 |

推荐先固定当前三个核心任务：

1. 任意物块放入任意盒子；
2. 任意物块叠放到另一物块；
3. 抓取失败或目标移动后的恢复。

再用同一批随机场景比较解析式原子、SmolVLA learned atom 和
OpenVLA-OFT/openpi 远程策略，避免跨 benchmark 的成功率直接比较。

## 10. 风险提示

1. **开源不等于可商用。** 模型权重、基础模型、数据、资产和代码许可证可能
   不同。
2. **论文成功率不可横向直接比较。** 机器人、动作空间、相机、任务定义和
   成功判据不同。
3. **零样本不等于零适配。** 相机标定、动作归一化、安全控制和机器人接口仍
   必须实现。
4. **VLA 不自动提供安全闭环。** 大多数 VLA 只预测下一段动作，不负责最终
   目标验证和异常恢复。
5. **仿真成功不等于实机成功。** 摩擦、回差、延迟、遮挡和深度噪声会改变结果。
6. **公开代码不代表完整复现。** 多个项目没有公开全部真实感知或预训练数据。
7. **Mac 适合现有仿真验证，不适合主流大 VLA 训练。** 训练和低延迟推理应
   规划独立 NVIDIA GPU 节点。

## 11. 官方来源

### 模块化 Agent

- TiPToP：[代码](https://github.com/tiptop-robot/tiptop) |
  [文档](https://tiptop-robot.readthedocs.io/en/latest/) |
  [限制](https://tiptop-robot.readthedocs.io/en/latest/limitations/) |
  [论文](https://arxiv.org/abs/2603.09971)
- VoxPoser：[代码](https://github.com/huangwl18/VoxPoser) |
  [项目页](https://voxposer.github.io/) |
  [论文](https://arxiv.org/abs/2307.05973)
- Code as Policies：[代码](https://github.com/google-research/google-research/tree/master/code_as_policies) |
  [项目页](https://code-as-policies.github.io/) |
  [论文](https://arxiv.org/abs/2209.07753)
- Inner Monologue：[项目页](https://innermonologue.github.io/) |
  [论文](https://arxiv.org/abs/2207.05608)
- ReKep：[代码](https://github.com/huangwl18/ReKep) |
  [项目页](https://rekep-robot.github.io/) |
  [论文](https://arxiv.org/abs/2409.01652)
- MALMM：[项目页](https://malmm1.github.io/) |
  [论文](https://arxiv.org/abs/2411.17636)

### VLA 与学习型策略

- OpenPI：[代码](https://github.com/Physical-Intelligence/openpi) |
  [π0](https://arxiv.org/abs/2410.24164) |
  [π0.5](https://arxiv.org/abs/2504.16054)
- Isaac GR00T：[代码](https://github.com/NVIDIA/Isaac-GR00T) |
  [产品页](https://developer.nvidia.com/isaac/gr00t)
- LeRobot/SmolVLA：[代码](https://github.com/huggingface/lerobot) |
  [文档](https://huggingface.co/docs/lerobot/) |
  [SmolVLA](https://huggingface.co/blog/smolvla) |
  [XLeRobot VLA 指南](https://xlerobot.readthedocs.io/en/latest/software/getting_started/VLA_ACT.html)
- OpenVLA：[代码](https://github.com/openvla/openvla) |
  [项目页](https://openvla.github.io/)
- OpenVLA-OFT：[代码](https://github.com/moojink/openvla-oft) |
  [项目页](https://openvla-oft.github.io/)
- Octo：[代码](https://github.com/octo-models/octo) |
  [项目页](https://octo-models.github.io/)
- RoboAgent：[代码](https://github.com/robopen/roboagent) |
  [项目页](https://robopen.github.io/)
- VIMA：[代码](https://github.com/vimalabs/VIMA) |
  [VIMA-Bench](https://github.com/vimalabs/VimaBench) |
  [项目页](https://vimalabs.github.io/)

### 基础设施

- ManiSkill：[代码](https://github.com/mani-skill/ManiSkill) |
  [文档](https://maniskill.readthedocs.io/en/latest/)
- RoboTwin：[代码](https://github.com/RoboTwin-Platform/RoboTwin) |
  [文档](https://robotwin-platform.github.io/doc/)
- Isaac Lab：[代码](https://github.com/isaac-sim/IsaacLab) |
  [文档](https://isaac-sim.github.io/IsaacLab/)
- Habitat：[Habitat-Lab](https://github.com/facebookresearch/habitat-lab) |
  [Habitat-Sim](https://github.com/facebookresearch/habitat-sim)
- Open X-Embodiment：[代码](https://github.com/google-deepmind/open_x_embodiment) |
  [项目页](https://robotics-transformer-x.github.io/)
- RLBench：[代码](https://github.com/stepjam/RLBench)
