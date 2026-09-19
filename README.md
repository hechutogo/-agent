# PickParts：LLM 指令驱动的 ManiSkill3 抓放 Demo

在 ManiSkill3 仿真中，用一句中文指令指挥桌面 **Panda 单臂**把指定零件抓进盒子：

> "把零件A放进盒子里" → LLM 规划 `pick(A) → place(box)` → 机械臂在仿真中真实抓放 → 回报结果

- **零件 A**：蓝色圆柱；**零件 B**：黄色方块；目标：桌面上的固定盒子
- **纯 CPU 可运行**（无需 GPU、无需显卡渲染），Windows 实测通过
- 大脑：Qwen-Agent + 任意 OpenAI 兼容 LLM（默认 DeepSeek）；小脑：pinocchio 逆解 + 笛卡尔直线运动规划

架构为四层：**大脑（LLM/Agent）— 技能层（move_to/pick/place）— 运动层（CLIK）— 身体（ManiSkill/SAPIEN）**，详见 [docs/技术文档.md](docs/技术文档.md)。

---

## 1. 目录结构

| 文件 | 作用 |
|---|---|
| `pick_parts_env.py` | 自定义 gym 环境 `PickParts-v1`：桌面、Panda、两个零件、盒子、入盒判定 |
| `panda_motion.py` | 运动层：pinocchio CLIK 逆解 + 笛卡尔直线 + 夹爪时序 |
| `sim_skills.py` | 三个 LLM 可调用技能：MoveTo / Pick / Place |
| `agent_main.py` | Agent 入口：指令 → LLM 规划 → 技能执行 → 汇报 |
| `run_demo.py` | 图形窗口演示（无需 LLM，按键直接触发抓放） |
| `test_pickparts.py` | 离屏批量回归测试（无窗口、无需 API Key） |
| `verify_maniskill.py` | ManiSkill 装机冒烟测试（官方 PickCube 跑 50 步） |
| `test_llm.py` | LLM API 连通性自检 |
| `启动仿真.bat` | Windows 一键启动 run_demo（**路径为本机绝对路径，需自行修改**） |
| `.env.example` | 环境变量模板（复制为 `.env` 后填 Key） |
| `docs/` | 技术文档、技术选型方案、执行记录 |

> 本项目未修改任何开源库源码；对 Qwen-Agent 的兼容处理是在 `agent_main.py` 中运行时打补丁。

---

## 2. 环境要求

- **操作系统**：Windows 10/11（代码也兼容 Linux/macOS，但安装步骤以 Windows 为例）
- **Python**：3.10（推荐用 Miniforge/conda 管理）
- **硬件**：无需 GPU；纯 CPU torch 即可运行（渲染也走 CPU）
- 磁盘：环境约 5 GB

---

## 3. 安装步骤

### 3.1 安装 Miniforge 并创建 Python 3.10 环境

下载 Miniforge（conda 的开源发行版）：**https://github.com/conda-forge/miniforge/releases**
（选 `Miniforge3-Windows-x86_64.exe`；也可用 Miniconda：https://docs.conda.io/en/latest/miniconda.html ）

```powershell
conda create -n maniskill python=3.10 -y
conda activate maniskill
```

> ⚠️ 如果系统里残留 `PYTHONHOME` / `PYTHONPATH` 环境变量（尤其指向已不存在的旧 Anaconda 目录），Python 会报 `No module named 'encodings'`。每个新终端先执行：
> `$env:PYTHONHOME=$null; $env:PYTHONPATH=$null`（PowerShell）

### 3.2 安装 CPU 版 PyTorch

官网（可选版本）：**https://pytorch.org/get-started/locally/**

```powershell
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

先装 CPU 版 torch，再装 ManiSkill，可避免 pip 拉取错误的 CUDA 版本。

### 3.3 安装 ManiSkill（仿真框架）

- 官网/文档：https://maniskill.readthedocs.io
- 源码：**https://github.com/haosulab/ManiSkill**（底层物理/渲染引擎 SAPIEN：https://github.com/haosulab/SAPIEN ）

```powershell
pip install mani_skill==3.0.1
```

本项目实测版本：`mani_skill 3.0.1`、`sapien 3.0.3`、`gymnasium 1.3.0`。

**机器人资产**：首次运行时 ManiSkill 会自动下载 Panda 等机器人资产。建议指定一个本地目录：

```powershell
setx MS_ASSET_DIR "D:\your\path\maniskill_data"   # 设置后重开终端
```

国内网络下载 GitHub 资产失败时，可开启系统代理或使用 GitHub 加速地址手动下载后解压到该目录。

### 3.4 安装 pinocchio（逆解运动学，必须用 conda 安装）

conda-forge 页面：**https://anaconda.org/conda-forge/pinocchio**

```powershell
conda install -c conda-forge pinocchio=4.1.0 -y
```

> ⚠️ PyPI 上的 `pinocchio`（0.4.3）是**同名无关假包**，切勿用 pip 安装。
> 若导入时遇到 OpenMP 重复加载报错（`OMP: Error #15`），设置环境变量 `KMP_DUPLICATE_LIB_OK=TRUE`（各入口脚本已内置默认值）。

### 3.5 安装 Agent 框架与其余依赖

- Qwen-Agent 源码：**https://github.com/QwenLM/Qwen-Agent**

```powershell
pip install qwen-agent==0.0.34 openai==3.16.2 transforms3d==0.4.2 python-dotenv==1.2.3 soundfile==0.14.0 imageio==2.37.4
```

实测版本：qwen-agent 0.0.34、openai 3.16.2、transforms3d 0.4.2、python-dotenv 1.2.3、soundfile 0.14.0、imageio 2.37.4、numpy 2.2.6。

> 注：`soundfile` 是 qwen-agent 0.0.34 未声明依赖但会无条件 import 的包，需手动安装。

### 3.6 配置 LLM API Key

复制模板并填入你自己的 Key：

```powershell
copy .env.example .env
```

编辑 `.env`（该文件已被 `.gitignore` 忽略，不会提交）：

```
LLM_API_KEY=sk-你的key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

- 默认使用 DeepSeek，Key 在 **https://platform.deepseek.com** 注册获取；
- 也可换成任何 **OpenAI 兼容**接口（如本地 vLLM、其他云厂商），改这三个变量即可，无需改代码。

---

## 4. 运行方式

所有 Python 命令均在已 `conda activate maniskill` 的终端、项目根目录下执行。

### 4.1 先做两项自检（推荐）

```powershell
python verify_maniskill.py   # 仿真器装机冒烟：官方 PickCube 离屏 50 步（无需 Key、无窗口）
python test_llm.py           # LLM API 连通性：成功打印模型回复
```

### 4.2 Agent 端到端：说一句话，机械臂照做

```powershell
python agent_main.py --instruction "把零件A放进盒子里"
python agent_main.py -i "把零件B放进盒子里"
python agent_main.py          # 不传参数则交互输入指令
```

执行完仿真窗口保持打开，按 **q** 退出。

### 4.3 纯图形演示（不调用 LLM，直接看抓放）

```powershell
python run_demo.py
```

窗口按键：**1** = 抓放零件 A（蓝圆柱）｜**2** = 抓放零件 B（黄方块）｜**r** = 重置场景｜**a** = 小幅随机扰动开关｜**q** = 退出

### 4.4 离屏回归测试（无窗口、无需 API Key）

```powershell
python test_pickparts.py --part both --seeds 3 7 11 21
```

`--part A|B|both` 选择零件，`--seeds` 给随机种子；全部抓起且入盒则退出码为 0。适合改代码后验证。

---

## 5. Windows 常见问题

| 现象 | 处理 |
|---|---|
| `No module named 'encodings'` | 清除残留的 `PYTHONHOME/PYTHONPATH`（见 3.1） |
| 一渲染就崩溃（0xC0000005） | 本项目已强制 `sim_backend="cpu", render_backend="cpu"`；自行调 gym.make 时也要显式指定 |
| `import pinocchio` 失败 / DLL 找不到 | 必须用 3.4 的 conda 命令安装；入口脚本已内置 `Library\bin` 的 DLL 自举，勿删文件头部代码块 |
| OMP Error #15 | `$env:KMP_DUPLICATE_LIB_OK="TRUE"`（脚本已设默认值） |
| 窗口最小化后"未响应" | 已知 SAPIEN/Windows 问题，`run_demo.py` 的 `render_retry` 已处理；恢复窗口即可 |
| 资产下载失败 | 设置 `MS_ASSET_DIR`，挂代理或用 GitHub 加速下载 |
| 双击 `启动仿真.bat` 跑不起来 | 该文件内是本机绝对路径（解释器与脚本路径），按你的实际路径修改，或直接用上面的 python 命令 |

---

## 6. 进一步阅读（docs/）

- [技术文档.md](docs/技术文档.md)：仓库结构、场景/动作/技能/Agent 设计、如何改进 Agent、如何新增技能（含示例）
- [技术选型方案.md](docs/技术选型方案.md)：为什么选 ManiSkill3/Qwen-Agent，与 Isaac Sim、MuJoCo、LangGraph 等的对比
- [执行记录.md](docs/执行记录.md)：从零搭建到跑通的完整过程、踩坑记录与环境铁律

## 7. 致谢与许可

- 仿真框架 [ManiSkill](https://github.com/haosulab/ManiSkill)（Apache-2.0）、[SAPIEN](https://github.com/haosulab/SAPIEN)
- Agent 框架 [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)、机器人模型 Franka Panda（ManiSkill 内置资产）
- 本项目代码供学习与研究使用。
