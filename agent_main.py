"""具身 Agent 主程序：文字/语音指令 -> Qwen-Agent(DeepSeek) 规划 -> 技能库 -> ManiSkill 仿真。

机器人：桌面固定单臂 Panda（无移动底盘），任务只有抓取零件并放入盒子。

运行方式：
    python agent_main.py                         # 交互输入指令
    python agent_main.py --instruction "把零件A放进盒子"

窗口按键（Agent 执行完后）：q 退出。
"""
import argparse
import os
import sys

# 资产目录必须在 import mani_skill 之前设置
os.environ.setdefault("MS_ASSET_DIR", r"D:\trae\111\maniskill_data")

# 未 conda activate 直接运行时，补齐 pinocchio 等原生 DLL 搜索路径；
# conda 版 pinocchio(libomp) 与 torch(libiomp) 共存需放开重复加载
_dll_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
if os.path.isdir(_dll_bin):
    os.environ["PATH"] = _dll_bin + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(_dll_bin)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
import gymnasium as gym
import numpy as np

import mani_skill.envs  # noqa: F401
import pick_parts_env  # noqa: F401  注册 PickParts-v1
import sim_skills
from sim_skills import MoveTo, Pick, Place

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


# ── 兼容补丁：Qwen-Agent 0.0.34 把工具结果转成 OpenAI 消息时写的是 'id'，
# 而 DeepSeek 严格要求 tool 消息带 'tool_call_id'，否则报 422。
# 在其基类上替换转换函数（不改动 site-packages 源码，升级框架不影响）。
def _patch_oai_messages():
    from qwen_agent.llm.base import BaseChatModel
    from qwen_agent.llm.schema import ASSISTANT, FUNCTION

    def _conv(messages):
        new_messages = []
        for msg in messages:
            if msg["role"] == ASSISTANT:
                if not new_messages or new_messages[-1]["role"] != ASSISTANT:
                    new_messages.append({"role": ASSISTANT, "content": ""})
                if msg.get("content"):
                    new_messages[-1]["content"] = msg["content"]
                if msg.get("function_call"):
                    new_messages[-1].setdefault("tool_calls", []).append({
                        "id": msg.get("extra", {}).get("function_id", "1"),
                        "type": "function",
                        "function": {
                            "name": msg["function_call"]["name"],
                            "arguments": msg["function_call"]["arguments"],
                        },
                    })
            elif msg["role"] == FUNCTION:
                new_messages.append({
                    "role": "tool",
                    "content": msg.get("content") or "",
                    "tool_call_id": msg.get("extra", {}).get("function_id", "1"),
                })
            else:
                new_messages.append(msg)
        return new_messages

    BaseChatModel._conv_qwen_agent_messages_to_oai = staticmethod(_conv)


_patch_oai_messages()

SYSTEM_MESSAGE = """你是一个固定在桌面上的单臂机器人 Panda 的操作大脑（你没有移动底盘，不需要也不能移动自身）。
场景：你正前方的桌面上有零件A（蓝色圆柱，在你的右前方）和零件B（黄色方块，在你的左前方），桌面远侧左角有一个空盒子。
你可以调用三个技能：pick（夹取零件，自动完成接近、下降、闭合、抬起）、place（放置并松爪，自动完成平移、下降、松开、撤离）、move_to（把夹爪移动到指定坐标，一般不需要单独调用）。
完成 "把零件X放进盒子" 的标准顺序是：pick(part=X) -> place(target=box)，两步即可，不要多余调用。
每次工具返回后，根据结果决定下一步；任务完成后用中文简短报告结果。
必须以工具返回的 "零件A入盒/零件B入盒=True" 作为任务成功的唯一依据；
若为 False，绝不允许声称任务成功，要如实说明当前状态。
注意：你目前只能看到工具返回的本体状态信息，看不到真实相机画面，坐标按工具描述中的粗略范围规划。"""

CONTROL_MODE = "pd_joint_pos"


def make_env():
    env = gym.make(
        "PickParts-v1",
        num_envs=1,
        obs_mode="rgbd",
        control_mode=CONTROL_MODE,
        render_mode="human",
        sim_backend="cpu",
        render_backend="cpu",
    )
    env.reset(seed=0)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", "-i", type=str, default=None, help="任务指令，不传则交互输入")
    args = parser.parse_args()
    instruction = args.instruction or input("请输入任务指令：").strip()
    if not instruction:
        print("指令为空，退出。")
        return

    env = make_env()
    sim_skills.set_env(env)

    llm_cfg = {
        "model": os.environ["LLM_MODEL"],
        "model_type": "oai",  # OpenAI 兼容接口（DeepSeek）
        "model_server": os.environ["LLM_BASE_URL"],
        "api_key": os.environ["LLM_API_KEY"],
        # DeepSeek 用 OpenAI 原生 function-calling 协议返回 tool_calls，
        # 不开这个 Qwen-Agent 会走文本模板，DeepSeek 回 DSML 标签导致工具不执行
        "generate_cfg": {"use_raw_api": True},
    }
    from qwen_agent.agents import Assistant
    agent = Assistant(
        llm=llm_cfg,
        function_list=[MoveTo(), Pick(), Place()],
        system_message=SYSTEM_MESSAGE,
    )

    print(f"\n用户指令：{instruction}\n" + "=" * 60, flush=True)
    messages = [{"role": "user", "content": instruction}]
    final = ""
    for responses in agent.run(messages=messages):
        msg = responses[-1]  # 每轮 yield 的最后一条消息
        role = getattr(msg, "role", "") or msg.get("role", "")
        content = getattr(msg, "content", "") or msg.get("content", "")
        fc = getattr(msg, "function_call", None)
        if fc:
            # LLM 决定调用技能（FunctionCall 消息）
            print(f"\n>> 调用技能 {fc.name}({fc.arguments})", flush=True)
        elif role == "function" and content:
            # 技能执行完返回的结果（Qwen-Agent 里工具角色名为 function）
            print(f"<< 技能返回：{content}", flush=True)
        elif role == "assistant" and content:
            # 思考/叙述文本：单行刷新，避免流式分片刷屏
            print(f"\r思考中：{content[-50:]}", end="", flush=True)
        final = content
    print("\n" + "=" * 60 + f"\n最终回复：{final}\n仿真窗口保持打开，按 q 退出。", flush=True)

    # 保持窗口，便于查看执行后的场景；最小化/首帧尺寸为 0 等情况由
    # render_retry 内部处理（Win32 消息泵 + 跳过零尺寸帧）
    from run_demo import render_retry
    try:
        while True:
            viewer = render_retry(env)
            if viewer is None:
                break
            if viewer.window.key_press("q"):
                break
            # pd_joint_pos 为绝对角度目标：保持当前关节角，避免机械臂漂走
            qpos = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
            hold = np.empty(env.action_space.shape[-1], dtype=np.float32)
            hold[:7] = qpos[:7]
            hold[7] = 1.0
            env.step(hold)
    except RuntimeError:
        pass
    env.close()


if __name__ == "__main__":
    main()
