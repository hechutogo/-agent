"""PickParts-v1 交互演示：桌面固定单臂 Panda + 零件A/B + 空盒子。

运行方式：
    python run_demo.py

窗口按键：
    q  退出     r  重置场景     a  开/关随机小动作（观察机器人响应）
鼠标左键旋转视角，右键平移，滚轮缩放。
"""
import os
import sys

# 资产目录（放在 import mani_skill 之前）
os.environ.setdefault("MS_ASSET_DIR", r"D:\trae\111\maniskill_data")

# 未 conda activate 直接运行时，补齐 pinocchio 等原生 DLL 搜索路径；
# conda 版 pinocchio(libomp) 与 torch(libiomp) 共存需放开重复加载
_dll_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
if os.path.isdir(_dll_bin):
    os.environ["PATH"] = _dll_bin + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(_dll_bin)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gymnasium as gym
import numpy as np
import time
import ctypes

import mani_skill.envs  # noqa: F401  注册 ManiSkill 内置环境
import pick_parts_env  # noqa: F401  注册 PickParts-v1
from panda_motion import PandaMotion, MotionPlanError

# Panda 关节位置控制：7 个臂关节绝对目标角 + 1 维归一化夹爪（共 8 维），
# 这是 ManiSkill 官方 motionplanning 示例使用的控制模式，便于直接执行关节轨迹
CONTROL_MODE = "pd_joint_pos"

# ── Windows 消息泵 ─────────────────────────────────────────────────────
# SAPIEN 窗口被最小化时其内部事件泵不再服务 Win32 消息队列，窗口会立刻
# "未响应"，恢复/关闭消息永远得不到处理。尺寸为 0 期间需要我们自己泵消息。
if sys.platform == "win32":
    import ctypes.wintypes

    class _MSG(ctypes.Structure):
        _fields_ = [
            ("hWnd", ctypes.wintypes.HWND),
            ("message", ctypes.wintypes.UINT),
            ("wParam", ctypes.wintypes.WPARAM),
            ("lParam", ctypes.wintypes.LPARAM),
            ("time", ctypes.wintypes.DWORD),
            ("pt", ctypes.wintypes.POINT),
        ]

    _user32 = ctypes.windll.user32

    def _pump_win32_messages():
        """处理当前线程所有待处理的窗口消息，返回是否收到退出请求。"""
        msg = _MSG()
        quit_requested = False
        while _user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE
            if msg.message == 0x0012:  # WM_QUIT
                quit_requested = True
                continue
            _user32.TranslateMessage(ctypes.byref(msg))
            _user32.DispatchMessageW(ctypes.byref(msg))
        return quit_requested
else:
    def _pump_win32_messages():
        return False


def render_retry(env):
    """渲染一帧，返回 viewer；窗口被关闭时返回 None。

    需要处理 SAPIEN 查看器在 Windows 上的两个时序坑：
    1. 窗口刚创建时尺寸可能为 0，原生 render 抛 "failed to resize"；
    2. 窗口最小化（尺寸 0）期间 SAPIEN 自身不泵 Win32 消息，窗口会未响应、
       无法恢复/关闭——此时手动泵 Win32 消息并跳过本帧渲染。
    """
    while True:
        viewer = getattr(env.unwrapped, "_viewer", None)
        if viewer is not None and viewer.window is not None:
            win = viewer.window
            size = win.size
            if size[0] == 0 or size[1] == 0:
                if _pump_win32_messages() or win.should_close:
                    viewer.close()
                    return None
                win.update_render()
                time.sleep(0.03)
                continue
            win.update_render()  # 正常路径：先泵送本帧窗口消息再渲染
            if win.should_close:
                viewer.close()
                return None
        try:
            return env.unwrapped.render_human()
        except RuntimeError as e:
            if "resize" not in str(e):
                raise
            # 首帧竞态：viewer 已建但窗口尺寸还没就绪，泵消息后重试
            if viewer is not None:
                if _pump_win32_messages() or viewer.closed:
                    viewer.close()
                    return None
                viewer.window.update_render()
            time.sleep(0.05)


def hold_action(env):
    """pd_joint_pos 是绝对角度目标，"不动" 的动作 = 当前 7 关节角 + 夹爪保持张开。"""
    qpos = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
    action = np.empty(8, dtype=np.float32)
    action[:7] = qpos[:7]
    action[7] = 1.0
    return action


def run_pick_place(env, part):
    """按键触发的真实抓放演示：part='A' 蓝圆柱 / 'B' 黄方块。"""
    b = env.unwrapped
    if part == "A":
        actor, half_h = b.part_a, pick_parts_env.PART_A_HALF_H
    else:
        actor, half_h = b.part_b, pick_parts_env.PART_B_HALF
    m = PandaMotion(env)
    try:
        z = m.pick_part(actor, half_h)
        print(f"[演示] 零件{part} 抓起后 z={z:.3f}", flush=True)
        m.place(pick_parts_env.BOX_C[0], pick_parts_env.BOX_C[1], target="box")
    except MotionPlanError as e:
        print(f"[演示] 运动规划失败：{e}", flush=True)
        return
    for _ in range(25):
        env.step(hold_action(env))
        render_retry(env)
    in_box = bool(b.evaluate()[f"part_{part}_in_box"][0])
    print(f"[演示] 零件{part} 入盒={in_box}", flush=True)


def main():
    env = gym.make(
        "PickParts-v1",
        num_envs=1,
        obs_mode="rgbd",
        control_mode=CONTROL_MODE,
        render_mode="human",
        sim_backend="cpu",
        render_backend="cpu",   # 本机 torch 为 CPU 版，必须指定，否则 Vulkan-CUDA 互操作崩溃
    )
    obs, info = env.reset(seed=0)
    print(f"场景已启动 | 控制模式: {CONTROL_MODE} | 动作维度: {env.action_space.shape}")
    print("按键: q=退出  r=重置  a=小幅随机扰动  1=抓放零件A演示  2=抓放零件B演示")

    random_motion = False
    while True:
        viewer = render_retry(env)
        if viewer is None:
            break
        if viewer.window.key_press("q"):
            break
        if viewer.window.key_press("r"):
            env.reset()
            random_motion = False
        if viewer.window.key_press("a"):
            random_motion = not random_motion
            print(f"小幅随机扰动: {'开' if random_motion else '关'}")
        if viewer.window.key_press("1"):
            run_pick_place(env, "A")
        if viewer.window.key_press("2"):
            run_pick_place(env, "B")

        if random_motion:
            # 绝对角度控制下的"小动作"= 当前角 ±0.02rad，限制在关节范围内
            qlim = env.unwrapped.agent.robot.get_qlimits()[0].cpu().numpy()
            qpos = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
            action = hold_action(env)
            action[:7] = np.clip(qpos[:7] + np.random.uniform(-0.02, 0.02, 7),
                                 qlim[:7, 0], qlim[:7, 1])
        else:
            action = hold_action(env)
        env.step(action)

    env.close()


if __name__ == "__main__":
    main()
