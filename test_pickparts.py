"""PickParts-v1 离屏测试：真实运动规划抓放验证（无窗口，CPU）。

对多个随机种子分别执行 零件A/零件B 的完整 pick -> place(box) 序列，
检查物体是否被真正抓起并最终进入盒子。
用法：
    python test_pickparts.py            # 测零件A和B（各若干种子）
    python test_pickparts.py --part A   # 只测零件A
"""
import argparse
import os
import sys

os.environ.setdefault("MS_ASSET_DIR", r"D:\trae\111\maniskill_data")

# 未 conda activate 直接运行时，补齐 pinocchio 等原生 DLL 搜索路径
_dll_bin = os.path.join(os.path.dirname(sys.executable), "Library", "bin")
if os.path.isdir(_dll_bin):
    os.environ["PATH"] = _dll_bin + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(_dll_bin)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gymnasium as gym
import numpy as np
import mani_skill.envs  # noqa: F401
import pick_parts_env  # noqa: F401  注册 PickParts-v1
from panda_motion import PandaMotion, MotionPlanError


def run_one(seed, part):
    env = gym.make(
        "PickParts-v1", obs_mode="none", render_mode="rgb_array",
        control_mode="pd_joint_pos",
        sim_backend="cpu", render_backend="cpu",
    )
    env.reset(seed=seed)
    benv = env.unwrapped
    actor = benv.part_a if part == "A" else benv.part_b
    half_h = pick_parts_env.PART_A_HALF_H if part == "A" else pick_parts_env.PART_B_HALF
    p0 = actor.pose.p[0].cpu().numpy()
    print(f"[seed={seed}] 零件{part} 初始位置=({p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f})", flush=True)

    motion = PandaMotion(env)
    print(f"[seed={seed}] TCP 初始={benv.agent.tcp.pose.p[0].cpu().numpy().round(3)}", flush=True)
    part_z = motion.pick_part(actor, half_h)
    print(f"[seed={seed}] 抓取后物体 z={part_z:.3f}（>{0.10} 视为抓起成功）", flush=True)
    motion.place(pick_parts_env.BOX_C[0], pick_parts_env.BOX_C[1], target="box")
    # 多等一会让物体落稳
    hold = np.empty(8, dtype=np.float32)
    hold[:7] = benv.agent.robot.get_qpos()[0, :7].cpu().numpy()
    hold[7] = 1.0
    for _ in range(25):
        env.step(hold)
    info = benv.evaluate()
    in_box = bool(info[f"part_{part}_in_box"][0])
    pf = actor.pose.p[0].cpu().numpy()
    print(f"[seed={seed}] 最终物体位置=({pf[0]:.3f},{pf[1]:.3f},{pf[2]:.3f}) 入盒={in_box}", flush=True)
    env.close()
    return in_box, part_z > 0.10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["A", "B", "both"], default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 7])
    args = parser.parse_args()

    parts = ["A", "B"] if args.part == "both" else [args.part]
    results = []
    for part in parts:
        for seed in args.seeds:
            try:
                in_box, lifted = run_one(seed, part)
            except MotionPlanError as e:
                print(f"[seed={seed}] 零件{part} 规划失败：{e}", flush=True)
                in_box, lifted = False, False
            results.append((part, seed, lifted, in_box))

    print("\n==== 汇总 ====", flush=True)
    for part, seed, lifted, in_box in results:
        print(f"零件{part} seed={seed}: 抓起={'✓' if lifted else '✗'} 入盒={'✓' if in_box else '✗'}", flush=True)
    n_ok = sum(1 for *_, b in results if b)
    print(f"入盒成功率：{n_ok}/{len(results)}", flush=True)
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
