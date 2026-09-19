"""ManiSkill 环境验证脚本：创建 PickCube-v1，离屏渲染并保存一帧 PNG。"""
import os

os.environ.setdefault("MS_ASSET_DIR", r"D:\trae\111\maniskill_data")

import gymnasium as gym
import mani_skill.envs  # noqa: F401  注册 ManiSkill 环境
import numpy as np

RENDER_PNG = r"D:\trae\111\verify_render.png"


def main():
    print("创建 PickCube-v1 环境 (cpu, rgb_array 离屏渲染)...")
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        render_mode="rgb_array",
        sim_backend="cpu",
        render_backend="cpu",
    )
    obs, info = env.reset(seed=42)
    print(f"reset 成功: obs 键 = {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")

    total_reward = 0.0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
    print(f"随机动作 50 步完成, 累计奖励 = {total_reward:.3f}")

    img = env.render()
    if img is not None:
        arr = np.asarray(img)
        print(f"渲染输出 shape = {arr.shape}, dtype = {arr.dtype}")
        # 保存 base RGB 帧
        frame = arr.squeeze()
        if frame.ndim == 4:  # 多相机时取第一个
            frame = frame[0]
        try:
            from PIL import Image
            Image.fromarray(frame[..., :3].astype("uint8")).save(RENDER_PNG)
            print(f"已保存渲染帧: {RENDER_PNG}")
        except ImportError:
            # 无 PIL 时用 imageio 或纯 numpy 回退
            try:
                import imageio
                imageio.imwrite(RENDER_PNG, frame[..., :3].astype("uint8"))
                print(f"已保存渲染帧(imageio): {RENDER_PNG}")
            except ImportError:
                np.save(RENDER_PNG.replace(".png", ".npy"), frame[..., :3])
                print(f"未安装 PIL/imageio, 已保存 .npy: {RENDER_PNG.replace('.png', '.npy')}")
    else:
        print("警告: render() 返回 None")

    env.close()
    print("验证完成 ✓")


if __name__ == "__main__":
    main()
