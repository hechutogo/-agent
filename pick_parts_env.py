"""PickParts-v1 自定义环境：桌面固定单臂 Panda + 零件A/B + 空盒子。

任务：按指令抓取指定零件（A=蓝圆柱 / B=黄方块）放入空盒子。
成功条件：目标零件进入盒子开口范围内。

坐标系（ManiSkill 标准桌面场景，TableSceneBuilder）：
    桌面高度 z=0；x 轴指向桌面深处（机械臂正前方），y 轴向左为正；
    Panda 基座固定在桌子近侧边缘 (x=-0.615, y=0)，不需要移动。
"""
from typing import Union

import math

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# ══════════════════════════════════════════════════════════════════
# 场景布局参数（桌面坐标 z=0；零件区和盒子都在 Panda 可达范围内）
# ══════════════════════════════════════════════════════════════════
PART_A_X = (-0.05, 0.10)   # 零件A 蓝圆柱 随机 x 范围
PART_A_Y = (-0.18, -0.05)  # 零件A 随机 y 范围（机械臂右侧）
PART_B_X = (-0.05, 0.10)   # 零件B 黄方块 随机 x 范围
PART_B_Y = (0.02, 0.08)    # 零件B 黄方块 随机 y 范围（盒墙近边 y≈0.14，指尖外探 5cm，必须 <0.09）
PART_A_HALF_H = 0.04       # 零件A：蓝圆柱 半高/半径
PART_A_R = 0.03
PART_B_HALF = 0.025        # 零件B：黄方块 半边长
BOX_C = [0.0, 0.25]        # 盒子中心 (x, y)：桌面远侧左角，Panda 可达
BOX_IN = 0.055             # 盒子内腔半宽
WALL_T = 0.006             # 壁厚
WALL_H = 0.03              # 壁高


@register_env("PickParts-v1", max_episode_steps=400)
class PickPartsEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        kwargs.setdefault("reward_mode", "sparse")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # 桌面斜上方俯视相机（模拟外部固定视角，仅视觉信息）
        pose = sapien_utils.look_at(eye=[0.3, 0.0, 0.6], target=[-0.1, 0.0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 1.0, 1.0], target=[0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # Panda 的基座位姿由 TableSceneBuilder.initialize 自动设置，无需手动摆放

        # 空盒子：底 + 四壁（kinematic，稳定），用 ActorBuilder 支持任意半尺寸
        def make_plate(name, center, half_extents, color):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(sapien.Pose(), half_extents)
            builder.add_box_visual(sapien.Pose(), half_extents, material=sapien.render.RenderMaterial(base_color=color))
            builder.set_initial_pose(sapien.Pose(p=center))
            return builder.build_kinematic(name=name)

        bx, by = BOX_C
        gray = [0.4, 0.4, 0.45, 1]
        self.box_bottom = make_plate(
            "box_bottom", [bx, by, 0.003],
            [BOX_IN + WALL_T, BOX_IN + WALL_T, 0.003], gray,
        )
        self.box_walls = [
            make_plate(f"box_wall{i}", p, hs, gray)
            for i, (p, hs) in enumerate([
                ([bx, by - BOX_IN - WALL_T / 2, 0.006 + WALL_H / 2], [BOX_IN * 2 + WALL_T * 3, WALL_T, WALL_H]),
                ([bx, by + BOX_IN + WALL_T / 2, 0.006 + WALL_H / 2], [BOX_IN * 2 + WALL_T * 3, WALL_T, WALL_H]),
                ([bx - BOX_IN - WALL_T / 2, by, 0.006 + WALL_H / 2], [WALL_T, BOX_IN * 2, WALL_H]),
                ([bx + BOX_IN + WALL_T / 2, by, 0.006 + WALL_H / 2], [WALL_T, BOX_IN * 2, WALL_H]),
            ])
        ]

        # 零件A：蓝圆柱。SAPIEN 圆柱碰撞体/视觉体轴沿本地 +X（PhysX 约定），
        # 需绕 Y 轴 -90° 把轴转到世界 +Z 直立摆放（q 为 wxyz）。
        cyl_q = [math.cos(-math.pi / 4), 0.0, math.sin(-math.pi / 4), 0.0]
        self._cyl_q = torch.tensor(cyl_q, dtype=torch.float32)
        self.part_a = actors.build_cylinder(
            self.scene, radius=PART_A_R, half_length=PART_A_HALF_H,
            color=[0.15, 0.35, 1.0, 1], name="part_A",
            initial_pose=sapien.Pose(p=[0.0, -0.1, PART_A_HALF_H], q=cyl_q),
        )
        # 零件B：黄方块
        self.part_b = actors.build_cube(
            self.scene, half_size=PART_B_HALF, color=[1.0, 0.85, 0.1, 1],
            name="part_B",
            initial_pose=sapien.Pose(p=[0.0, 0.08, PART_B_HALF]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # 零件随机摆放在 Panda 正前方桌面
            xyz_a = torch.zeros((b, 3))
            xyz_a[:, 0] = torch.rand((b,)) * (PART_A_X[1] - PART_A_X[0]) + PART_A_X[0]
            xyz_a[:, 1] = torch.rand((b,)) * (PART_A_Y[1] - PART_A_Y[0]) + PART_A_Y[0]
            xyz_a[:, 2] = PART_A_HALF_H
            # 圆柱固定直立（绕 Y -90° 把本地 +X 轴转到 +Z），旋转对称无需随机朝向
            qa = self._cyl_q.unsqueeze(0).expand(b, 4).clone()
            self.part_a.set_pose(Pose.create_from_pq(xyz_a, qa))

            xyz_b = torch.zeros((b, 3))
            xyz_b[:, 0] = torch.rand((b,)) * (PART_B_X[1] - PART_B_X[0]) + PART_B_X[0]
            xyz_b[:, 1] = torch.rand((b,)) * (PART_B_Y[1] - PART_B_Y[0]) + PART_B_Y[0]
            xyz_b[:, 2] = PART_B_HALF
            self.part_b.set_pose(Pose.create_from_pq(xyz_b, randomization.random_quaternions(b, lock_x=True, lock_y=True)))

    def _get_obs_extra(self, info: dict):
        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            part_A_in_box=info["part_A_in_box"],
            part_B_in_box=info["part_B_in_box"],
        )

    def _part_in_box(self, part) -> torch.Tensor:
        p = part.pose.p
        bx, by = BOX_C
        in_xy = (torch.abs(p[:, 0] - bx) < BOX_IN) & (torch.abs(p[:, 1] - by) < BOX_IN)
        in_z = (p[:, 2] > 0.0) & (p[:, 2] < 0.08)
        return in_xy & in_z

    def evaluate(self):
        return {
            "part_A_in_box": self._part_in_box(self.part_a),
            "part_B_in_box": self._part_in_box(self.part_b),
            "success": self._part_in_box(self.part_a) | self._part_in_box(self.part_b),
        }
