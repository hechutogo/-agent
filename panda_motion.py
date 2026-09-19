"""Panda 运动执行层：pinocchio CLIK 逆解 + 笛卡尔直线路径。

为什么不用 mplib：mplib 官方只提供 Ubuntu 预编译 wheel（nightly 也无 Windows
版），本机 Windows 无法 pip 安装。这里用 SAPIEN 自带的 pinocchio 运动学模型
实现 ManiSkill 官方 motionplanning 示例中等价的核心能力：

- 逆解：阻尼最小二乘 CLIK（PinocchioModel.compute_inverse_kinematics）
- 路径：固定"自上而下"抓取姿态，在笛卡尔空间走直线、逐点 IK 暖启动
  （等价于官方 plan_screw 的简化版本；本场景只有桌面抓放，不需要 RRT
   避障——统一在安全高度平移即可越过零件和盒壁）
- 执行：pd_joint_pos 绝对关节角目标 + 归一化夹爪指令（+1 张开 / -1 闭合），
  与官方 examples/motionplanning 的动作布局完全一致（8 维）

坐标系：环境世界坐标（桌面 z=0）。
"""
import os
import numpy as np
import sapien
from transforms3d.quaternions import mat2quat, qconjugate, qmult, rotate_vector

# 夹爪归一化指令（Panda mimic 控制器：+1=张开到 0.04，-1=闭合到 -0.01 保持夹持力）
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

APPROACHING = np.array([0.0, 0.0, -1.0])  # 自上而下抓取
STEP_M = 0.006          # 笛卡尔路径点间距（米）：约 0.12 m/s（控制周期 0.05s）
MAX_DQ = 0.06           # 单个控制步最大关节增量（弧度），超出则细分插值
SETTLE_STEPS = 3        # 到达目标后保持的控制步数（等 PD 收敛）
GRIP_STEPS = 8          # 夹爪开/合保持步数
SAFE_Z = 0.21           # 水平平移安全高度：指尖在 TCP 下方约 0.1m，须高于最高零件顶 0.08m
BOX_FLOOR_TOP_Z = 0.006  # 盒子底板上表面高度


class MotionPlanError(RuntimeError):
    """逆解失败或目标不可达。"""


class PandaMotion:
    def __init__(self, env):
        self.env = env
        benv = env.unwrapped
        self.agent = benv.agent
        self.robot = self.agent.robot
        self.tcp = self.agent.tcp
        # 从已加载的 articulation 导出运动学模型，坐标与仿真零偏差
        self.pin = self.robot.create_pinocchio_model()
        self.qlim = self.robot.get_qlimits()[0].cpu().numpy()  # (9, 2)
        self.gripper = GRIPPER_OPEN
        self.last_part_half_h = 0.03
        self._render_enabled = True

        # 固定顶抓姿态 R=[ortho, closing, approaching]（与官方
        # agent.build_grasp_pose 同一约定）；closing 取初始 TCP 的水平 y 轴
        r0 = self.tcp.pose.to_transformation_matrix()[0, :3, :3].cpu().numpy()
        closing = r0[:, 1].copy()
        closing[2] = 0.0
        closing /= max(np.linalg.norm(closing), 1e-8)
        ortho = np.cross(closing, APPROACHING)
        self.r_grasp = np.stack([ortho, closing, APPROACHING], axis=1)
        self.q_grasp = mat2quat(self.r_grasp)  # wxyz

    # ── 坐标变换 / IK ──────────────────────────────────────────────────
    def _world_to_base(self, p_world, q_world=None):
        """世界位姿 -> articulation base 坐标系（IK 要求在 base 系下）。"""
        bp = self.robot.pose.p[0].cpu().numpy()
        bq = self.robot.pose.q[0].cpu().numpy()
        p_local = rotate_vector(np.asarray(p_world, float) - bp, qconjugate(bq))
        q_local = qconjugate(bq) if q_world is None else qmult(qconjugate(bq), q_world)
        return sapien.Pose(p=p_local, q=q_local)

    def _ik(self, p_world, q_init, max_iterations=300, dt=0.2):
        """对世界坐标点求 7 轴关节角（手指保持不动），暖启动。"""
        target = self._world_to_base(p_world, self.q_grasp)
        mask = np.array([1] * 7 + [0] * 2)
        q, ok, err = self.pin.compute_inverse_kinematics(
            self.tcp.index,
            target,
            initial_qpos=q_init,
            active_qmask=mask,
            eps=1e-4,
            max_iterations=max_iterations,
            dt=dt,
            damp=1e-6,
        )
        q = np.asarray(q, dtype=np.float64)
        q[:7] = np.clip(q[:7], self.qlim[:7, 0], self.qlim[:7, 1])
        return q, bool(ok), float(err)

    # ── 仿真执行 ───────────────────────────────────────────────────────
    def _render(self):
        """有人机交互窗口时逐帧刷新；无窗口（离屏测试）时跳过。"""
        if not self._render_enabled or self.env.unwrapped.viewer is None:
            return
        try:
            from run_demo import render_retry
            if render_retry(self.env) is None:
                self._render_enabled = False
        except Exception:
            pass

    def _send(self, q_arm, repeat=1):
        """发送一帧（或重复若干帧）pd_joint_pos 动作：7 绝对关节角 + 夹爪。"""
        action = np.empty(8, dtype=np.float32)
        action[:7] = np.asarray(q_arm, dtype=np.float32)
        action[7] = self.gripper
        for _ in range(max(1, repeat)):
            self.env.step(action)
            self._render()

    # ── 原子动作 ───────────────────────────────────────────────────────
    def move_to(self, x, y, z):
        """TCP 以固定顶抓姿态沿笛卡尔直线移动到世界坐标点。"""
        p0 = self.tcp.pose.p[0].cpu().numpy().astype(np.float64)
        p1 = np.array([x, y, z], dtype=np.float64)
        dist = float(np.linalg.norm(p1 - p0))
        n = max(1, int(np.ceil(dist / STEP_M)))
        ts = np.arange(1, n + 1) / n

        # 逐点 IK（以上一点的解暖启动）
        q_prev = self.robot.get_qpos()[0].cpu().numpy()
        arm_waypoints = [q_prev[:7].copy()]
        for t in ts:
            pt = (1.0 - t) * p0 + t * p1
            q, ok, err = self._ik(pt, q_prev)
            if not ok:
                raise MotionPlanError(
                    f"IK 失败：目标=({x:.3f},{y:.3f},{z:.3f})，路径进度 t={t:.2f}，误差={err:.4f}"
                )
            arm_waypoints.append(q[:7])
            q_prev = q

        # 按关节速度上限细分相邻目标点
        dense = []
        for a, b in zip(arm_waypoints[:-1], arm_waypoints[1:]):
            k = max(1, int(np.ceil(np.abs(b - a).max() / MAX_DQ)))
            for j in range(1, k + 1):
                dense.append(a + (b - a) * (j / k))

        min_z = 1e9
        for q in dense:
            self._send(q)
            min_z = min(min_z, float(self.tcp.pose.p[0, 2].cpu()))
        self._send(dense[-1], repeat=SETTLE_STEPS)

        if os.environ.get("PANDA_DEBUG"):
            jumps = [float(np.abs(b - a).max())
                     for a, b in zip(arm_waypoints[:-1], arm_waypoints[1:])]
            print(f"[dbg move_to] ({p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f})->({x:.3f},{y:.3f},{z:.3f}) "
                  f"笛卡尔点={len(arm_waypoints)-1} 最大关节跳变={max(jumps):.3f}rad "
                  f"执行中TCP最低z={min_z:.3f}（直线最低={min(p0[2], z):.3f}）", flush=True)

        p = self.tcp.pose.p[0].cpu().numpy()
        return float(np.linalg.norm(p - p1))

    def set_gripper(self, open_gripper: bool):
        self.gripper = GRIPPER_OPEN if open_gripper else GRIPPER_CLOSE
        q = self.robot.get_qpos()[0].cpu().numpy()[:7]
        self._send(q, repeat=GRIP_STEPS)

    def pick_part(self, actor, half_h):
        """完整抓取序列：张开 -> 到正上方 -> 下降到物体中心 -> 闭合 -> 抬起。

        actor: 仿真中的物体 Actor（实时读取位姿）；
        half_h: 物体半高（圆柱 half_length / 方块 half_size），决定下降高度。
        返回抓取抬起后物体的实时高度，调用方据此判断是否真的抓起。
        """
        p = np.asarray(actor.pose.p[0].cpu(), dtype=np.float64)
        x, y = p[0], p[1]
        self.last_part_half_h = float(half_h)
        self.set_gripper(True)          # 先张开
        self.move_to(x, y, SAFE_Z)     # 水平修正到正上方（安全高度）
        self.move_to(x, y, half_h)     # 竖直下降，TCP 到物体中心
        self.set_gripper(False)        # 闭合夹持
        self.move_to(x, y, SAFE_Z)     # 竖直抬起
        return float(actor.pose.p[0, 2].cpu())

    def place(self, x, y, target="box"):
        """放置序列：目标上方 -> 下降到放置高度 -> 张开 -> 抬起撤离。"""
        if target == "box":
            z_release = BOX_FLOOR_TOP_Z + self.last_part_half_h + 0.01
        else:
            z_release = self.last_part_half_h + 0.02
        self.move_to(x, y, SAFE_Z + 0.02)
        self.move_to(x, y, z_release)
        self.set_gripper(True)
        self.move_to(x, y, SAFE_Z + 0.02)
