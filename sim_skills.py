"""机器人技能库（Qwen-Agent BaseTool 封装），桌面固定单臂 Panda 版本。

三个原子技能，对应 LLM 可调用的 "手"：
    move_to  移动夹爪到桌面坐标系下的目标点（真实笛卡尔直线运动）
    pick     夹取零件（A=蓝圆柱 / B=黄方块）：接近-下降-闭合-抬起
    place    把物体放到目标处（盒子/桌面）：平移-下降-张开-撤离

运动学/轨迹细节封装在 panda_motion.PandaMotion（pinocchio 逆解 +
pd_joint_pos 执行，8 维动作：7 绝对关节角 + 1 夹爪）。

约束：最终版只允许向 LLM 暴露相机视觉信息与本体状态，禁止直接读
零件真值坐标；当前阶段 pick/place 内部读取的物体位姿，后续应由
RGB-D 感知模块解算后替代。
"""
import numpy as np
from qwen_agent.tools.base import BaseTool

import pick_parts_env as env_def
from panda_motion import MotionPlanError

# 由 agent_main 启动时注入仿真环境，避免把 env 写死进工具类
ENV = None
MOTION = None


def set_env(env):
    """注入 gym 环境句柄。须在 env.reset() 之后、技能调用之前注入。"""
    global ENV, MOTION
    ENV = env
    MOTION = None  # 新环境需重新构建规划器（运动学模型绑定 articulation）


def _motion():
    global MOTION
    if MOTION is None:
        from panda_motion import PandaMotion
        MOTION = PandaMotion(ENV)
    return MOTION


def _part(part_id):
    benv = ENV.unwrapped
    if part_id == "A":
        return benv.part_a, env_def.PART_A_HALF_H
    return benv.part_b, env_def.PART_B_HALF


def _tcp_report() -> str:
    """读取本体状态：夹爪当前位姿 + 任务成功标志（允许暴露）。"""
    p = np.asarray(ENV.unwrapped.agent.tcp.pose.p[0].cpu())
    info = ENV.unwrapped.evaluate()
    return (
        f"夹爪位置=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})；"
        f"零件A入盒={bool(info['part_A_in_box'][0])}；"
        f"零件B入盒={bool(info['part_B_in_box'][0])}"
    )


class MoveTo(BaseTool):
    name = "move_to"
    description = (
        "移动机械臂夹爪到桌面坐标系下的目标点（真实运动）。"
        "坐标系：桌面高度 z=0，x 指向桌子深处（机械臂正前方），y 向左为正；"
        "机械臂基座固定在 (x=-0.615, y=0)。"
        "零件摆放在 x 约 -0.05~0.10（零件A 的 y 约 -0.18~-0.05，零件B 的 y 约 0.02~0.08），"
        "空盒子中心在 (0.0, 0.25)。"
        "在零件或盒子上方移动时 z 取约 0.21（安全高度）；不要直接把 z 设到桌面。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "目标 x 坐标（米）"},
            "y": {"type": "number", "description": "目标 y 坐标（米）"},
            "z": {"type": "number", "description": "目标 z 坐标（米，桌面为 0）"},
        },
        "required": ["x", "y", "z"],
    }

    def call(self, params, **kwargs):
        p = self._verify_json_format_args(params)
        try:
            err = _motion().move_to(float(p["x"]), float(p["y"]), float(p["z"]))
        except MotionPlanError as e:
            return f"move_to 失败（目标不可达/逆解无解）：{e}。请换一个更近或更高的目标点重试。{_tcp_report()}"
        return f"move_to 完成：夹爪 -> ({p['x']}, {p['y']}, {p['z']})，末端定位误差 {err*100:.1f} cm。{_tcp_report()}"


class Pick(BaseTool):
    name = "pick"
    description = (
        "抓取零件：part='A' 为蓝色圆柱，part='B' 为黄色方块。"
        "会自动完成 张开->移到零件正上方->下降->闭合->抬起 全过程，调用前无需先移动。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "part": {"type": "string", "enum": ["A", "B"], "description": "零件 A（蓝圆柱）或 B（黄方块）"},
        },
        "required": ["part"],
    }

    def call(self, params, **kwargs):
        p = self._verify_json_format_args(params)
        actor, half_h = _part(p["part"])
        try:
            part_z = _motion().pick_part(actor, half_h)
        except MotionPlanError as e:
            return f"pick 失败（运动规划无解）：{e}。{_tcp_report()}"
        lifted = part_z > 0.10
        flag = "成功抓起" if lifted else "未能抓起（夹爪闭合后物体未离开桌面，可能未对准，请重试）"
        return f"pick {flag}：零件 {p['part']}，抓取后物体高度 z={part_z:.2f}。{_tcp_report()}"


class Place(BaseTool):
    name = "place"
    description = "把当前夹持的物体放到目标处并张开夹爪。target='box' 放入空盒子，target='table' 放到桌面当前位置。会自动完成 上方平移->下降->松爪->撤离。"
    parameters = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": ["box", "table"], "description": "放置目标：空盒子或桌面"},
        },
        "required": ["target"],
    }

    def call(self, params, **kwargs):
        p = self._verify_json_format_args(params)
        m = _motion()
        if p["target"] == "box":
            x, y = env_def.BOX_C
        else:
            tcp = ENV.unwrapped.agent.tcp.pose.p[0].cpu().numpy()
            x, y = float(tcp[0]), float(tcp[1])
        try:
            m.place(float(x), float(y), target=p["target"])
        except MotionPlanError as e:
            return f"place 失败（运动规划无解）：{e}。{_tcp_report()}"
        return f"place 完成：物体已放到 {p['target']} 并松开夹爪。{_tcp_report()}"
