"""ManiSkill environment and the narrow sensor/actuation boundary."""
from ..runtime import configure
configure()

import numpy as np
import time
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from .perception import Frame
from .robots.xlerobot import PickPartsXLeRobot, RIGHT, LEFT, HEAD
from .objects import _SceneObjects


class PickPartsEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["pickparts_xlerobot"]

    def __init__(self, scene_objects, **kwargs):
        self._scene_objects = scene_objects
        super().__init__(robot_uids="pickparts_xlerobot", obs_mode="rgbd",
                         control_mode="pd_joint_pos", reward_mode="none",
                         sim_backend="cpu", render_backend="cpu", **kwargs)

    @property
    def _default_sensor_configs(self):
        return [CameraConfig("workspace", sapien_utils.look_at(
            [.45, -.90, 1.25], [0, -.25, .75]), 640, 480, .95, .01, 5.)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig("overview", sapien_utils.look_at(
            [1.7, -1.6, 1.55], [0, 0, .65]), 960, 720, 1., .01, 10.)

    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose())

    def _load_scene(self, options):
        b = self.scene.create_actor_builder()
        b.add_box_collision(half_size=[.30, .22, .025])
        b.add_box_visual(half_size=[.30, .22, .025],
                         material=[.55, .49, .40, 1])
        b.initial_pose = sapien.Pose([0, -.45, .695])
        b.build_static("table")
        for x in [-.25, .25]:
            for y in [-.62, -.28]:
                leg = self.scene.create_actor_builder()
                leg.add_box_collision(half_size=[.015, .015, .335])
                leg.add_box_visual(half_size=[.015, .015, .335],
                                   material=[.25, .26, .28, 1])
                leg.initial_pose = sapien.Pose([x, y, .335])
                leg.build_static(f"table_leg_{x}_{y}")
        self._scene_objects.build(self.scene)
        b = self.scene.create_actor_builder()
        b.add_box_collision(half_size=[2, 2, .02])
        b.add_box_visual(half_size=[2, 2, .02], material=[.19, .22, .26, 1])
        b.initial_pose = sapien.Pose([0, 0, -.02])
        b.build_static("floor")

    def _initialize_episode(self, env_idx, options):
        names = [j.name for j in self.agent.robot.get_active_joints()]
        rest = {"Pitch": 2.6, "Elbow": 2.8, "Wrist_Roll": 1.57,
                "Pitch_2": 2.6, "Elbow_2": 2.8, "Wrist_Roll_2": 1.57,
                "Jaw": .8}
        q = np.array([rest.get(n, 0.) for n in names], dtype=np.float32)
        self.agent.reset(q)

    def _get_obs_extra(self, info):
        return {}

    def evaluate(self):
        return {}


class Simulation:
    """Expose camera measurements and proprioception; keep scene private."""
    def __init__(self, seed=None, objects=None):
        self.on_frame = None
        self._last_frame = 0.
        self._scene_objects = _SceneObjects(seed=seed, objects=objects)
        self._env = PickPartsEnv(self._scene_objects)
        self._env.reset(seed=seed)
        self.joint_names = [j.name for j in self._env.agent.robot.get_active_joints()]
        self._target = self._env.agent.robot.get_qpos()[0].numpy().copy()
        self.hold(20)

    @property
    def object_specs(self):
        """Copy of public identity/color metadata, never scene measurements."""
        return self._scene_objects.specs

    def add_object(self, kind):
        """Add a reachable block or container; return its metadata spec."""
        spec = self._scene_objects.add(self._env.scene, kind)
        self.hold(20)
        return spec

    def _action(self):
        return np.concatenate([
            self._target[[self.joint_names.index(n) for n in group]]
            for group in (RIGHT, LEFT, HEAD, ["Jaw"], ["Jaw_2"])
        ]).astype(np.float32)

    def hold(self, steps=1):
        for _ in range(steps):
            self._env.step(self._action())
            if self.viewer_open:
                self._env.render_human()
            if self.on_frame is not None and time.monotonic() - self._last_frame >= .12:
                self._last_frame = time.monotonic()
                self.on_frame(self.observe())

    def enable_viewer(self):
        from pathlib import Path
        settings = Path(__file__).resolve().parents[2] / ".runtime" / "imgui.ini"
        settings.parent.mkdir(parents=True, exist_ok=True)
        sapien.render.set_imgui_ini_filename(str(settings))
        viewer = self._env.render_human()
        viewer.set_camera_pose(sapien_utils.look_at(
            [.9, -1.2, 1.35], [0, -.2, .65]).sp)
        viewer.render()

    @property
    def viewer_open(self):
        return self._env.viewer is not None and not self._env.viewer.closed

    def move_right(self, joints, jaw=None, steps=35):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (5,) or not np.isfinite(joints).all():
            raise ValueError("Expected five finite right-arm joint angles")
        indices = [self.joint_names.index(n) for n in RIGHT]
        start = self._target.copy()
        goal = start.copy()
        goal[indices] = joints
        if jaw is not None:
            if not 0 <= jaw <= 1.7:
                raise ValueError("Jaw angle outside URDF limits")
            goal[self.joint_names.index("Jaw")] = jaw
        for fraction in np.linspace(0, 1, steps):
            blend = fraction * fraction * (3 - 2 * fraction)
            self._target = start + blend * (goal - start)
            self.hold()
        self.hold(5)
        actual = self._env.agent.robot.get_qpos()[0].numpy()
        error = np.max(np.abs(actual[indices] - goal[indices]))
        if error > .12:
            raise RuntimeError(f"Joint tracking error {error:.3f} rad")

    def observe(self):
        obs = self._env.get_obs()
        data = obs["sensor_data"]["workspace"]
        params = obs["sensor_param"]["workspace"]
        def array(t):
            return t[0].detach().cpu().numpy().copy()
        extrinsic = np.eye(4)
        extrinsic[:3] = array(params["extrinsic_cv"])
        return Frame(
            rgb=array(data["rgb"]),
            depth=array(data["depth"]).squeeze(-1).astype(float) / 1000.,
            intrinsic=array(params["intrinsic_cv"]),
            camera_to_base=np.linalg.inv(extrinsic),
            qpos=array(obs["agent"]["qpos"]), qvel=array(obs["agent"]["qvel"]),
        )

    def close(self):
        self._env.close()
