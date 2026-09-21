"""XLeRobot model from Vector-Wangel/XLeRobot at the recorded revision."""
from pathlib import Path

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

URDF = Path(__file__).parent / "assets/xlerobot/xlerobot.urdf"
RIGHT = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
LEFT = [name + "_2" for name in RIGHT]
HEAD = ["head_pan_joint", "head_tilt_joint"]


@register_agent()
class PickPartsXLeRobot(BaseAgent):
    uid = "pickparts_xlerobot"
    urdf_path = str(URDF)
    fix_root_link = True
    urdf_config = {
        "_materials": {"gripper": {
            "static_friction": 2., "dynamic_friction": 2., "restitution": 0.,
        }},
        "link": {name: {"material": "gripper", "patch_radius": .01,
                        "min_patch_radius": .005}
                 for name in ("Fixed_Jaw", "Moving_Jaw", "Fixed_Jaw_2", "Moving_Jaw_2")},
    }

    @property
    def _controller_configs(self):
        def arm(names):
            return PDJointPosControllerConfig(
                names, None, None, 1000, 60, 30, normalize_action=False)

        def jaw(name):
            return PDJointPosControllerConfig(
                [name], 0., 1.7, 10, .3, .2, normalize_action=False)

        return {"pd_joint_pos": {
            "right": arm(RIGHT), "left": arm(LEFT), "head": arm(HEAD),
            "jaw": jaw("Jaw"), "jaw_left": jaw("Jaw_2"),
        }}
