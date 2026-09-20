"""Deterministic test doubles. No physics, network, or real cloud."""
import numpy as np

from pickparts_agent.atoms.base import AtomContext
from pickparts_agent.kinematics import ARM_NAMES
from pickparts_agent.state import WorldState


class FakeFrame:
    def __init__(self, qpos):
        self.qpos = np.asarray(qpos, dtype=float)
        self.rgb = np.zeros((4, 4, 3), np.uint8)
        self.depth = np.ones((4, 4))


class FakeSim:
    def __init__(self):
        self.joint_names = list(ARM_NAMES) + ["Jaw"]
        self.qpos = np.zeros(len(self.joint_names))
        # rest-like default
        defaults = {"Pitch": 2.6, "Elbow": 2.8, "Wrist_Roll": 1.57}
        for i, n in enumerate(ARM_NAMES):
            self.qpos[i] = defaults.get(n, 0.0)
        self.moves = []
        self.tracking_error = False

    def arm_qpos(self):
        idx = [self.joint_names.index(n) for n in ARM_NAMES]
        return self.qpos[idx]

    def observe(self):
        return FakeFrame(self.qpos)

    def hold(self, steps=1):
        return None

    def move_right(self, joints, jaw=None, steps=35):
        joints = np.asarray(joints, dtype=float)
        if self.tracking_error:
            raise RuntimeError("Joint tracking error 0.500 rad")
        idx = [self.joint_names.index(n) for n in ARM_NAMES]
        self.qpos[idx] = joints
        if jaw is not None:
            self.qpos[self.joint_names.index("Jaw")] = jaw
        self.moves.append({"joints": joints.tolist(), "jaw": jaw})

    def close(self):
        pass


class FakeKin:
    def __init__(self):
        self.q = np.array([0.0, 2.6, 2.8, 0.0, 1.57])
        self.position = np.array([0.0, -0.34, 0.84])
        self.fail_solve = False
        self.forward_offset = 0.0

    def solve(self, position, seed=None):
        if self.fail_solve:
            raise ValueError("IK unreachable: position error 0.2000 m")
        self.position = np.asarray(position, float)
        return self.q

    def forward(self, q):
        pose = np.eye(4)
        pose[:3, 3] = self.position + self.forward_offset
        pose[2, 1] = 0.99
        return pose


class FakeLocate:
    def __init__(self, points=None, bbox=None, confidence=0.9):
        self.points = {k: np.asarray(v, float) for k, v in (points or {}).items()}
        self.bbox = bbox or [1, 1, 4, 4]
        self.confidence = confidence
        self.calls = []
        self.fail_on = set()

    def add(self, target, point):
        self.points[target] = np.asarray(point, float)

    def __call__(self, frame, target):
        self.calls.append(target)
        if target in self.fail_on or target not in self.points:
            raise ValueError(f"{target} is not visible")
        return {"bbox": self.bbox, "point": self.points[target],
                "confidence": self.confidence}


class FakeChat:
    """Returns scripted dict responses; can fail on chosen calls."""
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def add(self, payload):
        self.responses.append(payload)

    def json(self, system, user):
        self.calls.append({"system": system, "user": user})
        if not self.responses:
            raise RuntimeError("FakeChat exhausted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return dict(item)


def make_context(sim=None, kin=None, locate=None, state=None,
                 stages=None, traces=None):
    state = state if state is not None else WorldState()
    locate = locate if locate is not None else FakeLocate({})
    return AtomContext(
        sim if sim is not None else FakeSim(),
        locate,
        kin if kin is not None else FakeKin(),
        state,
        (lambda s: stages.append(s)) if stages is not None else (lambda s: None),
        (lambda t: traces.append(t)) if traces is not None else (lambda t: None),
    )
