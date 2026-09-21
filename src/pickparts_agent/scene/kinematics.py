"""Robot-only URDF kinematics; never consumes scene object state."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ARM_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]


class ArmKinematics:
    def __init__(self):
        path = Path(__file__).parent / "robots/assets/xlerobot/xlerobot.urdf"
        root = ET.parse(path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], "Fixed_Jaw"
        while link in by_child:
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        self.chain, limits = [], {}
        for j in reversed(chain):
            origin = j.find("origin")
            t = np.eye(4)
            if origin is not None:
                t[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                t[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            name = j.get("name")
            axis = np.array([0., 0., 1.])
            if j.find("axis") is not None:
                axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
            self.chain.append((name, t, axis))
            if name in ARM_NAMES:
                limit = j.find("limit")
                limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        self.lower, self.upper = np.array([limits[n] for n in ARM_NAMES]).T
        # Open-gripper insertion point for a 24mm part. Mesh sections at
        # y=-.095 put the fixed inner face at x=.008 and lowest tip at y=-.106.
        # Leave 10mm lateral insertion clearance; closure seats against fixed jaw.
        self.tcp_offset = np.array([-.014, -.095, 0.])

    def forward(self, q):
        values = dict(zip(ARM_NAMES, q))
        t = np.eye(4)
        for name, origin, axis in self.chain:
            t = t @ origin
            if name in values:
                rot = np.eye(4)
                rot[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
                t = t @ rot
        t[:3, 3] += t[:3, :3] @ self.tcp_offset
        return t

    def solve(self, position, seed=None):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("IK target must be finite XYZ")
        def residual(q):
            pose = self.forward(q)
            return np.r_[pose[:3, 3] - position,
                         .12 * (pose[:3, 1] - [0, 0, 1]), .02 * q[4]]
        starts = ([seed] if seed is not None else []) + [
            [0., 1.5, 1.2, -1., 1.57], [0., .5, .5, -.5, 1.57],
            [0., 2.6, 2.8, 0., 1.57]]
        best = None
        for initial in starts:
            result = least_squares(residual, np.clip(initial, self.lower, self.upper),
                                   bounds=(self.lower, self.upper), max_nfev=150,
                                   ftol=1e-8, xtol=1e-8, gtol=1e-8)
            pose = self.forward(result.x)
            error = np.linalg.norm(pose[:3, 3] - position)
            if best is None or error < best[0]:
                best = error, result.x
            if error < .003 and pose[2, 1] > .97:
                return result.x
        raise ValueError(f"IK unreachable: position error {best[0]:.4f} m")
