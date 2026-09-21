import numpy as np
import pytest
from pickparts_agent.scene.kinematics import ArmKinematics


def test_reachable_target_matches_position_and_downward_approach():
    kin = ArmKinematics()
    q = kin.solve([.12, -.16, .81], seed=[0., 2., 1.2, -.8, 1.57])
    pose = kin.forward(q)
    np.testing.assert_allclose(pose[:3, 3], [.12, -.16, .81], atol=.003)
    assert pose[2, 1] > .97  # tool -Y approaches vertically downward


def test_unreachable_target_fails_instead_of_executing_nearest_pose():
    with pytest.raises(ValueError, match="IK"):
        ArmKinematics().solve([3, 0, 2])
