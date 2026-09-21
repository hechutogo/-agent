"""Real integration test; requires a working Vulkan driver."""
import numpy as np
import pytest


def test_xlerobot_has_real_rgbd_and_holds_pose():
    from pickparts_agent.scene.simulation import Simulation
    sim = Simulation(seed=3)
    try:
        frame = sim.observe()
        assert frame.rgb.shape == (480, 640, 3)
        assert np.std(frame.rgb) > 10
        assert np.count_nonzero(frame.depth > 0) > 10000
        assert np.isfinite(frame.qpos).all()
        before = frame.qpos.copy()
        sim.hold(20)
        assert np.max(np.abs(sim.observe().qpos - before)) < .15
        assert not any(n.startswith("root_") for n in sim.joint_names)
    finally:
        sim.close()


@pytest.mark.parametrize("seed,target", [(0, "A"), (1, "B"), (3, "B"), (7, "A"), (9, "B")])
def test_random_scene_real_atoms_grasp_lift_and_release(seed, target):
    from pickparts_agent.scene.simulation import Simulation
    from pickparts_agent.scene.perception import ScenePerception
    from pickparts_agent.scene.kinematics import ArmKinematics
    from pickparts_agent.agent.state import WorldState
    from pickparts_agent.agent.atoms.base import AtomContext
    from pickparts_agent.agent.atoms import (
        FindObject, ReachAbove, Grasp, Lift, CarryTo, ReleaseInto, ResetArm, VerifyState,
    )
    sim = Simulation(seed=seed)
    try:
        perception = ScenePerception(sim.object_specs)
        ctx = AtomContext(sim, perception.locate, ArmKinematics(),
                          WorldState(sim.object_specs), lambda _: None, lambda _: None)
        sequence = [
            (FindObject(), {"target": "box"}),
            (FindObject(), {"target": target}),
            (ReachAbove(), {"target": target}),
            (Grasp(), {"target": target}),
            (Lift(), {}),
            (CarryTo(), {"container": "box"}),
            (ReleaseInto(), {"container": "box"}),
            (ResetArm(), {}),
            (VerifyState(), {"target": target, "at": "box"}),
        ]
        for atom, args in sequence:
            result = atom.call(ctx, args)
            assert result.success, (seed, target, atom.name, result)
            if atom.name == "lift":
                assert result.observed["lift_m"] > .065
    finally:
        sim.close()


def test_dynamic_inventory_recreates_and_failed_add_leaves_scene_intact():
    from pickparts_agent.scene.simulation import Simulation
    from pickparts_agent.scene.perception import ScenePerception
    sim = Simulation(seed=3)
    try:
        initial = sim.object_specs
        added = sim.add_object("block")
        assert added["kind"] == "block"
        assert added["id"] not in {s["id"] for s in initial}
        specs = sim.object_specs
        assert len(specs) == 4
        before = sim.observe()
        seen = ScenePerception(specs).locate(before, added["id"])
        assert seen["top_z"] == pytest.approx(.756, abs=.004)
        with pytest.raises(ValueError):
            sim.add_object("sphere")
        assert sim.object_specs == specs
        np.testing.assert_array_equal(sim.observe().rgb, before.rgb)
        for _ in range(20):
            before_specs = sim.object_specs
            before_rgb = sim.observe().rgb
            try:
                sim.add_object("block")
            except ValueError:
                assert sim.object_specs == before_specs
                np.testing.assert_array_equal(sim.observe().rgb, before_rgb)
                break
        else:
            pytest.fail("Scene capacity must be finite")
    finally:
        sim.close()
    recreated = Simulation(seed=9, objects=specs)
    try:
        assert recreated.object_specs == specs
        assert ScenePerception(specs).locate(recreated.observe(), added["id"])
    finally:
        recreated.close()


def test_added_box_is_a_visible_static_accessible_container():
    from pickparts_agent.scene.simulation import Simulation
    from pickparts_agent.scene.perception import ScenePerception
    from pickparts_agent.scene.kinematics import ArmKinematics
    sim = Simulation(seed=3, objects=[])
    try:
        spec = sim.add_object("box")
        assert spec["kind"] == "box"
        perception = ScenePerception(sim.object_specs)
        before = perception.locate(sim.observe(), spec["id"])
        assert before["extent"][0] > .07
        assert before["extent"][1] > .07
        assert before["top_z"] == pytest.approx(.759, abs=.004)
        sim.hold(30)
        after = perception.locate(sim.observe(), spec["id"])
        np.testing.assert_allclose(before["point"], after["point"], atol=.001)
        destination = np.asarray(after["point"]).copy()
        destination[2] = after["top_z"] + .055
        ArmKinematics().solve(destination)
    finally:
        sim.close()


@pytest.mark.parametrize("seed,target", [(0, "A"), (3, "B"), (7, "A")])
def test_randomized_blocks_have_sensor_centered_stable_grasps(seed, target, tmp_path):
    from pickparts_agent.scene.simulation import Simulation
    from pickparts_agent.scene.perception import ScenePerception
    from pickparts_agent.baseline.motion import PickPlace
    sim = Simulation(seed=seed)
    try:
        perception = ScenePerception(sim.object_specs)
        before = perception.locate(sim.observe(), target)
        assert before["top_z"] == pytest.approx(.756, abs=.004)
        np.testing.assert_allclose(before["extent"][:2], [.024, .024], atol=.005)
        motion = PickPlace(sim, perception, output=tmp_path)
        motion._moves = []
        grasp = np.array(before["point"])
        grasp[2] -= .004
        hover = grasp + [0, 0, .10]
        motion._move(hover, jaw=.8)
        motion._move(grasp)
        sim.move_right(motion._arm(sim.observe()), jaw=0., steps=25)
        motion._move(hover)
        # Return the arm to a stationary hold before measuring lift.
        sim.hold(15)
        after = perception.locate(sim.observe(), target)
        assert after["point"][2] - before["point"][2] > .065
    finally:
        sim.close()
