"""Physical acceptance: image-derived lift followed by image-derived placement."""
from pickparts_agent.scene.simulation import Simulation
from pickparts_agent.scene.perception import ColorPerception


def test_a_is_lifted_and_placed_using_visual_feedback(tmp_path):
    from pickparts_agent.baseline.motion import PickPlace
    sim = Simulation(seed=3)
    try:
        result = PickPlace(sim, ColorPerception(), output=tmp_path).run("A")
        assert result["success"], result
        assert result["lift_m"] > .035
    finally:
        sim.close()
