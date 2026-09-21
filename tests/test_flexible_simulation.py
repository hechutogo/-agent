"""Physical acceptance of the same semantic atoms used by the web Agent."""
import numpy as np
import pytest

from pickparts_agent.agent.atoms import AtomContext, build_default_registry
from pickparts_agent.scene.kinematics import ArmKinematics
from pickparts_agent.agent.state import WorldState


@pytest.mark.parametrize("seed,source,destination,relation", [
    (0, "A", "B", "on"), (3, "B", "A", "on"), (7, "A", "box", "in"),
])
def test_random_scene_semantic_placement(seed, source, destination, relation, tmp_path):
    from PIL import Image
    from pickparts_agent.scene.simulation import Simulation
    from pickparts_agent.scene.perception import ScenePerception
    sim = Simulation(seed=seed)
    try:
        locate = ScenePerception(sim.object_specs).locate
        ctx = AtomContext(sim, locate, ArmKinematics(), WorldState(sim.object_specs),
                          lambda _: None, lambda _: None)
        registry = build_default_registry()
        calls = [
            ("find_object", {"target": source}),
            ("verify_state", {"target": source, "at": "table", "relation": "table"}),
            ("find_object", {"target": destination}),
            ("reach_above", {"target": source}),
            ("grasp", {"target": source}),
            ("lift", {}),
            ("carry_to", {"container": destination}),
            ("place_on" if relation == "on" else "release_into",
             {"target" if relation == "on" else "container": destination}),
            ("verify_state", {"target": source, "at": destination, "relation": relation}),
            ("reset_arm", {}),
        ]
        for name, args in calls:
            result = registry.get(name).call(ctx, args)
            if not result.success:
                Image.fromarray(sim.observe().rgb).save(tmp_path / "failure.png")
            assert result.success, (seed, name, result, tmp_path)
        # Independent final visual measurement: after arm reset, no held-state
        # memory or old placement inference can satisfy this check.
        frame = sim.observe()
        part, support = locate(frame, source), locate(frame, destination)
        delta = np.asarray(part["point"]) - support["point"]
        assert np.linalg.norm(delta[:2]) < (.014 if relation == "on" else .037)
        if relation == "on":
            assert .025 < delta[2] < .045
        Image.fromarray(frame.rgb).save(tmp_path / "final.png")
    finally:
        sim.close()
