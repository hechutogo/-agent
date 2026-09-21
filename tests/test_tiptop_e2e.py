"""Physical end-to-end acceptance: one-shot perceive -> TAMP -> open loop.

Grounding is fixed per case so the pipeline is validated offline; the language
step itself is covered by the grounding unit tests.
"""
import pytest

from pickparts_agent.scene.kinematics import ArmKinematics
from pickparts_agent.scene.simulation import Simulation
from tiptop_mac.agent import TiPToPAgent
from tiptop_mac.evaluation import judge_outcome
from tiptop_mac.executor import OpenLoopExecutor
from tiptop_mac.grounding import Goal
from tiptop_mac.tamp import TAMPLite
from tiptop_mac.types import Predicate


class FixedGrounder:
    def __init__(self, movable, support):
        self.goal = Goal((Predicate("on", (movable, support)),), "")

    def ground(self, instruction, specs):
        return self.goal


def build_agent(sim, movable, support):
    kin = ArmKinematics()
    from pickparts_agent.scene.perception import ScenePerception
    locate = ScenePerception(sim.object_specs).locate
    return TiPToPAgent(
        sim, FixedGrounder(movable, support), TAMPLite(kin),
        OpenLoopExecutor(sim, kin, locate=locate))


@pytest.mark.parametrize("seed,movable,support,relation", [
    (0, "A", "B", "on"),
    (3, "B", "A", "on"),
    (7, "A", "box", "in"),
])
def test_tiptop_open_loop_solves_task(seed, movable, support, relation):
    sim = Simulation(seed=seed)
    try:
        agent = build_agent(sim, movable, support)
        result = agent.run(f"把 {movable} 放到 {support}")
        assert result["success"], result
        goal = (Predicate("on", (movable, support)),)
        verdict = judge_outcome(sim.observe(), sim.object_specs, goal)
        assert verdict.success, verdict
        assert verdict.relation == relation
    finally:
        sim.close()
