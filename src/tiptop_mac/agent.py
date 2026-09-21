"""TiPToP-style agent: one-shot perceive -> ground -> global TAMP -> open loop."""
from pickparts_agent.agent.llm import JSONChat, supports_thinking
from pickparts_agent.scene.kinematics import ArmKinematics
from .executor import OpenLoopExecutor
from .grounding import Grounder
from .perception import build_scene_graph
from .tamp import TAMPError, TAMPLite


class TiPToPAgent:
    def __init__(self, sim, grounder, planner, executor, *,
                 perceive=build_scene_graph, on_stage=None, on_event=None):
        self.sim = sim
        self.grounder = grounder
        self.planner = planner
        self.executor = executor
        self.perceive = perceive
        self.on_stage = on_stage if on_stage is not None else (lambda s: None)
        self.on_event = on_event if on_event is not None else (lambda e: None)

    def run(self, instruction):
        self.on_event({"type": "activity", "kind": "understanding",
                       "text": f"一次性理解指令：{instruction}"})
        frame = self.sim.observe()
        specs = self.sim.object_specs
        try:
            goal = self.grounder.ground(instruction, specs)
        except ValueError as exc:
            return {"success": False, "aborted": False,
                    "message": f"指令无法接地：{exc}", "rationale": ""}
        self.on_event({"type": "activity", "kind": "plan",
                       "text": goal.rationale or "已生成目标谓词。"})
        scene = self.perceive(frame, specs)
        try:
            plan = self.planner.plan(scene, goal.predicates,
                                     rationale=goal.rationale)
        except TAMPError as exc:
            return {"success": False, "aborted": False,
                    "message": f"找不到可行规划：{exc}",
                    "rationale": goal.rationale}
        execution = self.executor.execute(plan)
        return {
            "success": execution["success"],
            "aborted": execution["aborted"],
            "message": execution["message"],
            "rationale": goal.rationale,
        }


def build_tiptop_agent(sim, llm_endpoint, *, on_stage=None, on_event=None):
    thinking = supports_thinking(llm_endpoint.model, llm_endpoint.base_url)
    chat = JSONChat(llm_endpoint.client(), llm_endpoint.model,
                    thinking=thinking)
    kin = ArmKinematics()
    from pickparts_agent.scene.perception import ScenePerception
    locate = ScenePerception(sim.object_specs).locate
    return TiPToPAgent(
        sim, Grounder(chat), TAMPLite(kin),
        OpenLoopExecutor(sim, kin, on_stage=on_stage, locate=locate),
        on_stage=on_stage, on_event=on_event)
