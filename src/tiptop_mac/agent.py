"""TiPToP-style agent: one-shot perceive -> ground -> global TAMP -> open loop."""
from contextlib import contextmanager

from pickparts_agent.agent.llm import JSONChat, supports_thinking
from pickparts_agent.observability import NullRecorder
from pickparts_agent.observability.events import redact
from pickparts_agent.scene.kinematics import ArmKinematics
from .executor import OpenLoopExecutor, record_artifact
from .grounding import Grounder
from .perception import build_scene_graph
from .tamp import TAMPLite


class TiPToPAgent:
    def __init__(self, sim, grounder, planner, executor, *,
                 perceive=build_scene_graph, on_stage=None, on_event=None,
                 recorder=None):
        self.sim = sim
        self.grounder = grounder
        self.planner = planner
        self.executor = executor
        self.perceive = perceive
        self.on_stage = on_stage if on_stage is not None else (lambda s: None)
        self.on_event = on_event if on_event is not None else (lambda e: None)
        self.recorder = recorder if recorder is not None else NullRecorder()
        self._owned_client = None

    def run(self, instruction):
        span = self.recorder.start_span("tiptop", verification="open_loop")
        result = {"success": False, "aborted": False, "message": "任务中止。"}
        try:
            result = self._run(instruction)
            result["recovery_required"] = bool(result["aborted"])
            result["verification"] = "open_loop"
            return result
        finally:
            self.recorder.end_span(
                span, "ok" if result["success"] else "error",
                message=result["message"])

    @contextmanager
    def _phase(self, name):
        span = self.recorder.start_span(name)
        try:
            yield
        except Exception as exc:
            self.recorder.log("error", f"TiPToP {name}: {type(exc).__name__}",
                              logger="tiptop.agent", exc_info=True)
            self.recorder.end_span(span, "error", type(exc).__name__)
            raise
        else:
            self.recorder.end_span(span, "ok")

    def _run(self, instruction):
        phase, rationale = "理解指令", ""
        try:
            with self._phase("ground"):
                self._activity(phase, "understanding",
                               f"一次性理解指令：{redact(instruction)}")
                specs = self.sim.object_specs
                goal = self.grounder.ground(instruction, specs)
                rationale = redact(goal.rationale)
            phase = "场景感知"
            with self._phase("perceive"):
                self._activity(phase, "observation", "读取一次 RGB-D 场景用于全局规划。")
                frame = self.sim.observe()
                record_artifact(self.recorder, "save_rgbd", frame,
                                note="TiPToP initial planning scene")
                scene = self.perceive(frame, specs)
                record_artifact(self.recorder, "save_state", {
                    "table": {"top_z": scene.table.top_z,
                              "normal": scene.table.normal.tolist(),
                              "bounds": scene.table.bounds.tolist()},
                    "objects": {
                        redact(key): {"kind": node.kind,
                                      "point": node.point.tolist(),
                                      "extent": node.extent.tolist(),
                                      "top_z": node.top_z,
                                      "bottom_z": node.bottom_z}
                        for key, node in scene.objects.items()},
                    "qpos": scene.qpos.tolist(),
                }, "tiptop-scene")
            phase = "TAMP 规划"
            with self._phase("plan"):
                self._activity(phase, "plan", rationale or "一次性生成全局动作轨迹。")
                plan = self.planner.plan(scene, goal.predicates,
                                         rationale=goal.rationale)
                event = {
                    "type": "plan", "goal": redact(instruction), "current": 0,
                    "steps": [
                        {"index": index, "atom": redact(step.kind),
                         "args": {
                             "position": (step.position.tolist()
                                          if step.position is not None else None),
                             "jaw": step.jaw, "label": redact(step.label)},
                         "status": "pending"}
                        for index, step in enumerate(plan.trajectory)],
                }
                record_artifact(self.recorder, "save_state", {
                    **event, "planning_time": plan.planning_time,
                    "rationale": rationale,
                    "predicates": [{"name": p.name, "args": [redact(a) for a in p.args]}
                                   for p in goal.predicates],
                }, "tiptop-plan")
                # The agent owns the instruction-bearing plan; the executor
                # publishes only step/activity/finish events.
                self.on_event(event)
        except Exception as exc:
            message = f"{phase}失败（{type(exc).__name__}）。"
            self.on_event({"type": "activity", "kind": "observation",
                           "text": message})
            self.on_event({"type": "finish", "success": False})
            return {"success": False, "aborted": False,
                    "message": message, "rationale": rationale}
        self._activity("开环执行", "action", "按既定轨迹执行，不重试、不重规划或验证最终结果。")
        execution = self.executor.execute(plan)
        return {
            "success": execution["success"],
            "aborted": execution["aborted"],
            "message": execution["message"],
            "rationale": rationale,
        }

    def _activity(self, stage, kind, text):
        self.on_stage(stage)
        self.on_event({"type": "activity", "kind": kind, "text": text})

    def close(self):
        """Release only the LLM client allocated by the factory."""
        client, self._owned_client = self._owned_client, None
        if client is not None:
            client.close()


def build_tiptop_agent(sim, llm_endpoint, *, on_stage=None, on_event=None,
                       recorder=None):
    recorder = recorder if recorder is not None else NullRecorder()
    thinking = supports_thinking(llm_endpoint.model, llm_endpoint.base_url)
    client = llm_endpoint.client()
    try:
        chat = JSONChat(client, llm_endpoint.model, thinking=thinking,
                        recorder=recorder, role="grounder")
        kin = ArmKinematics()
        from pickparts_agent.scene.perception import ScenePerception
        locate = ScenePerception(sim.object_specs).locate
        agent = TiPToPAgent(
            sim, Grounder(chat), TAMPLite(kin),
            OpenLoopExecutor(sim, kin, on_stage=on_stage, locate=locate,
                             on_event=on_event, recorder=recorder),
            on_stage=on_stage, on_event=on_event, recorder=recorder)
        agent._owned_client = client
        return agent
    except Exception:
        client.close()
        raise
