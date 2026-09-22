"""Ordered one-plan executions, visual checkpoints and bounded ReAct recovery."""
from tiptop_mac.agent import TiPToPAgent
from tiptop_mac.executor import record_artifact
from tiptop_mac.tamp import TAMPError
from pickparts_agent.observability.events import redact

from .grounding import GroundingError
from .executor import GripStateError


class OptimizedTiPToPAgent(TiPToPAgent):
    def __init__(self, *args, reactor, max_attempts=5, **kwargs):
        super().__init__(*args, **kwargs)
        if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        self.reactor = reactor
        self.max_attempts = max_attempts

    def bind_sim(self, sim):
        self.executor.bind_sim(sim)
        self.sim = sim

    def run(self, instruction):
        self._artifact_index = 0
        result = {"success": False, "aborted": False, "recovery_required": False,
                  "verification": "visual", "message": "尚未完成。",
                  "rationale": "", "completed_subtasks": 0,
                  "total_subtasks": 0, "attempts": 0}
        root = self.recorder.start_span("tiptop_optimized", verification="visual")
        rows = []
        try:
            with self._phase("decompose"):
                self._activity("任务拆解", "understanding",
                               "保留动作顺序，将指令拆成逐段规划和验证的子任务。")
                tasks = self.grounder.ground(instruction, self.sim.object_specs)
                if not tasks:
                    raise GroundingError("没有可执行的子任务")
                result["total_subtasks"] = len(tasks)
                result["rationale"] = redact(tasks[0].goal.rationale)
                rows = [{"index": i, "instruction": redact(t.instruction),
                         "status": "pending"} for i, t in enumerate(tasks)]
                self._subtasks(rows, 0)
                self._save_state(rows, "ordered-subtasks")
            for index, task in enumerate(tasks):
                rows[index]["status"] = "active"
                self._subtasks(rows, index)
                subspan = self.recorder.start_span(
                    f"subtask:{index + 1}", instruction=task.instruction)
                done = False
                try:
                    done = self._run_subtask(task, index, result)
                finally:
                    self.recorder.end_span(subspan, "ok" if done else "error")
                    rows[index]["status"] = "done" if done else "failed"
                    self._subtasks(rows, index + 1 if done else index)
                if not done:
                    break
                result["completed_subtasks"] += 1
            result["success"] = result["completed_subtasks"] == len(tasks)
            if result["success"]:
                result.update(aborted=False, recovery_required=False,
                              message=f"已按顺序完成 {len(tasks)} 个子任务，并逐段通过视觉验证。")
        except Exception as exc:
            self.recorder.log("error", f"TiPToP 优化版：{type(exc).__name__}",
                              logger="tiptop_optimized.agent", exc_info=True)
            result["message"] = f"任务未完成（{type(exc).__name__}）。"
            if result["attempts"] or self._grip_active():
                result["aborted"] = result["recovery_required"] = True
        finally:
            self.recorder.end_span(root, "ok" if result["success"] else "error",
                                   message=result["message"])
        self.on_event({"type": "finish", "success": result["success"]})
        return result

    def _subtasks(self, rows, current):
        self.on_event({"type": "subtasks", "current": current,
                       "steps": [dict(row) for row in rows]})

    def _grip_active(self):
        return (getattr(self.executor, "held", None) is not None
                or getattr(self.executor, "candidate", None) is not None)

    def _save_state(self, data, label):
        self._artifact_index += 1
        record_artifact(self.recorder, "save_state", data,
                        f"{label}-{self._artifact_index}")

    def _observe(self, phase):
        with self._phase(phase):
            self._activity("视觉验证" if phase == "verify" else "场景感知",
                           "observation", "读取当前 RGB-D，并检查物体关系和持物状态。")
            frame = self.sim.observe()
            record_artifact(self.recorder, "save_rgbd", frame,
                            note=f"TiPToP optimized {phase}")
            scene = self.perceive(frame, self.sim.object_specs)
            self._save_state({
                "qpos": scene.qpos.tolist(),
                "support": self.planner._initial_support(scene),
                "objects": {key: {"point": node.point.tolist(),
                                  "extent": node.extent.tolist(),
                                  "bottom_z": node.bottom_z,
                                  "top_z": node.top_z}
                            for key, node in scene.objects.items()},
            }, f"optimized-{phase}")
            try:
                held = self.executor.reconcile(scene)
            except GripStateError as exc:
                # Preserve fresh perception even though tuple assignment in
                # the caller cannot complete when reconciliation raises.
                exc.scene = scene
                exc.error_kind = "grip_unknown"
                raise
            return scene, held

    def _publish_plan(self, task, plan):
        event = {
            "type": "plan", "goal": redact(task.instruction), "current": 0,
            "steps": [
                {"index": index, "atom": step.kind, "status": "pending",
                 "args": {"position": (step.position.tolist()
                                      if step.position is not None else None),
                          "jaw": step.jaw, "label": redact(step.label)}}
                for index, step in enumerate(plan.trajectory)]}
        self.on_event(event)
        self._save_state({
            **event, "require_action": task.require_action,
            "operators": plan.operators, "planning_time": plan.planning_time,
            "predicates": [{"name": p.name, "args": p.args}
                           for p in task.goal.predicates],
        }, "optimized-plan")

    def _run_subtask(self, task, index, result):
        acted, execution_started = False, False
        failures, excluded_table_xy = [], []
        for attempt in range(1, self.max_attempts + 1):
            result["attempts"] += 1
            span = self.recorder.start_span(f"attempt:{attempt}", subtask=index + 1)
            scene, held, verified, plan, execution = None, None, False, None, None
            phase, grip_known = "perceive", False
            failure = {"error_kind": "verify", "message": "视觉目标尚未满足。"}
            try:
                scene, held = self._observe("perceive")
                grip_known = True
                phase = "plan"
                motion = self.executor.planning_context(scene, held=held)
                with self._phase("plan"):
                    self._activity("TAMP 规划", "plan",
                                   f"子任务 {index + 1}，第 {attempt} 次规划：{redact(task.instruction)}")
                    plan = self.planner.plan(
                        scene, task.goal.predicates, rationale=task.goal.rationale,
                        held=held, require_action=task.require_action and not acted,
                        arm_seed=motion["arm_seed"],
                        grasp_offset=motion["grasp_offset"],
                        excluded_table_xy=tuple(excluded_table_xy))
                    self._publish_plan(task, plan)
                execution_started = execution_started or bool(plan.trajectory)
                phase, grip_known = "execute", False
                execution = {**self.executor.execute(plan), "phase": "execute"}
                acted = acted or bool(execution.get("acted"))
                if not execution["success"]:
                    failure = execution
                    if execution.get("error_kind") in ("unreachable", "verify"):
                        failed_xy = self.planner.failed_table_xy(
                            plan, execution.get("failed_index"))
                        if failed_xy is not None:
                            excluded_table_xy.append(failed_xy)
                if execution.get("error_kind") == "fatal":
                    pass
                else:
                    # A release may succeed even when its retreat fails. Inspect
                    # the physical result before deciding whether to reexecute.
                    phase, scene, held = "verify", None, None
                    scene, held = self._observe("verify")
                    grip_known = True
                    verified = (self.planner.satisfies(scene, task.goal.predicates,
                                                      held=held)
                                and (not task.require_action or acted))
                    if not verified and execution["success"]:
                        failed_xy = self.planner.table_release_xy(plan)
                        if failed_xy is not None:
                            excluded_table_xy.append(failed_xy)
                    if not execution["success"]:
                        failure = execution
                    self._save_state({
                        "subtask": index, "attempt": attempt, "verified": verified,
                        "intermediate_action_observed": acted,
                    }, "optimized-verdict")
            except Exception as exc:
                if isinstance(exc, GripStateError):
                    scene = getattr(exc, "scene", scene)
                    held, grip_known = None, False
                kind = getattr(exc, "error_kind", None)
                if kind is None:
                    kind = ("planning" if isinstance(exc, TAMPError) else
                            "observation" if isinstance(exc, ValueError) else "fatal")
                observed_failure = {
                    "error_kind": kind, "phase": phase,
                    "message": (f"持物状态待确认：{str(exc)}"
                                if isinstance(exc, GripStateError) else
                                f"当前子任务未完成（{type(exc).__name__}）。")}
                if execution is None or execution.get("success"):
                    failure = observed_failure
                else:
                    failure = {**failure, "observation_error": observed_failure}
                self.recorder.log("error", observed_failure["message"],
                                  logger="tiptop_optimized.agent", exc_info=True)
            finally:
                self.recorder.end_span(span, "ok" if verified else "error",
                                       None if verified else failure["error_kind"])
            if verified:
                self._activity("子任务完成", "observation",
                               f"子任务 {index + 1} 已通过视觉验证。")
                return True
            result["aborted"] = (execution_started or self._grip_active()
                                 or held is not None or failure["error_kind"] == "fatal")
            result["recovery_required"] = result["aborted"]
            result["message"] = redact(failure["message"])
            if failure["error_kind"] == "fatal":
                return False
            if attempt == self.max_attempts:
                result["message"] = f"子任务 {index + 1} 达到 {attempt} 次尝试上限，尚未通过视觉验证。"
                return False
            failures.append({"error_kind": failure["error_kind"],
                             "message": redact(failure["message"]),
                             "phase": failure.get("phase", phase),
                             "observation_error": failure.get("observation_error")})
            with self._phase("react"):
                context = {
                    "instruction": redact(task.instruction), "attempt": attempt,
                    "completed_subtasks": index, "remaining_attempts": self.max_attempts - attempt,
                    "error_kind": failure["error_kind"], "failures": failures,
                    "held": held if grip_known else None,
                    "grip_state": ("holding" if held else "empty") if grip_known else "unknown",
                    "support": self.planner._initial_support(scene) if scene else {},
                    "goal": [{"name": p.name, "args": p.args}
                             for p in task.goal.predicates],
                }
                decision = self.reactor.decide(context)
                self._save_state({
                    **context, "decision": decision.action,
                    "reason": redact(decision.reason),
                }, "optimized-react")
                self.on_event({"type": "react", "attempt": attempt,
                               "decision": decision.action, "thought": "",
                               "detail": redact(decision.reason)})
                self._activity("ReAct 恢复", "recovery", redact(decision.reason))
            if decision.action not in ("replan", "reobserve"):
                result["message"] = "ReAct 停止当前子任务：" + redact(decision.reason)
                return False
        return False


def build_tiptop_optimized_agent(sim, llm_endpoint, *, on_stage=None,
                                 on_event=None, recorder=None):
    from pickparts_agent.agent.llm import JSONChat, supports_thinking
    from pickparts_agent.observability import NullRecorder
    from pickparts_agent.scene.kinematics import ArmKinematics
    from pickparts_agent.scene.perception import ScenePerception
    from .executor import RecoveringExecutor
    from .grounding import OrderedGrounder, RecoveryReactor
    from .tamp import TAMPLite
    recorder = recorder if recorder is not None else NullRecorder()
    client = llm_endpoint.client()
    try:
        chat = JSONChat(client, llm_endpoint.model,
                        thinking=supports_thinking(llm_endpoint.model, llm_endpoint.base_url),
                        recorder=recorder, role="tiptop_optimized")
        kin = ArmKinematics()
        agent = OptimizedTiPToPAgent(
            sim, OrderedGrounder(chat), TAMPLite(kin),
            RecoveringExecutor(sim, kin, on_stage=on_stage,
                               locate=ScenePerception(sim.object_specs).locate,
                               on_event=on_event, recorder=recorder),
            reactor=RecoveryReactor(chat), on_stage=on_stage,
            on_event=on_event, recorder=recorder)
        agent._owned_client = client
        return agent
    except Exception:
        client.close()
        raise
