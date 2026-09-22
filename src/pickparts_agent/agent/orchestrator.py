"""Six-step embodied loop: assess -> plan -> execute/react -> follow-up."""
from .atoms import build_default_registry
from .executor import ReActExecutor
from ..scene.kinematics import ArmKinematics
from .llm import JSONChat, supports_thinking
from .planner import Planner
from .reactor import Reactor
from .state import WorldState


def render_blockers(blockers):
    if not blockers:
        return "这个需求当前无法完成，请换一种说法。"
    parts = []
    for b in blockers:
        need, reason = b.get("need", ""), b.get("reason", "")
        parts.append(f"{need}（{reason}）" if reason else need)
    return "无法完成：" + "；".join(parts)


class Orchestrator:
    def __init__(self, state, registry, planner, reactor, executor,
                 on_event=None, recorder=None):
        self.state = state
        self.registry = registry
        self.planner = planner
        self.reactor = reactor
        self.executor = executor
        self.history = []
        self.on_event = on_event or (lambda e: None)
        if recorder is None:
            from ..observability import NullRecorder
            recorder = NullRecorder()
        self.recorder = recorder
        self.thinking = False

    def turn(self, text):
        self.on_event({"type": "activity", "kind": "understanding",
                       "text": f"用户的要求是：{text}。我会先确认对象和操作条件，再规划动作。"})
        with self.recorder.span("plan"):
            plan = self.planner.analyze(text, self.state)
        self.on_event({"type": "activity", "kind": "plan",
                       "text": plan.rationale or "已完成可行性评估。"})
        if not plan.feasible:
            return {"success": False, "message": render_blockers(plan.blockers)}
        with self.recorder.span("execute"):
            result = self.executor.execute(plan, goal=text)
        if result.get("success") and "还需要" not in result.get("message", ""):
            result["message"] += " 还需要我做什么？"
        self.history.append({"goal": text, **result})
        return result


def build_orchestrator(sim, locate, llm_endpoint,
                       on_stage=None, on_event=None, ask_user=None, recorder=None):
    if recorder is None:
        from ..observability import NullRecorder
        recorder = NullRecorder()
    state = WorldState(getattr(sim, "object_specs", None))
    registry = build_default_registry()
    thinking = supports_thinking(llm_endpoint.model, llm_endpoint.base_url)
    chat = JSONChat(llm_endpoint.client(), llm_endpoint.model, thinking=thinking,
                    recorder=recorder, role="plan")
    planner = Planner(chat, registry)
    reactor = Reactor(chat, registry)
    executor = ReActExecutor(
        sim, locate, ArmKinematics(), registry, state, planner, reactor,
        on_stage=on_stage, on_event=on_event, ask_user=ask_user,
        recorder=recorder)
    orchestrator = Orchestrator(state, registry, planner, reactor, executor,
                                on_event, recorder=recorder)
    orchestrator.thinking = thinking
    return orchestrator
