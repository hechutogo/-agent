"""Six-step embodied loop: assess -> plan -> execute/react -> follow-up."""
from .atoms import build_default_registry
from .executor import ReActExecutor
from .kinematics import ArmKinematics
from .llm import JSONChat
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
    def __init__(self, state, registry, planner, reactor, executor):
        self.state = state
        self.registry = registry
        self.planner = planner
        self.reactor = reactor
        self.executor = executor
        self.history = []

    def turn(self, text):
        plan = self.planner.analyze(text, self.state)
        if not plan.feasible:
            return {"success": False, "message": render_blockers(plan.blockers)}
        result = self.executor.execute(plan, goal=text)
        if result.get("success") and "还需要" not in result.get("message", ""):
            result["message"] += " 还需要我做什么？"
        self.history.append({"goal": text, **result})
        return result


def build_orchestrator(sim, locate, llm_endpoint,
                       on_stage=None, on_event=None, ask_user=None):
    state = WorldState()
    registry = build_default_registry()
    chat = JSONChat(llm_endpoint.client(), llm_endpoint.model)
    planner = Planner(chat, registry)
    reactor = Reactor(chat, registry)
    executor = ReActExecutor(
        sim, locate, ArmKinematics(), registry, state, planner, reactor,
        on_stage=on_stage, on_event=on_event, ask_user=ask_user)
    return Orchestrator(state, registry, planner, reactor, executor)
