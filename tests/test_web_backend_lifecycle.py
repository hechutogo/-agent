"""Exercise real backend ownership without loading physics or cloud clients."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from pickparts_agent.interfaces.web import RobotBackend
from pickparts_agent.observability import Recorder
from pickparts_agent.observability.query import get_run


class Resource:
    def __init__(self):
        self.close_calls = 0
        self.fail_close = False

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("private provider detail")


@pytest.fixture
def dependencies(monkeypatch):
    from pickparts_agent.agent import orchestrator
    from pickparts_agent.services import cloud
    from tiptop_mac import agent

    client, sim = Resource(), Resource()
    sim.object_specs = [{"id": "A", "kind": "block", "label": "red block",
                         "color": [.85, .045, .03]}]
    sim.observe = lambda: SimpleNamespace(rgb=None)
    endpoint = SimpleNamespace(model="test", base_url="https://example.test",
                               client=lambda: client)
    monkeypatch.setattr(cloud.Endpoint, "from_env", lambda prefix: endpoint)
    monkeypatch.setattr(cloud, "VisualLocator",
                        lambda *args: SimpleNamespace(locate=lambda: None))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args: None)
    engines = []
    state = SimpleNamespace(client=client, sim=sim, engines=engines, fail_build=False)

    def build(*args, **kwargs):
        if state.fail_build:
            raise ValueError("build failed")
        engine = Resource()
        engine.sim = args[0]
        engine.executor = SimpleNamespace(sim=args[0], _ctx=SimpleNamespace(sim=args[0]),
                                          locate=None)
        engine.held = "A"
        engine.bind_calls = 0

        def bind_sim(sim):
            engine.bind_calls += 1
            engine.sim = engine.executor.sim = engine.executor._ctx.sim = sim
            engine.held = None

        engine.bind_sim = bind_sim
        engine.thinking = False
        engine.planner = SimpleNamespace(chat=SimpleNamespace(client=engine))
        engine.turn = engine.run = lambda text: {"success": True, "message": text}
        engines.append(engine)
        return engine

    def build_optimized(sim, llm_endpoint, *, on_stage, on_event, recorder):
        state.optimized_args = (sim, llm_endpoint, on_stage, on_event, recorder)
        return build(sim)

    monkeypatch.setattr(orchestrator, "build_orchestrator", build)
    monkeypatch.setattr(agent, "build_tiptop_agent", build)
    optimized = ModuleType("tiptop_optimized.agent")
    optimized.build_tiptop_optimized_agent = build_optimized
    monkeypatch.setitem(sys.modules, optimized.__name__, optimized)
    module = ModuleType("pickparts_agent.scene.simulation")

    def simulation(objects=None):
        if objects is None:
            return sim
        replacement = Resource()
        replacement.object_specs = objects
        replacement.observe = sim.observe
        return replacement

    module.Simulation = simulation
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return state


@pytest.mark.parametrize("target", ["tiptop", "tiptop_optimized", "pickparts"])
def test_retirement_failure_keeps_new_engine_installed(dependencies, tmp_path, target):
    rec = Recorder(tmp_path)
    backend = RobotBackend(lambda frame: None, lambda stage: None, recorder=rec)
    if target == "pickparts":
        backend.select_agent("tiptop")
    old = dependencies.engines[-1]
    old.fail_close = True
    try:
        with rec.run("switch", target):
            backend.select_agent(target)
        new = dependencies.engines[-1]
        assert backend.agent == target
        assert (backend.orchestrator if target == "pickparts" else backend.tiptop) is new
        assert backend.turn("still usable")["success"]
        assert old.close_calls == 1 and new.close_calls == 0
        detail = get_run(tmp_path, rec.run_id)
        assert detail["status"] == "success"
        assert any(log["level"] == "warning" for log in detail["logs"])
        assert "private provider detail" not in str(detail)
    finally:
        old.fail_close = False
        backend.close()
        rec.sink.close()
    assert new.close_calls == 1
    assert dependencies.sim.close_calls == dependencies.client.close_calls == 1


@pytest.mark.parametrize("agent", ["pickparts", "tiptop", "tiptop_optimized"])
def test_shutdown_attempts_all_resources_when_closes_fail(dependencies, agent, caplog):
    backend = RobotBackend(lambda frame: None, lambda stage: None)
    backend.select_agent(agent)
    dependencies.engines[-1].fail_close = True
    dependencies.sim.fail_close = True
    backend.close()
    assert dependencies.engines[-1].close_calls == 1
    assert dependencies.sim.close_calls == dependencies.client.close_calls == 1
    assert "private provider detail" not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize("target", ["tiptop", "tiptop_optimized"])
def test_failed_build_preserves_usable_engine(dependencies, target):
    backend = RobotBackend(lambda frame: None, lambda stage: None)
    old = backend.orchestrator
    dependencies.fail_build = True
    try:
        with pytest.raises(ValueError, match="build failed"):
            backend.select_agent(target)
        assert backend.agent == "pickparts" and backend.orchestrator is old
        assert old.close_calls == 0
        assert backend.turn("still usable")["success"]
    finally:
        backend.close()


def test_initialization_failure_still_closes_client_after_sim_close_failure(dependencies):
    dependencies.fail_build = True
    dependencies.sim.fail_close = True
    with pytest.raises(ValueError, match="build failed"):
        RobotBackend(lambda frame: None, lambda stage: None)
    assert dependencies.sim.close_calls == dependencies.client.close_calls == 1


def test_optimized_dispatch_callbacks_and_visual_result(dependencies):
    publish, stage, view = lambda frame: None, lambda name: None, lambda event: None
    backend = RobotBackend(publish, stage, view)
    try:
        backend.select_agent("tiptop_optimized")
        assert dependencies.optimized_args == (
            dependencies.sim, backend.llm, stage, view, backend.recorder)
        assert backend.orchestrator is None
        assert backend.models["vision"] == "本地 RGB-D"
        assert backend.turn("move")["verification"] == "visual"
        expected = {"success": False, "aborted": True, "recovery_required": True,
                    "message": "stopped", "rationale": "still held",
                    "verification": "visual", "completed_subtasks": 1, "total_subtasks": 2}
        backend.tiptop.run = lambda text: dict(expected)
        assert backend.turn("move") == expected
        optimized = backend.tiptop
        backend.select_agent("tiptop")
        assert optimized.close_calls == 1
        assert backend.turn("move")["verification"] == "open_loop"
    finally:
        backend.close()


@pytest.mark.parametrize("agent", ["tiptop", "tiptop_optimized"])
def test_reset_rebinds_tiptop_and_only_optimized_clears_held_state(dependencies, agent):
    frames = []
    backend = RobotBackend(frames.append, lambda name: None)
    try:
        backend.select_agent(agent)
        engine, old_sim = backend.tiptop, backend.sim
        backend.reset()
        assert backend.sim is not old_sim
        assert old_sim.close_calls == 1
        assert backend.sim.object_specs == old_sim.object_specs
        assert engine.sim is engine.executor.sim is engine.executor._ctx.sim is backend.sim
        assert callable(engine.executor.locate)
        assert engine.bind_calls == (1 if agent == "tiptop_optimized" else 0)
        assert engine.held == (None if agent == "tiptop_optimized" else "A")
        assert len(frames) == 2
        assert backend.turn("next")["success"]
    finally:
        backend.close()


@pytest.mark.parametrize("field", ["held", "candidate"])
def test_switch_rejected_while_optimized_grip_is_owned_or_uncertain(dependencies, field):
    backend = RobotBackend(lambda frame: None, lambda name: None)
    try:
        backend.select_agent("tiptop_optimized")
        original = backend.tiptop
        setattr(original.executor, field, "A")
        with pytest.raises(ValueError):
            backend.select_agent("tiptop")
        assert backend.tiptop is original
        assert original.close_calls == 0
        assert backend.agent == "tiptop_optimized"
    finally:
        backend.close()
