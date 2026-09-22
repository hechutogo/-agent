"""TiPToP trace and event contracts, without physics or network calls."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from helpers import FakeChat, FakeKin, FakeSim
from pickparts_agent.agent.atoms.base import AtomResult
from pickparts_agent.observability import NullRecorder, Recorder
from test_tiptop_agent import SPECS, make_scene
from test_tiptop_executor import make_plan
from tiptop_mac.agent import TiPToPAgent, build_tiptop_agent
from tiptop_mac.executor import OpenLoopExecutor
from tiptop_mac.grounding import Grounder, GroundingError
from tiptop_mac.tamp import TAMPLite, TAMPError
from tiptop_mac.types import MotionStep


class ObservedSim(FakeSim):
    def __init__(self):
        super().__init__()
        self.object_specs = SPECS
        self.observations = 0
        self.closes = 0

    def observe(self):
        self.observations += 1
        frame = super().observe()
        frame.intrinsic = np.eye(3)
        frame.camera_to_base = np.eye(4)
        return frame

    def close(self):
        self.closes += 1


def read_trace(root):
    return [json.loads(line) for path in root.glob("trace-*.jsonl")
            for line in path.read_text().splitlines()]


def spans(root):
    trace = read_trace(root)
    ends = {e["span_id"]: e for e in trace if e["kind"] == "span_end"}
    return {e["name"]: (e, ends[e["span_id"]])
            for e in trace if e["kind"] == "span_start"}


def instrumented_agent(recorder, *, support="B"):
    sim = ObservedSim()
    chat = FakeChat([{"goal": [{"predicate": "on", "args": ["A", support]}],
                     "rationale": "Place A"}])
    scene = make_scene()
    events, stages, perceptions = [], [], []

    def perceive(frame, specs):
        perceptions.append(frame)
        return scene

    def locate(frame, target):
        node = scene.objects[target]
        return {"point": node.point, "bottom_z": node.bottom_z}

    executor = OpenLoopExecutor(
        sim, FakeKin(), locate=locate, recorder=recorder,
        on_event=events.append, on_stage=stages.append)
    agent = TiPToPAgent(
        sim, Grounder(chat), TAMPLite(FakeKin()), executor,
        perceive=perceive, recorder=recorder,
        on_event=events.append, on_stage=stages.append)
    return agent, events, stages, perceptions, chat


def test_agent_publishes_one_plan_and_open_loop_result_with_nested_trace(tmp_path):
    recorder = Recorder(tmp_path)
    agent, events, stages, perceptions, chat = instrumented_agent(recorder)
    with recorder.run("command", "Place A on B"):
        result = agent.run("Place A on B")
    assert result["success"] and not result["recovery_required"]
    assert result["verification"] == "open_loop"
    assert len(chat.calls) == len(perceptions) == 1
    plans = [event for event in events if event["type"] == "plan"]
    assert len(plans) == 1
    plan = plans[0]
    assert plan["goal"] == "Place A on B" and plan["current"] == 0
    assert plan["steps"][0] == {
        "index": 0, "atom": "move",
        "args": {"position": [0.0, -0.45, 0.859], "jaw": 0.8, "label": ""},
        "status": "pending",
    }
    step_events = [e for e in events if e["type"] == "step"]
    assert step_events == [
        {"type": "step", "index": i, "status": status}
        for i in range(len(plan["steps"])) for status in ("active", "done")]
    assert events.index(plan) < events.index(step_events[0])
    assert [e for e in events if e["type"] == "finish"] == [
        {"type": "finish", "success": True}]
    assert {"理解指令", "场景感知", "TAMP 规划", "开环执行"} <= set(stages)
    assert {"understanding", "observation", "plan", "action"} <= {
        e["kind"] for e in events if e["type"] == "activity"}
    recorded = spans(tmp_path)
    agent_span = recorded["tiptop"][0]["span_id"]
    for name in ("ground", "perceive", "plan", "execute"):
        assert recorded[name][0]["parent_id"] == agent_span
        assert recorded[name][1]["status"] == "ok"
    assert recorded["atom:calibrate"][0]["parent_id"] == recorded["execute"][0]["span_id"]
    artifacts = [e for e in read_trace(tmp_path) if e["kind"] == "artifact"]
    assert sum(e["type"] == "rgbd" for e in artifacts) == 2
    states = [json.loads((tmp_path / e["path"]).read_text())
              for e in artifacts if e["type"] == "state"]
    assert any("objects" in state and "qpos" in state for state in states)
    assert any(state.get("goal") == "Place A on B" and "steps" in state
               for state in states)
    assert any(state.get("verification") == "open_loop" for state in states)
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize("stage,error", [
    ("ground", GroundingError("private backend detail sk-123456789secret")),
    ("perceive", ValueError("private backend detail sk-123456789secret")),
    ("plan", TAMPError("private backend detail sk-123456789secret")),
])
def test_early_failure_ends_failed_spans_without_motion_or_recovery(tmp_path, stage, error):
    recorder = Recorder(tmp_path)
    agent, events, _, _, _ = instrumented_agent(recorder)

    def fail(*args, **kwargs):
        raise error

    if stage == "ground":
        agent.grounder.ground = fail
    elif stage == "perceive":
        agent.perceive = fail
    else:
        agent.planner.plan = fail
    with recorder.run("command"):
        result = agent.run("Place A on B")
    assert not result["success"] and not result["aborted"]
    assert not result["recovery_required"] and result["verification"] == "open_loop"
    assert agent.sim.moves == []
    assert not any(e["type"] in ("plan", "step") for e in events)
    assert events[-1] == {"type": "finish", "success": False}
    public = json.dumps([events, result])
    assert "private backend detail" not in public and "sk-123456789secret" not in public
    recorded = spans(tmp_path)
    assert recorded[stage][1]["status"] == recorded["tiptop"][1]["status"] == "error"
    assert recorded[stage][1]["error_kind"] == type(error).__name__


@pytest.mark.parametrize("failure", ["tracking", "returned", "unknown", "calibration"])
def test_executor_failure_marks_actual_step_and_skips_rest(tmp_path, monkeypatch, failure):
    recorder, sim, events = Recorder(tmp_path), ObservedSim(), []
    failed_step = MotionStep("move", np.array([0., -.45, .9]), .8)
    if failure == "returned":
        monkeypatch.setattr("tiptop_mac.executor._cartesian",
                            lambda *a, **k: AtomResult(False, "verify", "private detail"))
    elif failure == "unknown":
        failed_step = MotionStep("teleport", None, .8)
    elif failure == "calibration":
        failed_step = MotionStep("calibrate", None, None, "invalid")
    else:
        original_move = sim.move_right

        def fail_second_move(*args, **kwargs):
            if sim.moves:
                raise RuntimeError("private backend detail sk-123456789secret")
            return original_move(*args, **kwargs)

        sim.move_right = fail_second_move
    plan = make_plan([MotionStep("reset", None, .8), failed_step,
                      MotionStep("reset", None, .8)])
    executor = OpenLoopExecutor(sim, FakeKin(), recorder=recorder, on_event=events.append)
    with recorder.run("command"):
        result = executor.execute(plan)
    assert not result["success"] and result["aborted"]
    assert len(sim.moves) == 1
    assert [e for e in events if e["type"] == "step"] == [
        {"type": "step", "index": 0, "status": "active"},
        {"type": "step", "index": 0, "status": "done"},
        {"type": "step", "index": 1, "status": "active"},
        {"type": "step", "index": 1, "status": "failed"},
        {"type": "step", "index": 2, "status": "skipped"},
    ]
    assert events[-1] == {"type": "finish", "success": False}
    assert "private" not in json.dumps([events, result])
    recorded = spans(tmp_path)
    assert recorded["atom:" + failed_step.kind][1]["status"] == "error"
    assert recorded["atom:reset"][1]["status"] == "ok"
    assert recorded["execute"][1]["status"] == "error"
    assert any(e["kind"] == "artifact" and e["type"] == "rgbd"
               for e in read_trace(tmp_path))


def test_agent_abort_requires_recovery_and_does_not_repeat_grounding(tmp_path):
    recorder = Recorder(tmp_path)
    agent, events, _, perceptions, chat = instrumented_agent(recorder)
    agent.sim.tracking_error = True
    with recorder.run("command"):
        result = agent.run("Place A on B")
    assert result["aborted"] and result["recovery_required"]
    assert result["verification"] == "open_loop"
    assert len(perceptions) == len(chat.calls) == 1
    assert spans(tmp_path)["tiptop"][1]["status"] == "error"
    assert len([e for e in events if e["type"] == "finish"]) == 1


def test_satisfied_goal_emits_empty_plan_and_finish_without_new_observations():
    agent, events, _, perceptions, _ = instrumented_agent(NullRecorder(), support="table")
    result = agent.run("Leave A on table")
    assert result["success"] and result["verification"] == "open_loop"
    assert agent.sim.moves == [] and agent.sim.observations == 1
    assert len(perceptions) == 1
    assert [e for e in events if e["type"] == "plan"] == [
        {"type": "plan", "goal": "Leave A on table", "current": 0, "steps": []}]
    assert events[-1] == {"type": "finish", "success": True}


class Client:
    def __init__(self):
        self.closes = 0
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.complete))

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        content = '{"goal":[{"predicate":"on","args":["A","table"]}],"rationale":""}'
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content))])

    def close(self):
        self.closes += 1


def test_factory_wires_grounder_trace_and_closes_only_owned_client(tmp_path, monkeypatch):
    client, sim, events = Client(), ObservedSim(), []
    endpoint = SimpleNamespace(client=lambda: client, model="test", base_url="https://example.com/v1")
    monkeypatch.setattr("tiptop_mac.agent.ArmKinematics", FakeKin)
    recorder = Recorder(tmp_path)
    agent = build_tiptop_agent(sim, endpoint, recorder=recorder, on_event=events.append)
    agent.perceive = lambda frame, specs: make_scene()
    with recorder.run("command"):
        assert agent.run("Leave A on table")["success"]
    recorded = spans(tmp_path)
    assert recorded["llm:grounder"][0]["parent_id"] == recorded["ground"][0]["span_id"]
    assert len(client.calls) == 1
    assert list((tmp_path / "artifacts").rglob("*.resp.txt"))
    assert len([e for e in events if e["type"] == "finish"]) == 1
    agent.close()
    agent.close()
    assert client.closes == 1 and sim.closes == 0


def test_factory_cleanup_on_assembly_failure(monkeypatch):
    client = Client()
    endpoint = SimpleNamespace(client=lambda: client, model="test", base_url="https://example.com/v1")

    def fail():
        raise RuntimeError("kinematics unavailable")

    monkeypatch.setattr("tiptop_mac.agent.ArmKinematics", fail)
    sim = ObservedSim()
    with pytest.raises(RuntimeError, match="kinematics unavailable"):
        build_tiptop_agent(sim, endpoint)
    assert client.closes == 1 and sim.closes == 0


def test_direct_agent_close_does_not_close_borrowed_resources():
    agent, _, _, _, _ = instrumented_agent(NullRecorder())
    client = Client()
    agent.grounder.chat = client
    agent.close()
    assert client.closes == agent.sim.closes == 0
