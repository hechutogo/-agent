"""Agent selection must serialize with simulation work and retain run attribution."""
import threading

import pytest
from fastapi.testclient import TestClient

from pickparts_agent.interfaces.web import Console, create_app
from pickparts_agent.observability import NullRecorder, Recorder
from test_web import Backend, wait_for


class DualBackend(Backend):
    scene_objects = [{"id": "A", "kind": "block"}]

    def __init__(self, publish, stage, view=None, recorder=None):
        super().__init__(publish, stage, view, recorder)
        self.agent = "pickparts"
        self.view = view
        self.models = {"llm": "test-pickparts"}

    def select_agent(self, agent):
        assert threading.get_ident() == self.owner
        self.agent = agent
        self.models = {"llm": "test-" + agent}

    def turn(self, text):
        self.view({"type": "activity", "kind": "plan", "text": "生成计划"})
        result = super().turn(text)
        result["message"] = self.agent + ":" + text
        result["verification"] = "open_loop" if self.agent == "tiptop" else "visual"
        result["recovery_required"] = text == "abort"
        result["success"] = text != "abort"
        return result


def test_selection_preserves_scene_and_dispatches_selected_agent(tmp_path):
    backends = []

    def factory(*args, **kwargs):
        backend = DualBackend(*args, **kwargs)
        backends.append(backend)
        return backend

    with TestClient(create_app(factory, Recorder(tmp_path))) as client:
        initial = wait_for(client, lambda s: s["status"] == "ready")
        assert initial.get("agent") == "pickparts"
        assert client.post("/api/agent", json={"agent": "invalid"}).status_code == 422
        assert client.post("/api/agent", json={"agent": "tiptop"}).status_code == 202
        selected = wait_for(client, lambda s: s["status"] == "ready" and s["agent"] == "tiptop")
        assert selected["scene_objects"] == initial["scene_objects"]
        assert selected["models"]["llm"] == "test-tiptop"
        assert len(backends) == 1
        client.post("/api/command", json={"text": "move"})
        busy = wait_for(client, lambda s: s["stage"] == "抓取")
        assert busy["run_id"] != initial["run_id"]
        live = client.get("/api/runs/" + busy["run_id"]).json()
        assert live["agent"] == "tiptop" and live["status"] == "running"
        assert client.post("/api/agent", json={"agent": "pickparts"}).status_code == 409
        backends[0].release.set()
        done = wait_for(client, lambda s: s["status"] == "ready")
        assert done["messages"][-1]["text"] == "tiptop:move"
        assert done["messages"][-1]["agent"] == "tiptop"
        assert done["messages"][-1]["verification"] == "open_loop"
        detail = client.get("/api/runs/" + done["run_id"]).json()
        assert detail["status"] == "success"
        assert any(log["message"] == "生成计划" for log in detail["logs"])
        assert client.get("/api/runs").json()["runs"][0]["agent"] == "tiptop"
        client.post("/api/command", json={"text": "abort"})
        wait_for(client, lambda s: s["recovery_required"])
        assert client.post("/api/agent", json={"agent": "pickparts"}).status_code == 409


def test_switch_failure_keeps_previous_agent_and_can_recover(tmp_path):
    class Failing(DualBackend):
        def select_agent(self, agent):
            raise RuntimeError("private provider detail")

    with TestClient(create_app(Failing, Recorder(tmp_path))) as client:
        wait_for(client, lambda s: s["status"] == "ready")
        assert client.post("/api/agent", json={"agent": "tiptop"}).status_code == 202
        state = wait_for(client, lambda s: s["status"] == "ready" and bool(s["error"]))
        assert state["agent"] == "pickparts"
        assert "private provider" not in str(state)
        assert not state["recovery_required"]


def test_history_query_rejects_invalid_date_and_limit(tmp_path):
    with TestClient(create_app(DualBackend, Recorder(tmp_path))) as client:
        assert client.get("/api/runs?day=../../bad").status_code == 422
        assert client.get("/api/runs?limit=100000").status_code == 422


def test_failed_switch_preserves_completed_workflow(tmp_path):
    class Completed(DualBackend):
        def turn(self, text):
            self.view({"type": "plan", "goal": text, "current": 0,
                       "steps": [{"index": 0, "status": "done"}]})
            self.view({"type": "finish", "success": True})
            return {"success": True, "message": "done"}

        def select_agent(self, agent):
            raise RuntimeError("private provider detail")

    with TestClient(create_app(Completed, Recorder(tmp_path))) as client:
        wait_for(client, lambda s: s["status"] == "ready")
        client.post("/api/command", json={"text": "move"})
        done = wait_for(client, lambda s: s["status"] == "ready" and s["workflow"])
        assert done["workflow"]["status"] == "done"
        client.post("/api/agent", json={"agent": "tiptop"})
        failed = wait_for(client, lambda s: s["status"] == "ready" and s["error"])
        assert failed["workflow"] == done["workflow"]
        assert failed["messages"] == done["messages"]
        assert failed["agent"] == "pickparts"


def test_subtask_snapshots_replace_without_overwriting_motion_or_react():
    console = Console(Backend, NullRecorder())
    first = {"type": "subtasks", "current": 0, "steps": [
        {"index": 0, "instruction": "stack A", "status": "active"},
        {"index": 1, "instruction": "store B", "status": "pending"}]}
    console.view(first)
    first["steps"][0]["status"] = "failed"
    assert console.snapshot()["subtasks"]["steps"][0]["status"] == "active"
    console.view({"type": "plan", "goal": "stack A", "current": 0,
                  "steps": [{"index": 0, "atom": "move", "args": {}, "status": "pending"}]})
    console.view({"type": "step", "index": 0, "status": "done"})
    console.view({"type": "react", "attempt": 1, "thought": "observe",
                  "decision": "retry", "detail": "retry grasp"})
    second = {"type": "subtasks", "current": 1, "steps": [
        {"index": 0, "instruction": "stack A", "status": "done"},
        {"index": 1, "instruction": "store B", "status": "active"}]}
    console.view(second)
    snapshot = console.snapshot()
    assert snapshot["subtasks"] == {"current": 1, "steps": second["steps"]}
    assert snapshot["workflow"]["steps"][0]["status"] == "done"
    assert snapshot["react"][0]["decision"] == "retry"
    console.view({"type": "plan", "goal": "store B", "current": 0, "steps": []})
    assert console.snapshot()["subtasks"] == snapshot["subtasks"]
    console.state["status"] = "ready"
    console.submit("command", "next task")
    assert console.snapshot()["subtasks"] is None
    assert console.snapshot()["workflow"] is None
    assert console.snapshot()["react"] == []


@pytest.mark.parametrize("cleanup", ["reset", "agent"])
@pytest.mark.parametrize("success", [True, False])
def test_optimized_command_result_history_and_lifecycle(tmp_path, cleanup, success):
    result = {"success": success, "aborted": not success, "recovery_required": not success,
              "verification": "visual", "message": "verified" if success else "stopped",
              "rationale": "observed scene", "completed_subtasks": 2 if success else 1,
              "total_subtasks": 2}

    class Optimized(DualBackend):
        def turn(self, text):
            assert threading.get_ident() == self.owner
            self.view({"type": "subtasks", "current": 1, "steps": [
                {"index": 0, "instruction": "stack A", "status": "done"},
                {"index": 1, "instruction": "store B", "status": "done" if success else "failed"}]})
            return dict(result)

    with TestClient(create_app(Optimized, Recorder(tmp_path))) as client:
        initial = wait_for(client, lambda s: s["status"] == "ready")
        labels = {item["id"]: item["name"] for item in initial["agents"]}
        assert labels["tiptop"] == "TiPToP 初版"
        assert labels["tiptop_optimized"] == "TiPToP 优化版"
        assert client.post("/api/agent", json={"agent": "tiptop_optimized"}).status_code == 202
        wait_for(client, lambda s: s["status"] == "ready" and s["agent"] == "tiptop_optimized")
        assert client.post("/api/command", json={"text": "two tasks"}).status_code == 202
        done = wait_for(client, lambda s: s["status"] == "ready" and bool(s["messages"]))
        message = done["messages"][-1]
        assert {key: message[key if key != "message" else "text"] for key in result} == result
        assert done["subtasks"]["steps"][1]["status"] == ("done" if success else "failed")
        assert done["recovery_required"] is not success
        detail = client.get("/api/runs/" + done["run_id"]).json()
        assert detail["agent"] == "tiptop_optimized"
        assert detail["status"] == ("success" if success else "incomplete")
        if not success:
            assert client.post("/api/agent", json={"agent": "tiptop"}).status_code == 409
        if cleanup == "reset" or not success:
            assert client.post("/api/reset").status_code == 202
            cleared = wait_for(client, lambda s: s["status"] == "ready" and not s["messages"])
            assert cleared["agent"] == "tiptop_optimized"
        else:
            assert client.post("/api/agent", json={"agent": "tiptop"}).status_code == 202
            cleared = wait_for(client, lambda s: s["status"] == "ready" and s["agent"] == "tiptop")
        assert cleared["subtasks"] is None
