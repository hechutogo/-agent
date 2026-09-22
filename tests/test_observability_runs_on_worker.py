import json
import time

import pytest
from fastapi.testclient import TestClient


class ScriptedBackend:
    def __init__(self, publish, stage, view=None, recorder=None):
        self.models = {}
        self.scene_objects = []

    def turn(self, text):
        return {"success": True, "message": text + "完成。"}

    def add_object(self, kind):
        pass

    def reset(self):
        pass

    def close(self):
        pass


def _wait(client, predicate):
    deadline = time.monotonic() + 3
    state = client.get("/api/state").json()
    while time.monotonic() < deadline:
        state = client.get("/api/state").json()
        if predicate(state):
            return state
        time.sleep(.01)
    pytest.fail(f"State transition timed out: {state}")


def _events(tmp_path):
    path = next(tmp_path.glob("trace-*.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_worker_records_one_run_per_job(tmp_path):
    from pickparts_agent.interfaces.web import create_app
    from pickparts_agent.observability import Recorder
    recorder = Recorder(tmp_path)
    app = create_app(factory=ScriptedBackend, recorder=recorder)
    with TestClient(app) as client:
        _wait(client, lambda s: s["status"] == "ready")
        assert client.post("/api/command", json={"text": "把 A 放进盒子"}).status_code == 202
        _wait(client, lambda s: s["status"] == "ready" and any(
            m["role"] == "assistant" for m in s["messages"]))
        assert client.post("/api/objects", json={"kind": "block"}).status_code == 202
        _wait(client, lambda s: s["status"] == "ready")
        assert client.post("/api/reset").status_code == 202
        _wait(client, lambda s: s["status"] == "ready")

    events = _events(tmp_path)
    tasks = [e["task"] for e in events if e["kind"] == "run_start"]
    assert tasks == ["reset", "command", "add", "reset"]
    assert sum(e["kind"] == "run_end" for e in events) == 4
    command_end = [e for e in events
                   if e["kind"] == "run_end" and e.get("message") == "把 A 放进盒子完成。"]
    assert command_end and command_end[0]["status"] == "success"
