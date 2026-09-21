"""Service contracts: serialized robot commands, reconnectable history, recovery."""
import importlib
import threading
import time

import pytest
from fastapi.testclient import TestClient


def module():
    try:
        return importlib.import_module("pickparts_agent.interfaces.web")
    except ImportError:
        pytest.fail("Web service is not implemented")


def wait_for(client, predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.get("/api/state").json()
        if predicate(state):
            return state
        time.sleep(.01)
    pytest.fail(f"State transition timed out: {state}")


class Backend:
    """Only replaces external physics/cloud, retains real service scheduling."""
    def __init__(self, publish, stage, view=None):
        self.stage = stage
        self.release = threading.Event()
        self.owner = threading.get_ident()
        self.closed = False

    def turn(self, text):
        assert threading.get_ident() == self.owner
        self.stage("抓取")
        assert self.release.wait(2)
        return {"success": True, "message": text + "完成。还需要什么？"}

    def close(self):
        assert threading.get_ident() == self.owner
        self.closed = True


def test_commands_serialize_history_survives_reconnect_and_reset():
    web = module()
    backends = []
    def factory(publish, stage, view=None):
        backend = Backend(publish, stage, view)
        backends.append(backend)
        return backend
    app = web.create_app(factory=factory)
    with TestClient(app) as client:
        wait_for(client, lambda s: s["status"] == "ready")
        initial = client.get("/api/state").json()
        assert initial["workflow"] is None and initial["react"] == []
        assert client.post("/api/command", json={"text": "把零件 A 放进盒子"}).status_code == 202
        wait_for(client, lambda s: s["stage"] == "抓取")
        assert client.post("/api/command", json={"text": "把零件 B 放进盒子"}).status_code == 409
        assert client.post("/api/reset").status_code == 409
        backends[0].release.set()
        state = wait_for(client, lambda s: s["status"] == "ready")
        assert [m["role"] for m in state["messages"]] == ["user", "assistant"]
        assert state["messages"][-1]["success"] is True
        assert client.get("/api/state").json()["messages"] == state["messages"]
        assert client.post("/api/command", json={"text": "把零件 B 放进盒子"}).status_code == 202
        state = wait_for(client, lambda s: len(s["messages"]) == 4)
        assert "B" in state["messages"][-1]["text"]
        assert client.post("/api/reset").status_code == 202
        state = wait_for(client, lambda s: s["status"] == "ready" and not s["messages"])
        assert backends[0].closed
    assert backends[-1].closed


def test_input_origin_and_initialization_failure():
    web = module()
    def fail(*args):
        raise RuntimeError("private-token")
    with TestClient(web.create_app(factory=fail)) as client:
        state = wait_for(client, lambda s: s["status"] == "error")
        assert "private-token" not in str(state)
        assert client.post("/api/command", json={"text": "   "}).status_code == 422
        assert client.post("/api/command", json={"text": "A"}).status_code == 409
        assert client.post("/api/reset", headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.get("/api/frame").status_code == 204


def test_object_add_runs_on_worker_and_scene_inventory_is_published():
    web = module()
    class SceneBackend(Backend):
        scene_objects = [{"id": "A", "kind": "block"}]

        def add_object(self, kind):
            assert threading.get_ident() == self.owner
            self.scene_objects = self.scene_objects + [{"id": "C", "kind": kind}]

    with TestClient(web.create_app(factory=SceneBackend)) as client:
        wait_for(client, lambda s: s["status"] == "ready")
        assert client.post("/api/objects", json={"kind": "sphere"}).status_code == 422
        assert client.post("/api/objects", json={"kind": "block"}).status_code == 202
        state = wait_for(client, lambda s: s["status"] == "ready" and len(s.get("scene_objects", [])) == 2)
        assert state["scene_objects"][-1]["id"] == "C"


def test_activity_events_are_reconnectable_and_react_is_bounded():
    console = module().Console(Backend)
    console.view({"type": "activity", "kind": "plan", "text": "先定位，再抓取"})
    assert console.snapshot()["messages"][-1]["role"] == "activity"
    for i in range(120):
        console.view({"type": "react", "attempt": i, "thought": "重新观测",
                      "decision": "retry", "detail": "定位丢失"})
    assert len(console.snapshot()["react"]) <= 100
    assert console.snapshot()["react"][-1]["attempt"] == 119
