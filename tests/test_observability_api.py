import time

import pytest
from fastapi.testclient import TestClient

from pickparts_agent.observability import Recorder
from pickparts_agent.observability.query import (
    artifact_path, get_run, list_runs)
from test_observability_runs_on_worker import ScriptedBackend


def _record_runs(tmp_path):
    rec = Recorder(tmp_path)
    with rec.run("reset"):
        pass
    with rec.run("command", "把 A 放到 B"):
        with rec.span("atom:find_object", target="A"):
            pass
    runs = list_runs(tmp_path)
    assert [r["task"] for r in runs] == ["command", "reset"]
    command = runs[0]
    assert command["goal"] == "把 A 放到 B" and command["status"] == "success"
    return command["run_id"]


def test_list_runs_newest_first(tmp_path):
    _record_runs(tmp_path)


def test_get_run_builds_spans(tmp_path):
    run_id = _record_runs(tmp_path)
    detail = get_run(tmp_path, run_id)
    names = [s["name"] for s in detail["spans"]]
    assert names[0] == "run:command" and "atom:find_object" in names
    atom = next(s for s in detail["spans"] if s["name"] == "atom:find_object")
    assert atom["attrs"].get("target") == "A"
    assert get_run(tmp_path, "00000101-000000-dead") is None


def test_artifact_path_blocks_traversal(tmp_path):
    rec = Recorder(tmp_path)
    with rec.run("command", "g"):
        rec.save_state({"x": 1}, "probe")
    run_id = list_runs(tmp_path)[0]["run_id"]
    path = artifact_path(tmp_path, run_id, "probe.json")
    assert path is not None and path.is_file()
    assert artifact_path(tmp_path, run_id, "missing.json") is None
    assert artifact_path(tmp_path, run_id, "../../etc/passwd") is None
    assert artifact_path(tmp_path, "../evil", "x") is None
    assert artifact_path(tmp_path, run_id, "/etc/passwd") is None


class ArtifactBackend(ScriptedBackend):
    def __init__(self, publish, stage, view=None, recorder=None):
        super().__init__(publish, stage, view, recorder)
        self._recorder = recorder

    def turn(self, text):
        self._recorder.save_state({"ok": 1}, "probe")
        return {"success": True, "message": "done"}


def _wait(client, predicate):
    deadline = time.monotonic() + 3
    state = client.get("/api/state").json()
    while time.monotonic() < deadline:
        state = client.get("/api/state").json()
        if predicate(state):
            return state
        time.sleep(.01)
    pytest.fail(f"timed out: {state}")


def test_runs_api_endpoints(tmp_path):
    from pickparts_agent.interfaces.web import create_app
    recorder = Recorder(tmp_path)
    app = create_app(factory=ArtifactBackend, recorder=recorder)
    with TestClient(app) as client:
        _wait(client, lambda s: s["status"] == "ready")
        client.post("/api/command", json={"text": "把 A 放进盒子"})
        _wait(client, lambda s: s["status"] == "ready" and any(
            m["role"] == "assistant" for m in s["messages"]))

        runs = client.get("/api/runs").json()["runs"]
    command = next(r for r in runs if r["task"] == "command")
    run_id = command["run_id"]
    detail = client.get(f"/api/runs/{run_id}").json()
    assert any(a["type"] == "state" for a in detail["artifacts"])
    rel = next(a["path"] for a in detail["artifacts"] if a["type"] == "state")
    name = rel.split("/", 2)[-1]  # drop artifacts/<run_id>/
    resp = client.get(f"/api/runs/{run_id}/artifacts/{name}")
    assert resp.status_code == 200 and resp.json() == {"ok": 1}
    bad = client.get(f"/api/runs/{run_id}/artifacts/../../etc/passwd")
    assert bad.status_code in (403, 404)
    assert client.get("/api/runs/19990101-000000-dead").status_code == 404
