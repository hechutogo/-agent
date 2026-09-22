"""Manual daily fixtures capture run boundaries independently of Recorder."""
import json
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from pickparts_agent.interfaces.web import create_app
from pickparts_agent.observability import Recorder
from pickparts_agent.observability import query
from test_observability_runs_on_worker import ScriptedBackend


def write_day(root, day, events):
    path = root / f"trace-{day}.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events),
                    encoding="utf-8")


@pytest.fixture
def midnight(tmp_path):
    today = date.today()
    yesterday = today - timedelta(days=1)
    write_day(tmp_path, yesterday, [
        {"kind": "run_start", "run_id": "overnight", "seq": 1,
         "ts": f"{yesterday}T23:59:58+00:00", "task": "command",
         "goal": "move", "agent": "tiptop"},
        {"kind": "span_start", "run_id": "overnight", "seq": 2,
         "ts": f"{yesterday}T23:59:59+00:00", "span_id": "root",
         "name": "run:command"},
        {"kind": "log", "run_id": "overnight", "seq": 3,
         "ts": f"{yesterday}T23:59:59.500+00:00", "message": "before"},
    ])
    # Deliberately not in timestamp order: merging must be chronological.
    write_day(tmp_path, today, [
        {"kind": "run_end", "run_id": "overnight", "seq": 7,
         "ts": f"{today}T00:00:02+00:00", "status": "incomplete",
         "message": "not done", "duration_ms": 4000, "counts": {"atoms": 1}},
        {"kind": "span_end", "run_id": "overnight", "seq": 6,
         "ts": f"{today}T00:00:01+00:00", "span_id": "root",
         "status": "incomplete", "duration_ms": 2000},
        {"kind": "log", "run_id": "overnight", "seq": 4,
         "ts": f"{today}T00:00:00+00:00", "message": "after"},
        {"kind": "artifact", "run_id": "overnight", "seq": 5,
         "ts": f"{today}T00:00:00.500+00:00", "type": "state",
         "path": "artifacts/overnight/probe.json"},
        {"kind": "run_start", "run_id": "new-day", "seq": 8,
         "ts": f"{today}T01:00:00+00:00", "task": "reset", "agent": "pickparts"},
    ])
    return yesterday, today


def test_get_run_merges_start_spans_logs_and_completion(tmp_path, midnight):
    detail = query.get_run(tmp_path, "overnight")
    assert detail["goal"] == "move" and detail["agent"] == "tiptop"
    assert detail["status"] == "incomplete" and detail["duration_ms"] == 4000
    assert [log["message"] for log in detail["logs"]] == ["before", "after"]
    assert len(detail["spans"]) == 1
    assert detail["spans"][0]["status"] == "incomplete"
    assert detail["spans"][0]["duration_ms"] == 2000
    assert detail["artifacts"][0]["path"] == "artifacts/overnight/probe.json"


def test_list_runs_filters_start_date_and_includes_later_completion(tmp_path, midnight):
    yesterday, today = midnight
    runs = query.list_runs(tmp_path, yesterday)
    assert [run["run_id"] for run in runs] == ["overnight"]
    assert runs[0]["status"] == "incomplete"
    assert runs[0]["duration_ms"] == 4000 and runs[0]["counts"] == {"atoms": 1}
    assert [run["run_id"] for run in query.list_runs(tmp_path, today)] == ["new-day"]


def test_list_uses_shared_cached_reads_and_refreshes_changed_file(tmp_path, midnight, monkeypatch):
    yesterday, today = midnight
    query._CACHE.clear()
    original_open = type(tmp_path).open
    reads = []

    def counted_open(path, *args, **kwargs):
        if args and args[0] == "r":
            reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "open", counted_open)
    query.list_runs(tmp_path, yesterday)
    query.get_run(tmp_path, "overnight")
    query.list_runs(tmp_path, today)
    assert len(reads) == 2
    path = tmp_path / f"trace-{today}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "kind": "run_end", "run_id": "new-day",
            "ts": f"{today}T01:00:01+00:00", "status": "success",
        }) + "\n")
    assert query.list_runs(tmp_path, today)[0]["status"] == "success"
    assert len(reads) == 3


def test_listing_scans_each_retained_day_once_not_once_per_run(tmp_path, midnight, monkeypatch):
    yesterday, _ = midnight
    original_read = query._read_cached
    paths = []

    def read(path):
        paths.append(path)
        return original_read(path)

    monkeypatch.setattr(query, "_read_cached", read)
    runs = query.list_runs(tmp_path, yesterday, retention_days=1)
    assert runs[0]["status"] == "incomplete"
    assert len(paths) == len(set(paths)) == 2


def test_api_listing_honors_recorder_retention(tmp_path):
    old = date.today() - timedelta(days=8)
    write_day(tmp_path, old, [{
        "kind": "run_start", "run_id": "long-retained",
        "ts": f"{old}T23:59:59+00:00", "task": "command",
    }])
    write_day(tmp_path, date.today(), [{
        "kind": "run_end", "run_id": "long-retained",
        "ts": f"{date.today()}T00:00:01+00:00", "status": "success",
    }])
    rec = Recorder(tmp_path, retention_days=10)
    with TestClient(create_app(ScriptedBackend, rec)) as client:
        runs = client.get(f"/api/runs?day={old}").json()["runs"]
        assert len(runs) == 1 and runs[0]["status"] == "success"
        assert client.get("/api/runs/long-retained").json()["task"] == "command"
    rec.sink.close()
    assert query.list_runs(tmp_path, old, retention_days=7) == []
