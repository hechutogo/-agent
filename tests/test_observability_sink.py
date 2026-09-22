import json
from datetime import date, timedelta

from pickparts_agent.observability.sink import JSONLSink


def test_write_appends_today_file(tmp_path):
    sink = JSONLSink(tmp_path, retention_days=7)
    sink.write({"kind": "run_start", "run_id": "r", "seq": 1})
    sink.write({"kind": "run_end", "run_id": "r", "seq": 2})
    sink.close()
    path = tmp_path / f"trace-{date.today().isoformat()}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["kind"] == "run_start" and rows[1]["seq"] == 2


def test_cleanup_removes_old_files_only(tmp_path):
    old = tmp_path / f"trace-{(date.today()-timedelta(days=10)).isoformat()}.jsonl"
    keep = tmp_path / f"trace-{date.today().isoformat()}.jsonl"
    old.write_text("{}\n"); keep.write_text("{}\n")
    JSONLSink(tmp_path, retention_days=7).cleanup()
    assert not old.exists() and keep.exists()


def test_cleanup_removes_old_artifact_dirs(tmp_path):
    artifacts = tmp_path / "artifacts"
    stale = artifacts / "old-run"
    fresh = artifacts / "fresh-run"
    stale.mkdir(parents=True); fresh.mkdir(parents=True)
    import os
    old_ts = (date.today() - timedelta(days=30)).strftime("%Y%m%d%H%M%S")
    os.utime(stale, (0, 0))
    JSONLSink(tmp_path, retention_days=7).cleanup()
    assert not stale.exists() and fresh.exists()
