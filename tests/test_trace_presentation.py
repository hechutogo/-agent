"""Trace data accurately represents unfinished work, failures, and safe output."""
from pickparts_agent.observability import Recorder
from pickparts_agent.observability.query import get_run, list_runs


def test_run_metadata_and_root_outcome(tmp_path):
    rec = Recorder(tmp_path)
    with rec.run("command", "move", agent="tiptop") as handle:
        live = get_run(tmp_path, rec.run_id)
        assert live["agent"] == "tiptop"
        assert live["status"] == "running"
        assert live["duration_ms"] >= 0
        handle.set_result("incomplete", "规划无解")
    run = get_run(tmp_path, rec.run_id)
    assert run["spans"][0]["status"] == "incomplete"
    assert run["spans"][0]["end_ts"]
    assert list_runs(tmp_path)[0]["agent"] == "tiptop"


def test_metadata_and_log_events_are_redacted(tmp_path):
    rec = Recorder(tmp_path)
    with rec.run("command", "token sk-abcdefgh123456", agent="pickparts"):
        rec.log("info", "token sk-abcdefgh123456")
    assert "sk-abcdefgh123456" not in next(tmp_path.glob("trace-*.jsonl")).read_text()
    detail = get_run(tmp_path, rec.run_id)
    assert detail["logs"][0]["seq"] > 0


def test_day_queries_cannot_escape_trace_directory(tmp_path):
    # Query functions are used outside HTTP too, so validate at their boundary.
    assert list_runs(tmp_path, "../../private") == []


def test_exception_details_stay_in_local_log(tmp_path):
    import pytest

    rec = Recorder(tmp_path)
    with pytest.raises(RuntimeError):
        with rec.run("command", "move", agent="pickparts"):
            with rec.span("provider"):
                raise RuntimeError("private provider response")
    assert "private provider response" in next(tmp_path.glob("trace-*.jsonl")).read_text()
    detail = get_run(tmp_path, rec.run_id)
    assert "private provider response" not in str(detail)
    assert "private provider response" not in str(list_runs(tmp_path))
    assert detail["logs"][0]["exc_type"] == "RuntimeError"
