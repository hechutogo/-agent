import json

from pickparts_agent.observability import Recorder, NullRecorder


def _rows(root):
    path = next(root.glob("trace-*.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_records_lifecycle_and_counts(tmp_path):
    rec = Recorder(tmp_path)
    with rec.run("command", "把 A 放 B") as job:
        with rec.span("plan"):
            with rec.span("llm:plan", model="m"):
                pass
        with rec.span("execute"):
            sp = rec.start_span("atom:grasp", target="A")
            rec.end_span(sp, "ok")
        job.set_result("success", "已完成")
    rows = _rows(tmp_path)
    assert rows[0]["kind"] == "run_start" and rows[-1]["kind"] == "run_end"
    end = rows[-1]
    assert end["status"] == "success" and end["counts"]["atoms"] == 1
    assert end["counts"]["llm_calls"] == 1


def test_exception_marks_run_error_and_reraises(tmp_path):
    rec = Recorder(tmp_path)
    try:
        with rec.run("reset"):
            raise RuntimeError("x")
    except RuntimeError:
        pass
    assert _rows(tmp_path)[-1]["status"] == "error"


def test_null_recorder_is_usable():
    rec = NullRecorder()
    with rec.run("command", "g") as job:
        with rec.span("atom:x", target="A"):
            rec.log("info", "m"); job.set_result("success", "ok")


def test_save_state_and_level_gate(tmp_path):
    rec = Recorder(tmp_path, level="warning")
    with rec.run("add"):
        rec.save_state({"x": 1}, "state-after")
        rec.log("info", "hidden")
    rows = _rows(tmp_path)
    assert any(r["kind"] == "artifact" for r in rows)
    assert not any(r["kind"] == "log" for r in rows)


def test_save_rgbd_writes_artifact_event(tmp_path):
    import numpy as np

    class _Frame:
        rgb = np.zeros((4, 4, 3), np.uint8); depth = np.ones((4, 4), np.float32)
        intrinsic = np.eye(3); camera_to_base = np.eye(4); qpos = np.zeros(2)

    rec = Recorder(tmp_path)
    with rec.run("command", "g"):
        sp = rec.start_span("atom:grasp", target="A")
        rec.save_rgbd(_Frame(), "失败现场")
        rec.end_span(sp, "error", "lost_object", "未定位")
    rows = _rows(tmp_path)
    assert any(r["kind"] == "artifact" and r["type"] == "rgbd" for r in rows)
