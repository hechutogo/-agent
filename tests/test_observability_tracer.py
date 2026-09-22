import pytest

from pickparts_agent.observability.tracer import Tracer


def collect():
    events = []
    return events, lambda kind, payload: events.append((kind, payload))


def test_begin_end_emits_pair_and_parent():
    events, emit = collect()
    tr = Tracer(emit, "r1")
    root = tr.begin("run:command")
    child = tr.begin("plan")
    assert child.parent_id == root.span_id
    tr.end(child, "ok"); tr.end(root, "ok")
    kinds = [k for k, _ in events]
    assert kinds == ["span_start", "span_start", "span_end", "span_end"]
    end = events[2][1]
    assert end["status"] == "ok" and end["duration_ms"] >= 0


def test_span_context_marks_error_and_reraises():
    events, emit = collect()
    tr = Tracer(emit, "r1")
    with pytest.raises(ValueError):
        with tr.span("atom:grasp", target="A"):
            raise ValueError("boom")
    assert events[1][1]["status"] == "error"
    assert events[1][1]["error_kind"] == "ValueError"


def test_nested_context_restores_parent():
    events, emit = collect()
    tr = Tracer(emit, "r1")
    with tr.span("root"):
        with tr.span("child"):
            assert tr.current().name == "child"
        assert tr.current().name == "root"
