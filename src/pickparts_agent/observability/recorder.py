"""Recorder facade: one run = one root span plus logs, child spans, artifacts."""
from __future__ import annotations

import os
import sys
import time
import traceback as tb
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .artifacts import ArtifactStore
from .events import now_iso, redact
from .sink import JSONLSink
from .tracer import Span, Tracer

_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}


def _default_root() -> Path:
    return Path(__file__).resolve().parents[3] / "logs"


class RunHandle:
    def __init__(self):
        self.status = "success"
        self.message = ""
        self.recovery_required = False

    def set_result(self, status, message, recovery_required=False):
        self.status = status
        self.message = message
        self.recovery_required = recovery_required


class Recorder:
    def __init__(self, root=None, retention_days=7, level="info"):
        self.root = Path(root) if root else _default_root()
        self.sink = JSONLSink(self.root, retention_days)
        self.artifacts = ArtifactStore(self.root)
        self._level = _LEVELS.get(str(level).lower(), 20)
        self._seq = 0
        self.run_id = None
        self.tracer = None
        self._counts = {"atoms": 0, "react_iters": 0, "llm_calls": 0}
        self._llm_index = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(self, kind: str, payload: dict) -> None:
        self.sink.write({"ts": now_iso(), "run_id": self.run_id,
                         "seq": self._next_seq(), "kind": kind, **payload})

    @contextmanager
    def run(self, task, goal=None, *, agent=None):
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        self.tracer = Tracer(self._emit, self.run_id)
        self._counts = {"atoms": 0, "react_iters": 0, "llm_calls": 0}
        self._llm_index = 0
        start = time.perf_counter()
        self._emit("run_start", {"task": task, "goal": redact(goal) if goal is not None else None,
                                 "agent": agent})
        root = self.tracer.begin(f"run:{task}", task=task)
        handle = RunHandle()
        try:
            yield handle
        except Exception as exc:
            self.log("error", f"{type(exc).__name__}: {exc}", logger="worker", exc_info=True)
            self.tracer.end(root, "error", type(exc).__name__, str(exc))
            self._write_run_end("error", redact(str(exc)), False, start)
            raise
        else:
            self.tracer.end(root, "ok" if handle.status == "success" else handle.status,
                            message=handle.message)
            self._write_run_end(handle.status, handle.message,
                                handle.recovery_required, start)

    def _write_run_end(self, status, message, recovery_required, start):
        self._emit("run_end", {"status": status, "message": redact(message),
                               "recovery_required": bool(recovery_required),
                               "counts": dict(self._counts),
                               "duration_ms": max(0, round((time.perf_counter() - start) * 1000, 2))})

    def _count(self, name):
        if name.startswith("atom:"):
            self._counts["atoms"] += 1
        elif name == "react":
            self._counts["react_iters"] += 1
        elif name.startswith("llm:"):
            self._counts["llm_calls"] += 1

    @contextmanager
    def span(self, name, **attrs):
        self._count(name)
        with self.tracer.span(name, **_clean_attrs(attrs)):
            yield

    def start_span(self, name, **attrs) -> Span:
        self._count(name)
        return self.tracer.begin(name, **_clean_attrs(attrs))

    def end_span(self, span, status, error_kind=None, message=""):
        self.tracer.end(span, status, error_kind, message)

    def _current_span_id(self):
        current = self.tracer.current() if self.tracer else None
        return current.span_id if current else None

    def log(self, level, message, *, logger="", exc_info=False):
        value = _LEVELS.get(str(level).lower(), 20)
        if value < self._level:
            return
        payload = {"level": str(level).lower(), "message": redact(message),
                   "logger": logger, "span_id": self._current_span_id()}
        if exc_info:
            exc = sys.exc_info()[1]
            if exc is not None:
                payload["exc_type"] = type(exc).__name__
                payload["exc_traceback"] = redact("".join(tb.format_exception(exc)))
        self._emit("log", payload)

    def save_rgbd(self, frame, note=""):
        span_id = self._current_span_id()
        rel = self.artifacts.write_rgbd(self.run_id, span_id, frame)
        self._emit("artifact", {"span_id": span_id, "type": "rgbd",
                                "path": rel, "note": note})
        return rel

    def save_state(self, snapshot, label):
        span_id = self._current_span_id()
        rel = self.artifacts.write_state(self.run_id, label, snapshot)
        self._emit("artifact", {"span_id": span_id, "type": "state",
                                "path": rel, "note": ""})
        return rel

    def save_llm(self, system, user, raw, label):
        self._llm_index += 1
        unique = f"llm-{label}-{self._llm_index}"
        span_id = self._current_span_id()
        rels = self.artifacts.write_llm(self.run_id, unique, system, user, raw)
        for rel in rels:
            self._emit("artifact", {"span_id": span_id, "type": "llm",
                                    "path": rel, "note": label})
        return rels

    def trace(self, payload):
        self.log("debug", payload if isinstance(payload, str) else str(payload),
                 logger="atom")


def _clean_attrs(attrs):
    clean = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, (int, float)):
            clean[key] = value
        elif isinstance(value, str):
            clean[key] = redact(value)
        else:
            clean[key] = redact(str(value))
    return clean


class _NullCM:
    def __enter__(self):
        return RunHandle()

    def __exit__(self, *args):
        return False


class NullRecorder:
    def run(self, task, goal=None, *, agent=None):
        return _NullCM()

    @contextmanager
    def span(self, name, **attrs):
        yield None

    def start_span(self, name, **attrs):
        return None

    def end_span(self, span, status, error_kind=None, message=""):
        pass

    def log(self, level, message, *, logger="", exc_info=False):
        pass

    def save_rgbd(self, frame, note=""):
        return ""

    def save_state(self, snapshot, label):
        return ""

    def save_llm(self, system, user, raw, label):
        return []

    def trace(self, payload):
        pass


def build_recorder(root=None) -> Recorder:
    root = root or os.getenv("PICKPARTS_LOG_DIR") or None
    return Recorder(root, int(os.getenv("PICKPARTS_LOG_RETENTION_DAYS", "7")),
                    os.getenv("PICKPARTS_LOG_LEVEL", "info"))
