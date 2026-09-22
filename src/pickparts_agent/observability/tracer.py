"""Explicit span tracking via contextvars (single worker thread)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable

from .events import redact


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    attrs: dict
    start: float


class Tracer:
    def __init__(self, emit: Callable[[str, dict], None], run_id: str):
        self._emit = emit
        self._run_id = run_id
        self._current: ContextVar[Span | None] = ContextVar(
            f"span-{run_id}", default=None)
        self._tokens: list = []
        self._n = 0

    def _new_id(self) -> str:
        self._n += 1
        return f"{self._run_id}-s{self._n}"

    def current(self) -> Span | None:
        return self._current.get()

    def begin(self, name: str, **attrs) -> Span:
        parent = self._current.get()
        span = Span(self._new_id(), parent.span_id if parent else None,
                    name, dict(attrs), time.perf_counter())
        self._emit("span_start", {"span_id": span.span_id,
                                  "parent_id": span.parent_id,
                                  "name": span.name, "attrs": span.attrs})
        self._tokens.append(self._current.set(span))
        return span

    def end(self, span: Span, status: str, error_kind: str | None = None,
            message: str = "") -> None:
        payload = {"span_id": span.span_id, "status": status,
                   "duration_ms": max(0, round((time.perf_counter() - span.start) * 1000, 2))}
        if error_kind:
            payload["error_kind"] = error_kind
        if message:
            payload["message"] = redact(message)
        self._emit("span_end", payload)
        if self._tokens:
            self._current.reset(self._tokens.pop())

    @contextmanager
    def span(self, name: str, **attrs):
        span = self.begin(name, **attrs)
        try:
            yield span
        except Exception as exc:
            self.end(span, "error", type(exc).__name__, str(exc))
            raise
        else:
            self.end(span, "ok")
