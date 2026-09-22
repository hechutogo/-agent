"""Event schema, timestamps and secret redaction for the observability log."""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

_KEY_RE = re.compile(r"sk-[A-Za-z0-9_.\-]{8,}")
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}")
_NAMED_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|secret|password|token|authorization)\b"
    r"[\"']?\s*[:=]\s*[\"']?)(?:bearer\s+)?"
    r"[A-Za-z0-9._~+/\-=]{8,}")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def redact(value: Any) -> str:
    text = value if isinstance(value, str) else "" if value is None else str(value)
    text = _NAMED_SECRET_RE.sub(r"\1***REDACTED***", text)
    text = _KEY_RE.sub("sk-***REDACTED***", text)
    text = _BEARER_RE.sub(r"\1***REDACTED***", text)
    return text


def make_event(kind: str, run_id: str, seq: int, **fields: Any) -> dict:
    event = {"ts": now_iso(), "run_id": run_id, "seq": int(seq), "kind": kind}
    event.update(fields)
    return event


def dumps(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
