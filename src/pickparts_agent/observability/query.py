"""Read-only queries over the daily JSONL trace files and artifact folders."""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

_RUN_ID_RE = re.compile(r"^[0-9A-Za-z._-]+$")

# (path, mtime_ns, size) -> parsed events; avoids re-reading live files.
_CACHE: dict = {}
_CACHE_MAX = 64


def _read_cached(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return []
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    events = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = events
    return events


def _day_file(log_dir, day) -> Path:
    return Path(log_dir) / f"trace-{day}.jsonl"


def _retained_events(log_dir, retention_days):
    today = date.today()
    for delta in range(int(retention_days), -1, -1):
        yield from _read_cached(_day_file(log_dir, today - timedelta(days=delta)))


def _event_order(event):
    try:
        timestamp = datetime.fromisoformat(event.get("ts", "")).timestamp()
    except (TypeError, ValueError, OverflowError):
        timestamp = 0
    return timestamp, event.get("seq", 0)


def _public_result(event):
    # Unexpected provider exceptions may contain request data beyond API keys.
    return ("操作异常，请查看本地日志。" if event.get("status") == "error"
            else event.get("message"))


def list_runs(log_dir, day=None, limit=100, retention_days=7):
    """List runs by start date, including their later retained completion."""
    day = day or date.today().isoformat()
    try:
        day = date.fromisoformat(str(day)).isoformat()
    except ValueError:
        return []
    runs: dict = {}
    order: list = []
    # Scan cached days once for the entire listing, not once per run.
    events = sorted(
        (event for event in _retained_events(log_dir, retention_days)
         if event.get("kind") in ("run_start", "run_end")),
        key=_event_order)
    for event in events:
        run_id = event.get("run_id")
        if not run_id:
            continue
        kind = event.get("kind")
        if kind == "run_start":
            if str(event.get("ts", ""))[:10] != day:
                continue
            if run_id not in runs:
                runs[run_id] = {"run_id": run_id}
                order.append(run_id)
            record = runs[run_id]
            record.update(task=event.get("task"), goal=event.get("goal"),
                          start_ts=event.get("ts"), agent=event.get("agent"),
                          status="running")
        elif kind == "run_end" and run_id in runs:
            record = runs[run_id]
            record.update(status=event.get("status"), message=_public_result(event),
                          recovery_required=event.get("recovery_required"),
                          counts=event.get("counts"),
                          duration_ms=event.get("duration_ms"), end_ts=event.get("ts"))
    result = [_live_timing(runs[run_id]) for run_id in reversed(order)]
    return result[: max(1, int(limit))]


def _live_timing(record):
    if record.get("status") == "running" and record.get("start_ts"):
        try:
            start = datetime.fromisoformat(record["start_ts"])
            record["duration_ms"] = max(
                0, round((datetime.now(start.tzinfo) - start).total_seconds() * 1000, 2))
        except (TypeError, ValueError):
            pass
    return record


def _build_run(run_id, events):
    info = {"run_id": run_id, "spans": [], "logs": [], "artifacts": []}
    spans: dict = {}
    span_order: list = []
    for event in events:
        kind = event.get("kind")
        if kind == "run_start":
            info.update(task=event.get("task"), goal=event.get("goal"),
                        start_ts=event.get("ts"), agent=event.get("agent"),
                        status="running")
        elif kind == "run_end":
            info.update(status=event.get("status"), message=_public_result(event),
                        recovery_required=event.get("recovery_required"),
                        counts=event.get("counts"),
                        duration_ms=event.get("duration_ms"), end_ts=event.get("ts"))
        elif kind == "span_start":
            span = {"span_id": event["span_id"],
                    "parent_id": event.get("parent_id"),
                    "name": event.get("name"),
                    "attrs": event.get("attrs", {}), "start_ts": event.get("ts")}
            spans[event["span_id"]] = span
            span_order.append(event["span_id"])
        elif kind == "span_end":
            span = spans.get(event["span_id"])
            if span is not None:
                error_kind = event.get("error_kind")
                exception = (isinstance(error_kind, str)
                             and error_kind[:1].isupper())
                span.update(status=event.get("status"),
                            duration_ms=event.get("duration_ms"),
                            error_kind=error_kind,
                            error_message=error_kind if exception else event.get("message"),
                            end_ts=event.get("ts"))
        elif kind == "log":
            entry = {"ts": event.get("ts"), "seq": event.get("seq"), "level": event.get("level"),
                     "message": event.get("message"),
                     "logger": event.get("logger"),
                     "span_id": event.get("span_id")}
            if event.get("exc_type"):
                # Type name only; the full traceback stays in the local file.
                entry["exc_type"] = event["exc_type"]
                entry["message"] = f"操作异常（{event['exc_type']}），详细信息见本地日志。"
            info["logs"].append(entry)
        elif kind == "artifact":
            info["artifacts"].append({
                "span_id": event.get("span_id"), "type": event.get("type"),
                "path": event.get("path"), "note": event.get("note")})
    info["spans"] = [spans[span_id] for span_id in span_order]
    return _live_timing(info)


def get_run(log_dir, run_id, retention_days=7):
    if not _RUN_ID_RE.fullmatch(str(run_id)) or ".." in str(run_id):
        return None
    events = [event for event in _retained_events(log_dir, retention_days)
              if event.get("run_id") == run_id]
    return _build_run(run_id, sorted(events, key=_event_order)) if events else None


def artifact_path(log_dir, run_id, name):
    run_id, name = str(run_id), str(name)
    if not _RUN_ID_RE.fullmatch(run_id) or ".." in run_id:
        return None
    base = (Path(log_dir) / "artifacts" / run_id).resolve()
    target = (base / name).resolve()
    if target != base and base not in target.parents:
        return None
    if not target.is_file():
        return None
    return target
