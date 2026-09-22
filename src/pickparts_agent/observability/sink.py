"""Append-only JSONL sink with daily rotation and age-based cleanup."""
from __future__ import annotations

import shutil
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from .events import dumps


class JSONLSink:
    def __init__(self, root, retention_days: int = 7):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention_days = int(retention_days)
        self._day: date | None = None
        self._fh = None

    def _file_for(self, day: date) -> Path:
        return self.root / f"trace-{day.isoformat()}.jsonl"

    def _ensure_open(self) -> None:
        today = date.today()
        if self._fh is None or self._day != today:
            if self._fh is not None:
                self._fh.close()
            self._day = today
            self._fh = self._file_for(today).open("a", encoding="utf-8")
            self.cleanup(today)

    def write(self, event: dict) -> None:
        try:
            self._ensure_open()
            self._fh.write(dumps(event) + "\n")
            self._fh.flush()
            if event.get("kind") == "run_end":
                import os
                try:
                    os.fsync(self._fh.fileno())
                except OSError:
                    pass
        except Exception:
            print("observability sink write failed", file=sys.stderr)

    def cleanup(self, today: date | None = None) -> None:
        today = today or date.today()
        cutoff = today - timedelta(days=self.retention_days)
        for path in self.root.glob("trace-*.jsonl"):
            try:
                day = date.fromisoformat(path.stem[len("trace-"):])
            except ValueError:
                continue
            if day < cutoff:
                try:
                    path.unlink()
                except OSError:
                    pass
        artifacts = self.root / "artifacts"
        if artifacts.is_dir():
            cutoff_ts = datetime.combine(cutoff, datetime.min.time()).timestamp()
            for child in artifacts.iterdir():
                try:
                    if child.is_dir() and child.stat().st_mtime < cutoff_ts:
                        shutil.rmtree(child, ignore_errors=True)
                except OSError:
                    pass

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
