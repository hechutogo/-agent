"""Artifact storage: RGB-D captures, state snapshots, LLM transcripts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .events import redact


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)
        self.dir = self.root / "artifacts"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        path = self.dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def write_rgbd(self, run_id, span_id, frame) -> str:
        folder = self._run_dir(run_id)
        Image.fromarray(np.asarray(frame.rgb)).save(folder / f"{span_id}-error.png")
        np.savez_compressed(
            folder / f"{span_id}-error.npz", depth=np.asarray(frame.depth),
            intrinsic=np.asarray(frame.intrinsic),
            camera_to_base=np.asarray(frame.camera_to_base),
            qpos=np.asarray(frame.qpos))
        return self._rel(folder / f"{span_id}-error.png")

    def write_state(self, run_id, label, snapshot) -> str:
        path = self._run_dir(run_id) / f"{label}.json"
        path.write_text(redact(json.dumps(snapshot, ensure_ascii=False, indent=2,
                                          allow_nan=False)), encoding="utf-8")
        return self._rel(path)

    def write_llm(self, run_id, label, system, user, raw):
        folder = self._run_dir(run_id)
        req = folder / f"{label}.req.txt"
        resp = folder / f"{label}.resp.txt"
        req.write_text(redact(f"[SYSTEM]\n{system}\n\n[USER]\n{user}"), encoding="utf-8")
        resp.write_text(redact(raw), encoding="utf-8")
        return [self._rel(req), self._rel(resp)]
