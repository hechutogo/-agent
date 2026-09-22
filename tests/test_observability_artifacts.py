import json
import numpy as np

from pickparts_agent.observability.artifacts import ArtifactStore


class _Frame:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    intrinsic = np.eye(3)
    camera_to_base = np.eye(4)
    qpos = np.zeros(2)


def test_write_rgbd_creates_png_and_npz(tmp_path):
    rel = ArtifactStore(tmp_path).write_rgbd("r1", "r1-s3", _Frame())
    base = tmp_path / rel
    assert base.exists() and base.with_suffix(".npz").exists()
    assert np.load(base.with_suffix(".npz"))["depth"].shape == (4, 4)


def test_write_state_roundtrip(tmp_path):
    rel = ArtifactStore(tmp_path).write_state("r1", "state-before", {"a": 1})
    assert json.loads((tmp_path / rel).read_text()) == {"a": 1}


def test_write_llm_pair(tmp_path):
    rels = ArtifactStore(tmp_path).write_llm("r1", "llm-plan-1", "sys", "usr", "raw")
    assert len(rels) == 2
    assert "[SYSTEM]" in (tmp_path / rels[0]).read_text()
    assert (tmp_path / rels[1]).read_text() == "raw"


def test_linked_text_artifacts_do_not_expose_secrets(tmp_path):
    store = ArtifactStore(tmp_path)
    secret = "sk-abcdefgh123456"
    opaque = "0123456789abcdef0123456789abcdef"
    paths = store.write_llm(
        "r1", "llm", secret, f'{{"api_key":"{opaque}"}}', secret)
    paths.append(store.write_state(
        "r1", "scene", {"goal": secret, "access_token": opaque, "count": 2}))
    for path in paths:
        content = (tmp_path / path).read_text()
        assert secret not in content and "abcdefgh123456" not in content
        assert opaque not in content
    assert json.loads((tmp_path / paths[-1]).read_text())["count"] == 2
