"""Exercise real geometry, SDK serialization and config; fake only remote calls."""
import base64
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import numpy as np
from openai import OpenAI
from PIL import Image
import pytest

from pickparts_agent.scene.perception import Frame


@pytest.fixture
def cloud():
    # Missing implementation is an explicit test failure during the first red run.
    try:
        return importlib.import_module("pickparts_agent.services.cloud")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Cloud implementation is missing: {exc}")


@pytest.fixture
def frame():
    return Frame(
        rgb=np.full((6, 8, 3), [255, 0, 0], dtype=np.uint8),
        depth=np.full((6, 8), 2.0),
        intrinsic=np.array([[2., 0, 0], [0, 2., 0], [0, 0, 1]]),
        camera_to_base=np.array([[1., 0, 0, 10], [0, 1., 0, 20],
                                 [0, 0, 1., 30], [0, 0, 0, 1.]]),
        qpos=np.zeros(2), qvel=np.zeros(2),
    )


def remote_client(payload, requests):
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "vision", "object": "chat.completion", "created": 0,
            "model": "vision", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": payload}}],
        })
    return OpenAI(api_key="test-key", base_url="https://cloud.invalid/v1",
                  http_client=httpx.Client(transport=httpx.MockTransport(handle)),
                  max_retries=0)


def test_cloud_projects_only_visible_depth_and_sends_rgb(cloud, frame):
    requests = []
    payload = json.dumps({"visible": True, "bbox": [2, 1, 6, 5], "confidence": .9})
    frame.depth[:] = np.nan
    frame.depth[2, 3] = 2.
    frame.depth[0, 0] = 100.  # Outside bbox: must never influence the result.
    with remote_client(payload, requests) as client:
        result = cloud.CloudPerception(client, "vision").locate(frame, "A")
    assert result["bbox"] == [2, 1, 6, 5]
    assert result["confidence"] == .9
    np.testing.assert_allclose(result["point"], [13, 22, 32])
    body = json.loads(requests[0].content)
    assert body["model"] == "vision"
    contents = body["messages"][-1]["content"]
    image = next(item["image_url"]["url"] for item in contents if item["type"] == "image_url")
    with Image.open(io.BytesIO(base64.b64decode(image.split(",")[1]))) as rgb:
        np.testing.assert_array_equal(np.asarray(rgb), frame.rgb)
    assert "depth" not in body and "qpos" not in body


@pytest.mark.parametrize("payload,reason", [
    ("not json", "Cloud localization failed .*JSONDecodeError"),
    ("[]", "Target is not visibly localized"),
    ("null", "Target is not visibly localized"),
    ('{"visible": false, "bbox": [1,1,4,4], "confidence": 0.99}',
     "Target is not visibly localized"),
    ('{"visible": true, "bbox": [1,1,4,4], "confidence": 0.2}',
     "Invalid or low localization confidence"),
    ('{"visible": true, "bbox": [-1,1,4,4], "confidence": 0.9}',
     "Bbox outside image bounds or empty"),
    ('{"visible": true, "bbox": [1,1,9,4], "confidence": 0.9}',
     "Bbox outside image bounds or empty"),
    ('{"visible": true, "bbox": [4,1,1,4], "confidence": 0.9}',
     "Bbox outside image bounds or empty"),
    ('{"visible": true, "bbox": [1,1,1,4], "confidence": 0.9}',
     "Bbox outside image bounds or empty"),
    ('{"visible": true, "bbox": [1,1,4], "confidence": 0.9}',
     "Malformed numeric bbox"),
    ('{"visible": true, "bbox": ["1",1,4,4], "confidence": 0.9}',
     "Malformed numeric bbox"),
    ('{"visible": true, "bbox": [true,1,4,4], "confidence": 0.9}',
     "Malformed numeric bbox"),
    ('{"visible": true, "bbox": [1,1,4,4], "confidence": true}',
     "Invalid or low localization confidence"),
    ('{"visible": true, "bbox": [1,1,4,4], "confidence": NaN}',
     "Invalid or low localization confidence"),
    ('{"visible": true, "bbox": [1,1,4,4], "confidence": 1.1}',
     "Invalid or low localization confidence"),
    ('{"bbox": [1,1,4,4], "confidence": 0.9}', "Target is not visibly localized"),
])
def test_cloud_rejects_malformed_invisible_or_uncertain_boxes(cloud, frame, payload, reason):
    with remote_client(payload, []) as client:
        with pytest.raises(cloud.PerceptionError, match=reason):
            cloud.CloudPerception(client, "vision").locate(frame, "A")


def test_cloud_matching_color_localization_control(cloud, frame):
    payload = '{"visible": true, "bbox": [1,1,4,4], "confidence": 0.9}'
    with remote_client(payload, []) as client:
        result = cloud.CloudPerception(client, "vision").locate(frame, "A")
    assert result["bbox"] == [1, 1, 4, 4]
    assert result["confidence"] == .9
    assert result["refinement"] == "rgb-color"
    np.testing.assert_allclose(result["point"], [12, 22, 32])


@pytest.mark.parametrize("depth", [0., -1., np.nan, np.inf])
def test_cloud_rejects_invalid_depth(cloud, frame, depth):
    frame.depth[:] = depth
    payload = json.dumps({"visible": True, "bbox": [2, 1, 6, 5], "confidence": .9})
    with remote_client(payload, []) as client:
        with pytest.raises(cloud.PerceptionError, match="depth"):
            cloud.CloudPerception(client, "vision").locate(frame, "box")


def test_cloud_locates_open_description_without_hardcoded_color(cloud, frame):
    requests = []
    payload = '{"visible": true, "bbox": [1,1,4,4], "confidence": 0.9}'
    with remote_client(payload, requests) as client:
        result = cloud.CloudPerception(client, "vision").locate(frame, "黄色圆柱 C")
    assert result["confidence"] == .9
    assert "黄色圆柱 C" in json.loads(requests[0].content)["messages"][0]["content"][0]["text"]


@pytest.fixture
def generic_frame():
    return Frame(
        rgb=np.full((24, 44, 3), 100, dtype=np.uint8),
        depth=np.full((24, 44), .7),
        intrinsic=np.diag([100., 100., 1.]),
        camera_to_base=np.eye(4),
        qpos=np.zeros(2), qvel=np.zeros(2),
    )


@pytest.mark.parametrize("target,refine_color", [
    ("gray cylinder", True), ("A", False),
])
@pytest.mark.parametrize("valid_pixels", [1, 600])
def test_generic_bbox_rejects_insufficient_depth_coverage(
        cloud, generic_frame, target, refine_color, valid_pixels):
    # Valid depth outside the 800-pixel bbox cannot rescue sparse measurements.
    region = generic_frame.depth[2:22, 2:42]
    region[:] = np.nan
    region.flat[:valid_pixels] = .7
    payload = '{"visible": true, "bbox": [2,2,42,22], "confidence": 0.99}'
    with remote_client(payload, []) as client:
        with pytest.raises(cloud.PerceptionError, match="depth coverage"):
            cloud.CloudPerception(client, "vision", refine_color=refine_color).locate(
                generic_frame, target)


@pytest.mark.parametrize("target,refine_color", [
    ("gray cylinder", True), ("A", False),
])
@pytest.mark.parametrize("other_depth", [.9, .72, .7])
def test_generic_bbox_rejects_disconnected_depth_surfaces(
        cloud, generic_frame, target, refine_color, other_depth):
    generic_frame.depth[2:22, 22:42] = other_depth
    if other_depth == .7:
        # Even coplanar islands must not become a target in the unmeasured gap.
        generic_frame.depth[2:22, 21:23] = np.nan
    payload = '{"visible": true, "bbox": [2,2,42,22], "confidence": 0.99}'
    with remote_client(payload, []) as client:
        with pytest.raises(cloud.PerceptionError, match="depth surfaces"):
            cloud.CloudPerception(client, "vision", refine_color=refine_color).locate(
                generic_frame, target)


@pytest.mark.parametrize("target,refine_color", [
    ("gray cylinder", True), ("A", False),
])
@pytest.mark.parametrize("outliers", [False, True])
def test_generic_bbox_projects_coherent_measured_surface(
        cloud, generic_frame, target, refine_color, outliers):
    generic_frame.depth[:2] = 10.  # Outside bbox.
    if outliers:
        generic_frame.depth[2, 2] = np.nan
        generic_frame.depth[21, 41] = np.nan
        generic_frame.depth[2, 41] = .9
        generic_frame.depth[21, 2] = .9
    payload = '{"visible": true, "bbox": [2,2,42,22], "confidence": 0.99}'
    with remote_client(payload, []) as client:
        result = cloud.CloudPerception(client, "vision", refine_color=refine_color).locate(
            generic_frame, target)
    assert result["bbox"] == [2, 2, 42, 22]
    assert result["refinement"] == "none"
    np.testing.assert_allclose(result["point"], [.1505, .0805, .7])
    assert result["top_z"] == pytest.approx(.7)
    assert result["bottom_z"] == pytest.approx(.7)
    assert result["extent"][2] == pytest.approx(0.)


def test_asr_uses_cloud_audio_upload(cloud, tmp_path):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"text": "move A to box"})
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF-test-audio")
    with OpenAI(api_key="test-key", base_url="https://cloud.invalid/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        assert cloud.transcribe(client, "whisper-1", audio) == "move A to box"
    assert requests[0].url.path == "/v1/audio/transcriptions"
    assert b"RIFF-test-audio" in requests[0].content
    assert b"whisper-1" in requests[0].content


def test_missing_configuration_names_variables_not_values(cloud, monkeypatch):
    for key in list(os.environ):
        if key.startswith(("LLM_", "VLM_", "ASR_", "OPENAI_")):
            monkeypatch.delenv(key)
    with pytest.raises(cloud.ConfigurationError, match="LLM_API_KEY"):
        cloud.Endpoint.from_env("LLM")


def test_role_specific_configuration_falls_back_to_llm_endpoint(cloud, monkeypatch):
    for key in list(os.environ):
        if key.startswith(("LLM_", "VLM_", "ASR_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("LLM_API_KEY", "private-token")
    monkeypatch.setenv("LLM_BASE_URL", "https://cloud.invalid/v1")
    monkeypatch.setenv("VLM_MODEL", "vision")
    config = cloud.Endpoint.from_env("VLM")
    assert config.api_key == "private-token" and config.model == "vision"
    assert "private-token" not in repr(config)


def test_cli_help_does_not_import_simulation_or_cloud():
    command = (
        "import sys; from pickparts_agent.app import main; "
        "\ntry: main(['--help'])"
        "\nexcept SystemExit as e: assert e.code == 0"
        "\nassert 'sapien' not in sys.modules"
        "\nassert 'torch' not in sys.modules"
        "\nassert 'qwen_agent' not in sys.modules"
        "\nassert 'sounddevice' not in sys.modules"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", command], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for flag in ("--smoke", "--demo", "--audio", "--record", "--output", "--seed", "--view"):
        assert flag in result.stdout


def test_cli_missing_config_exits_before_simulation():
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("LLM_", "VLM_", "ASR_", "OPENAI_"))}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-m", "pickparts_agent.app"],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "LLM_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr and "sapien" not in result.stderr


def test_save_frame_writes_rgb_and_metric_depth(frame, tmp_path):
    try:
        app = importlib.import_module("pickparts_agent.interfaces.cli")
    except ModuleNotFoundError as exc:
        pytest.fail(f"CLI implementation is missing: {exc}")
    app.save_frame(frame, tmp_path)
    with Image.open(tmp_path / "rgb.png") as image:
        np.testing.assert_array_equal(image, frame.rgb)
    with np.load(tmp_path / "depth.npz") as saved:
        np.testing.assert_array_equal(saved["depth"], frame.depth)
        np.testing.assert_array_equal(saved["intrinsic"], frame.intrinsic)
        np.testing.assert_array_equal(saved["camera_to_base"], frame.camera_to_base)


def test_cloud_rejects_huge_json_integer_without_numeric_crash(cloud, frame):
    payload = json.dumps({"visible": True, "bbox": [1, 1, 10**100, 4], "confidence": .9})
    with remote_client(payload, []) as client:
        with pytest.raises(cloud.PerceptionError):
            cloud.CloudPerception(client, "vision").locate(frame, "A")


def test_cloud_rejects_non_numeric_depth(cloud, frame):
    malformed = Frame(frame.rgb, np.full((6, 8), "bad"), frame.intrinsic,
                      frame.camera_to_base, frame.qpos, frame.qvel)
    payload = json.dumps({"visible": True, "bbox": [1, 1, 4, 4], "confidence": .9})
    with remote_client(payload, []) as client:
        with pytest.raises(cloud.PerceptionError, match="depth"):
            cloud.CloudPerception(client, "vision").locate(malformed, "A")


def test_credentials_do_not_fall_back_to_a_different_host(cloud, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "private-token")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.invalid/v1")
    monkeypatch.setenv("ASR_BASE_URL", "https://asr.invalid/v1")
    monkeypatch.setenv("ASR_MODEL", "whisper-1")
    monkeypatch.delenv("ASR_API_KEY", raising=False)
    with pytest.raises(cloud.ConfigurationError, match="ASR_API_KEY"):
        cloud.Endpoint.from_env("ASR")


@pytest.mark.parametrize("shape", ["fractional", "singular", "nonfinite"])
def test_projection_boundary_cases(cloud, frame, shape):
    bbox = [2.2, 1.2, 3.2, 2.2]
    if shape == "singular":
        frame.intrinsic[:] = 0
    elif shape == "nonfinite":
        frame.camera_to_base[0, 0] = np.inf
    payload = json.dumps({"visible": True, "bbox": bbox, "confidence": .9})
    with remote_client(payload, []) as client:
        perception = cloud.CloudPerception(client, "vision")
        if shape == "fractional":
            np.testing.assert_allclose(perception.locate(frame, "A")["point"], [13, 22, 32])
        else:
            with pytest.raises(cloud.PerceptionError, match="calibration"):
                perception.locate(frame, "A")


@pytest.mark.parametrize("response", [{"text": ""}, {"text": None}, {"text": "  "}])
def test_asr_rejects_empty_transcripts(cloud, tmp_path, response):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF-test")
    with OpenAI(api_key="test-key", base_url="https://cloud.invalid/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=response)))) as client:
        with pytest.raises(cloud.CloudError, match="ASR"):
            cloud.transcribe(client, "whisper-1", audio)


@pytest.mark.parametrize("args", [
    ["--record", "0"], ["--record", "-1"], ["--record", "nan"],
    ["--record", "121"], ["--audio", "/does/not/exist.wav"],
    ["--smoke", "--demo", "A"],
])
def test_cli_rejects_invalid_arguments_before_loading_runtime(args):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-m", "pickparts_agent.app", *args],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "error:" in result.stderr and "Traceback" not in result.stderr


@pytest.mark.parametrize("args", [
    ["--smoke"], ["--demo", "A", "B"], [],
])
def test_cli_viewer_is_opt_in_and_composes_with_modes(args):
    from pickparts_agent.interfaces.cli import _parser

    assert _parser().parse_args(args).view is False
    assert _parser().parse_args([*args, "--view"]).view is True


@pytest.mark.parametrize("target,color", [
    ("A", [230, 20, 20]), ("B", [20, 20, 230]), ("box", [20, 200, 20]),
])
def test_cloud_color_refinement_removes_background_and_face_bias(cloud, frame, target, color):
    frame.rgb[:] = [100, 100, 100]
    frame.depth[:] = 10
    for y, x in [(1, 1), (2, 1), (3, 1), (4, 1), (4, 5)]:
        frame.rgb[y, x] = color
        frame.depth[y, x] = 2
    frame.rgb[0, 7] = color  # Correct color outside bbox must not influence XY.
    payload = json.dumps({"visible": True, "bbox": [1, 1, 6, 5], "confidence": .9})
    with remote_client(payload, []) as client:
        result = cloud.CloudPerception(client, "vision").locate(frame, target)
    # Projected X: [11,11,11,11,15], Y: [21,22,23,24,24]; percentile interpolation.
    np.testing.assert_allclose(result["point"], [12.92, 22.52, 32])
    assert result["bbox"] == [1, 1, 6, 5]
    assert result["refinement"] == "rgb-color"


def test_cloud_color_refinement_can_be_disabled(cloud, frame):
    frame.rgb[:] = [100, 100, 100]
    payload = json.dumps({"visible": True, "bbox": [2, 1, 6, 5], "confidence": .9})
    with remote_client(payload, []) as client:
        result = cloud.CloudPerception(client, "vision", refine_color=False).locate(frame, "A")
    np.testing.assert_allclose(result["point"], [13.5, 22.5, 32])
    assert result["refinement"] == "none"


def test_cloud_color_refinement_rejects_no_matching_visible_pixels(cloud, frame):
    payload = json.dumps({"visible": True, "bbox": [2, 1, 6, 5], "confidence": .9})
    with remote_client(payload, []) as client:
        with pytest.raises(cloud.PerceptionError, match="color"):
            cloud.CloudPerception(client, "vision").locate(frame, "B")
