"""Cloud-only AI boundaries (endpoints + non-privileged cloud perception)."""
import base64
from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from ..scene.perception import Frame, backproject


class ConfigurationError(ValueError):
    pass


class CloudError(RuntimeError):
    pass


class PerceptionError(CloudError):
    pass


class Detection(TypedDict):
    bbox: list[float]
    point: np.ndarray
    confidence: float
    refinement: str


@dataclass(frozen=True)
class Endpoint:
    api_key: str = field(repr=False)
    base_url: str
    model: str

    @classmethod
    def from_env(cls, role):
        if role not in ("LLM", "VLM", "ASR"):
            raise ConfigurationError("Unknown cloud role")
        llm_url = os.getenv("LLM_BASE_URL", "").strip()
        role_url = os.getenv(f"{role}_BASE_URL", "").strip() or llm_url
        # Never forward one provider's secret to a separately configured provider.
        same_origin = urlsplit(role_url)[:2] == urlsplit(llm_url)[:2]
        values = {}
        missing = []
        for name in ("API_KEY", "BASE_URL", "MODEL"):
            value = os.getenv(f"{role}_{name}", "").strip()
            if (not value and role != "LLM"
                    and (name == "BASE_URL" or (name == "API_KEY" and same_origin))):
                value = os.getenv(f"LLM_{name}", "").strip()
            if not value or value in ("sk-your-api-key-here", "your-api-key"):
                missing.append(f"{role}_{name}")
            values[name] = value
        if missing:
            raise ConfigurationError("Missing cloud configuration: " + ", ".join(missing))
        return cls(values["API_KEY"], values["BASE_URL"], values["MODEL"])

    def client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url,
                      timeout=60, max_retries=0)

    def qwen_config(self):
        # Qwen-Agent 0.0.34 serializes tool results as `id`; native OpenAI
        # providers require tool_call_id. Keep the upstream converter otherwise.
        from qwen_agent.llm.base import BaseChatModel
        converter = BaseChatModel._conv_qwen_agent_messages_to_oai
        if not getattr(converter, "_pickparts_fixed", False):
            def compatible(messages):
                converted = converter(messages)
                for message in converted:
                    if message.get("role") == "tool" and "id" in message:
                        message["tool_call_id"] = message.pop("id")
                return converted
            compatible._pickparts_fixed = True
            BaseChatModel._conv_qwen_agent_messages_to_oai = staticmethod(compatible)
        return {
            "model": self.model, "model_type": "oai",
            "model_server": self.base_url, "api_key": self.api_key,
            "generate_cfg": {"max_retries": 0, "request_timeout": 60,
                             "use_raw_api": True},
        }


class CloudPerception:
    """Cloud bbox with optional, explicit local RGB-color depth refinement."""

    def __init__(self, client, model, min_confidence=.7, *, refine_color=True):
        if not 0 < min_confidence <= 1:
            raise ValueError("Confidence threshold must be in (0, 1]")
        self.client = client
        self.model = model
        self.min_confidence = min_confidence
        self.refine_color = refine_color

    def locate(self, frame: Frame, target: str) -> Detection:
        descriptions = {"A": "red part A", "B": "blue part B", "box": "green destination box"}
        if not isinstance(target, str) or not target.strip() or len(target) > 160:
            raise ValueError("Target must be a nonempty visual description")
        refine_color = self.refine_color and target in descriptions
        rgb, depth = np.asarray(frame.rgb), np.asarray(frame.depth)
        if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8
                or depth.shape != rgb.shape[:2] or not rgb.size):
            raise PerceptionError("Invalid RGB/depth frame")
        if not (np.issubdtype(depth.dtype, np.floating) or np.issubdtype(depth.dtype, np.integer)):
            raise PerceptionError("Invalid numeric depth")
        height, width = depth.shape
        image = io.BytesIO()
        Image.fromarray(rgb).save(image, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")
        prompt = (
            f"Locate the visible {descriptions.get(target, target)} in this {width}x{height} RGB image. "
            "Return only JSON: {\"visible\":true,\"bbox\":[x1,y1,x2,y2],\"confidence\":0.9}. "
            "bbox must tightly bound the object in native image pixels, not normalized "
            "coordinates; x2/y2 are exclusive. Use visible:false if absent or occluded. "
            "Do not infer hidden objects or output 3D coordinates."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=0,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ]}],
            )
            _raw = response.choices[0].message.content
            data = json.loads(_raw)
        except Exception as exc:
            raise PerceptionError(f"Cloud localization failed ({type(exc).__name__})") from None
        if not isinstance(data, dict) or data.get("visible") is not True:
            raise PerceptionError("Target is not visibly localized")
        bbox, confidence = data.get("bbox"), data.get("confidence")
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(type(v) not in (int, float) for v in bbox)):
            raise PerceptionError("Malformed numeric bbox")
        if (type(confidence) not in (int, float)
                or not self.min_confidence <= confidence <= 1):
            raise PerceptionError("Invalid or low localization confidence")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise PerceptionError("Bbox outside image bounds or empty")
        # Use only measured positive depths at pixel centers inside the returned box.
        yy, xx = np.mgrid[int(np.ceil(y1)):int(np.ceil(y2)),
                          int(np.ceil(x1)):int(np.ceil(x2))]
        measured = depth[yy, xx]
        valid = np.isfinite(measured) & (measured > 0)
        if refine_color:
            r, g, b = rgb[yy, xx].astype(float).transpose(2, 0, 1)
            if target == "A":
                color = (r > 65) & (r > g * 1.8) & (r > b * 1.8)
            elif target == "B":
                color = (b > 65) & (b > r * 1.8) & (b > g * 1.35)
            else:
                color = (g > 65) & (g > r * 1.6) & (g > b * 1.3)
            valid &= color
            if not valid.any():
                raise PerceptionError("No matching RGB color pixels with valid depth inside bbox")
        if not valid.any():
            raise PerceptionError("No valid visible depth inside bbox")
        if not refine_color:
            minimum = max(8, .8 * valid.size)
            if valid.sum() < minimum:
                raise PerceptionError("Insufficient valid depth coverage inside bbox")
            # Image neighbors connect only across small measured depth changes.
            # A bbox alone cannot distinguish competing surfaces or depth islands.
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            ys, xs = np.nonzero(valid)
            depths = measured[valid].astype(float)
            indices = np.full(valid.shape, -1, dtype=int)
            indices[ys, xs] = np.arange(len(xs))
            edges = []
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                ny, nx = ys + dy, xs + dx
                inside = ((ny < valid.shape[0]) & (nx >= 0)
                          & (nx < valid.shape[1]))
                source = np.flatnonzero(inside)
                dest = indices[ny[inside], nx[inside]]
                connected = (dest >= 0) & (np.abs(depths[source] - depths[dest]) < .012)
                edges.append((source[connected], dest[connected]))
            rows = np.concatenate([edge[0] for edge in edges])
            cols = np.concatenate([edge[1] for edge in edges])
            graph = coo_matrix((np.ones(len(rows)), (rows, cols)),
                               shape=(len(xs), len(xs)))
            _, regions = connected_components(graph, directed=False)
            dominant = regions == np.argmax(np.bincount(regions))
            if dominant.sum() < max(minimum, .9 * len(xs)):
                raise PerceptionError("Ambiguous disconnected depth surfaces inside bbox")
            valid[ys[~dominant], xs[~dominant]] = False
        try:
            points = backproject(np.column_stack((xx[valid], yy[valid])), measured[valid],
                                 frame.intrinsic, frame.camera_to_base)
        except (ValueError, np.linalg.LinAlgError):
            raise PerceptionError("Invalid depth or camera calibration") from None
        point = np.median(points, axis=0)
        if refine_color:
            # Match the visible-extent estimator: facing surfaces otherwise bias XY.
            point[:2] = np.mean(np.percentile(points[:, :2], [1, 99], axis=0), axis=0)
        if not np.isfinite(point).all():
            raise PerceptionError("Invalid projected depth")
        return {"bbox": bbox, "point": point, "confidence": float(confidence),
                "refinement": "rgb-color" if refine_color else "none",
                "top_z": float(np.percentile(points[:, 2], 98)),
                "bottom_z": float(np.percentile(points[:, 2], 2)),
                "extent": (np.percentile(points, 98, axis=0)
                           - np.percentile(points, 2, axis=0)).tolist()}


class VisualLocator:
    """Use measured full-frame RGB-D for catalog objects, VLM for open descriptions."""

    def __init__(self, specs, client, model):
        from ..scene.perception import ScenePerception
        self.local = ScenePerception(specs)
        self.ids = {s["id"] for s in specs}
        self.cloud = CloudPerception(client, model)

    def locate(self, frame, target):
        if target in self.ids:
            # Do not let a remote guess override a missing/ambiguous known object.
            return self.local.locate(frame, target)
        return self.cloud.locate(frame, target)


def transcribe(client, model, path):
    try:
        with Path(path).open("rb") as audio:
            response = client.audio.transcriptions.create(model=model, file=audio)
        text = response.text
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty transcription")
        return text.strip()
    except Exception as exc:
        raise CloudError(f"Cloud ASR failed ({type(exc).__name__})") from None
