"""Local HTTP console. Only the worker thread touches the simulator/Agent."""
import argparse
import copy
from contextlib import asynccontextmanager
import io
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[3]


class RobotBackend:
    def __init__(self, publish, stage, view=None):
        from dotenv import load_dotenv
        from ..services.cloud import Endpoint, VisualLocator
        from ..agent.orchestrator import build_orchestrator
        from ..scene.simulation import Simulation

        load_dotenv(ROOT / ".env")
        llm, vision = Endpoint.from_env("LLM"), Endpoint.from_env("VLM")
        self.client = vision.client()
        self.sim = None
        try:
            self.sim = Simulation()
            self.sim.on_frame = publish
            self.vision = vision
            self.stage, self.view, self.publish = stage, view or (lambda e: None), publish
            locate = VisualLocator(self.sim.object_specs, self.client, vision.model).locate
            self.orchestrator = build_orchestrator(
                self.sim, locate, llm,
                on_stage=stage, on_event=view or (lambda e: None))
            self.models = {"llm": llm.model, "vision": vision.model,
                           "thinking": self.orchestrator.thinking}
            publish(self.sim.observe())
        except Exception:
            self.close()
            raise

    def turn(self, text):
        result = self.orchestrator.turn(text)
        result.setdefault("recovery_required", False)
        self.sim.on_frame(self.sim.observe())
        return result

    @property
    def scene_objects(self):
        return self.sim.object_specs

    def reset(self):
        from ..scene.simulation import Simulation
        from ..agent.state import WorldState
        inventory = self.sim.object_specs
        self.sim.close()
        self.sim = Simulation(objects=inventory)
        self.sim.on_frame = self.publish
        state = WorldState(self.sim.object_specs)
        self.orchestrator.state = self.orchestrator.executor.state = state
        self.orchestrator.executor.sim = self.sim
        self.orchestrator.history.clear()
        self._refresh_locator()
        self.publish(self.sim.observe())

    def _refresh_locator(self):
        from ..services.cloud import VisualLocator
        self.orchestrator.executor.locate = VisualLocator(
            self.sim.object_specs, self.client, self.vision.model).locate

    def add_object(self, kind):
        self.sim.add_object(kind)
        state = self.orchestrator.state
        state.catalog = {s["id"]: s for s in self.sim.object_specs}
        # Newly spawned bodies can occlude old observations.
        for name in list(state.objects):
            state.mark_missing(name)
        self._refresh_locator()
        self.publish(self.sim.observe())

    def close(self):
        if self.sim is not None:
            self.sim.close()
        self.client.close()


class Console:
    def __init__(self, factory):
        self.factory = factory
        self.lock = threading.RLock()
        self.jobs = queue.Queue()
        self.frame = None
        self.state = {
            "status": "starting", "stage": "初始化", "messages": [],
            "events": [], "frame_id": 0, "models": {}, "error": None,
            "recovery_required": False, "workflow": None, "react": [],
            "scene_objects": [],
        }
        self.thread = threading.Thread(target=self._worker, daemon=True,
                                       name="pickparts-simulation")
        self._sequence = 0

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def publish(self, frame):
        from PIL import Image
        output = io.BytesIO()
        Image.fromarray(frame.rgb).save(output, format="JPEG", quality=85)
        with self.lock:
            self.frame = output.getvalue()
            self.state["frame_id"] += 1
            self.state["frame_time"] = time.time()

    def stage(self, name):
        with self.lock:
            self.state["stage"] = name
            self.state["events"].append({"stage": name, "time": time.time()})
            self.state["events"] = self.state["events"][-100:]

    def view(self, event):
        kind = event["type"]
        with self.lock:
            if kind == "plan":
                self.state["workflow"] = {
                    "goal": event["goal"], "current": event["current"],
                    "status": "running", "steps": event["steps"]}
            elif kind == "step":
                wf = self.state.get("workflow")
                if wf:
                    for step in wf["steps"]:
                        if step["index"] == event["index"]:
                            step["status"] = event["status"]
            elif kind == "react":
                self.state.setdefault("react", []).append({
                    "attempt": event["attempt"], "thought": event["thought"],
                    "decision": event["decision"], "detail": event["detail"]})
                self.state["react"] = self.state["react"][-100:]
            elif kind == "activity":
                self._sequence += 1
                self.state["messages"].append({
                    "role": "activity", "kind": event["kind"], "text": event["text"],
                    "seq": self._sequence})
                self.state["messages"] = self.state["messages"][-250:]
            elif kind == "finish":
                wf = self.state.get("workflow")
                if wf:
                    wf["status"] = "done" if event["success"] else "failed"

    def submit(self, kind, text=None):
        with self.lock:
            if self.state["status"] in ("starting", "busy", "resetting", "adding"):
                raise HTTPException(409, "当前操作尚未结束，请稍后继续。")
            if kind in ("command", "add") and (
                    self.state["status"] != "ready" or self.state["recovery_required"]):
                raise HTTPException(409, "请先重置场景，恢复后再发送指令。")
            self.state["status"] = {"command": "busy", "add": "adding"}.get(kind, "resetting")
            self.state["stage"] = {"command": "理解指令", "add": "生成可达位置"}.get(kind, "初始化")
            self.state["events"] = []
            if kind == "command":
                self.state.update(workflow=None, react=[], error=None)
                self.state["messages"].append({"role": "user", "text": text})
            self.jobs.put((kind, text))

    def _worker(self):
        backend = None
        kind, text = "reset", None
        try:
            while True:
                try:
                    if kind == "reset":
                        can_reset = backend is not None and hasattr(backend, "reset")
                        if backend is not None and not can_reset:
                            backend.close()
                            backend = None
                        with self.lock:
                            self.frame = None
                            self.state.update(messages=[], events=[], error=None,
                                              recovery_required=False,
                                              workflow=None, react=[])
                        if can_reset:
                            backend.reset()
                        else:
                            backend = self.factory(self.publish, self.stage, self.view)
                        with self.lock:
                            self.state.update(status="ready", stage="等待指令",
                                              models=getattr(backend, "models", {}),
                                              scene_objects=getattr(backend, "scene_objects", []))
                    elif kind == "add":
                        try:
                            backend.add_object(text)
                            self.view({"type": "activity", "kind": "observation",
                                       "text": "已在可达区域添加物体，可通过下方标识下达指令。"})
                        except ValueError as exc:
                            message = str(exc)
                            if message.startswith("Scene capacity"):
                                message = "可达区域已无足够空位，或已达到 6 个物体上限。可随机重置后再试。"
                            self.view({"type": "activity", "kind": "observation",
                                       "text": message})
                        with self.lock:
                            self.state.update(status="ready", stage="等待指令",
                                              scene_objects=getattr(backend, "scene_objects", []))
                    elif kind == "command":
                        result = backend.turn(text)
                        with self.lock:
                            self.state["messages"].append({
                                "role": "assistant", "text": result["message"],
                                "success": result["success"]})
                            self.state["messages"] = self.state["messages"][-250:]
                            recovery = result.get("recovery_required", False)
                            self.state.update(
                                status="ready", recovery_required=recovery,
                                stage="已完成" if result["success"] else "未完成")
                except Exception as exc:
                    # Provider exception strings can contain API keys/request data.
                    from ..services.cloud import ConfigurationError
                    message = (str(exc) if isinstance(exc, ConfigurationError)
                               else f"操作异常（{type(exc).__name__}），请重置后重试。")
                    with self.lock:
                        self.state.update(status="error", error=message, stage="异常")
                        if self.state.get("workflow"):
                            self.state["workflow"]["status"] = "failed"
                kind, text = self.jobs.get()
                if kind == "close":
                    break
        finally:
            if backend is not None:
                backend.close()

    def close(self):
        self.jobs.put(("close", None))
        self.thread.join(timeout=5)


class Command(BaseModel):
    text: str = Field(min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, text):
        if not text.strip():
            raise ValueError("请输入指令")
        return text.strip()


class NewObject(BaseModel):
    kind: Literal["block", "box"]


def create_app(factory=RobotBackend):
    console = Console(factory)

    @asynccontextmanager
    async def lifespan(app):
        console.thread.start()
        yield
        console.close()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin:
            if urlsplit(origin).netloc != request.headers.get("host"):
                return Response("Cross-origin control is disabled", status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static, check_dir=False), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/api/state")
    def state():
        return console.snapshot()

    @app.get("/api/frame")
    def frame():
        with console.lock:
            content = console.frame
        return Response(content, media_type="image/jpeg") if content else Response(status_code=204)

    @app.post("/api/command", status_code=202)
    def command(body: Command):
        console.submit("command", body.text)
        return {"accepted": True}

    @app.post("/api/reset", status_code=202)
    def reset():
        console.submit("reset")
        return {"accepted": True}

    @app.post("/api/objects", status_code=202)
    def add_object(body: NewObject):
        console.submit("add", body.kind)
        return {"accepted": True}

    return app


def warm_up_renderer():
    """Create a main-thread render anchor that must stay referenced.

    On macOS, SAPIEN's first Vulkan device creation aborts (SIGTRAP) on a
    non-main thread, and destroying the last Simulation tears down the
    process-wide Vulkan state. Building one here on the main thread and
    keeping the returned object alive lets the simulation worker create and
    release its own simulations safely.
    """
    from ..scene.simulation import Simulation
    sim = Simulation()
    sim.close()
    return sim


def main():
    parser = argparse.ArgumentParser(description="XLeRobot live Agent web console")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    render_anchor = warm_up_renderer() if sys.platform == "darwin" else None
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
