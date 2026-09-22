"""Local HTTP console. Only the worker thread touches the simulator/Agent."""
import argparse
import copy
from contextlib import asynccontextmanager, contextmanager
from datetime import date
import io
import logging
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[3]
AGENTS = [
    {"id": "pickparts", "name": "自研 Agent",
     "description": "逐步执行、视觉校验，遇到异常通过 ReAct 纠偏。"},
    {"id": "tiptop", "name": "TiPToP 初版",
     "description": "一次感知与全局规划，按预生成轨迹开环执行。"},
    {"id": "tiptop_optimized", "name": "TiPToP 优化版",
     "description": "按有序子任务逐个规划与执行，视觉校验结果，并通过 ReAct 纠偏。"},
]


class RobotBackend:
    def __init__(self, publish, stage, view=None, recorder=None):
        from dotenv import load_dotenv
        from ..services.cloud import Endpoint
        from ..scene.simulation import Simulation

        load_dotenv(ROOT / ".env")
        self.llm = Endpoint.from_env("LLM")
        vision = Endpoint.from_env("VLM")
        self.client = vision.client()
        self.sim = None
        self.orchestrator = None
        self.tiptop = None
        self.agent = None
        self.recorder = recorder
        try:
            self.sim = Simulation()
            self.sim.on_frame = publish
            self.vision = vision
            self.stage, self.view, self.publish = stage, view or (lambda e: None), publish
            self.select_agent("pickparts")
            publish(self.sim.observe())
        except Exception:
            self.close()
            raise

    def turn(self, text):
        result = (self.tiptop.run(text) if self.agent in ("tiptop", "tiptop_optimized")
                  else self.orchestrator.turn(text))
        result.setdefault("recovery_required", False)
        result.setdefault("verification", "open_loop" if self.agent == "tiptop" else "visual")
        self.sim.on_frame(self.sim.observe())
        return result

    def select_agent(self, agent):
        """Build first so a failed switch leaves the previous engine usable."""
        if agent not in {item["id"] for item in AGENTS}:
            raise ValueError("Unknown agent")
        if agent == self.agent:
            return
        if self.agent == "tiptop_optimized" and self.tiptop is not None:
            executor = self.tiptop.executor
            if (getattr(executor, "held", None) is not None
                    or getattr(executor, "candidate", None) is not None):
                raise ValueError("请先放下当前物体或重置场景，再切换 Agent。")
        if agent in ("tiptop", "tiptop_optimized"):
            if agent == "tiptop_optimized":
                from tiptop_optimized.agent import build_tiptop_optimized_agent as build_agent
            else:
                from tiptop_mac.agent import build_tiptop_agent as build_agent
            from ..agent.llm import supports_thinking
            thinking = supports_thinking(self.llm.model, self.llm.base_url)
            engine = build_agent(
                self.sim, self.llm, on_stage=self.stage,
                on_event=self.view, recorder=self.recorder)
        else:
            from ..agent.orchestrator import build_orchestrator
            from ..services.cloud import VisualLocator
            locate = VisualLocator(
                self.sim.object_specs, self.client, self.vision.model).locate
            engine = build_orchestrator(
                self.sim, locate, self.llm, on_stage=self.stage,
                on_event=self.view, recorder=self.recorder)
            thinking = engine.thinking
        old_orchestrator, old_tiptop = self.orchestrator, self.tiptop
        self.orchestrator = engine if agent == "pickparts" else None
        self.tiptop = engine if agent in ("tiptop", "tiptop_optimized") else None
        self.agent = agent
        self.models = {"llm": self.llm.model,
                       "vision": "本地 RGB-D" if self.tiptop is not None else self.vision.model,
                       "thinking": thinking}
        # Retirement cannot roll back an already installed, usable engine.
        self._close_engine(old_orchestrator, old_tiptop)

    def _close_resource(self, resource, name):
        if resource is None:
            return
        try:
            resource.close()
        except Exception as exc:
            message = f"Resource cleanup failed for {name} ({type(exc).__name__})"
            logging.getLogger(__name__).warning(message)
            if self.recorder is not None:
                try:
                    self.recorder.log("warning", message, logger="backend")
                except Exception:
                    logging.getLogger(__name__).warning("Cleanup warning could not be recorded")

    def _close_engine(self, orchestrator, tiptop):
        if orchestrator is not None:
            self._close_resource(orchestrator.planner.chat.client, "pickparts")
        self._close_resource(tiptop, "tiptop")

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
        if self.orchestrator is not None:
            state = WorldState(self.sim.object_specs)
            self.orchestrator.state = self.orchestrator.executor.state = state
            self.orchestrator.executor.sim = self.sim
            self.orchestrator.history.clear()
        if self.tiptop is not None:
            if self.agent == "tiptop_optimized":
                self.tiptop.bind_sim(self.sim)
            else:
                self.tiptop.sim = self.tiptop.executor.sim = self.sim
                self.tiptop.executor._ctx.sim = self.sim
        self._refresh_locator()
        self.publish(self.sim.observe())

    def _refresh_locator(self):
        from ..services.cloud import VisualLocator
        if self.orchestrator is not None:
            self.orchestrator.executor.locate = VisualLocator(
                self.sim.object_specs, self.client, self.vision.model).locate
        if self.tiptop is not None:
            from ..scene.perception import ScenePerception
            self.tiptop.executor.locate = ScenePerception(self.sim.object_specs).locate

    def add_object(self, kind):
        self.sim.add_object(kind)
        if self.orchestrator is not None:
            state = self.orchestrator.state
            state.catalog = {s["id"]: s for s in self.sim.object_specs}
            # Newly spawned bodies can occlude old observations.
            for name in list(state.objects):
                state.mark_missing(name)
        self._refresh_locator()
        self.publish(self.sim.observe())

    def close(self):
        self._close_engine(self.orchestrator, self.tiptop)
        self._close_resource(self.sim, "simulation")
        self._close_resource(self.client, "vision client")


class Console:
    def __init__(self, factory, recorder=None):
        if recorder is None:
            from ..observability import build_recorder
            recorder = build_recorder()
        self.recorder = recorder
        self.factory = factory
        self.lock = threading.RLock()
        self.jobs = queue.Queue()
        self.frame = None
        self.state = {
            "status": "starting", "stage": "初始化", "messages": [],
            "events": [], "frame_id": 0, "models": {}, "error": None,
            "recovery_required": False, "workflow": None, "subtasks": None, "react": [],
            "scene_objects": [], "agent": "pickparts", "agents": AGENTS,
            "run_id": None, "run_task": None,
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
        if kind == "activity":
            self.recorder.log("info", event["text"], logger=event["kind"])
        elif kind == "react":
            self.recorder.log("warning", event["detail"], logger="recovery")
        elif kind == "step":
            self.recorder.log(
                "warning" if event["status"] == "failed" else "info",
                f"步骤 {event['index'] + 1} · {event['status']}", logger="execution")
        with self.lock:
            if kind == "subtasks":
                self.state["subtasks"] = {
                    "current": event["current"], "steps": copy.deepcopy(event["steps"])}
            elif kind == "plan":
                self.state["workflow"] = {
                    "goal": event["goal"], "current": event["current"],
                    "status": "running", "steps": copy.deepcopy(event["steps"])}
            elif kind == "step":
                wf = self.state.get("workflow")
                if wf:
                    for step in wf["steps"]:
                        if step["index"] == event["index"]:
                            step["status"] = event["status"]
                            if event["status"] == "active":
                                wf["current"] = event["index"]
            elif kind == "react":
                self.state.setdefault("react", []).append({
                    "attempt": event["attempt"], "thought": event["thought"],
                    "decision": event["decision"], "detail": event["detail"]})
                self.state["react"] = self.state["react"][-100:]
            elif kind == "activity":
                self._sequence += 1
                self.state["messages"].append({
                    "role": "activity", "kind": event["kind"], "text": event["text"],
                    "seq": self._sequence, "agent": self.state["agent"]})
                self.state["messages"] = self.state["messages"][-250:]
            elif kind == "finish":
                wf = self.state.get("workflow")
                if wf:
                    wf["status"] = "done" if event["success"] else "failed"

    def submit(self, kind, text=None):
        with self.lock:
            if self.state["status"] in ("starting", "busy", "resetting", "adding", "switching"):
                raise HTTPException(409, "当前操作尚未结束，请稍后继续。")
            if kind in ("command", "add", "agent") and (
                    self.state["status"] != "ready" or self.state["recovery_required"]):
                raise HTTPException(409, "请先重置场景，恢复后再发送指令。")
            self.state["status"] = {"command": "busy", "add": "adding",
                                    "agent": "switching"}.get(kind, "resetting")
            self.state["stage"] = {"command": "理解指令", "add": "生成可达位置",
                                   "agent": "切换 Agent"}.get(kind, "初始化")
            self.state["events"] = []
            self.state["error"] = None
            if kind == "command":
                self.state.update(workflow=None, subtasks=None, react=[], error=None)
                self.state["messages"].append({"role": "user", "text": text,
                                               "agent": self.state["agent"]})
            self.jobs.put((kind, text))

    @contextmanager
    def _record_job(self, task, goal=None, agent=None):
        with self.recorder.run(task, goal, agent=agent or self.state["agent"]) as job:
            with self.lock:
                self.state.update(run_id=getattr(self.recorder, "run_id", None),
                                  run_task=task)
            yield job

    def _worker(self):
        backend = None
        kind, text = "reset", None
        try:
            while True:
                try:
                    if kind == "reset":
                        with self._record_job("reset"):
                            can_reset = backend is not None and hasattr(backend, "reset")
                            if backend is not None and not can_reset:
                                backend.close()
                                backend = None
                            with self.lock:
                                self.frame = None
                                self.state.update(messages=[], events=[], error=None,
                                                  recovery_required=False,
                                                  workflow=None, subtasks=None, react=[])
                            if can_reset:
                                backend.reset()
                            else:
                                backend = self.factory(
                                    self.publish, self.stage, self.view,
                                    recorder=self.recorder)
                                if self.state["agent"] != "pickparts":
                                    backend.select_agent(self.state["agent"])
                            with self.lock:
                                self.state.update(stage="等待指令",
                                                  models=getattr(backend, "models", {}),
                                                  scene_objects=getattr(backend, "scene_objects", []))
                    elif kind == "agent":
                        with self._record_job("switch", text, agent=text) as job:
                            backend.select_agent(text)
                            with self.lock:
                                self.state.update(
                                    agent=text, messages=[], workflow=None, subtasks=None, react=[],
                                    error=None, stage="等待指令",
                                    models=getattr(backend, "models", {}))
                            job.set_result("success", "已切换 Agent，场景保持不变。")
                    elif kind == "add":
                        with self._record_job("add", text) as job:
                            try:
                                backend.add_object(text)
                                self.view({"type": "activity", "kind": "observation",
                                           "text": "已在可达区域添加物体，可通过下方标识下达指令。"})
                                job.set_result("success", "已添加物体")
                            except ValueError as exc:
                                message = str(exc)
                                if message.startswith("Scene capacity"):
                                    message = "可达区域已无足够空位，或已达到 6 个物体上限。可随机重置后再试。"
                                self.view({"type": "activity", "kind": "observation",
                                           "text": message})
                                job.set_result("incomplete", message)
                            with self.lock:
                                self.state.update(stage="等待指令",
                                                  scene_objects=getattr(backend, "scene_objects", []))
                    elif kind == "command":
                        with self._record_job("command", text) as job:
                            result = backend.turn(text)
                            with self.lock:
                                self.state["messages"].append({
                                    "role": "assistant", "text": result["message"],
                                    "success": result["success"], "agent": self.state["agent"],
                                    "verification": result.get("verification", "visual"),
                                    **{key: result[key] for key in (
                                        "aborted", "recovery_required", "rationale",
                                        "completed_subtasks", "total_subtasks") if key in result}})
                                self.state["messages"] = self.state["messages"][-250:]
                                recovery = result.get("recovery_required", False)
                                self.state.update(
                                    recovery_required=recovery,
                                    stage="已完成" if result["success"] else "未完成")
                            job.set_result(
                                "success" if result["success"] else "incomplete",
                                result["message"],
                                result.get("recovery_required", False))
                    # The final trace is durable before controls become available.
                    with self.lock:
                        self.state["status"] = "ready"
                except Exception as exc:
                    # Provider exception strings can contain API keys/request data.
                    from ..services.cloud import ConfigurationError
                    message = (str(exc) if isinstance(exc, ConfigurationError)
                               else f"操作异常（{type(exc).__name__}），请重置后重试。")
                    with self.lock:
                        if kind == "agent":
                            message = f"Agent 切换失败（{type(exc).__name__}），已保留原 Agent，可重试。"
                        self.state.update(status="ready" if kind == "agent" else "error",
                                          error=message, stage="切换失败" if kind == "agent" else "异常")
                        if kind == "command" and self.state.get("workflow"):
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


class AgentSelection(BaseModel):
    agent: Literal["pickparts", "tiptop", "tiptop_optimized"]


def create_app(factory=RobotBackend, recorder=None):
    console = Console(factory, recorder)
    from ..observability.query import artifact_path, get_run, list_runs

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

    @app.get("/history")
    def history():
        return FileResponse(static / "history.html")

    @app.get("/api/state")
    def state():
        return console.snapshot()

    @app.get("/api/frame")
    def frame():
        with console.lock:
            content = console.frame
        return Response(content, media_type="image/jpeg") if content else Response(status_code=204)

    @app.get("/api/runs")
    def runs(day: date | None = None, limit: int = Query(100, ge=1, le=1000)):
        log_dir = getattr(console.recorder, "root", None)
        retention = getattr(getattr(console.recorder, "sink", None),
                            "retention_days", 7)
        return {"runs": list_runs(log_dir, day.isoformat() if day else None,
                                  limit, retention) if log_dir else []}

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        log_dir = getattr(console.recorder, "root", None)
        if log_dir is None:
            raise HTTPException(404)
        retention = getattr(getattr(console.recorder, "sink", None),
                            "retention_days", 7)
        detail = get_run(log_dir, run_id, retention)
        if detail is None:
            raise HTTPException(404)
        return detail

    @app.get("/api/runs/{run_id}/artifacts/{path:path}")
    def run_artifact(run_id: str, path: str):
        log_dir = getattr(console.recorder, "root", None)
        if log_dir is None:
            raise HTTPException(404)
        target = artifact_path(log_dir, run_id, path)
        if target is None:
            raise HTTPException(404)
        return FileResponse(target)

    @app.post("/api/command", status_code=202)
    def command(body: Command):
        console.submit("command", body.text)
        return {"accepted": True}

    @app.post("/api/reset", status_code=202)
    def reset():
        console.submit("reset")
        return {"accepted": True}

    @app.post("/api/agent", status_code=202)
    def select_agent(body: AgentSelection):
        console.submit("agent", body.agent)
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
