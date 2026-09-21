"""Two-architecture benchmark: ReAct pickparts agent vs open-loop tiptop_mac."""
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time

from .types import Predicate
from .evaluation import judge_outcome


@dataclass(frozen=True)
class Case:
    seed: int
    instruction: str
    movable: str
    support: str
    relation: str


BENCHMARK_CASES = (
    Case(0, "把 A 叠到 B 上", "A", "B", "on"),
    Case(3, "把 B 叠到 A 上", "B", "A", "on"),
    Case(7, "把 A 放进盒子里", "A", "box", "in"),
)


@dataclass(frozen=True)
class CaseResult:
    system: str
    seed: int
    status: str
    success: bool
    aborted: bool
    elapsed: float
    verdict_relation: str
    reason: str


def _skip(system, case, exc):
    return CaseResult(system, case.seed, "skipped", False, False,
                      0.0, "", str(exc))


def run_tiptop(case):
    from dotenv import load_dotenv
    load_dotenv()
    from pickparts_agent.services.cloud import ConfigurationError, Endpoint
    try:
        llm = Endpoint.from_env("LLM")
    except ConfigurationError as exc:
        return _skip("tiptop_mac", case, exc)
    sim = None
    try:
        from pickparts_agent.scene.simulation import Simulation
        sim = Simulation(seed=case.seed)
        from .agent import build_tiptop_agent
        agent = build_tiptop_agent(sim, llm)
        started = time.perf_counter()
        output = agent.run(case.instruction)
        elapsed = time.perf_counter() - started
        goal = (Predicate("on", (case.movable, case.support)),)
        verdict = judge_outcome(sim.observe(), sim.object_specs, goal)
        return CaseResult("tiptop_mac", case.seed, "completed",
                          verdict.success, output.get("aborted", False),
                          round(elapsed, 3), verdict.relation,
                          verdict.reason)
    except Exception as exc:
        return CaseResult("tiptop_mac", case.seed, "error", False, False,
                          0.0, "", f"{type(exc).__name__}: {exc}")
    finally:
        if sim is not None:
            sim.close()


def run_pickparts(case):
    from dotenv import load_dotenv
    load_dotenv()
    from pickparts_agent.services.cloud import ConfigurationError, Endpoint
    try:
        llm = Endpoint.from_env("LLM")
        vision = Endpoint.from_env("VLM")
    except ConfigurationError as exc:
        return _skip("pickparts", case, exc)
    sim = None
    try:
        from pickparts_agent.scene.simulation import Simulation
        sim = Simulation(seed=case.seed)
        with ExitStack() as stack:
            client = stack.enter_context(vision.client())
            from pickparts_agent.services.cloud import VisualLocator
            from pickparts_agent.agent.orchestrator import build_orchestrator
            locate = VisualLocator(sim.object_specs, client,
                                   vision.model).locate
            orchestrator = build_orchestrator(sim, locate, llm)
            started = time.perf_counter()
            output = orchestrator.turn(case.instruction)
            elapsed = time.perf_counter() - started
        goal = (Predicate("on", (case.movable, case.support)),)
        verdict = judge_outcome(sim.observe(), sim.object_specs, goal)
        return CaseResult("pickparts", case.seed, "completed",
                          verdict.success, False, round(elapsed, 3),
                          verdict.relation,
                          verdict.reason if verdict.success
                          else output.get("message", verdict.reason))
    except Exception as exc:
        return CaseResult("pickparts", case.seed, "error", False, False,
                          0.0, "", f"{type(exc).__name__}: {exc}")
    finally:
        if sim is not None:
            sim.close()


DEFAULT_RUNNERS = {"tiptop_mac": run_tiptop, "pickparts": run_pickparts}


def _render(payload):
    lines = [
        "# 双架构对比评测",
        f"生成时间：{payload['generated_at']}",
        "",
        "| seed | 任务 | 系统 | 状态 | 裁判关系 | 成功 | 中止 | 耗时(s) | 说明 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in payload["runs"]:
        lines.append("| {seed} | {instruction} | {system} | {status} | "
                     "{relation} | {success} | {aborted} | {elapsed} | "
                     "{reason} |".format(
                         relation=run["verdict_relation"], **run))
    return "\n".join(lines) + "\n"


class Benchmark:
    def __init__(self, cases=BENCHMARK_CASES, *, runners=None,
                 clock=datetime.now):
        self.cases = tuple(cases)
        self.runners = runners if runners is not None else DEFAULT_RUNNERS
        self.clock = clock

    def run(self, output_root):
        rows = []
        for case in self.cases:
            for runner in self.runners.values():
                rows.append(runner(case))
        stamp = self.clock().strftime("%Y%m%d-%H%M%S")
        path = Path(output_root) / "benchmark" / stamp
        path.mkdir(parents=True, exist_ok=True)
        payload = {"generated_at": stamp,
                   "runs": [{**asdict(r),
                             "instruction": next(
                                 c.instruction for c in self.cases
                                 if c.seed == r.seed)}
                            for r in rows]}
        (path / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2))
        (path / "report.md").write_text(_render(payload))
        return path
