"""Command line interface for the TiPToP Mac/CPU comparison system."""
import argparse
from pathlib import Path
import sys


def _parser():
    parser = argparse.ArgumentParser(
        prog="tiptop_mac",
        description="TiPToP-style one-shot agent on Mac/CPU (ManiSkill 3)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one instruction once (open loop)")
    run.add_argument("instruction")
    run.add_argument("--seed", type=int, default=0)

    bench = sub.add_parser("benchmark",
                           help="Compare tiptop_mac against the ReAct agent")
    bench.add_argument("--output", type=Path, default=Path("runs"))
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    from pickparts_agent.runtime import configure
    configure()

    if args.command == "benchmark":
        from .benchmark import Benchmark
        path = Benchmark().run(args.output)
        print(f"Benchmark report: {path / 'report.md'}")
        return 0

    from dotenv import load_dotenv
    load_dotenv()
    from pickparts_agent.services.cloud import ConfigurationError, Endpoint
    try:
        llm = Endpoint.from_env("LLM")
    except ConfigurationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    sim = None
    try:
        from pickparts_agent.scene.simulation import Simulation
        sim = Simulation(seed=args.seed)
        from .agent import build_tiptop_agent
        result = build_tiptop_agent(sim, llm).run(args.instruction)
        print(result["message"])
        return 0 if result["success"] else 1
    finally:
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
