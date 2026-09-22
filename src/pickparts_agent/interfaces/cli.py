"""CLI entry point. Heavy simulator and cloud imports stay behind argument parsing."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys


def save_frame(frame, output):
    import numpy as np
    from PIL import Image

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.rgb).save(output / "rgb.png")
    np.savez_compressed(output / "depth.npz", depth=frame.depth,
                        intrinsic=frame.intrinsic, camera_to_base=frame.camera_to_base)


def _record(seconds, output):
    if not 0 < seconds <= 120:
        raise ValueError("Recording duration must be between 0 and 120 seconds")
    try:
        import sounddevice
    except ImportError:
        raise RuntimeError("Microphone support requires: pip install '.[microphone]'") from None
    import soundfile

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    samples = sounddevice.rec(int(seconds * sample_rate), samplerate=sample_rate,
                              channels=1, dtype="float32")
    sounddevice.wait()
    path = output / "recording.wav"
    soundfile.write(path, samples, sample_rate, subtype="PCM_16")
    return path


def _parser():
    parser = argparse.ArgumentParser(description="XLeRobot sensor-only cloud Agent / RGB-D demo")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true", help="Save RGB PNG and metric depth NPZ")
    modes.add_argument("--demo", nargs="*", choices=["A", "B"], metavar="PART",
                       help="Deterministic OpenCV RGB-D baseline, default A B (not cloud AI)")
    modes.add_argument("--audio", type=Path, metavar="FILE", help="Cloud ASR of an audio file")
    modes.add_argument("--record", type=float, metavar="SECONDS",
                       help="Record microphone audio, then use cloud ASR (microphone extra)")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", action="store_true",
                        help="Open simulator viewer; smoke/demo wait until it closes (default headless)")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.record is not None and not 0 < args.record <= 120:
        parser.error("--record must be greater than 0 and at most 120 seconds")
    if args.audio is not None and not args.audio.is_file():
        parser.error("--audio must name an existing file")
    sim = None
    try:
        with ExitStack() as stack:
            if not args.smoke and args.demo is None:
                from dotenv import load_dotenv
                from ..services.cloud import VisualLocator, Endpoint, transcribe

                load_dotenv()
                llm = Endpoint.from_env("LLM")
                vision = Endpoint.from_env("VLM")
                vision_client = stack.enter_context(vision.client())
                command = None
                if args.audio is not None or args.record is not None:
                    asr = Endpoint.from_env("ASR")
                    asr_client = stack.enter_context(asr.client())
                    audio = args.audio if args.audio is not None else _record(args.record, args.output)
                    command = transcribe(asr_client, asr.model, audio)

            from ..scene.simulation import Simulation

            sim = Simulation(seed=args.seed)
            if args.view:
                sim.enable_viewer()
            if args.smoke:
                sim.hold(10)
                save_frame(sim.observe(), args.output)
                print(f"Simulation RGB-D smoke artifacts: {args.output}")
                if args.view:
                    while sim.viewer_open:
                        sim.hold(1)
                return 0

            from ..baseline.motion import PickPlace

            if args.demo is not None:
                from ..scene.perception import ColorPerception

                print("Deterministic OpenCV RGB-D baseline (NOT cloud Agent validation)")
                controller = PickPlace(sim, ColorPerception(), output=args.output)
                successful = True
                for index, target in enumerate(args.demo or ["A", "B"]):
                    result = controller.run(target)
                    save_frame(sim.observe(), args.output / f"{index:02d}-{target}")
                    print(result["message"])
                    if result["success"] and "还需要什么" not in result["message"]:
                        print("还需要什么")
                    if not result["success"]:
                        successful = False
                        break
                if args.view:
                    while sim.viewer_open:
                        sim.hold(1)
                return 0 if successful else 1

            from ..agent.orchestrator import build_orchestrator
            from ..observability import build_recorder

            recorder = build_recorder()
            locate = VisualLocator(sim.object_specs, vision_client, vision.model).locate
            orchestrator = build_orchestrator(sim, locate, llm, recorder=recorder)
            print("Flexible atom Agent mode. Describe the task in natural language; quit to exit.")
            print("Perception: full-frame RGB-D for catalog objects; cloud vision for open descriptions.")
            turn = 0
            while True:
                if command is None:
                    try:
                        text = input("> ")
                    except EOFError:
                        return 0
                    if text.strip().lower() in ("quit", "exit"):
                        return 0
                else:
                    text = command
                with recorder.run("command", text) as job:
                    result = orchestrator.turn(text)
                    job.set_result(
                        "success" if result["success"] else "incomplete",
                        result["message"],
                        result.get("recovery_required", False))
                print(result["message"])
                save_frame(sim.observe(), args.output / f"turn-{turn:03d}")
                turn += 1
                if command is not None:
                    return 0 if result["success"] else 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Provider exceptions can embed credentials or request bodies.
        from ..services.cloud import CloudError, ConfigurationError

        message = str(exc) if isinstance(exc, (CloudError, ConfigurationError)) else type(exc).__name__
        if isinstance(exc, RuntimeError) and str(exc).startswith("Microphone support requires:"):
            message = str(exc)
        print(f"Error: {message}", file=sys.stderr)
        return 1
    finally:
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
