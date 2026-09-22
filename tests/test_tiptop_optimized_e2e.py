"""Real RGB-D and physics acceptance; only the remote language model is fixed."""
import pytest

from helpers import FakeChat
from pickparts_agent.scene.kinematics import ArmKinematics
from pickparts_agent.scene.perception import ScenePerception
from pickparts_agent.scene.simulation import Simulation
from tiptop_mac.perception import build_scene_graph
from tiptop_mac.types import Predicate


@pytest.mark.parametrize("inject_failure", [False, True])
def test_box_table_box_round_trip_with_visual_checkpoints(inject_failure, monkeypatch, tmp_path):
    from tiptop_optimized.agent import OptimizedTiPToPAgent
    from tiptop_optimized.executor import RecoveringExecutor
    from tiptop_optimized.grounding import OrderedGrounder, RecoveryReactor
    from tiptop_optimized.tamp import TAMPLite
    from pickparts_agent.observability import Recorder

    def request(supports):
        return {"subtasks": [
            {"instruction": f"把 A 放到 {support}",
             "goal": [{"predicate": "on", "args": ["A", support]}]}
            for support in supports]}

    sim = Simulation(seed=7)
    try:
        kin = ArmKinematics()
        planner = TAMPLite(kin)
        executor = RecoveringExecutor(
            sim, kin, locate=ScenePerception(sim.object_specs).locate)
        checkpoints, events = [], []
        def event(payload):
            events.append(payload)
            if payload["type"] == "subtasks" and payload["steps"][0]["status"] == "done":
                scene = build_scene_graph(sim.observe(), sim.object_specs)
                checkpoints.append(planner._initial_support(scene).get("A"))
        recorder = Recorder(tmp_path)
        executor.recorder = recorder
        executor.on_event = event
        agent = OptimizedTiPToPAgent(
            sim, OrderedGrounder(FakeChat([request(["box"]), request(["table", "box"])])),
            planner, executor,
            reactor=RecoveryReactor(FakeChat([
                {"action": "replan", "reason": "重新观测，规划当前子任务。"}] * 6)),
            on_event=event, recorder=recorder)
        with recorder.run("physical-setup", "入盒", agent="tiptop_optimized"):
            setup = agent.run("先把 A 放入盒子")
        if not setup["success"]:
            scene = build_scene_graph(sim.observe(), sim.object_specs)
            diagnostic = {
                "objects": {k: {"point": n.point.tolist(), "extent": n.extent.tolist(),
                                "bottom": n.bottom_z, "top": n.top_z}
                            for k, n in scene.objects.items()},
                "qpos": scene.qpos.tolist(),
                "joints": sim.joint_names,
                "held": executor.held, "candidate": executor.candidate,
            }
            pytest.fail(str((setup, diagnostic)))
        checkpoints.clear()
        events.clear()
        if inject_failure:
            import tiptop_optimized.executor as module
            original = module._cartesian
            remaining = [1]
            def flaky(*args, **kwargs):
                if remaining[0]:
                    remaining[0] -= 1
                    raise ValueError("Injected transient IK failure")
                return original(*args, **kwargs)
            monkeypatch.setattr(module, "_cartesian", flaky)
        with recorder.run("physical-roundtrip", "先桌面再回盒子", agent="tiptop_optimized"):
            result = agent.run("A 已在盒子里，先把 A 放到桌面，再把 A 放回盒子")
        assert result["success"], str(result)
        assert result["completed_subtasks"] == 2
        assert "table" in checkpoints and checkpoints[-1] == "box", checkpoints
        assert result["attempts"] == (3 if inject_failure else 2)
        assert len([e for e in events if e["type"] == "react"]) == int(inject_failure)
        final = build_scene_graph(sim.observe(), sim.object_specs)
        assert planner.satisfies(final, (Predicate("on", ("A", "box")),))
    finally:
        sim.close()


@pytest.mark.parametrize("seed", [7, 17, 27])
def test_stack_then_box_with_fresh_visual_checkpoints(seed, tmp_path):
    from tiptop_optimized.agent import OptimizedTiPToPAgent
    from tiptop_optimized.executor import RecoveringExecutor
    from tiptop_optimized.grounding import OrderedGrounder, RecoveryReactor
    from tiptop_optimized.tamp import TAMPLite
    from pickparts_agent.observability import Recorder

    sim = Simulation(seed=seed)
    try:
        kin, recorder = ArmKinematics(), Recorder(tmp_path)
        planner = TAMPLite(kin)
        executor = RecoveringExecutor(
            sim, kin, locate=ScenePerception(sim.object_specs).locate,
            recorder=recorder)
        checkpoints = []

        def event(payload):
            if payload["type"] == "subtasks":
                completed = sum(row["status"] == "done" for row in payload["steps"])
                if completed > len(checkpoints):
                    scene = build_scene_graph(sim.observe(), sim.object_specs)
                    checkpoints.append(planner.observed_support(
                        scene, held=executor.reconcile(scene)).get("A"))

        agent = OptimizedTiPToPAgent(
            sim, OrderedGrounder(FakeChat([{"subtasks": [
                {"instruction": f"把 A 放到 {support}",
                 "goal": [{"predicate": "on", "args": ["A", support]}]}
                for support in ("B", "box")]}])),
            planner, executor, recorder=recorder, on_event=event,
            reactor=RecoveryReactor(FakeChat([
                {"action": "reobserve", "reason": "先刷新观测再规划当前子任务"}
            ] * 8)))
        with recorder.run("physical-stack-box", f"stack then box seed={seed}",
                          agent="tiptop_optimized"):
            result = agent.run("把A放到B上面，再把A放到盒子里")
        assert result["success"], str((result, checkpoints))
        assert checkpoints == ["B", "box"]
        assert result["completed_subtasks"] == 2
        assert executor.held is executor.candidate is None
    finally:
        sim.close()
