"""No single-frame success and no actuation while resolving visual noise."""
import pytest


def test_transient_occlusion_then_two_positive_frames_confirms():
    from pickparts_agent.scene.verification import sample_until_stable
    values = iter([ValueError("occluded"), True, True])
    waits = []

    def sample():
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    result = sample_until_stable(sample, bool, lambda: waits.append(True))
    assert result.confirmed and result.samples == 3
    assert len(waits) == 2
    assert result.error is None


def test_isolated_positive_frames_do_not_confirm():
    from pickparts_agent.scene.verification import sample_until_stable
    values = iter([True, False, True])
    result = sample_until_stable(lambda: next(values), bool, lambda: None)
    assert not result.confirmed and result.samples == 3


def test_permanent_occlusion_is_bounded_and_retains_error():
    from pickparts_agent.scene.verification import sample_until_stable
    def sample():
        raise ValueError("occluded")
    result = sample_until_stable(sample, bool, lambda: None)
    assert not result.confirmed and result.samples == 3
    assert str(result.error) == "occluded"


def test_backend_failure_does_not_become_visual_retry():
    from pickparts_agent.scene.verification import sample_until_stable
    def sample():
        raise RuntimeError("renderer unavailable")
    with pytest.raises(RuntimeError, match="renderer"):
        sample_until_stable(sample, bool, lambda: None)


def test_optimized_transient_verification_does_not_reexecute_or_call_llm():
    from test_tiptop_optimized_agent import build
    agent, events, _ = build(["box"])
    original = agent.executor.reconcile
    calls = []
    def reconcile(scene):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("temporary occlusion")
        return original(scene)
    agent.executor.reconcile = reconcile
    result = agent.run("入盒")
    assert result["success"] and result["attempts"] == 1
    assert len([event for event in events if event["type"] == "plan"]) == 1
    assert not [event for event in events if event["type"] == "react"]


def test_optimized_single_positive_frame_does_not_finish():
    from test_tiptop_optimized_agent import build
    agent, _, _ = build(["box"], ["false_success"] * 5)
    frames = iter([True, False, True] * 5)
    agent.planner.satisfies = lambda *args, **kwargs: next(frames)
    result = agent.run("入盒")
    assert not result["success"] and result["completed_subtasks"] == 0


def test_self_agent_does_not_accept_a_single_positive_relation():
    from helpers import make_context
    from pickparts_agent.agent.atoms import VerifyState
    from test_placement_verification import detection
    frames = iter([True, False, True])
    def locate(frame, target):
        if target == "box":
            return detection(bottom=.720, kind="box")
        return detection(x=0. if next(frames) else .10)
    ctx = make_context(locate=locate)
    result = VerifyState().call(ctx, {"target": "A", "at": "box"})
    assert not result.success
    assert ctx.state.objects["A"].placement is None


def test_box_occlusion_uses_visible_containment_consistently():
    from pickparts_agent.agent.atoms import VerifyState
    from test_placement_verification import context_with_measurements, detection
    part = detection(bottom=.757)
    part.update(top_z=.762, point=[0., 0., .761], extent=[.024, .024, .005])
    ctx = context_with_measurements(part)
    result = VerifyState().call(ctx, {"target": "A", "at": "box"})
    assert result.success
