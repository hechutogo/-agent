import json
from types import SimpleNamespace

import pytest

from helpers import FakeChat
from pickparts_agent.agent.atoms import build_default_registry
from pickparts_agent.agent.llm import LLMError
from pickparts_agent.agent.planner import Planner
from pickparts_agent.agent.state import WorldState


def planner(chat):
    return Planner(chat, build_default_registry())


def valid_steps():
    return [
        {"atom": "find_object", "args": {"target": "A"}},
        {"atom": "find_object", "args": {"target": "box"}},
        {"atom": "reach_above", "args": {"target": "A"}},
        {"atom": "grasp", "args": {"target": "A"}},
        {"atom": "lift", "args": {"clearance": 0.1}},
        {"atom": "carry_to", "args": {"container": "box"}},
        {"atom": "release_into", "args": {"container": "box"}},
        {"atom": "verify_state", "args": {"target": "A", "at": "box"}},
        {"atom": "reset_arm", "args": {}},
    ]


def test_feasible_plan_returns_validated_steps():
    chat = FakeChat([{"feasible": True, "blockers": [], "steps": valid_steps(),
                      "rationale": "all visible"}])
    plan = planner(chat).analyze("把 A 放进盒子", WorldState())
    assert plan.feasible and len(plan.steps) == 9
    assert "find_object" in chat.calls[0]["system"]


def test_infeasible_returns_blockers_and_no_steps():
    chat = FakeChat([{"feasible": False,
                      "blockers": [{"need": "目标可见", "reason": "没有零件 C"}],
                      "steps": [], "rationale": ""}])
    plan = planner(chat).analyze("把 C 放进盒子", WorldState())
    assert not plan.feasible and plan.steps == []
    assert plan.blockers[0]["need"] == "目标可见"


def test_bad_json_returns_infeasible_plan():
    plan = planner(FakeChat([LLMError("x")])).analyze("do it", WorldState())
    assert not plan.feasible and plan.steps == []


def test_unknown_atom_in_plan_fails_closed():
    steps = valid_steps() + [{"atom": "explode", "args": {}}]
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("goal", WorldState())
    assert not plan.feasible and plan.steps == []


@pytest.mark.parametrize("feasible", ["false", "true", 0, 1, None, [], {}])
def test_feasibility_requires_a_json_boolean(feasible):
    plan = planner(FakeChat([{"feasible": feasible, "steps": valid_steps()}])
                   ).analyze("move A", WorldState())
    assert plan.feasible is False and plan.steps == []
    assert plan.blockers


@pytest.mark.parametrize("patch", [
    {"steps": []}, {"steps": None}, {"steps": {}},
    {"blockers": "blocked"}, {"blockers": [None]},
    {"blockers": [{"need": "permission", "reason": 5}]},
    {"blockers": [{"need": "permission", "reason": "not granted"}]},
    {"rationale": {"private": "not a summary"}},
])
def test_invalid_success_payload_fails_closed(patch):
    payload = {"feasible": True, "blockers": [], "steps": valid_steps(),
               "rationale": "Locate both objects, then move A into the box."}
    payload.update(patch)
    plan = planner(FakeChat([payload])).analyze("move A", WorldState())
    assert plan.feasible is False and plan.steps == []
    assert plan.blockers


@pytest.mark.parametrize("removed", [0, 1, 2, 4, 5, 7])
def test_missing_pick_place_precondition_or_verification_fails_closed(removed):
    steps = valid_steps()
    del steps[removed]
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("move A", WorldState())
    assert plan.feasible is False and plan.steps == []


def stack_steps(source="part-7", destination="support-9"):
    return [
        {"atom": "find_object", "args": {"target": source}},
        {"atom": "find_object", "args": {"target": destination}},
        {"atom": "reach_above", "args": {"target": source}},
        {"atom": "grasp", "args": {"target": source}},
        {"atom": "lift", "args": {"clearance": 0.1}},
        {"atom": "carry_to", "args": {"container": destination}},
        {"atom": "place_on", "args": {"target": destination}},
        {"atom": "verify_state", "args": {
            "target": source, "at": destination, "relation": "on"}},
        {"atom": "reset_arm", "args": {}},
    ]


def table_steps(source="part-7"):
    return [
        {"atom": "find_object", "args": {"target": source}},
        {"atom": "reach_above", "args": {"target": source}},
        {"atom": "grasp", "args": {"target": source}},
        {"atom": "lift", "args": {"clearance": 0.1}},
        {"atom": "place_on_table", "args": {}},
        {"atom": "verify_state", "args": {
            "target": source, "at": "table", "relation": "table"}},
        {"atom": "reset_arm", "args": {}},
    ]


def test_table_placement_uses_runtime_grounded_table_atom():
    steps = table_steps()
    plan = planner(FakeChat([{"feasible": True, "steps": steps,
                             "rationale": "Place the part on a safe table area."}])
                   ).analyze("move the part onto the table", WorldState())
    assert plan.feasible and plan.steps == steps


def test_table_cannot_be_localized_or_used_as_an_object_destination():
    steps = [
        {"atom": "find_object", "args": {"target": "A"}},
        {"atom": "find_object", "args": {"target": "table"}},
        {"atom": "reach_above", "args": {"target": "A"}},
        {"atom": "grasp", "args": {"target": "A"}},
        {"atom": "lift", "args": {}},
        {"atom": "carry_to", "args": {"container": "table"}},
        {"atom": "place_on", "args": {"target": "table"}},
        {"atom": "verify_state", "args": {
            "target": "A", "at": "table", "relation": "table"}},
    ]
    plan = planner(FakeChat([{"feasible": True, "steps": steps,
                             "rationale": "Move A to the table."}])
                   ).analyze("move A to the table", WorldState())
    assert not plan.feasible and not plan.steps


def test_table_cannot_be_verified_as_the_movable_target():
    steps = [{"atom": "verify_state", "args": {
        "target": "table", "at": "table", "relation": "table"}}]
    plan = planner(FakeChat([{"feasible": True, "steps": steps,
                             "rationale": "Verify table."}])
                   ).analyze("verify table", WorldState())
    assert not plan.feasible and not plan.steps


@pytest.mark.parametrize("source,destination", [
    ("A", "B"), ("part-7", "support-9"),
    ("small yellow part near the edge", "wide purple block"),
])
def test_dynamic_and_visual_description_stack_plan(source, destination):
    steps = stack_steps(source, destination)
    chat = FakeChat([{"feasible": True, "steps": steps,
                      "rationale": "Stack the selected part on its support."}])
    plan = planner(chat).analyze("stack", WorldState())
    assert plan.feasible and plan.steps == steps


@pytest.mark.parametrize("patch", [
    {"target": "another-part", "at": "support-9", "relation": "on"},
    {"target": "part-7", "at": "table", "relation": "on"},
    {"target": "part-7", "at": "support-9", "relation": "sideways"},
])
def test_stack_verification_must_match_placed_source_and_destination(patch):
    steps = stack_steps()
    steps[7]["args"] = patch
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("stack", WorldState())
    assert not plan.feasible and not plan.steps


def test_snapshot_catalog_and_observation_age_are_supplied_to_planner():
    snapshot = {"catalog": {
        "part-7": {"id": "part-7", "kind": "block", "label": "yellow part",
                   "color": "yellow"}},
        "frame_id": 12, "objects": {"part-7": {
            "id": "part-7", "visible": False, "frame_id": 3,
            "placement": "table"}}}
    chat = FakeChat([{"feasible": True, "steps": [
        {"atom": "find_object", "args": {"target": "part-7"}}],
        "rationale": "Refresh the yellow part's location."}])
    plan = planner(chat).analyze(
        "locate yellow part", SimpleNamespace(snapshot=lambda: snapshot))
    assert plan.feasible
    assert json.loads(chat.calls[0]["user"])["world"] == snapshot


@pytest.mark.parametrize("kind,relation", [("block", "on"), ("box", "in")])
def test_placement_relation_matches_real_catalog(kind, relation):
    state = WorldState(catalog=[
        {"id": "part-7", "kind": "block", "label": "yellow part", "color": [1, 1, 0]},
        {"id": "support-9", "kind": kind, "label": "destination", "color": [1, 0, 1]},
    ])
    steps = stack_steps()
    if kind == "box":
        steps[6] = {"atom": "release_into", "args": {"container": "support-9"}}
    steps[7]["args"]["relation"] = relation
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("place", state)
    assert plan.feasible and plan.steps == steps
    steps[7]["args"]["relation"] = "in" if relation == "on" else "on"
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("place", state)
    assert not plan.feasible and not plan.steps


def test_held_source_replan_can_continue_without_regrasping():
    state = WorldState()
    state.set_held("A")
    steps = [stack_steps("A", "B")[1], *stack_steps("A", "B")[5:]]
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("A onto B", state)
    assert plan.feasible and plan.steps == steps


def test_held_object_cannot_be_released_with_generic_gripper_open():
    state = WorldState()
    state.set_held("A")
    steps = [
        {"atom": "set_gripper", "args": {"open": True}},
        {"atom": "verify_state", "args": {
            "target": "A", "at": "table", "relation": "table"}},
    ]
    plan = planner(FakeChat([{"feasible": True, "steps": steps,
                             "rationale": "Release A."}])
                   ).analyze("place A safely on table", state)
    assert not plan.feasible and not plan.steps


def test_destination_located_only_after_grasp_is_rejected():
    steps = stack_steps("A", "B")
    destination = steps.pop(1)
    steps.insert(4, destination)
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("stack", WorldState())
    assert not plan.feasible and not plan.steps


def test_non_object_plan_response_fails_closed():
    chat = SimpleNamespace(json=lambda *args: None)
    plan = planner(chat).analyze("stack", WorldState())
    assert not plan.feasible and plan.blockers and not plan.steps


def test_closing_gripper_between_reach_and_grasp_requires_reopening():
    state = WorldState()
    steps = stack_steps("A", "B")
    steps.insert(3, {"atom": "set_gripper", "args": {"open": False}})
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("stack", state)
    assert not plan.feasible
    steps.insert(4, {"atom": "set_gripper", "args": {"open": True}})
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("stack", state)
    assert plan.feasible


def test_opening_gripper_after_lift_invalidates_planned_hold():
    steps = stack_steps("A", "B")
    steps.insert(5, {"atom": "set_gripper", "args": {"open": True}})
    plan = planner(FakeChat([{"feasible": True, "steps": steps}])
                   ).analyze("stack", WorldState())
    assert not plan.feasible
