from pickparts_agent.agent.atoms import build_default_registry

EXPECTED = {"find_object", "reach_above", "grasp", "lift", "carry_to",
            "release_into", "verify_state", "reset_arm", "set_gripper",
            "place_on", "place_on_table"}


def test_default_registry_registers_all_motion_and_relation_atoms():
    registry = build_default_registry()
    assert set(registry.atoms) == EXPECTED


def test_catalog_is_non_empty_text():
    assert "find_object" in build_default_registry().catalog_for_prompt()
