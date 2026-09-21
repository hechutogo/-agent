import importlib.util
from pathlib import Path


def test_domain_packages_are_importable():
    from pickparts_agent.agent import Orchestrator, WorldState
    from pickparts_agent.agent.atoms import FindObject, PlaceOn
    from pickparts_agent.scene import Frame, Simulation
    from pickparts_agent.services.cloud import Endpoint

    assert all((Orchestrator, WorldState, FindObject, PlaceOn,
                Frame, Simulation, Endpoint))


def test_old_internal_modules_are_removed():
    for name in (
        "pickparts_agent.planner",
        "pickparts_agent.executor",
        "pickparts_agent.reactor",
        "pickparts_agent.state",
        "pickparts_agent.atoms",
        "pickparts_agent.simulation",
        "pickparts_agent.perception",
        "pickparts_agent.cloud",
    ):
        assert importlib.util.find_spec(name) is None


def test_scene_does_not_import_higher_layers():
    forbidden = ("pickparts_agent.agent", "pickparts_agent.interfaces",
                 "pickparts_agent.services")
    for path in Path("src/pickparts_agent/scene").rglob("*.py"):
        text = path.read_text()
        assert not any(name in text for name in forbidden), path


def test_root_package_contains_only_entry_points_and_runtime():
    root = Path("src/pickparts_agent")
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py", "app.py", "runtime.py", "web.py",
    }


def test_assets_live_with_their_owning_domains():
    root = Path("src/pickparts_agent")
    assert (root / "interfaces/static/index.html").is_file()
    assert (root / "scene/robots/assets/xlerobot/xlerobot.urdf").is_file()
    assert not (root / "static").exists()
    assert not (root / "robots").exists()
