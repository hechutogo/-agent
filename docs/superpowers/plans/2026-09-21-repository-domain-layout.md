# Repository Domain Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Reorganize the flat `pickparts_agent` package into explicit Agent,
scene, service, interface, and baseline domains without changing behavior.

**Architecture:** Move domain logic into `agent/`, `scene/`, `services/`,
`interfaces/`, and `baseline/`. Split the large atom library by responsibility,
hard-migrate internal imports, and retain only thin root CLI/Web entry wrappers.

**Tech Stack:** Python 3.11, setuptools, pytest, ManiSkill 3, SAPIEN, FastAPI.

**Spec:** `docs/superpowers/specs/2026-09-21-repository-domain-layout-design.md`

## Global Constraints

- Preserve the non-privileged observation boundary.
- Preserve all ten atoms and the flat AtomCall contract.
- Preserve `python -m pickparts_agent.app` and
  `python -m pickparts_agent.web`.
- Do not add dependencies or change runtime behavior.
- Do not keep compatibility modules for old internal Agent paths.
- Preserve unrelated and pre-existing uncommitted changes.
- Do not create commits during this migration because moved files already
  contain uncommitted implementation that must remain intact.

---

## File Map

### Create

```text
src/pickparts_agent/__init__.py
src/pickparts_agent/agent/__init__.py
src/pickparts_agent/agent/atoms/__init__.py
src/pickparts_agent/agent/atoms/base.py
src/pickparts_agent/agent/atoms/registry.py
src/pickparts_agent/agent/atoms/perception.py
src/pickparts_agent/agent/atoms/manipulation.py
src/pickparts_agent/agent/atoms/placement.py
src/pickparts_agent/agent/planner.py
src/pickparts_agent/agent/executor.py
src/pickparts_agent/agent/reactor.py
src/pickparts_agent/agent/orchestrator.py
src/pickparts_agent/agent/state.py
src/pickparts_agent/agent/llm.py
src/pickparts_agent/scene/__init__.py
src/pickparts_agent/scene/simulation.py
src/pickparts_agent/scene/objects.py
src/pickparts_agent/scene/perception.py
src/pickparts_agent/scene/kinematics.py
src/pickparts_agent/scene/robots/__init__.py
src/pickparts_agent/services/__init__.py
src/pickparts_agent/services/cloud.py
src/pickparts_agent/interfaces/__init__.py
src/pickparts_agent/interfaces/cli.py
src/pickparts_agent/interfaces/web.py
src/pickparts_agent/baseline/__init__.py
src/pickparts_agent/baseline/motion.py
tests/test_package_layout.py
```

### Move

```text
src/pickparts_agent/robots/assets -> src/pickparts_agent/scene/robots/assets
src/pickparts_agent/static -> src/pickparts_agent/interfaces/static
```

### Replace with thin wrappers

```text
src/pickparts_agent/app.py
src/pickparts_agent/web.py
```

### Delete after migration

```text
src/pickparts_agent/atoms/
src/pickparts_agent/cloud.py
src/pickparts_agent/executor.py
src/pickparts_agent/kinematics.py
src/pickparts_agent/llm.py
src/pickparts_agent/motion.py
src/pickparts_agent/orchestrator.py
src/pickparts_agent/perception.py
src/pickparts_agent/planner.py
src/pickparts_agent/reactor.py
src/pickparts_agent/robots/
src/pickparts_agent/scene_objects.py
src/pickparts_agent/simulation.py
src/pickparts_agent/state.py
```

---

### Task 1: Add package-layout acceptance tests

**Files:**
- Create: `tests/test_package_layout.py`

**Interfaces:**
- Consumes: existing importable classes and entry points.
- Produces: structural acceptance tests for canonical imports and forbidden old
  imports.

- [x] **Step 1: Write canonical import tests**

```python
import importlib.util


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
    ):
        assert importlib.util.find_spec(name) is None
```

- [x] **Step 2: Run the tests and confirm they fail before migration**

```bash
.venv/bin/python -m pytest tests/test_package_layout.py -q
```

Expected: canonical domain imports fail and old modules still exist.

---

### Task 2: Move the scene domain

**Files:**
- Create: `src/pickparts_agent/scene/__init__.py`
- Move: `kinematics.py`, `perception.py`, `simulation.py`
- Rename: `scene_objects.py` to `scene/objects.py`
- Move: `robots/` to `scene/robots/`
- Modify: scene files and scene-focused tests

**Interfaces:**
- Produces:

```python
from pickparts_agent.scene import (
    ArmKinematics, Frame, ScenePerception, Simulation, backproject,
)
```

- [x] **Step 1: Move files without changing behavior**

```text
kinematics.py    -> scene/kinematics.py
perception.py    -> scene/perception.py
simulation.py    -> scene/simulation.py
scene_objects.py -> scene/objects.py
robots/          -> scene/robots/
```

- [x] **Step 2: Add the scene public API**

```python
from .kinematics import ARM_NAMES, ArmKinematics
from .perception import ColorPerception, Frame, ScenePerception, backproject
from .simulation import Simulation

__all__ = [
    "ARM_NAMES", "ArmKinematics", "ColorPerception", "Frame",
    "ScenePerception", "Simulation", "backproject",
]
```

- [x] **Step 3: Correct scene-local imports**

```python
# scene/simulation.py
from ..runtime import configure
from .objects import _SceneObjects
from .perception import Frame
from .robots.xlerobot import PickPartsXLeRobot, RIGHT, LEFT, HEAD

# scene/objects.py
from .kinematics import ArmKinematics
```

- [x] **Step 4: Update scene test imports**

Replace root imports with `pickparts_agent.scene.*` in:

```text
tests/helpers.py
tests/test_color_perception.py
tests/test_kinematics.py
tests/test_observation_geometry.py
tests/test_scene_objects.py
tests/test_scene_perception.py
tests/test_simulation.py
tests/test_pick_place.py
tests/test_flexible_simulation.py
```

- [x] **Step 5: Run scene tests**

```bash
.venv/bin/python -m pytest \
  tests/test_kinematics.py tests/test_observation_geometry.py \
  tests/test_scene_objects.py tests/test_scene_perception.py \
  tests/test_simulation.py -q
```

Expected: all pass.

---

### Task 3: Move Agent orchestration and state

**Files:**
- Create: `src/pickparts_agent/agent/__init__.py`
- Move: `src/pickparts_agent/atoms/` to
  `src/pickparts_agent/agent/atoms/` without splitting it yet
- Move: `planner.py`, `executor.py`, `reactor.py`, `orchestrator.py`,
  `state.py`, `llm.py`
- Modify: moved imports and Agent-focused tests

**Interfaces:**
- Produces:

```python
from pickparts_agent.agent import (
    Orchestrator, Planner, ReActExecutor, Reactor, WorldState,
    build_orchestrator,
)
```

- [x] **Step 1: Move Agent modules and the intact atom package**

Move the six orchestration/state files and existing `atoms/` directory into
`agent/` without changing class implementations. This keeps the Agent package
importable before the library split.

- [x] **Step 2: Add Agent public exports**

```python
from .executor import ReActExecutor
from .orchestrator import Orchestrator, build_orchestrator
from .planner import Plan, Planner
from .reactor import Decision, Reactor
from .state import ObjectRecord, WorldState

__all__ = [
    "Decision", "ObjectRecord", "Orchestrator", "Plan", "Planner",
    "ReActExecutor", "Reactor", "WorldState", "build_orchestrator",
]
```

- [x] **Step 3: Update imports inside moved modules**

```python
# agent/orchestrator.py
from ..scene.kinematics import ArmKinematics
from .atoms import build_default_registry

# planner.py and reactor.py
from .llm import LLMError

# executor.py
from .atoms.base import AtomContext

# agent/atoms/library.py
from ...scene.kinematics import ARM_NAMES
```

- [x] **Step 4: Update Agent test imports**

Use `pickparts_agent.agent.*` in state, LLM, planner, reactor, executor, and
orchestrator tests.

- [x] **Step 5: Run non-atom Agent tests that can import**

```bash
.venv/bin/python -m pytest \
  tests/test_state.py tests/test_llm_json.py \
  tests/test_planner.py tests/test_reactor.py \
  tests/test_executor.py tests/test_orchestrator.py -q
```

Expected: all pass.

---

### Task 4: Move and split the atom package

**Files:**
- Create: `src/pickparts_agent/agent/atoms/perception.py`
- Create: `src/pickparts_agent/agent/atoms/manipulation.py`
- Create: `src/pickparts_agent/agent/atoms/placement.py`
- Modify: `src/pickparts_agent/agent/atoms/__init__.py`
- Delete: `src/pickparts_agent/agent/atoms/library.py`
- Modify: all atom imports in source and tests

**Interfaces:**
- Produces the same ten concrete atom classes and:

```python
def build_default_registry() -> AtomRegistry
```

- [x] **Step 1: Extract perception atoms**

Move `FindObject` and `VerifyState` to `agent/atoms/perception.py`. Use:

```python
from ...scene.perception import table_height
from .base import Atom, AtomResult, Check
```

Import `ResetArm` locally inside `VerifyState.run()` only when the camera must
be cleared.

- [x] **Step 2: Extract manipulation atoms**

Move `_guard`, `REST_Q`, `_arm_qpos`, `_cartesian`, `SetGripper`,
`ReachAbove`, `Grasp`, `Lift`, and `CarryTo` into
`agent/atoms/manipulation.py`.

```python
from ...scene.kinematics import ARM_NAMES
from .base import Atom, AtomResult, Check
from .perception import FindObject
```

- [x] **Step 3: Extract placement atoms**

Move `ReleaseInto`, `ResetArm`, and `PlaceOn` into
`agent/atoms/placement.py`.

```python
from .base import Atom, AtomResult, Check
from .manipulation import REST_Q, _arm_qpos, _cartesian, _guard
```

- [x] **Step 4: Build one public atom surface**

`agent/atoms/__init__.py` imports all ten atoms and constructs the existing
registry in the existing order.

- [x] **Step 5: Update source and test imports**

Tests may import concrete implementations from their owning module or use the
public `pickparts_agent.agent.atoms` surface. No test may import
`pickparts_agent.atoms`.

- [x] **Step 6: Run all Agent tests**

```bash
.venv/bin/python -m pytest \
  tests/test_atom_base.py tests/test_registry.py \
  tests/test_atom_registry_default.py tests/test_atoms_library.py \
  tests/test_placement_verification.py tests/test_state.py \
  tests/test_llm_json.py tests/test_planner.py tests/test_reactor.py \
  tests/test_executor.py tests/test_orchestrator.py \
  tests/test_flexible_simulation.py -q
```

Expected: all pass.

---

### Task 5: Move cloud and baseline services

**Files:**
- Create: `src/pickparts_agent/services/{__init__,cloud}.py`
- Create: `src/pickparts_agent/baseline/{__init__,motion}.py`
- Delete: root `cloud.py` and `motion.py`
- Modify: service/baseline imports and tests

**Interfaces:**
- Produces:

```python
from pickparts_agent.services.cloud import Endpoint, VisualLocator, transcribe
from pickparts_agent.baseline.motion import PickPlace
```

- [x] **Step 1: Move cloud integration**

Update the geometry import:

```python
from ..scene.perception import Frame, backproject
```

- [x] **Step 2: Move fixed baseline**

Update motion imports:

```python
from ..scene.kinematics import ARM_NAMES, ArmKinematics
```

- [x] **Step 3: Update tests**

Update `test_cloud.py`, `test_deepseek.py`, `test_pick_place.py`, and
`test_motion_recovery.py` to canonical paths.

- [x] **Step 4: Run focused tests**

```bash
.venv/bin/python -m pytest \
  tests/test_cloud.py tests/test_deepseek.py \
  tests/test_pick_place.py tests/test_motion_recovery.py -q
```

Expected: all pass.

---

### Task 6: Move interface implementations and preserve entry points

**Files:**
- Create: `src/pickparts_agent/__init__.py`
- Create: `src/pickparts_agent/interfaces/{__init__,cli,web}.py`
- Move: `static/` to `interfaces/static/`
- Replace: `src/pickparts_agent/app.py`
- Replace: `src/pickparts_agent/web.py`
- Modify: `pyproject.toml`, `run.sh`, `web.sh`, `setup.sh`, interface tests

**Interfaces:**
- Preserves:

```bash
python -m pickparts_agent.app
python -m pickparts_agent.web
pickparts
pickparts-web
```

- [x] **Step 1: Move CLI and Web implementations**

Update imports to:

```python
from ..agent import build_orchestrator
from ..baseline.motion import PickPlace
from ..scene import ColorPerception, Simulation
from ..services.cloud import Endpoint, VisualLocator, transcribe
```

In `interfaces/web.py`, update the repository root calculation for the extra
package level:

```python
ROOT = Path(__file__).resolve().parents[3]
```

- [x] **Step 2: Add the package marker and thin executable wrappers**

```python
# pickparts_agent/__init__.py
"""PickParts XLeRobot embodied Agent."""
```

```python
# pickparts_agent/app.py
from .interfaces.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# pickparts_agent/web.py
from .interfaces.web import main

if __name__ == "__main__":
    main()
```

- [x] **Step 3: Update package metadata**

```toml
[project.scripts]
pickparts = "pickparts_agent.interfaces.cli:main"
pickparts-web = "pickparts_agent.interfaces.web:main"

[tool.setuptools.package-data]
"pickparts_agent.scene" = ["robots/assets/**/*"]
"pickparts_agent.interfaces" = ["static/*"]
```

- [x] **Step 4: Update interface tests**

Tests that inspect `_parser`, `create_app`, `Console`, or `RobotBackend` import
from `pickparts_agent.interfaces.cli` or `.web`. Subprocess tests continue to
exercise root `-m` wrappers.

- [x] **Step 5: Verify entry points and interfaces**

```bash
.venv/bin/python -m pytest tests/test_cloud.py tests/test_web.py \
  tests/test_runtime_install.py -q
.venv/bin/python -m pickparts_agent.app --help
.venv/bin/python -m pickparts_agent.web --help
```

Expected: tests pass and both commands exit 0.

---

### Task 7: Enforce dependency direction and remove old modules

**Files:**
- Modify: `tests/test_package_layout.py`
- Delete: all superseded root domain files and old directories

**Interfaces:**
- Consumes the target structure from Tasks 2-6.
- Produces a clean root package with no duplicate implementation.

- [x] **Step 1: Add source dependency checks**

```python
def test_scene_does_not_import_higher_layers():
    forbidden = ("pickparts_agent.agent", "pickparts_agent.interfaces",
                 "pickparts_agent.services")
    for path in Path("src/pickparts_agent/scene").rglob("*.py"):
        text = path.read_text()
        assert not any(name in text for name in forbidden)
```

- [x] **Step 2: Search for old imports**

```bash
rg -n 'pickparts_agent\\.(planner|executor|reactor|orchestrator|state|llm|atoms|simulation|scene_objects|perception|kinematics|cloud|motion)' \
  src tests --glob '*.py'
```

Expected: no matches.

- [x] **Step 3: Run package-layout tests**

```bash
.venv/bin/python -m pytest tests/test_package_layout.py -q
```

Expected: all pass.

---

### Task 8: Update repository documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/技术文档.md`
- Modify: current non-historical documentation that references source paths

**Interfaces:**
- Documents only canonical paths.

- [x] **Step 1: Replace the repository tree and file map**

Document the five domain packages, thin entry wrappers, and test ownership.

- [x] **Step 2: Update all technical-document links**

Examples:

```text
../src/pickparts_agent/planner.py
-> ../src/pickparts_agent/agent/planner.py

../src/pickparts_agent/simulation.py
-> ../src/pickparts_agent/scene/simulation.py
```

- [x] **Step 3: Check links and stale paths**

```bash
rg -n 'src/pickparts_agent/(planner|executor|reactor|orchestrator|state|llm|atoms|simulation|scene_objects|perception|kinematics|cloud|motion)\\.py' \
  README.md docs/技术文档.md
```

Expected: no matches.

Historical files under `docs/superpowers/` are excluded.

---

### Task 9: Final verification

**Files:**
- Verify all moved and modified files.

- [x] **Step 1: Run formatting and whitespace checks**

```bash
git diff --check
```

Expected: exit 0.

- [x] **Step 2: Run the complete test suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: the existing 351 tests plus the new package-layout tests pass.

- [x] **Step 3: Run smoke imports**

```bash
.venv/bin/python - <<'PY'
from pickparts_agent.agent import build_orchestrator
from pickparts_agent.agent.atoms import build_default_registry
from pickparts_agent.scene import Simulation
from pickparts_agent.services.cloud import VisualLocator
print(len(build_default_registry().atoms))
PY
```

Expected: prints `10`.

- [x] **Step 4: Run CLI/Web help**

```bash
.venv/bin/python -m pickparts_agent.app --help
.venv/bin/python -m pickparts_agent.web --help
```

Expected: both exit 0.

- [x] **Step 5: Inspect final tree**

```bash
find src/pickparts_agent -maxdepth 3 -type f \
  ! -path '*/__pycache__/*' | sort
```

Expected: only runtime and thin wrappers remain at root; all implementation is
inside the five domain packages.
