# Repository Domain Layout Design

## 1. Goal

Reorganize `pickparts_agent` into explicit domain packages so an engineer can
find Agent, scene, cloud, interface, and baseline code without scanning a flat
module list. Preserve runtime behavior and public command entry points while
hard-migrating internal Python imports to the new package paths.

## 2. Constraints

- Preserve the sensor-only boundary: Agent code must not access actors, poses,
  environment rewards, segmentation IDs, or simulator success state.
- Preserve the flat AtomCall plan contract and all ten current atoms.
- Preserve `python -m pickparts_agent.app`, `python -m pickparts_agent.web`,
  `pickparts`, `pickparts-web`, `run.sh`, and `web.sh`.
- Do not add dependencies or change behavior.
- Do not keep root-level compatibility modules for old internal Agent imports.
- Preserve all current uncommitted implementation work.
- Update current documentation and tests to use only new canonical paths.

## 3. Options Considered

### Option A: Move only Agent files

Create `agent/` but leave simulation, perception, cloud, Web, and baseline files
at package root.

This has the smallest migration surface but leaves the root package mixing
domain logic, integrations, and interfaces. It does not fully satisfy the
repository organization goal.

### Option B: Domain packages

Create `agent/`, `scene/`, `services/`, `interfaces/`, and `baseline/`.
Keep only runtime configuration and thin executable wrappers at package root.

This requires more import updates but gives each file one clear owner and
enforces a comprehensible dependency direction. This is the selected design.

### Option C: Rename files without packages

Add prefixes such as `agent_planner.py` and `scene_simulation.py`.

This avoids package moves but makes imports noisy and encodes structure in file
names instead of actual module boundaries. It is rejected.

## 4. Target Structure

```text
src/pickparts_agent/
  __init__.py                  # package marker and public metadata
  app.py                       # thin CLI compatibility entry
  web.py                       # thin Web compatibility entry
  runtime.py                   # process-wide macOS Vulkan setup
  agent/
    __init__.py                # public Agent API
    orchestrator.py
    planner.py
    executor.py
    reactor.py
    state.py
    llm.py
    atoms/
      __init__.py              # ten-atom registry and public exports
      base.py
      registry.py
      perception.py            # FindObject, VerifyState
      manipulation.py          # SetGripper, ReachAbove, Grasp, Lift, CarryTo
      placement.py             # ReleaseInto, PlaceOn, ResetArm
  scene/
    __init__.py                # public scene API
    simulation.py
    objects.py
    perception.py
    kinematics.py
    robots/
      __init__.py
      xlerobot.py
      assets/xlerobot/...
  services/
    __init__.py
    cloud.py
  interfaces/
    __init__.py
    cli.py
    web.py
    static/
      index.html
      app.js
      style.css
  baseline/
    __init__.py
    motion.py
```

## 5. Responsibilities

### 5.1 `agent/`

Owns semantic task handling:

- `state.py`: non-privileged blackboard and observation records.
- `planner.py`: feasibility and flat AtomCall planning.
- `executor.py`: atom execution, budgets, events, recovery application, final
  goal verification.
- `reactor.py`: retry, replace, replan, ask_user, and abort decisions.
- `orchestrator.py`: one user turn and component assembly.
- `llm.py`: strict JSON model transport and supported Think toggle.
- `atoms/`: atom contracts, validation, registry, and implementations.

`agent/` may import public APIs from `scene/` but must not import
`interfaces/`.

### 5.2 `scene/`

Owns physical and sensor concerns:

- `simulation.py`: ManiSkill environment and sensor/actuator boundary.
- `objects.py`: private actor construction and random reachable layouts.
- `perception.py`: RGB-D geometry and catalog-object perception.
- `kinematics.py`: URDF-only FK/IK.
- `robots/`: XLeRobot registration and assets.

`scene/` must not import `agent/`, `services/`, or `interfaces/`.

### 5.3 `services/`

Owns provider integrations:

- `cloud.py`: endpoint configuration, open-description VLM localization,
  `VisualLocator`, and ASR.

It may import sensor DTOs and geometry helpers from `scene/`, but no UI code.

### 5.4 `interfaces/`

Owns user-facing entry implementations:

- `cli.py`: command parsing, smoke/demo/Agent modes, artifact saving.
- `web.py`: FastAPI, worker queue, simulator backend, and HTTP API.
- `static/`: Web frontend assets.

Root `app.py` and `web.py` delegate to these modules and contain no domain
logic.

### 5.5 `baseline/`

Owns the deterministic non-Agent baseline:

- `motion.py`: fixed OpenCV RGB-D pick/place controller.

This keeps demonstration code separate from the flexible Agent.

## 6. Atom Split

The current `atoms/library.py` is split by behavior:

- `perception.py`: `FindObject`, `VerifyState`.
- `manipulation.py`: motion guard, rest pose, qpos/TCP helpers,
  `SetGripper`, `ReachAbove`, `Grasp`, `Lift`, `CarryTo`.
- `placement.py`: `ReleaseInto`, `PlaceOn`, `ResetArm`.

`placement.py` consumes the private motion helpers from `manipulation.py`.
`perception.py` imports `ResetArm` lazily inside `VerifyState.run()` to avoid a
module cycle. `agent.atoms.__init__` remains the only public import surface for
concrete atoms.

## 7. Dependency Direction

```text
root entry wrappers
  -> interfaces
       -> agent
       -> services
       -> scene
       -> baseline

agent
  -> scene

services
  -> scene

baseline
  -> scene

scene
  -> runtime
```

Forbidden directions:

- `scene -> agent`
- `scene -> interfaces`
- `agent -> interfaces`
- `services -> interfaces`

## 8. Import and Entry-Point Policy

Canonical examples:

```python
from pickparts_agent.agent import build_orchestrator
from pickparts_agent.agent.atoms import FindObject
from pickparts_agent.scene import Simulation, ScenePerception
from pickparts_agent.services.cloud import Endpoint, VisualLocator
```

The project does not preserve old internal imports such as
`pickparts_agent.planner` or `pickparts_agent.atoms`. Tests and documentation
must use canonical imports.

Compatibility is intentionally limited to executable entry points:

```text
pickparts_agent.app -> pickparts_agent.interfaces.cli
pickparts_agent.web -> pickparts_agent.interfaces.web
```

This keeps existing shell scripts and installed commands working without
leaving duplicate domain modules at package root.

## 9. Migration and Failure Control

Move one domain at a time:

1. Create `scene/`, update scene-local imports and scene tests.
2. Create `agent/`, split atoms, update Agent tests.
3. Create `services/` and update cloud tests.
4. Create `baseline/` and update baseline tests.
5. Create `interfaces/`, add the root package marker, retain thin root wrappers,
   and update package data.
6. Update all remaining imports and documentation.

Each phase must run its focused tests before the next move. The final checks
must reject:

- imports from deleted root domain modules;
- references to old paths in current README/technical documentation;
- missing static/URDF package data;
- import cycles;
- changed CLI/Web entry behavior.

## 10. Testing

Focused suites:

- Scene: `test_kinematics.py`, `test_scene_objects.py`,
  `test_scene_perception.py`, `test_simulation.py`.
- Agent: atom, state, planner, reactor, executor, orchestrator, and flexible
  simulation tests.
- Services: `test_cloud.py`, `test_deepseek.py`.
- Baseline: `test_pick_place.py`, `test_motion_recovery.py`.
- Interfaces: `test_web.py`, runtime and CLI subprocess tests.

Final acceptance:

```bash
.venv/bin/python -m pytest -q
```

The existing 351-test behavior baseline must remain green. `run.sh --smoke`,
`python -m pickparts_agent.app --help`, and
`python -m pickparts_agent.web --help` must still work.

## 11. Documentation

Update:

- `README.md`: target tree and primary entry paths.
- `docs/技术文档.md`: all source links and architecture descriptions.
- Current operational documentation that names canonical source paths.

Historical design/plan documents under `docs/superpowers/` remain historical
records and are not rewritten to pretend the old path layout never existed.
