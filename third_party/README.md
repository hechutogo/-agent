# Upstream provenance

The repository was derived from https://github.com/hechutogo/-agent.git
at base commit `5d0f880`. The original Panda proof-of-concept entry points
were removed after the XLeRobot package became the supported implementation.

XLeRobot assets are copied from:

- Repository: https://github.com/Vector-Wangel/XLeRobot
- Revision: `a7ee564294f03484783ed053ab1550bccc3c6c09`
- Source: `simulation/Maniskill/assets/xlerobot/`
- Destination: `src/pickparts_agent/scene/robots/assets/xlerobot/`
- Upstream license: Apache-2.0, reproduced as `XLeRobot-LICENSE`.

Local modification, 2026-09-20: in `xlerobot.urdf`, joints
`root_x_axis_joint`, `root_y_axis_joint`, and `root_z_rotation_joint` are
fixed to lock the cart for tabletop operation. Mesh dimensions, visual
geometry, masses and inertias have not been manually changed. SAPIEN may
generate additional `.convex.stl` cache files during loading; these are not
source assets and are ignored by Git.

The XLeRobot project describes hardware v0.3; the exact hardware-version
equivalence of this simulation URDF has not been independently certified.
This implementation promises this pinned simulation model, not an
independent dimensional certification of physical v0.3 hardware.

The controller configuration in `robots/xlerobot.py` is project code:
right/left joint names are explicit, gripper link names match the URDF,
and the gripping drive is retuned for the small demo parts.

ManiSkill 3, SAPIEN, Qwen-Agent, SciPy and other Python dependencies retain
their respective upstream licenses in their installed distributions.
MoltenVK and libcxx runtime packages are downloaded from conda-forge into
the ignored `.runtime/` folder; package license metadata is extracted there.
