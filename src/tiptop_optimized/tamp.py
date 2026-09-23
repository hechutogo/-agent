"""Single-subgoal TAMP with measured supports and explicit action history.

Temporal ordering belongs to the caller: perceive again between subgoals. A
plan performs at most one transfer so inherited grasp geometry never goes stale.
No simulator poses or object-catalog dimensions are used here.
"""
from collections import deque
from dataclasses import dataclass
import json
import time

import numpy as np

from pickparts_agent.agent.atoms.manipulation import _cartesian_waypoints
from pickparts_agent.scene.support import in_box, on_block, on_table, stable_block_overlap
from tiptop_mac.tamp import TAMPLite as _BaseTAMPLite
from tiptop_mac.tamp import TAMPError, _Infeasible
from tiptop_mac.types import MotionStep, TAMPPlan


class CartesianPathError(ValueError):
    def __init__(self, step_index):
        super().__init__(f"Cartesian path is infeasible at step {step_index}")
        self.step_index = step_index


def validate_cartesian_path(kin, trajectory, arm_seed, grasp_offset,
                            *, index_offset=0):
    """Validate the exact waypoint sequence used by the runtime executor."""
    shift = np.asarray(grasp_offset, dtype=float)
    seed = np.asarray(arm_seed, dtype=float).copy()
    for local_index, step in enumerate(trajectory):
        if step.kind != "move":
            continue
        target = np.asarray(step.position, dtype=float) + shift
        try:
            for waypoint in _cartesian_waypoints(kin, target, seed):
                seed = kin.solve(np.asarray(waypoint, dtype=float), seed=seed)
        except Exception as exc:
            raise CartesianPathError(index_offset + local_index) from exc
    return seed


@dataclass(frozen=True)
class _State:
    held: str | None
    support: tuple[tuple[str, str], ...]
    moved: frozenset[str] = frozenset()


class TAMPLite(_BaseTAMPLite):
    """Reuse baseline IK, fit tests, path reconstruction and grasp trajectories."""

    FAILED_TARGET_RADIUS = .05
    MAX_TABLE_CANDIDATES = 256

    def plan(self, scene, goal, *, rationale="", require_action=False, held=None,
             arm_seed=None, grasp_offset=None, excluded_table_xy=()):
        started = time.perf_counter()
        goal = tuple(goal)
        self._validate_goal(scene, goal)
        arm_seed = self._motion_vector(arm_seed, "arm_seed", size=5)
        grasp_offset = self._motion_vector(
            grasp_offset, "grasp_offset", size=3)
        excluded_table_xy = tuple(
            self._motion_vector(point, "excluded_table_xy", size=2)
            for point in excluded_table_xy)
        initial = self.observed_support(scene, held=held)
        target = goal[0].args[0]
        start = _State(held, tuple(sorted(initial.items())))
        if self._satisfied(start, goal, require_action=require_action):
            return TAMPPlan(goal, (), (), 0., rationale)
        if held is not None and held != target:
            raise TAMPError("No feasible single transfer while another object is held")
        queue = deque([start])
        parents = {start: (None, None)}
        expansions = 0
        while queue and expansions < self.MAX_STATES:
            state = queue.popleft()
            expansions += 1
            for operator, successor in self._successors(scene, state):
                if operator[1][0] != target or successor in parents:
                    continue
                parents[successor] = (state, operator)
                if self._satisfied(successor, goal, require_action=require_action):
                    path = self._path(parents, successor)
                    try:
                        trajectory = self._build_trajectory(
                            scene, path, goal, arm_seed=arm_seed,
                            grasp_offset=grasp_offset,
                            excluded_table_xy=excluded_table_xy)
                    except _Infeasible:
                        continue
                    return TAMPPlan(
                        goal, tuple(op for op, _ in path), trajectory,
                        time.perf_counter() - started, rationale)
                queue.append(successor)
        raise TAMPError("TAMP-lite found no feasible single-subgoal plan")

    @staticmethod
    def _motion_vector(value, name, *, size):
        if value is None:
            return None
        vector = np.asarray(value, dtype=float)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise TAMPError(f"{name} must be a finite {size}-vector")
        return vector.copy()

    def satisfies(self, scene, goal, *, held=None):
        """Verify the current observation, not a predicted state or past action.

        The caller must separately record completion of earlier temporal steps.
        Invalid geometry/goals raise TAMPError rather than asserting success.
        """
        goal = tuple(goal)
        self._validate_goal(scene, goal)
        support = self.observed_support(scene, held=held)
        return self._satisfied(_State(held, tuple(sorted(support.items()))), goal)

    def observed_support(self, scene, *, held=None):
        """Return measured block -> support IDs; omit held/unsupported blocks.

        Box support means measured containment below the rim, not proof of
        contact with an occluded interior floor. Small gaps allow RGB-D noise.
        """
        self._validate_geometry(scene)
        if held is not None and (
                held not in scene.objects or scene.objects[held].kind != "block"):
            raise TAMPError("Held object requires measured block geometry")
        return {b: s for b, s in self._initial_support(scene).items()
                if b != held and s != held}

    @staticmethod
    def _validate_goal(scene, goal):
        if len(goal) != 1:
            raise TAMPError("Expected a single predicate; perceive between subgoals")
        predicate = goal[0]
        arity = {"on": 2, "holding": 1}.get(predicate.name)
        if arity is None or len(predicate.args) != arity:
            raise TAMPError("Unsupported single-subgoal predicate")
        target = predicate.args[0]
        if target not in scene.objects or scene.objects[target].kind != "block":
            raise TAMPError("Goal requires a measured movable block")
        if predicate.name == "on":
            support = predicate.args[1]
            if support == target or (
                    support != "table" and (
                        support not in scene.objects
                        or scene.objects[support].kind not in ("block", "box"))):
                raise TAMPError("Goal requires a valid measured support")

    @staticmethod
    def _validate_geometry(scene):
        bounds = np.asarray(scene.table.bounds, dtype=float)
        if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
                or not np.isfinite(scene.table.top_z)
                or np.any(bounds[0, :2] >= bounds[1, :2])):
            raise TAMPError("Invalid measured table geometry")
        for node in scene.objects.values():
            point = np.asarray(node.point, dtype=float)
            extent = np.asarray(node.extent, dtype=float)
            if (point.shape != (3,) or extent.shape != (3,)
                    or not np.isfinite(point).all()
                    or not np.isfinite(extent).all() or np.any(extent <= 0)
                    or not np.isfinite([node.bottom_z, node.top_z]).all()
                    or node.top_z <= node.bottom_z):
                raise TAMPError("Invalid measured object geometry")

    def _successors(self, scene, state):
        support = dict(state.support)
        if state.held is None:
            # A completed transfer must be followed by fresh perception.
            if state.moved:
                return
            for b, node in scene.objects.items():
                if node.kind != "block" or b in support.values():
                    continue
                remaining = tuple(sorted((k, v) for k, v in support.items()
                                         if k != b))
                yield ("pick", (b,)), _State(b, remaining, state.moved | {b})
        else:
            b = state.held
            forbidden = self._descendants(b, support)
            candidates = ["table"] + [
                i for i, node in scene.objects.items()
                if i != b and i not in forbidden and node.kind in ("block", "box")
            ]
            for destination in candidates:
                placed = dict(support, **{b: destination})
                yield ("place", (b, destination)), _State(
                    None, tuple(sorted(placed.items())), state.moved | {b})

    @staticmethod
    def _satisfied(state, goal, *, require_action=False):
        # Matching final geometry alone cannot establish the requested milestone.
        if require_action and any(p.args[0] not in state.moved for p in goal):
            return False
        return _BaseTAMPLite._satisfied(state, goal)

    def _initial_support(self, scene):
        support = {}
        for b, node in scene.objects.items():
            if node.kind != "block":
                continue
            blocks, boxes = [], []
            for other_id, other in scene.objects.items():
                if other_id == b:
                    continue
                if other.kind == "block" and on_block(node, other):
                    blocks.append((other.top_z, other_id))
                elif other.kind == "box" and in_box(node, other):
                    boxes.append((other.top_z, other_id))
            if blocks:
                support[b] = max(blocks)[1]
            elif boxes:
                support[b] = max(boxes)[1]
            elif (on_table(node, scene.table.top_z)
                  and self._inside_table(scene, node.point[:2], node.extent[:2])):
                support[b] = "table"
        return support

    @staticmethod
    def _inside_table(scene, xy, extent, margin=0.):
        bounds = np.asarray(scene.table.bounds, dtype=float)
        half = np.asarray(extent, dtype=float) / 2 + margin
        return bool(np.all(np.asarray(xy) - half >= bounds[0, :2])
                    and np.all(np.asarray(xy) + half <= bounds[1, :2]))

    @staticmethod
    def _check_collisions(scene, release, held, support_id):
        node = scene.objects[held]
        for other_id, other in scene.objects.items():
            if other_id in (held, support_id):
                continue
            clearance = (np.asarray(node.extent[:2])
                         + np.asarray(other.extent[:2])) / 2 + .01
            if support_id == "table":
                # Descending next to a rim needs space for the jaw body too.
                clearance += .03
            if np.all(np.abs(release[:2] - other.point[:2]) < clearance):
                raise _Infeasible

    def _placement_steps(self, scene, b, destination, release):
        self._check_collisions(scene, release, b, destination)
        half_h = max((scene.objects[b].top_z - scene.objects[b].bottom_z) / 2,
                     .012)
        obstacle_top = max([scene.table.top_z] + [
            node.top_z for i, node in scene.objects.items() if i != b])
        above = np.array([release[0], release[1],
                          max(release[2] + .07, obstacle_top + half_h + .075)])
        retreat = release + np.array([0., 0., .07])
        return (
            MotionStep("move", above, 0.),
            MotionStep("move", release, 0.),
            MotionStep("gripper", None, .8),
            MotionStep("move", retreat, .8),
        )

    def _table_steps(self, scene, b, excluded_table_xy):
        node = scene.objects[b]
        bounds = np.asarray(scene.table.bounds, dtype=float)
        inset = np.asarray(node.extent[:2]) / 2 + .01
        low, high = bounds[0, :2] + inset, bounds[1, :2] - inset
        if np.any(low > high):
            raise _Infeasible
        # Search the observed table, including edges, rather than returning to
        # the mover's old XY (which may still be inside a box).
        axes = [np.linspace(lo, hi, min(61, max(2, int(np.ceil((hi - lo) / .02)) + 1)))
                for lo, hi in zip(low, high)]
        grid = np.array(np.meshgrid(*axes)).reshape(2, -1).T
        preferred = np.clip(np.asarray(node.point[:2]), low, high)
        center = (low + high) / 2
        fractions = np.array([.2, .5, .8])
        anchor_axes = [
            lo + fractions * (hi - lo) for lo, hi in zip(low, high)]
        anchors = np.array(np.meshgrid(*anchor_axes)).reshape(2, -1).T
        order = np.argsort(np.linalg.norm(grid - preferred, axis=1),
                           kind="stable")
        candidates = np.vstack([preferred, center, anchors, grid[order]])
        half_h = max((node.top_z - node.bottom_z) / 2, .012)
        seen = set()
        checked = 0
        for xy in candidates:
            key = tuple(np.round(xy, 6))
            if key in seen:
                continue
            seen.add(key)
            if any(np.linalg.norm(xy - failed) < self.FAILED_TARGET_RADIUS
                   for failed in excluded_table_xy):
                continue
            if checked >= self.MAX_TABLE_CANDIDATES:
                break
            checked += 1
            # RGB-D bottoms and grasp offsets carry millimetre-scale error.
            # Release just above the measured table, then let gravity settle.
            release = np.array([*xy, scene.table.top_z + half_h + .006])
            try:
                yield self._placement_steps(scene, b, "table", release)
            except _Infeasible:
                continue

    def _validate_trajectory(self, trajectory, *, arm_seed, grasp_offset):
        shift = (np.asarray(grasp_offset, dtype=float).copy()
                 if grasp_offset is not None else np.zeros(3))
        if arm_seed is None:
            try:
                for step in trajectory:
                    if step.kind == "move":
                        self.kin.solve(np.asarray(step.position) + shift)
            except Exception as exc:
                raise _Infeasible from exc
            return None
        try:
            return validate_cartesian_path(
                self.kin, trajectory, arm_seed, shift)
        except CartesianPathError as exc:
            raise _Infeasible from exc

    def _build_trajectory(self, scene, path, goal, *, arm_seed=None,
                          grasp_offset=None, excluded_table_xy=()):
        operators = [op for op, _ in path]
        kinds = [op[0] for op in operators]
        if (kinds not in (["pick"], ["place"], ["pick", "place"])
                or len({op[1][0] for op in operators}) != 1):
            raise _Infeasible
        steps = []
        for operator, state in path:
            if operator[0] == "pick":
                steps.extend(self._pick_steps(
                    scene, operator[1][0], arm_seed=arm_seed))
                continue
            b, destination = operator[1]
            if destination == "table":
                prefix = tuple(steps)
                prefix_seed = arm_seed
                if prefix and arm_seed is not None:
                    prefix_seed = self._validate_trajectory(
                        prefix, arm_seed=arm_seed,
                        grasp_offset=grasp_offset)
                for table_steps in self._table_steps(
                        scene, b, excluded_table_xy):
                    try:
                        self._validate_trajectory(
                            table_steps, arm_seed=prefix_seed,
                            grasp_offset=grasp_offset)
                    except _Infeasible:
                        continue
                    return prefix + table_steps
                raise _Infeasible
            node, receiver = scene.objects[b], scene.objects[destination]
            half_h = max((node.top_z - node.bottom_z) / 2, .012)
            if receiver.kind == "box":
                if not self._fits_on(node, receiver, margin=-.006):
                    raise _Infeasible
            elif not stable_block_overlap(
                    node, receiver, center=receiver.point[:2]):
                raise _Infeasible
            release = np.asarray(receiver.point, dtype=float).copy()
            release[2] = receiver.top_z + half_h + .004
            if receiver.kind == "box":
                release[2] = max(release[2], receiver.point[2] + .055)
            steps.extend(self._placement_steps(scene, b, destination, release))
        trajectory = tuple(steps)
        self._validate_trajectory(
            trajectory, arm_seed=arm_seed, grasp_offset=grasp_offset)
        return trajectory

    def _pick_steps(self, scene, target, *, arm_seed):
        node = scene.objects[target]
        if not node.grasps:
            raise _Infeasible
        grasp = np.asarray(node.point, dtype=float).copy()
        grasp[2] -= .004
        assumed = max((node.top_z - node.bottom_z) / 2, .012)
        obstacle_top = max([scene.table.top_z] + [
            other.top_z for other in scene.objects.values()])
        for clearance in (.10, .08, .06):
            hover = grasp.copy()
            # Preserve vertical lift evidence and clearance above observed
            # obstacles when the default 10 cm approach exceeds the workspace.
            hover[2] = max(grasp[2] + clearance, obstacle_top + .03)
            steps = (
                MotionStep("move", hover, .8),
                MotionStep("move", grasp, .8),
                MotionStep("gripper", None, 0.),
                MotionStep("move", hover, 0.),
                MotionStep("calibrate", None, None,
                           label=json.dumps({"held": target, "assumed": assumed})),
            )
            try:
                self._validate_trajectory(
                    steps, arm_seed=arm_seed, grasp_offset=None)
            except _Infeasible:
                continue
            return steps
        raise _Infeasible

    @staticmethod
    def table_release_xy(plan):
        if not any(
                kind == "place" and args[1] == "table"
                for kind, args in plan.operators):
            return None
        opening = next((
            index for index, step in enumerate(plan.trajectory)
            if step.kind == "gripper" and step.jaw == .8), None)
        if opening is None:
            return None
        release = plan.trajectory[opening - 1]
        if release.kind != "move" or release.position is None:
            return None
        return np.asarray(release.position[:2], dtype=float).copy()

    @staticmethod
    def failed_table_xy(plan, failed_index):
        release_xy = TAMPLite.table_release_xy(plan)
        if release_xy is None or failed_index is None:
            return None
        opening = next(
            index for index, step in enumerate(plan.trajectory)
            if step.kind == "gripper" and step.jaw == .8)
        if failed_index >= opening:
            return None
        calibrations = [
            index for index, step in enumerate(plan.trajectory[:opening])
            if step.kind == "calibrate"]
        placement_start = calibrations[-1] + 1 if calibrations else 0
        if failed_index < placement_start:
            return None
        return release_xy
