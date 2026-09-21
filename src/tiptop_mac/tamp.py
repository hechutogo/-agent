"""CPU TAMP-lite: symbolic BFS with IK and geometric feasibility checks."""
from collections import deque
from dataclasses import dataclass
import json
import time

import numpy as np

from .types import MotionStep, Predicate, TAMPPlan


class TAMPError(RuntimeError):
    pass


class _Infeasible(Exception):
    pass


@dataclass(frozen=True)
class _State:
    held: str | None
    support: tuple[tuple[str, str], ...]


class TAMPLite:
    MAX_STATES = 500

    def __init__(self, kin):
        self.kin = kin

    def plan(self, scene, goal, *, rationale=""):
        started = time.perf_counter()
        initial = self._initial_support(scene)
        start_state = _State(None, tuple(sorted(initial.items())))
        if self._satisfied(start_state, goal):
            return TAMPPlan(goal=tuple(goal), operators=(), trajectory=(),
                            planning_time=0.0, rationale=rationale)
        queue = deque([start_state])
        parents = {start_state: (None, None)}
        expansions = 0
        while queue and expansions < self.MAX_STATES:
            state = queue.popleft()
            expansions += 1
            for operator, new_state in self._successors(scene, state):
                if new_state in parents:
                    continue
                parents[new_state] = (state, operator)
                if self._satisfied(new_state, goal):
                    path = self._path(parents, new_state)
                    try:
                        trajectory = self._build_trajectory(scene, path, goal)
                    except _Infeasible:
                        continue
                    return TAMPPlan(
                        goal=tuple(goal),
                        operators=tuple(op for op, _ in path),
                        trajectory=trajectory,
                        planning_time=time.perf_counter() - started,
                        rationale=rationale)
                queue.append(new_state)
        raise TAMPError("TAMP-lite found no feasible plan")

    # -- symbolic search ---------------------------------------------------

    def _successors(self, scene, state):
        blocks = [i for i, n in scene.objects.items() if n.kind == "block"]
        outcomes = []
        if state.held is None:
            support = dict(state.support)
            for b in blocks:
                remaining = tuple(sorted(
                    (k, v) for k, v in support.items() if k != b))
                outcomes.append((("pick", (b,)), _State(b, remaining)))
        else:
            b = state.held
            forbidden = self._descendants(b, dict(state.support))
            supports = ["table"] + [
                i for i, n in scene.objects.items()
                if i != b and i not in forbidden
                and n.kind in ("block", "box")]
            for s in supports:
                new_support = dict(state.support)
                new_support[b] = s
                outcomes.append((("place", (b, s)),
                                 _State(None, tuple(sorted(new_support.items())))))
        return outcomes

    @staticmethod
    def _descendants(root, support):
        result, frontier = set(), [root]
        while frontier:
            current = frontier.pop()
            children = [k for k, v in support.items()
                        if v == current and k not in result]
            result.update(children)
            frontier.extend(children)
        return result

    @staticmethod
    def _satisfied(state, goal):
        support = dict(state.support)
        wants_holding = False
        for predicate in goal:
            if predicate.name == "holding":
                wants_holding = True
                if state.held != predicate.args[0]:
                    return False
            elif support.get(predicate.args[0]) != predicate.args[1]:
                return False
        if not wants_holding and state.held is not None:
            return False
        return True

    @staticmethod
    def _path(parents, state):
        path = []
        while parents[state][0] is not None:
            previous, operator = parents[state]
            path.append((operator, state))
            state = previous
        path.reverse()
        return path

    # -- initial relations from measured geometry --------------------------

    def _initial_support(self, scene):
        support = {}
        blocks = [(i, n) for i, n in scene.objects.items()
                  if n.kind == "block"]
        for object_id, node in blocks:
            choices = []
            for other_id, other in blocks:
                if other_id == object_id:
                    continue
                gap = node.bottom_z - other.top_z
                if not -.003 <= gap <= .015:
                    continue
                if self._footprint_inside(node, other, margin=.003):
                    choices.append(other)
            if choices:
                support[object_id] = max(choices, key=lambda n: n.top_z).id
            else:
                support[object_id] = "table"
        return support

    @staticmethod
    def _footprint_inside(mover, support, *, margin):
        mover_half = np.asarray(mover.extent[:2], dtype=float) / 2
        support_half = np.asarray(support.extent[:2], dtype=float) / 2
        delta = np.abs(np.asarray(mover.point[:2])
                       - np.asarray(support.point[:2]))
        return bool(np.all(delta + mover_half <= support_half + margin))

    @staticmethod
    def _fits_on(mover, support, *, margin):
        """Footprint fit at the planned release point (support center)."""
        mover_half = np.asarray(mover.extent[:2], dtype=float) / 2
        support_half = np.asarray(support.extent[:2], dtype=float) / 2
        return bool(np.all(mover_half <= support_half + margin))

    # -- geometric feasibility + full trajectory ---------------------------

    def _ik(self, position):
        try:
            self.kin.solve(np.asarray(position, dtype=float))
        except Exception:
            raise _Infeasible

    @staticmethod
    def _check_collisions(scene, release, held, support_id):
        node = scene.objects[held]
        for other_id, other in scene.objects.items():
            if other_id in (held, support_id):
                continue
            gap = float(np.linalg.norm(
                release[:2] - np.asarray(other.point[:2])))
            need = float(np.max(
                (np.asarray(node.extent[:2])
                 + np.asarray(other.extent[:2])) / 2)) + .01
            if gap < need:
                raise _Infeasible

    def _build_trajectory(self, scene, path, goal):
        steps = []
        support = self._initial_support(scene)
        for operator, _ in path:
            if operator[0] == "pick":
                b = operator[1][0]
                node = scene.objects[b]
                if not node.grasps:
                    raise _Infeasible
                grasp = np.asarray(node.point, dtype=float).copy()
                grasp[2] -= .004
                hover = grasp.copy()
                hover[2] += .10
                assumed = max((node.top_z - node.bottom_z) / 2, .012)
                self._ik(grasp)
                self._ik(hover)
                steps.extend([
                    MotionStep("move", hover, .8),
                    MotionStep("move", grasp, .8),
                    MotionStep("gripper", None, 0.),
                    MotionStep("move", hover, 0.),
                    MotionStep("calibrate", None, None,
                               label=json.dumps({"held": b,
                                                 "assumed": assumed})),
                ])
                support.pop(b, None)
            else:
                b, s = operator[1]
                node = scene.objects[b]
                half_h = max((node.top_z - node.bottom_z) / 2, .012)
                if s == "table":
                    current = np.asarray(node.point, dtype=float)
                    release = np.array([current[0], current[1],
                                        scene.table.top_z + half_h])
                    top = scene.table.top_z
                else:
                    rec = scene.objects[s]
                    target = np.asarray(rec.point, dtype=float)
                    if rec.kind == "box":
                        release = np.array([target[0], target[1],
                                           rec.point[2] + .055])
                        if not self._fits_on(node, rec, margin=-.006):
                            raise _Infeasible
                    else:
                        release = np.array([target[0], target[1],
                                           rec.top_z + half_h + .004])
                        if not self._fits_on(node, rec, margin=.003):
                            raise _Infeasible
                    top = rec.top_z
                self._check_collisions(scene, release, b, s)
                above = np.array([release[0], release[1],
                                  max(.835, top + .075)])
                self._ik(above)
                self._ik(release)
                steps.extend([
                    MotionStep("move", above, 0.),
                    MotionStep("move", release, 0.),
                    MotionStep("gripper", None, .8),
                    MotionStep("move",
                               np.array([release[0], release[1],
                                         release[2] + .07]), .8),
                ])
                support[b] = s
        return tuple(steps)
