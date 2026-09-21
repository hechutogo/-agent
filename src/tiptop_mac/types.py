"""TiPToP-style object-centric data structures."""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Grasp:
    position: np.ndarray
    top_z: float
    width: float
    confidence: float


@dataclass
class ObjectNode:
    id: str
    kind: str
    label: str
    color: tuple[float, float, float]
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    point: np.ndarray
    top_z: float
    bottom_z: float
    extent: np.ndarray
    cloud: np.ndarray
    grasps: list[Grasp] = field(default_factory=list)


@dataclass(frozen=True)
class TableSurface:
    normal: np.ndarray
    top_z: float
    bounds: np.ndarray


@dataclass(frozen=True)
class SceneGraph:
    table: TableSurface
    objects: dict[str, ObjectNode]
    qpos: np.ndarray


@dataclass(frozen=True)
class Predicate:
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class MotionStep:
    kind: str
    position: np.ndarray | None
    jaw: float | None
    label: str = ""


@dataclass(frozen=True)
class TAMPPlan:
    goal: tuple[Predicate, ...]
    operators: tuple[tuple[str, tuple], ...]
    trajectory: tuple[MotionStep, ...]
    planning_time: float
    rationale: str
