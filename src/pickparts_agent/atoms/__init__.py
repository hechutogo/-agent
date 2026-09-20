"""Semantic atom package."""
from .base import Atom, AtomContext, AtomResult, Check
from .library import (
    CarryTo, FindObject, Grasp, Lift, ReachAbove, ReleaseInto, ResetArm,
    SetGripper, VerifyState, REST_Q,
)
from .registry import AtomRegistry


def build_default_registry():
    return AtomRegistry([
        FindObject(), ReachAbove(), Grasp(), Lift(), CarryTo(),
        ReleaseInto(), VerifyState(), ResetArm(), SetGripper(),
    ])


__all__ = [
    "Atom", "AtomContext", "AtomResult", "Check", "AtomRegistry",
    "CarryTo", "FindObject", "Grasp", "Lift", "ReachAbove",
    "ReleaseInto", "ResetArm", "SetGripper", "VerifyState", "REST_Q",
    "build_default_registry",
]
