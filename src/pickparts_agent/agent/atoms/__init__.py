"""Semantic atom package."""
from .base import Atom, AtomContext, AtomResult, Check
from .manipulation import CarryTo, Grasp, Lift, ReachAbove, SetGripper, REST_Q
from .perception import FindObject, VerifyState
from .placement import PlaceOn, ReleaseInto, ResetArm
from .registry import AtomRegistry


def build_default_registry():
    return AtomRegistry([
        FindObject(), ReachAbove(), Grasp(), Lift(), CarryTo(),
        ReleaseInto(), VerifyState(), ResetArm(), SetGripper(), PlaceOn(),
    ])


__all__ = [
    "Atom", "AtomContext", "AtomResult", "Check", "AtomRegistry",
    "CarryTo", "FindObject", "Grasp", "Lift", "ReachAbove",
    "ReleaseInto", "ResetArm", "SetGripper", "VerifyState", "PlaceOn", "REST_Q",
    "build_default_registry",
]
