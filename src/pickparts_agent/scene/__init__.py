"""Simulation, robot, kinematics, and sensor-domain APIs."""

_EXPORTS = {
    "ARM_NAMES": (".kinematics", "ARM_NAMES"),
    "ArmKinematics": (".kinematics", "ArmKinematics"),
    "ColorPerception": (".perception", "ColorPerception"),
    "Frame": (".perception", "Frame"),
    "ScenePerception": (".perception", "ScenePerception"),
    "Simulation": (".simulation", "Simulation"),
    "backproject": (".perception", "backproject"),
    "table_height": (".perception", "table_height"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
