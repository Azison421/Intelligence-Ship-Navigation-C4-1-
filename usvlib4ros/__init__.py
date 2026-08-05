"""Package exports with ROS-facing modules loaded only when requested."""

from usvlib4ros.msg import DictToObject, GlobalData


__all__ = [
    "GlobalData",
    "DictToObject",
    "USVMathUtil",
    "LogUtil",
    "NMEAUtil",
    "USVRosbridgeClient",
]


def __getattr__(name: str):
    if name in {"USVMathUtil", "LogUtil", "NMEAUtil", "USVRosbridgeClient"}:
        from usvlib4ros import usvRosUtil

        value = getattr(usvRosUtil, name)
    else:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    globals()[name] = value
    return value
