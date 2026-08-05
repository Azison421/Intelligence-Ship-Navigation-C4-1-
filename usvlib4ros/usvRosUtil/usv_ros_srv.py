"""Blocking ROS service-call adapter used by the current bridge."""

import roslibpy

from usvlib4ros.usvRosUtil.usv_ros_util import USVRosbridgeClient


class RosSrvCallProxy:
    def __init__(self, serviceName, srvType):
        self.service = roslibpy.Service(
            USVRosbridgeClient.ros,
            serviceName,
            srvType,
        )

    def callService(self, request, timeout=3):
        return self.service.call(request, timeout=timeout)


__all__ = ["RosSrvCallProxy"]
