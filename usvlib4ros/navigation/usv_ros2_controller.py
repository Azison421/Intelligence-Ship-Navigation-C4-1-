"""Minimal ROS bridge used by the immutable competition entry and calibration."""

from usvlib4ros.msg.global_data import Constants, DictToObject, GlobalData
from usvlib4ros.usvRosUtil import (
    LogUtil,
    RosPublisherProxy,
    RosSrvCallProxy,
    RosSubscriberProxy,
    USVRosbridgeClient,
)


class Ros2Controller:
    def __init__(self, host, port, deviceId, globalData: GlobalData):
        USVRosbridgeClient.initRoslibpyLogger()
        USVRosbridgeClient.initUSVRosBridgeConnection(host=host, port=port)
        self.deviceId = deviceId
        self.globalData = globalData

        self.deviceManageSrvCall = RosSrvCallProxy(
            serviceName="usv/service/device/manage",
            srvType="message_pkg/DeviceManage",
        )
        self.create_ship()
        self.deviceStatusSubscriber = RosSubscriberProxy(
            topicName=f"usv/device/status/{deviceId}",
            msgType="message_pkg/DeviceStatus",
            callback=self.__listenerDeviceStatusCallback,
        )
        self.scadaStatusSubscriber = RosSubscriberProxy(
            topicName=f"usv/scada/status/{deviceId}",
            msgType="message_pkg/ScadaStatus",
            callback=self.__listenerScadaStatusCallback,
        )
        self.laserSubscriber = RosSubscriberProxy(
            topicName=f"usv/{deviceId}/laser/scan",
            msgType="sensor_msgs/LaserScan",
            callback=self.__listenerLaserCallback,
        )
        self.navigationStatusPublisher = RosPublisherProxy(
            topicName=f"usv/navigation/status/{deviceId}",
            msgType="message_pkg/NavigationStatus",
        )
        self.navigationStatusPublisher.startPublicTopicThread(
            frequency=10,
            messageProvider=globalData.navigation_output_snapshot,
        )
        self.deviceControllerSrvCall = RosSrvCallProxy(
            serviceName=f"usv/server/{deviceId}",
            srvType="message_pkg/DeviceController",
        )

    def __listenerDeviceStatusCallback(self, message):
        self.globalData.device_data = DictToObject(**message)

    def __listenerScadaStatusCallback(self, message):
        self.globalData.scada_data = DictToObject(**message)

    def __listenerLaserCallback(self, message):
        self.globalData.laser_data = DictToObject(**message)

    def initParameterList(self):
        response = self.deviceControllerSrvCall.callService(
            {
                "client_id": "navigation",
                "action": Constants.Request_Action.Parameter_Init,
                "data": "some",
            }
        )
        return response["code"] == 1

    def create_ship(self):
        response = self.deviceManageSrvCall.callService(
            {"device_id": self.deviceId, "action": 1}
        )
        LogUtil.info(response)
        return response["result"] == 1

    def reset_unity(self):
        response = self.deviceControllerSrvCall.callService(
            {
                "client_id": "navigation",
                "action": Constants.Request_Action.Reset,
                "data": "1",
            }
        )
        return response["code"] == 1

    def set_task(self):
        """Request task start; DeviceStatus is the acknowledgement."""

        return self.deviceControllerSrvCall.callService(
            {
                "client_id": "navigation",
                "action": Constants.Request_Action.Set_Task,
                "data": "1",
            }
        )

    def set_auto_work(self):
        response = self.deviceControllerSrvCall.callService(
            {
                "client_id": "navigation",
                "action": Constants.Request_Action.Set_Work_Model,
                "data": "2",
            }
        )
        return response["code"] == 1


__all__ = ["Ros2Controller"]
