"""ROS topic adapters for three read-only inputs and NavigationStatus output."""

import threading
import time

import roslibpy

from usvlib4ros.msg.global_data import DictToObject
from usvlib4ros.usvRosUtil.usv_ros_util import USVRosbridgeClient


class RosSubscriberProxy:
    def __init__(self, topicName, msgType, callback=None, defaultMsg=None):
        self.callback = callback
        self.msgData = defaultMsg
        self.cond = threading.Condition()
        self.topic = roslibpy.Topic(USVRosbridgeClient.ros, topicName, msgType)
        self.topic.subscribe(callback=self.__defaultSubscriberCallback)

    def __defaultSubscriberCallback(self, message):
        with self.cond:
            self.msgData = DictToObject(**message)
            if self.callback is not None:
                self.callback(message)
            self.cond.notify()

    def getMsgData(self):
        return self.msgData

    def wait_for_message(self, timeout=None):
        with self.cond:
            self.cond.wait(timeout=timeout)
        return self.msgData


class RosPublisherProxy:
    def __init__(self, topicName, msgType):
        self.topic = roslibpy.Topic(USVRosbridgeClient.ros, topicName, msgType)
        self.th = None

    def publish(self, msgData):
        self.topic.publish(msgData)

    def startPublicTopicThread(self, frequency, messageProvider):
        if frequency <= 0 or not callable(messageProvider):
            raise ValueError("publisher requires a positive frequency and provider")
        self.th = threading.Thread(
            target=self.__run,
            args=(1.0 / frequency, messageProvider),
            daemon=True,
            name="navigation-status-publisher",
        )
        self.th.start()

    def __run(self, period, messageProvider):
        while USVRosbridgeClient.ros.is_connected:
            self.publish(messageProvider())
            time.sleep(period)


__all__ = ["RosPublisherProxy", "RosSubscriberProxy"]
