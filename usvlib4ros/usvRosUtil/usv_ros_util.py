"""ROS bridge connection and minimal console logging."""

import inspect
import logging
import traceback

import roslibpy


class LogUtil:
    @classmethod
    def info(cls, message):
        frame = inspect.stack()[1]
        print(f"{frame.filename}({frame.lineno}):{message}")

    @classmethod
    def error(cls, message):
        traceback.print_exc()
        print(message)

    @classmethod
    def debug(cls, message):
        frame = inspect.stack()[1]
        print(f"{frame.filename}({frame.lineno}):{message}")


class USVRosbridgeClient:
    ros = None

    @classmethod
    def initUSVRosBridgeConnection(cls, host, port):
        cls.ros = roslibpy.Ros(host=host, port=port)
        cls.ros.run()

    @classmethod
    def initRoslibpyLogger(cls):
        logger = logging.getLogger("twisted")
        logger.setLevel(logging.INFO)
        if not any(
            getattr(handler, "_national_test_handler", False)
            for handler in logger.handlers
        ):
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler._national_test_handler = True
            logger.addHandler(handler)


__all__ = ["LogUtil", "USVRosbridgeClient"]
