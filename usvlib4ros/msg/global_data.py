"""Shared ROS input snapshots and the sole NavigationStatus output payload."""

import threading


class DictToObject:
    def __init__(self, **values):
        for key, value in values.items():
            if isinstance(value, dict):
                value = DictToObject(**value)
            elif isinstance(value, list):
                value = [
                    DictToObject(**item) if isinstance(item, dict) else item
                    for item in value
                ]
            setattr(self, key, value)

    def to_dict(self):
        result = {}
        for key, value in vars(self).items():
            if isinstance(value, DictToObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, DictToObject) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


class Constants:
    class Request_Action:
        Set_Work_Model = 1
        Reset = 31
        Parameter_Init = 20


class GlobalData:
    Instance = None

    @classmethod
    def getInstance(cls):
        if cls.Instance is None:
            cls.Instance = cls()
        return cls.Instance

    def __init__(self):
        self.device_data = DictToObject(
            time=0.0,
            work_model=0,
            task_status=0,
            throttle_percent=0,
            rudder_percent=0,
            reset_status=0,
            reset_request_time=0.0,
        )
        self.scada_data = DictToObject(
            time=0.0,
            pose={
                "lng": 0.0,
                "lat": 0.0,
                "yaw": 0.0,
                "speed": 0.0,
                "rotate_speed": 0.0,
            },
        )
        self.laser_data = DictToObject()
        self.navigation_output_data = DictToObject(
            time=0.0,
            advise_throttle=0,
            advise_rudder=0,
            advise_heading=0.0,
            point_index=0,
            distance=0.0,
            e=0,
            step=0,
            score=0,
            loss=0.0,
            max_e=0,
            status=0,
        )
        self._navigation_lock = threading.Lock()

    def updateThrottleRudderOutput(
        self,
        adviseSpeed,
        adviseRotate,
        advisedHeading,
        nextPointIndex,
        shipToNextWPDistance,
    ):
        with self._navigation_lock:
            output = self.navigation_output_data
            output.advise_throttle = int(adviseSpeed)
            output.advise_rudder = int(adviseRotate)
            output.advise_heading = float(advisedHeading)
            output.point_index = int(nextPointIndex)
            output.distance = float(shipToNextWPDistance)
            output.time = float(getattr(self.device_data, "time", 0.0) or 0.0)

    def updateAlgorithmOutput(self, e, step, score, loss, max_e, status):
        with self._navigation_lock:
            output = self.navigation_output_data
            output.e = int(e)
            output.step = int(step)
            output.score = int(score)
            output.loss = float(loss)
            output.max_e = int(max_e)
            output.status = int(status)

    def navigation_output_snapshot(self):
        with self._navigation_lock:
            return self.navigation_output_data.to_dict()


__all__ = ["Constants", "DictToObject", "GlobalData"]
