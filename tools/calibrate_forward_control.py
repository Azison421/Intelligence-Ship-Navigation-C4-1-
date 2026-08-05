"""Bounded two-sided NavigationStatus calibration for National_Test."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from usvlib4ros.msg.global_data import GlobalData
from usvlib4ros.navigation.fixed_map_runtime import (
    LiveInputAdapter,
    build_fixed_route_context,
)
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.planning.forward_control_profile import (
    MINIMUM_STEERAGE_YAW_RATE,
    STRAIGHT_PROBE_THROTTLES,
    TARGET_CRUISE_SPEED_MPS,
    ForwardProbe,
    ForwardProbeSafetySample,
    action_protocol_hash,
    build_forward_control_profile,
    forward_probe_abort_reason,
    supplemental_turn_probe_controls,
)
from usvlib4ros.usvRosUtil import USVRosbridgeClient


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "logs"
PERIOD_S = 0.1
BASELINE_S = 2.0
COMMAND_S = 2.0
COAST_S = 5.0
SETTLE_TIMEOUT_S = 15.0
STATIONARY_SPEED_MPS = 0.03
STATIONARY_YAW_RATE_RAD_S = 0.02
STATIONARY_CYCLES = 10


def _publish(data: GlobalData, throttle: int, rudder: int) -> None:
    """Write the sole actuator command through NavigationStatus."""

    data.updateThrottleRudderOutput(throttle, rudder, 0.0, 0, 0.0)
    data.updateAlgorithmOutput(0, 0, 0, 0.0, 1_000, 2)


def _wait_for_reset(
    data: GlobalData,
    initial_request_time: float,
    timeout_s: float = 30.0,
) -> bool:
    deadline = time.monotonic() + timeout_s
    observed_request = False
    while time.monotonic() < deadline:
        reset_status = int(getattr(data.device_data, "reset_status", 0) or 0)
        request_time = float(
            getattr(data.device_data, "reset_request_time", 0.0) or 0.0
        )
        if reset_status == 1 or request_time != initial_request_time:
            observed_request = True
        if observed_request and reset_status == 2:
            return True
        _publish(data, 0, 0)
        time.sleep(PERIOD_S)
    return False


def _wait_for_auto(data: GlobalData, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if int(getattr(data.device_data, "work_model", 0) or 0) == 2:
            return True
        _publish(data, 0, 0)
        time.sleep(PERIOD_S)
    return False


def _wait_for_pose(data: GlobalData, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pose = getattr(data.scada_data, "pose", None)
        latitude = float(getattr(pose, "lat", 0.0) or 0.0)
        longitude = float(getattr(pose, "lng", 0.0) or 0.0)
        if (
            math.isfinite(latitude)
            and math.isfinite(longitude)
            and abs(latitude) > 1e-9
            and abs(longitude) > 1e-9
        ):
            return pose
        _publish(data, 0, 0)
        time.sleep(PERIOD_S)
    raise TimeoutError("POSE_UNAVAILABLE")


def _wait_for_task(data: GlobalData, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if int(getattr(data.device_data, "task_status", 0) or 0) != 0:
            return
        _publish(data, 0, 0)
        time.sleep(PERIOD_S)
    raise TimeoutError("TRAINING_NOT_STARTED")


def _minimum_valid_laser(runtime_input) -> float:
    values = [
        float(value)
        for value, valid in zip(
            runtime_input.laser_ranges,
            runtime_input.laser_valid_mask,
        )
        if valid and math.isfinite(float(value))
    ]
    if not values:
        raise RuntimeError("LASER_INVALID")
    return min(values)


def _derived_speed(samples) -> tuple[float, ...]:
    values = []
    for (_, first), (_, second) in zip(samples, samples[1:]):
        duration = second.stamp_sim - first.stamp_sim
        if duration > 1e-6:
            values.append(
                math.hypot(second.x - first.x, second.y - first.y) / duration
            )
    return tuple(values)


def _derived_yaw_rate(samples) -> tuple[float, ...]:
    values = []
    for (_, first), (_, second) in zip(samples, samples[1:]):
        duration = second.stamp_sim - first.stamp_sim
        if duration > 1e-6:
            delta = (second.yaw - first.yaw + math.pi) % (2.0 * math.pi) - math.pi
            values.append(delta / duration)
    return tuple(values)


def _tail_mean(values: tuple[float, ...]) -> float:
    if not values:
        raise RuntimeError("INSUFFICIENT_CALIBRATION_SAMPLES")
    return statistics.fmean(values[len(values) // 2 :])


def _steerage_throttle(probes: list[ForwardProbe]) -> float | None:
    for throttle in sorted({probe.throttle for probe in probes}):
        rows = [probe for probe in probes if probe.throttle == throttle]
        if any(
            probe.yaw_rate_rad_s >= MINIMUM_STEERAGE_YAW_RATE for probe in rows
        ) and any(
            probe.yaw_rate_rad_s <= -MINIMUM_STEERAGE_YAW_RATE for probe in rows
        ):
            return throttle
    return None


def _cruise_throttle(probes: list[ForwardProbe]) -> float:
    straight = sorted(
        (probe for probe in probes if abs(probe.rudder) <= 1e-12),
        key=lambda probe: probe.throttle,
    )
    return next(
        (
            probe.throttle
            for probe in straight
            if probe.speed_mps >= TARGET_CRUISE_SPEED_MPS
        ),
        max(straight, key=lambda probe: probe.speed_mps).throttle,
    )


def main() -> int:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = OUTPUT_DIR / f"forward-control-v2-{stamp}.json"
    result: dict[str, object] = {
        "schema_version": "national-test-forward-calibration-v2",
        "contains_sensitive_connection_data": False,
        "command_topic": "NavigationStatus",
        "device_status_role": "read_only_feedback",
        "verdict": "aborted",
        "stop_reason": "INITIALIZATION_ERROR",
        "limits": {
            "baseline_s": BASELINE_S,
            "command_s": COMMAND_S,
            "maximum_coast_s": COAST_S,
            "settle_timeout_s": SETTLE_TIMEOUT_S,
            "stationary_speed_mps": STATIONARY_SPEED_MPS,
            "stationary_yaw_rate_rad_s": STATIONARY_YAW_RATE_RAD_S,
            "stationary_cycles": STATIONARY_CYCLES,
            "start_clearance_m": 3.0,
            "start_laser_m": 3.0,
            "active_clearance_m": 0.5,
            "active_laser_m": 0.8,
            "maximum_displacement_m": 1.5,
        },
        "trials": [],
    }
    data = GlobalData.getInstance()
    probes: list[ForwardProbe] = []

    try:
        config = json.loads(
            (PROJECT_ROOT / "config.json").read_text(encoding="utf-8")
        )["ros2"]
        controller = Ros2Controller(
            host=config["host"],
            port=int(config["port"]),
            deviceId=config["deviceId"],
            globalData=data,
        )
        controller.initParameterList()
        _wait_for_task(data)
        context = build_fixed_route_context(
            session_id=f"forward-calibration-{stamp}",
        )

        def run_trial(throttle: float, rudder: float) -> ForwardProbe:
            result["active_trial"] = {"throttle": throttle, "rudder": rudder}
            initial_request_time = float(
                getattr(data.device_data, "reset_request_time", 0.0) or 0.0
            )
            _publish(data, 0, 0)
            if not controller.reset_unity():
                raise RuntimeError("UNITY_RESET_REQUEST_FAILED")
            if not _wait_for_reset(data, initial_request_time):
                raise TimeoutError("UNITY_RESET_TIMEOUT")
            if not controller.set_auto_work():
                raise RuntimeError("AUTO_MODE_REQUEST_FAILED")
            if not _wait_for_auto(data):
                raise TimeoutError("AUTO_MODE_TIMEOUT")
            _wait_for_pose(data)
            adapter = LiveInputAdapter(data, context)
            snapshot = context.compiled_map.snapshot
            settle_started = time.monotonic()
            settle_deadline = settle_started + SETTLE_TIMEOUT_S
            stationary_cycles = 0
            first_settle_speed = None
            while time.monotonic() < settle_deadline:
                _publish(data, 0, 0)
                first_input = adapter.build()
                state = first_input.vessel_state
                if first_settle_speed is None:
                    first_settle_speed = abs(state.speed)
                reason = forward_probe_abort_reason(
                    ForwardProbeSafetySample(
                        pose_age_s=first_input.pose_age_s,
                        scan_age_s=first_input.scan_age_s,
                        clearance_m=snapshot.clearance_at(state),
                        minimum_laser_m=_minimum_valid_laser(first_input),
                        displacement_m=0.0,
                        feedback_ok=(
                            state.is_finite()
                            and first_input.work_model == 2
                            and first_input.task_status != 0
                        ),
                    ),
                    starting=True,
                )
                if reason is not None:
                    raise RuntimeError(reason)
                if (
                    abs(state.speed) <= STATIONARY_SPEED_MPS
                    and abs(state.yaw_rate) <= STATIONARY_YAW_RATE_RAD_S
                ):
                    stationary_cycles += 1
                    if stationary_cycles >= STATIONARY_CYCLES:
                        break
                else:
                    stationary_cycles = 0
                time.sleep(PERIOD_S)
            else:
                raise TimeoutError("UNITY_DID_NOT_SETTLE")

            settle_s = time.monotonic() - settle_started
            origin = first_input.vessel_state

            def safety(runtime_input, *, starting: bool) -> None:
                state = runtime_input.vessel_state
                reason = forward_probe_abort_reason(
                    ForwardProbeSafetySample(
                        pose_age_s=runtime_input.pose_age_s,
                        scan_age_s=runtime_input.scan_age_s,
                        clearance_m=snapshot.clearance_at(state),
                        minimum_laser_m=_minimum_valid_laser(runtime_input),
                        displacement_m=math.hypot(
                            state.x - origin.x,
                            state.y - origin.y,
                        ),
                        feedback_ok=(
                            state.is_finite()
                            and runtime_input.work_model == 2
                            and runtime_input.task_status != 0
                        ),
                    ),
                    starting=starting,
                )
                if reason is not None:
                    raise RuntimeError(reason)

            safety(first_input, starting=True)
            trace: list[dict[str, object]] = []

            def phase(
                name: str,
                duration_s: float,
                throttle_percent: int,
                rudder_percent: int,
                *,
                distance_limited: bool = False,
            ):
                samples = []
                started = time.monotonic()
                while time.monotonic() - started < duration_s:
                    _publish(data, throttle_percent, rudder_percent)
                    runtime_input = adapter.build()
                    state = runtime_input.vessel_state
                    displacement = math.hypot(
                        state.x - origin.x,
                        state.y - origin.y,
                    )
                    safety(runtime_input, starting=False)
                    elapsed = time.monotonic() - started
                    samples.append((elapsed, state))
                    trace.append(
                        {
                            "phase": name,
                            "elapsed_s": elapsed,
                            "x_m": state.x,
                            "y_m": state.y,
                            "yaw_rad": state.yaw,
                            "speed_mps": state.speed,
                            "yaw_rate_rad_s": state.yaw_rate,
                            "command_throttle_percent": throttle_percent,
                            "command_rudder_percent": rudder_percent,
                            "feedback_throttle_percent": float(
                                getattr(data.device_data, "throttle_percent", 0.0)
                                or 0.0
                            ),
                            "feedback_rudder_percent": float(
                                getattr(data.device_data, "rudder_percent", 0.0)
                                or 0.0
                            ),
                            "clearance_m": snapshot.clearance_at(state),
                            "minimum_laser_m": _minimum_valid_laser(runtime_input),
                            "pose_age_s": runtime_input.pose_age_s,
                            "scan_age_s": runtime_input.scan_age_s,
                        }
                    )
                    if distance_limited and displacement >= 1.2:
                        _publish(data, 0, 0)
                        break
                    time.sleep(PERIOD_S)
                return tuple(samples)

            baseline = phase("baseline", BASELINE_S, 0, 0)
            command = phase(
                "command",
                COMMAND_S,
                int(round(throttle * 100.0)),
                int(round(rudder * 100.0)),
                distance_limited=True,
            )
            if not command:
                raise RuntimeError("NO_COMMAND_SAMPLES")
            command_end = command[-1][1]
            coast = phase("coast", COAST_S, 0, 0, distance_limited=True)
            _publish(data, 0, 0)
            baseline_speed = _tail_mean(_derived_speed(baseline))
            speed = _tail_mean(_derived_speed(command))
            yaw_rate = _tail_mean(_derived_yaw_rate(command))
            coast_distance = (
                0.0
                if not coast
                else math.hypot(
                    coast[-1][1].x - command_end.x,
                    coast[-1][1].y - command_end.y,
                )
            )
            probe = ForwardProbe(
                throttle,
                rudder,
                max(0.0, speed),
                yaw_rate,
                coast_distance,
            )
            result["trials"].append(
                {
                    "throttle": throttle,
                    "rudder": rudder,
                    "baseline_speed_mps": baseline_speed,
                    "settle_s": settle_s,
                    "settle_initial_speed_mps": first_settle_speed,
                    "settle_final_speed_mps": abs(origin.speed),
                    "speed_mps": probe.speed_mps,
                    "yaw_rate_rad_s": probe.yaw_rate_rad_s,
                    "coast_distance_m": probe.coast_distance_m,
                    "trace": trace,
                }
            )
            result["active_trial"] = None
            return probe

        effective_rows: list[float] = []
        for throttle in STRAIGHT_PROBE_THROTTLES:
            probe = run_trial(throttle, 0.0)
            probes.append(probe)
            baseline_speed = float(result["trials"][-1]["baseline_speed_mps"])
            if probe.speed_mps >= max(0.05, baseline_speed + 0.03):
                effective_rows.append(throttle)
        effective = tuple(effective_rows)
        if not effective:
            raise RuntimeError("NO_EFFECTIVE_FORWARD_RESPONSE")

        steerage = None
        for throttle in effective:
            probes.append(run_trial(throttle, 0.3))
            probes.append(run_trial(throttle, -0.3))
            steerage = _steerage_throttle(probes)
            if steerage is not None:
                break
        if steerage is None:
            raise RuntimeError("NO_TWO_SIDED_STEERAGE")
        cruise = _cruise_throttle(probes)
        existing = {(probe.throttle, probe.rudder) for probe in probes}
        for throttle, rudder in supplemental_turn_probe_controls(steerage, cruise):
            if (throttle, rudder) not in existing:
                probes.append(run_trial(throttle, rudder))
                existing.add((throttle, rudder))

        profile = build_forward_control_profile(probes)
        result.update(
            {
                "verdict": "calibrated",
                "stop_reason": None,
                "profile": asdict(profile),
                "action_protocol_hash": action_protocol_hash(profile),
            }
        )
    except Exception as exc:
        result["stop_reason"] = f"{type(exc).__name__}:{exc}"
    finally:
        for _ in range(3):
            _publish(data, 0, 0)
            time.sleep(0.05)
        ros = USVRosbridgeClient.ros
        if ros is not None:
            ros.terminate()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(str(target))
    return 0 if result["verdict"] == "calibrated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
