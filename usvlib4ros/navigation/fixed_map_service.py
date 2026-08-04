"""Official-sample lifecycle wrapped around the fixed-map controller core."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from usvlib4ros.navigation.fixed_map_runtime import (
    DEFAULT_CHECKPOINT,
    FixedMapControllerCore,
    LiveInputAdapter,
    RuntimeDecision,
    approved_fixed_route_fallback,
    build_live_route_context,
    load_live_ready_policy,
    load_offline_ready_policy,
    load_tested_candidate_policy,
)
from usvlib4ros.navigation.training_reports import (
    DEFAULT_REPORTS_DIR,
    EpisodeReport,
    TrainingReportLogger,
)
from usvlib4ros.planning.fixed_route import SIDECAR_PATH
from usvlib4ros.policy.checkpoint_promotion import PolicyMode
from usvlib4ros.usvRosUtil import LogUtil


MAX_EPOCH = 4_000
MAX_STEPS = 3_000
MAX_EPISODE_SECONDS = 300.0
CONTROL_PERIOD_S = 0.1
FAILURE_CONFIRMATION_SECONDS = 5.0
CHECKPOINT_PROMOTION_PENDING = (
    "SAC checkpoint has not passed offline and Unity promotion"
)
UNSAFE_STOP_REASONS = frozenset(
    {
        "NO_SAFE_ACTION",
        "LATEST_INPUT_UNSAFE",
        "MAP_INVALID",
        "DYNAMICS_INVALID",
        "LASER_EMERGENCY_STOP",
    }
)


def policy_loader_for_mode(policy_mode: PolicyMode):
    """Map an explicit policy mode to its fail-closed checkpoint loader.

    ``LIVE`` is the strictest gate: the checkpoint must have passed both
    offline evaluation and Unity live validation.  Candidate loaders are
    reachable only through explicit validation modes.
    """

    if policy_mode == PolicyMode.UNITY_TEST:
        return load_tested_candidate_policy
    if policy_mode == PolicyMode.OFFLINE_VALIDATION:
        return load_offline_ready_policy
    return load_live_ready_policy


def preflight_assets(checkpoint_path: Optional[Path] = None) -> None:
    """Fail fast before connecting when any runtime asset is missing."""

    checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT)
    required = (
        ("checkpoint", checkpoint),
        (
            "checkpoint manifest",
            checkpoint.with_suffix(checkpoint.suffix + ".json"),
        ),
        ("static world sidecar", SIDECAR_PATH),
        (
            "live affine profile",
            SIDECAR_PATH.parent / "national_test_live_profile.json",
        ),
    )
    missing = [
        f"  {name}: {path}" for name, path in required if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Navigation assets are missing:\n" + "\n".join(missing)
        )


def advance_failure_streak(
    previous: int,
    reason: str,
    failure_reasons,
) -> int:
    """Count only consecutive failure samples; one recovery clears the streak."""

    return previous + 1 if reason in failure_reasons else 0


def is_collision_evidence(
    reason: str,
    minimum_laser_m,
) -> bool:
    """Confirm impact only when map invalidity and laser evidence agree."""

    return (
        reason == "MAP_INVALID"
        and minimum_laser_m is not None
        and minimum_laser_m <= 0.6
    )


def advance_collision_confirmation(
    laser_stop_started_s: Optional[float],
    reason: str,
    minimum_laser_m,
    *,
    now_s: float,
) -> tuple[Optional[float], bool]:
    """Require direct dual evidence or five seconds of continuous laser stop."""

    if reason == "LASER_EMERGENCY_STOP":
        if laser_stop_started_s is None:
            laser_stop_started_s = now_s
    else:
        laser_stop_started_s = None
    return (
        laser_stop_started_s,
        is_collision_evidence(reason, minimum_laser_m)
        or (
            laser_stop_started_s is not None
            and now_s - laser_stop_started_s
            >= FAILURE_CONFIRMATION_SECONDS
        ),
    )


class FixedMapNavigationService:
    """Keep the sample reset/auto/route UI while replacing its PPO internals."""

    def __init__(
        self,
        ros_ctrl,
        global_data,
        *,
        checkpoint_path: Optional[Path] = None,
        action_bridge=None,
        policy_mode: PolicyMode = PolicyMode.LIVE,
        single_episode: bool = True,
        reports_dir: Optional[Path] = None,
    ) -> None:
        self.ros_ctrl = ros_ctrl
        self.global_data = global_data
        self.action_bridge = action_bridge
        self.policy_mode = PolicyMode(policy_mode)
        self.single_episode = bool(single_episode)
        self.checkpoint_path = Path(
            checkpoint_path or DEFAULT_CHECKPOINT
        )
        self.reports_dir = Path(reports_dir or DEFAULT_REPORTS_DIR)
        self.last_episode_metrics = None
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        """Ask the run loop to exit promptly; callers still publish zero."""

        self._stop_event.set()

    def _record_episode_report(
        self,
        logger: TrainingReportLogger,
        episode: int,
    ) -> None:
        metrics = self.last_episode_metrics
        if not isinstance(metrics, dict):
            return
        try:
            global_episode = logger.record_episode(
                EpisodeReport(
                    episode=episode,
                    total_steps=int(metrics.get("total_steps", 0)),
                    completed=bool(metrics.get("completed", False)),
                    completed_waypoints=int(
                        metrics.get("completed_waypoints", 0)
                    ),
                    duration_s=float(metrics.get("duration_s", 0.0)),
                    waypoint_reached_steps=tuple(
                        metrics.get("waypoint_reached_steps", ())
                    ),
                    waypoint_min_distances_m=tuple(
                        metrics.get("waypoint_min_distances_m", ())
                    ),
                    stop_reason=str(metrics.get("stop_reason", "")),
                    replans=int(metrics.get("replans", 0)),
                    collisions=int(metrics.get("collisions", 0)),
                    laser_emergency_stops=int(
                        metrics.get("laser_emergency_stops", 0)
                    ),
                    unrecovered_unsafe_events=int(
                        metrics.get("unrecovered_unsafe_events", 0)
                    ),
                )
            )
        except (OSError, ValueError) as exc:
            LogUtil.error(f"training report write failed: {exc}")
            return
        print(
            f"Training report episode {global_episode} written to "
            f"{logger.reports_dir}"
        )

    def _task_status_active(self) -> bool:
        return int(
            getattr(self.global_data.device_data, "task_status", 0)
            or 0
        ) != 0

    def _publish_zero(
        self,
        *,
        mission_index: int = 0,
        distance_m: float = 0.0,
        heading_deg: float = 0.0,
    ) -> None:
        self.global_data.updateThrottleRudderOutput(
            0,
            0,
            heading_deg,
            mission_index,
            distance_m,
        )
        if self.action_bridge is not None:
            self.action_bridge.set_command(0, 0)

    def _publish_decision(
        self,
        decision: RuntimeDecision,
        *,
        episode: int,
        step: int,
    ) -> None:
        if decision.stop or decision.control is None:
            throttle = 0
            rudder = 0
        else:
            throttle = int(
                max(
                    -100,
                    min(100, round(decision.control.throttle * 100.0)),
                )
            )
            rudder = int(
                max(
                    -100,
                    min(100, round(decision.control.rudder * 100.0)),
                )
            )
        self.global_data.updateThrottleRudderOutput(
            throttle,
            rudder,
            decision.advised_heading_deg,
            decision.mission_index,
            decision.distance_to_goal_m,
        )
        if self.action_bridge is not None:
            self.action_bridge.set_command(throttle, rudder)
        score = int(
            round(
                100.0
                * decision.mission_index
                / max(1, 12)
            )
        )
        self.global_data.updateAlgorithmOutput(
            episode,
            step,
            score,
            0.0,
            MAX_EPOCH,
            2,
        )

    def _wait_for_reset(
        self,
        timeout_s: float = 30.0,
        *,
        initial_request_time: Optional[float] = None,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        if initial_request_time is None:
            initial_request_time = float(
                getattr(
                    self.global_data.device_data,
                    "reset_request_time",
                    0.0,
                )
                or 0.0
            )
        else:
            initial_request_time = float(initial_request_time)
        observed_current_request = False
        while (
            time.monotonic() < deadline
            and not self._stop_event.is_set()
        ):
            if int(
                getattr(
                    self.global_data.device_data,
                    "task_status",
                    0,
                )
                or 0
            ) == 0:
                return False
            reset_status = int(
                getattr(
                    self.global_data.device_data,
                    "reset_status",
                    0,
                )
                or 0
            )
            request_time = float(
                getattr(
                    self.global_data.device_data,
                    "reset_request_time",
                    0.0,
                )
                or 0.0
            )
            if (
                reset_status == 1
                or request_time != initial_request_time
            ):
                observed_current_request = True
            if observed_current_request and reset_status == 2:
                return True
            self._publish_zero()
            time.sleep(0.1)
        return False

    def _wait_for_auto(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while (
            time.monotonic() < deadline
            and not self._stop_event.is_set()
        ):
            if int(
                getattr(
                    self.global_data.device_data,
                    "work_model",
                    0,
                )
                or 0
            ) == 2:
                return True
            self._publish_zero()
            time.sleep(0.1)
        return False

    def _wait_for_pose(self, timeout_s: float = 10.0):
        deadline = time.monotonic() + timeout_s
        while (
            time.monotonic() < deadline
            and not self._stop_event.is_set()
        ):
            pose = getattr(self.global_data.scada_data, "pose", None)
            lat = float(getattr(pose, "lat", 0.0) or 0.0)
            lng = float(getattr(pose, "lng", 0.0) or 0.0)
            if abs(lat) > 1e-9 and abs(lng) > 1e-9:
                return pose
            self._publish_zero()
            time.sleep(0.1)
        raise TimeoutError("no live ship pose received after Unity reset")

    def _run_episode(
        self,
        route,
        episode: int,
        *,
        max_seconds: float = MAX_EPISODE_SECONDS,
    ) -> bool:
        pose = self._wait_for_pose()
        context = build_live_route_context(
            route,
            pose,
            session_id=f"unity-episode-{episode}-{int(time.time())}",
        )
        manifest_path = self.checkpoint_path.with_suffix(
            self.checkpoint_path.suffix + ".json"
        )
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        print(
            f"[{self.policy_mode.value}] loading policy "
            f"{self.checkpoint_path} "
            f"sha256={manifest.get('checkpoint_sha256')} "
            f"offline_ready={manifest.get('offline_ready')} "
            f"live_ready={manifest.get('live_ready')}"
        )
        loader = policy_loader_for_mode(self.policy_mode)
        policy = loader(
            self.checkpoint_path,
            context,
        )
        adapter = LiveInputAdapter(self.global_data, context)
        core = FixedMapControllerCore(context, policy)
        started = time.monotonic()
        last_reason = ""
        points = tuple(
            (
                point[0] - context.compiled_map.manifest.origin_enu[0],
                point[1] - context.compiled_map.manifest.origin_enu[1],
            )
            for point in context.compiled_map.manifest.route_points_enu
        )
        waypoint_min_distances = [float("inf")] * len(points)
        waypoint_reached_steps = [None] * len(points)
        laser_stops = 0
        unsafe_events = 0
        collision_indicators = 0
        replans = 0
        telemetry = []
        laser_stop_started_s = None
        unsafe_streak = 0
        completed = False
        total_steps = 0
        try:
            for step in range(MAX_STEPS):
                if self._stop_event.is_set():
                    self._publish_zero(mission_index=core.mission_index)
                    return False
                if not self._task_status_active():
                    print(f"Stop train step {step}...")
                    self._publish_zero(mission_index=core.mission_index)
                    return False
                if time.monotonic() - started > max_seconds:
                    last_reason = "TIMEOUT"
                    self._publish_zero(mission_index=core.mission_index)
                    return False

                tick_started = time.monotonic()
                sample = adapter.build()
                for index, (x, y) in enumerate(points):
                    waypoint_min_distances[index] = min(
                        waypoint_min_distances[index],
                        (
                            (sample.vessel_state.x - x) ** 2
                            + (sample.vessel_state.y - y) ** 2
                        )
                        ** 0.5,
                    )
                previous_mission_index = core.mission_index
                decision = core.step(sample)
                total_steps = step + 1
                for reached_index in range(
                    previous_mission_index,
                    min(decision.mission_index, len(points)),
                ):
                    if waypoint_reached_steps[reached_index] is None:
                        waypoint_reached_steps[reached_index] = step + 1
                if decision.completed:
                    waypoint_reached_steps[-1] = step + 1
                valid_laser = tuple(
                    value
                    for value, valid in zip(
                        sample.laser_ranges,
                        sample.laser_valid_mask,
                    )
                    if valid
                )
                minimum_laser = (
                    min(valid_laser) if valid_laser else None
                )
                laser_stops += int(
                    decision.reason == "LASER_EMERGENCY_STOP"
                )
                (
                    laser_stop_started_s,
                    confirmed_collision,
                ) = advance_collision_confirmation(
                    laser_stop_started_s,
                    decision.reason,
                    minimum_laser,
                    now_s=time.monotonic(),
                )
                unsafe_streak = advance_failure_streak(
                    unsafe_streak,
                    decision.reason,
                    UNSAFE_STOP_REASONS,
                )
                collision_indicators = max(
                    collision_indicators,
                    int(confirmed_collision),
                )
                replans += int(decision.replanned)
                self._publish_decision(
                    decision,
                    episode=episode,
                    step=step,
                )
                if step % 10 == 0 or decision.reason != last_reason:
                    telemetry.append(
                        {
                            "step": step,
                            "elapsed_s": time.monotonic() - started,
                            "mission_index": decision.mission_index,
                            "maneuver_phase": decision.maneuver_phase,
                            "x_m": sample.vessel_state.x,
                            "y_m": sample.vessel_state.y,
                            "yaw_rad": sample.vessel_state.yaw,
                            "speed_mps": sample.vessel_state.speed,
                            "yaw_rate_rad_s": (
                                sample.vessel_state.yaw_rate
                            ),
                            "distance_to_goal_m": (
                                decision.distance_to_goal_m
                            ),
                            "map_clearance_m": (
                                context.compiled_map.snapshot.clearance_at(
                                    sample.vessel_state
                                )
                            ),
                            "minimum_laser_m": (
                                minimum_laser
                            ),
                            "throttle": (
                                decision.control.throttle
                                if decision.control is not None
                                else 0.0
                            ),
                            "rudder": (
                                decision.control.rudder
                                if decision.control is not None
                                else 0.0
                            ),
                            "reason": decision.reason,
                            "replanned": decision.replanned,
                        }
                    )
                if (
                    step % 20 == 0
                    or decision.replanned
                    or decision.reason != last_reason
                ):
                    print(
                        f"Step: {step}, Action: {decision.action}, "
                        f"Reason: {decision.reason}, "
                        f"Point: {decision.mission_index}, "
                        f"Distance: {decision.distance_to_goal_m:.2f}"
                    )
                last_reason = decision.reason
                if confirmed_collision:
                    unsafe_events = 1
                    last_reason = "CONFIRMED_COLLISION"
                    self._publish_zero(
                        mission_index=decision.mission_index,
                        heading_deg=decision.advised_heading_deg,
                    )
                    return False
                if decision.completed:
                    completed = True
                    print(
                        f"Episode ended at step {step}, "
                        "National_Test route completed"
                    )
                    self._publish_zero(
                        mission_index=decision.mission_index,
                        heading_deg=decision.advised_heading_deg,
                    )
                    return True
                elapsed = time.monotonic() - tick_started
                time.sleep(
                    max(0.0, CONTROL_PERIOD_S - elapsed)
                )
            self._publish_zero(mission_index=core.mission_index)
            return False
        finally:
            self.last_episode_metrics = {
                "total_steps": total_steps,
                "completed": completed,
                "duration_s": time.monotonic() - started,
                "completed_waypoints": sum(
                    step is not None for step in waypoint_reached_steps
                ),
                "waypoint_reached_steps": waypoint_reached_steps,
                "waypoint_min_distances_m": waypoint_min_distances,
                "collisions": collision_indicators,
                "laser_emergency_stops": laser_stops,
                "unrecovered_unsafe_events": max(
                    unsafe_events,
                    int(unsafe_streak > 0 and not completed),
                ),
                "replans": replans,
                "final_mission_index": core.mission_index,
                "stop_reason": last_reason,
                "telemetry": telemetry,
            }

    def run(self) -> None:
        try:
            self.ros_ctrl.initParameterList()
        except Exception as exc:
            LogUtil.error(exc)
        print(
            f"policy_mode={self.policy_mode.value} "
            f"single_episode={self.single_episode} "
            f"checkpoint={self.checkpoint_path}"
        )
        while not self._stop_event.is_set():
            try:
                print("wait train button trigger ...")
                while (
                    not self._task_status_active()
                    and not self._stop_event.is_set()
                ):
                    self._publish_zero()
                    time.sleep(1.0)
                if self._stop_event.is_set():
                    break

                report_logger = TrainingReportLogger.for_train_click(
                    self.reports_dir
                )
                for episode in range(MAX_EPOCH):
                    if (
                        self._stop_event.is_set()
                        or not self._task_status_active()
                    ):
                        print("Stop train ...")
                        break
                    print(f"train {episode} ...")
                    print("Reset unity ...")
                    reset_request_time = float(
                        getattr(
                            self.global_data.device_data,
                            "reset_request_time",
                            0.0,
                        )
                        or 0.0
                    )
                    if not self.ros_ctrl.reset_unity():
                        raise RuntimeError("Unity reset request failed")
                    if not self._wait_for_reset(
                        initial_request_time=reset_request_time,
                    ):
                        raise TimeoutError("Unity reset did not complete")
                    if not self.ros_ctrl.set_auto_work():
                        raise RuntimeError("automatic work-mode request failed")
                    if not self._wait_for_auto():
                        raise TimeoutError(
                            "device did not enter automatic work mode"
                        )
                    route = self.ros_ctrl.getRoute()
                    if not getattr(route, "points", None):
                        print(
                            "Unity route service returned no points; using "
                            "approved fixed route fallback"
                        )
                        route = approved_fixed_route_fallback()
                    print(
                        f"Route {getattr(route, 'name', 'unnamed')}: "
                        f"{len(route.points)} points, "
                        f"{len(getattr(route, 'obstacles', ()) or ())} "
                        "obstacles"
                    )
                    self.global_data.route = route
                    self.last_episode_metrics = None
                    try:
                        try:
                            completed = self._run_episode(route, episode)
                        finally:
                            self._record_episode_report(
                                report_logger,
                                episode,
                            )
                    except ValueError as exc:
                        if str(exc) != CHECKPOINT_PROMOTION_PENDING:
                            raise
                        print(
                            "ROS/Unity connection OK; SAC checkpoint is not "
                            "live-ready. Stop and restart training after "
                            "promotion."
                        )
                        while (
                            self._task_status_active()
                            and not self._stop_event.is_set()
                        ):
                            self._publish_zero()
                            time.sleep(1.0)
                        break
                    if completed and self.single_episode:
                        print(
                            "Mission completed; single-episode mode, "
                            "holding zero control until the task stops"
                        )
                        while (
                            self._task_status_active()
                            and not self._stop_event.is_set()
                        ):
                            self._publish_zero()
                            time.sleep(0.1)
                        break
            except Exception as exc:
                self._publish_zero()
                LogUtil.error(exc)
            finally:
                self._publish_zero()
                time.sleep(0.02)


__all__ = [
    "FixedMapNavigationService",
    "policy_loader_for_mode",
    "preflight_assets",
]
