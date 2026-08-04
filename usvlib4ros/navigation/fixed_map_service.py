"""Official-sample lifecycle wrapped around the fixed-map controller core."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import random
import threading
import time
from pathlib import Path
from typing import Optional

import torch

from usvlib4ros.navigation.fixed_map_runtime import (
    DEFAULT_CHECKPOINT,
    FixedMapControllerCore,
    LiveInputAdapter,
    RuntimeDecision,
    approved_fixed_route_fallback,
    build_live_route_context,
    load_live_ready_policy,
    load_offline_ready_policy,
    load_runtime_maneuver_profile,
    load_runtime_safety_profile,
    load_tested_candidate_policy,
)
from usvlib4ros.navigation.training_reports import (
    DEFAULT_REPORTS_DIR,
    EpisodeReport,
    SelfTrainingEpisodeReport,
    SelfTrainingGenerationReport,
    SelfTrainingReportLogger,
    TrainingReportLogger,
)
from usvlib4ros.navigation.reverse_control_calibration import (
    reverse_control_profile_from_dict,
)
from usvlib4ros.planning.forward_control_profile import (
    forward_control_profile_from_dict,
)
from usvlib4ros.planning.fixed_route import SIDECAR_PATH
from usvlib4ros.policy.checkpoint_promotion import PolicyMode
from usvlib4ros.policy.recurrent_sac import RecurrentDiscreteSAC
from usvlib4ros.policy.fixed_map_trainer import (
    EpisodeInterrupted,
    FixedMapSACTrainer,
)
from usvlib4ros.policy.self_training import (
    ActiveCheckpointRegistry,
    EvaluationSummary as SelfTrainingEvaluationSummary,
    GenerationEvidence,
    SelfTrainingConfig,
    SelfTrainingCursor,
    SelfTrainingLearner,
    SelfTrainingStage,
    SelfTrainingStateStore,
    UnityTransitionRecorder,
    operational_profile_from_manifest,
    promotion_decision,
    save_generation_checkpoint,
)
from usvlib4ros.usvRosUtil import LogUtil


MAX_EPOCH = 4_000
MAX_STEPS = 5_000
MAX_EPISODE_SECONDS = 600.0
CONTROL_PERIOD_S = 0.1
FAILURE_CONFIRMATION_SECONDS = 5.0
CHECKPOINT_PROMOTION_PENDING = (
    "SAC checkpoint has not passed offline and Unity promotion"
)
ACTIVE_CHECKPOINT_PATH = (
    DEFAULT_CHECKPOINT.parent / "national_test_sac_active.json"
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
            reason == "LASER_EMERGENCY_STOP"
            and minimum_laser_m is not None
            and minimum_laser_m <= 0.0
        )
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
        self_training: bool = False,
        validate_only: bool = False,
    ) -> None:
        self.ros_ctrl = ros_ctrl
        self.global_data = global_data
        self.action_bridge = action_bridge
        self.policy_mode = PolicyMode(policy_mode)
        self.single_episode = bool(single_episode)
        if type(self_training) is not bool or type(validate_only) is not bool:
            raise ValueError("training execution flags must be boolean")
        if self_training and validate_only:
            raise ValueError("self-training and validate-only are mutually exclusive")
        if self_training and self.policy_mode is not PolicyMode.UNITY_TEST:
            raise ValueError("self-training is restricted to unity_test mode")
        self.self_training = self_training
        self.validate_only = validate_only
        self.checkpoint_path = Path(
            checkpoint_path or DEFAULT_CHECKPOINT
        )
        self.reports_dir = Path(reports_dir or DEFAULT_REPORTS_DIR)
        self.last_episode_metrics = None
        self.last_episode_transitions = ()
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
        score_override: Optional[float] = None,
        loss: float = 0.0,
        max_epoch: int = MAX_EPOCH,
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
        score = (
            int(round(score_override))
            if score_override is not None
            else int(
                round(
                    100.0
                    * decision.mission_index
                    / max(1, 12)
                )
            )
        )
        self.global_data.updateAlgorithmOutput(
            episode,
            step,
            score,
            float(loss),
            int(max_epoch),
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
        policy_override: Optional[RecurrentDiscreteSAC] = None,
        deterministic_policy: bool = True,
        collect_transitions: bool = False,
        display_episode: Optional[int] = None,
        display_max_epoch: int = MAX_EPOCH,
        latest_critic_loss: float = 0.0,
    ) -> bool:
        pose = self._wait_for_pose()
        safety_profile = load_runtime_safety_profile(
            self.checkpoint_path,
            self.policy_mode,
        )
        maneuver_profile = load_runtime_maneuver_profile(
            self.checkpoint_path,
            self.policy_mode,
        )
        context = build_live_route_context(
            route,
            pose,
            session_id=f"unity-episode-{episode}-{int(time.time())}",
            safety_profile=safety_profile,
            maneuver_profile=maneuver_profile,
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
            f"live_ready={manifest.get('live_ready')} "
            f"safety_profile={safety_profile.profile_id} "
            f"clearance_m={safety_profile.required_clearance_m} "
            "laser_stop_m="
            f"{safety_profile.laser_emergency_distance_m} "
            f"maneuver_profile={maneuver_profile.profile_id} "
            "point4_throttle_cap="
            f"{maneuver_profile.approach_throttle_cap} "
            "point4_rudder_cap="
            f"{maneuver_profile.approach_rudder_cap} "
            "point4_to_5_control="
            f"({maneuver_profile.turn_throttle},"
            f"{maneuver_profile.turn_rudder})"
        )
        if policy_override is None:
            loader = policy_loader_for_mode(self.policy_mode)
            policy = loader(
                self.checkpoint_path,
                context,
            )
        else:
            policy = policy_override
        adapter = LiveInputAdapter(self.global_data, context)
        if (
            policy_override is None
            and deterministic_policy
            and not collect_transitions
        ):
            core = FixedMapControllerCore(context, policy)
        else:
            core = FixedMapControllerCore(
                context,
                policy,
                deterministic_policy=deterministic_policy,
                full_safe_action_authority=bool(
                    getattr(policy, "full_safe_action_authority", False)
                ),
            )
        recorder = UnityTransitionRecorder() if collect_transitions else None
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
        episode_committed = False
        episode_transitions = ()
        try:
            for step in range(MAX_STEPS):
                if self._stop_event.is_set():
                    if recorder is not None:
                        recorder.discard_partial()
                    self._publish_zero(mission_index=core.mission_index)
                    return False
                if not self._task_status_active():
                    print(f"Stop train step {step}...")
                    if recorder is not None:
                        recorder.discard_partial()
                    self._publish_zero(mission_index=core.mission_index)
                    return False
                if time.monotonic() - started > max_seconds:
                    last_reason = "TIMEOUT"
                    if recorder is not None:
                        episode_transitions = recorder.finish(timeout=True)
                    episode_committed = True
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
                if recorder is not None and decision.training_trace is not None:
                    recorder.observe(decision.training_trace)
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
                    episode=(
                        episode
                        if display_episode is None
                        else display_episode
                    ),
                    step=step,
                    score_override=(
                        None
                        if recorder is None
                        else sum(
                            transition.reward
                            for transition in recorder.transitions
                        )
                    ),
                    loss=latest_critic_loss,
                    max_epoch=display_max_epoch,
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
                    if recorder is not None:
                        episode_transitions = recorder.finish(collision=True)
                    episode_committed = True
                    self._publish_zero(
                        mission_index=decision.mission_index,
                        heading_deg=decision.advised_heading_deg,
                    )
                    return False
                if decision.completed:
                    completed = True
                    if recorder is not None:
                        episode_transitions = recorder.finish(completed=True)
                    episode_committed = True
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
            if recorder is not None:
                last_reason = "TIMEOUT_MAX_STEPS"
                episode_transitions = recorder.finish(timeout=True)
            episode_committed = True
            self._publish_zero(mission_index=core.mission_index)
            return False
        finally:
            self.last_episode_transitions = tuple(episode_transitions)
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
                "episode_committed": episode_committed,
                "total_reward": sum(
                    transition.reward
                    for transition in episode_transitions
                ),
            }

    @staticmethod
    def _checkpoint_digest(path: Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _checkpoint_manifest(path: Path) -> dict:
        manifest_path = Path(path).with_suffix(Path(path).suffix + ".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("SAC checkpoint manifest must be an object")
        return manifest

    def _build_self_training_trainer(
        self,
        checkpoint_path: Path,
        config: SelfTrainingConfig,
    ) -> FixedMapSACTrainer:
        manifest = self._checkpoint_manifest(checkpoint_path)
        profile = operational_profile_from_manifest(manifest)
        try:
            forward_profile = forward_control_profile_from_dict(
                manifest["forward_control_profile"]
            )
            reverse_profile = reverse_control_profile_from_dict(
                manifest["reverse_control_profile"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("self-training checkpoint calibration is invalid") from exc
        trainer = FixedMapSACTrainer(
            forward_profile=forward_profile,
            reverse_profile=reverse_profile,
            calibration_status="calibrated",
            reverse_calibration_status="calibrated",
            operational_profile=profile,
            full_safe_action_authority=True,
            seed=31,
            hidden_dim=int(manifest.get("hidden_dim", 0)),
        )
        trainer.sac.load_checkpoint(checkpoint_path)
        torch.set_num_threads(config.cpu_threads)
        return trainer

    def _load_self_training_session(self):
        config = SelfTrainingConfig()
        store = SelfTrainingStateStore(
            self.reports_dir / "self_training_state"
        )
        if store.pointer_path.is_file():
            snapshot = store.load()
            cursor = snapshot.cursor
            champion = Path(cursor.champion_path)
            if (
                not champion.is_file()
                or self._checkpoint_digest(champion)
                != cursor.champion_sha256
            ):
                raise ValueError("persisted self-training champion hash is invalid")
            metadata = dict(snapshot.metadata)
            if metadata.get("config") != asdict(config):
                raise ValueError("persisted self-training config is incompatible")
            if snapshot.training_state is None:
                raise ValueError("persisted self-training network state is missing")
            trainer = self._build_self_training_trainer(champion, config)
            working_state = snapshot.training_state
            trainer.sac.load_training_state_dict(working_state)
            random.setstate(snapshot.python_random_state)
            torch.set_rng_state(snapshot.torch_random_state)
            trainer_rng_state = metadata.get("trainer_rng_state")
            if trainer_rng_state is None:
                raise ValueError("persisted trainer random state is missing")
            trainer.rng.setstate(trainer_rng_state)
            learner = SelfTrainingLearner(
                trainer.sac,
                config,
                offline_replay=snapshot.offline_replay,
                unity_replay=snapshot.unity_replay,
                on_fault=self._publish_zero,
            )
            current = trainer.sac.training_state_dict()
            trainer.sac.load_checkpoint(champion)
            champion_state = trainer.sac.training_state_dict()
            trainer.sac.load_training_state_dict(current)
            learner.set_champion_training_state(champion_state)
            self.checkpoint_path = champion
            return config, store, cursor, metadata, trainer, learner

        checkpoint = Path(self.checkpoint_path)
        digest = self._checkpoint_digest(checkpoint)
        manifest = self._checkpoint_manifest(checkpoint)
        if manifest.get("checkpoint_sha256") != digest:
            raise ValueError("self-training seed checkpoint hash is invalid")
        cursor = SelfTrainingCursor.new(
            config,
            champion_path=str(checkpoint.resolve()),
            champion_sha256=digest,
        )
        trainer = self._build_self_training_trainer(checkpoint, config)
        learner = SelfTrainingLearner(
            trainer.sac,
            config,
            on_fault=self._publish_zero,
        )
        metadata = {
            "config": asdict(config),
            "baseline_unity": [],
            "offline_evaluation": [],
            "unity_validation": [],
            "champion_unity": [],
            "trainer_rng_state": trainer.rng.getstate(),
        }
        self._persist_self_training(
            store,
            cursor,
            metadata,
            trainer,
            learner,
        )
        return config, store, cursor, metadata, trainer, learner

    @staticmethod
    def _persist_self_training(
        store: SelfTrainingStateStore,
        cursor: SelfTrainingCursor,
        metadata: dict,
        trainer: FixedMapSACTrainer,
        learner: SelfTrainingLearner,
    ) -> None:
        metadata["trainer_rng_state"] = trainer.rng.getstate()
        store.save(
            cursor,
            offline_replay=learner.offline_replay,
            unity_replay=learner.unity_replay,
            training_state=learner.sac.training_state_dict(),
            metadata=metadata,
        )

    def _prepare_unity_route(self):
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
        if not self._wait_for_reset(initial_request_time=reset_request_time):
            raise TimeoutError("Unity reset did not complete")
        if not self.ros_ctrl.set_auto_work():
            raise RuntimeError("automatic work-mode request failed")
        if not self._wait_for_auto():
            raise TimeoutError("device did not enter automatic work mode")
        route = self.ros_ctrl.getRoute()
        if not getattr(route, "points", None):
            print(
                "Unity route service returned no points; using approved "
                "fixed route fallback"
            )
            route = approved_fixed_route_fallback()
        self.global_data.route = route
        return route

    @staticmethod
    def _offline_metric(summary) -> dict[str, object]:
        return {
            "completed": bool(summary.completed),
            "total_steps": int(summary.steps),
            "total_reward": float(summary.total_reward),
            "safety_stops": int(summary.safety_stop),
            "timeouts": int(summary.timeout),
            "collisions": 0,
            "laser_stops": 0,
            "unrecovered_unsafe_events": int(summary.safety_stop),
            "stop_reason": str(summary.stop_reason),
        }

    @staticmethod
    def _unity_metric(metrics: dict) -> dict[str, object]:
        return {
            "completed": bool(metrics.get("completed", False)),
            "total_steps": int(metrics.get("total_steps", 0)),
            "total_reward": float(metrics.get("total_reward", 0.0)),
            "safety_stops": int(
                bool(metrics.get("unrecovered_unsafe_events", 0))
            ),
            "timeouts": int(
                str(metrics.get("stop_reason", "")).startswith("TIMEOUT")
            ),
            "collisions": int(metrics.get("collisions", 0)),
            "laser_stops": int(metrics.get("laser_emergency_stops", 0)),
            "unrecovered_unsafe_events": int(
                metrics.get("unrecovered_unsafe_events", 0)
            ),
            "stop_reason": str(metrics.get("stop_reason", "")),
        }

    @staticmethod
    def _evaluation_summary(rows: list[dict]) -> SelfTrainingEvaluationSummary:
        completed_rows = [row for row in rows if row["completed"]]
        return SelfTrainingEvaluationSummary(
            completed=len(completed_rows),
            attempted=len(rows),
            total_steps=tuple(
                int(row["total_steps"]) for row in completed_rows
            ),
            collisions=sum(int(row["collisions"]) for row in rows),
            laser_stops=sum(int(row["laser_stops"]) for row in rows),
            safety_stops=sum(int(row["safety_stops"]) for row in rows),
            timeouts=sum(int(row["timeouts"]) for row in rows),
            unrecovered_unsafe_events=sum(
                int(row["unrecovered_unsafe_events"]) for row in rows
            ),
        )

    def _record_self_training_metric(
        self,
        logger: SelfTrainingReportLogger,
        cursor: SelfTrainingCursor,
        metric: dict,
        learner: SelfTrainingLearner,
    ) -> None:
        latest = learner.latest_metrics
        logger.record_episode(
            SelfTrainingEpisodeReport(
                session_id=cursor.session_id,
                generation=cursor.generation,
                stage=cursor.stage.value,
                episode=cursor.completed_training_episodes,
                total_steps=int(metric["total_steps"]),
                total_reward=float(metric["total_reward"]),
                completed=bool(metric["completed"]),
                actor_loss=float(latest.get("actor_loss", 0.0)),
                critic_loss=float(latest.get("critic_loss", 0.0)),
                training_step=learner.sac.training_step,
                collisions=int(metric["collisions"]),
                laser_emergency_stops=int(metric["laser_stops"]),
                unrecovered_unsafe_events=int(
                    metric["unrecovered_unsafe_events"]
                ),
                stop_reason=str(metric["stop_reason"]),
            )
        )

    def _finalize_self_training_generation(
        self,
        *,
        config: SelfTrainingConfig,
        cursor: SelfTrainingCursor,
        metadata: dict,
        learner: SelfTrainingLearner,
        report_logger: SelfTrainingReportLogger,
    ) -> SelfTrainingCursor:
        offline = self._evaluation_summary(metadata["offline_evaluation"])
        unity = self._evaluation_summary(metadata["unity_validation"])
        champion_rows = metadata.get("champion_unity") or metadata.get(
            "baseline_unity"
        )
        champion_unity = self._evaluation_summary(champion_rows)
        decision = promotion_decision(
            champion_unity,
            offline,
            unity,
            generation_collision=cursor.generation_collision,
            minimum_step_improvement=config.minimum_step_improvement,
        )
        parent = Path(cursor.champion_path)
        target = parent.parent / (
            f"national_test_sac_{cursor.session_id}_g{cursor.generation:04d}_v5.pt"
        )
        checkpoint, manifest_path = save_generation_checkpoint(
            learner.sac,
            target,
            parent_checkpoint=parent,
            cursor=cursor,
            config=config,
            evidence=GenerationEvidence(
                offline=offline,
                unity=unity,
                champion_unity=champion_unity,
            ),
            decision=decision,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate_digest = str(manifest["checkpoint_sha256"])
        next_cursor = cursor.advance(config)
        if decision.promote:
            ActiveCheckpointRegistry(ACTIVE_CHECKPOINT_PATH).promote(checkpoint)
            learner.mark_champion()
            next_cursor = next_cursor.with_champion(
                checkpoint,
                candidate_digest,
            )
            metadata["champion_unity"] = list(metadata["unity_validation"])
            self.checkpoint_path = checkpoint
        else:
            learner.rollback_to_champion()
        report_logger.record_generation(
            SelfTrainingGenerationReport(
                session_id=cursor.session_id,
                generation=cursor.generation,
                completed_training_episodes=cursor.completed_training_episodes,
                parent_sha256=cursor.champion_sha256,
                candidate_sha256=candidate_digest,
                promoted=decision.promote,
                promotion_reason=decision.reason,
                offline_completed=offline.completed,
                unity_completed=unity.completed,
                unity_median_steps=unity.median_steps,
            )
        )
        metadata["offline_evaluation"] = []
        metadata["unity_validation"] = []
        return next_cursor

    def _run_self_training_click(self) -> None:
        (
            config,
            store,
            cursor,
            metadata,
            trainer,
            learner,
        ) = self._load_self_training_session()
        if cursor.stage is SelfTrainingStage.WAITING:
            cursor = cursor.extend_target(config)
            self._persist_self_training(
                store, cursor, metadata, trainer, learner
            )
        self_logger = SelfTrainingReportLogger(self.reports_dir)
        legacy_logger = TrainingReportLogger.for_train_click(self.reports_dir)

        while self._task_status_active() and not self._stop_event.is_set():
            if cursor.stage is SelfTrainingStage.WAITING:
                self._publish_zero()
                time.sleep(0.1)
                continue
            stage = cursor.stage
            print(
                f"self-train generation={cursor.generation} "
                f"stage={stage.value} index={cursor.stage_index + 1} "
                f"E={cursor.completed_training_episodes}/"
                f"{cursor.target_training_episodes}"
            )
            if stage in (
                SelfTrainingStage.OFFLINE_TRAIN,
                SelfTrainingStage.OFFLINE_EVAL,
            ):
                self._publish_zero()
                try:
                    transitions, summary = trainer.run_episode(
                        episode=(
                            cursor.generation * 100_000
                            + cursor.stage_index
                        ),
                        nominal_action_probability=(
                            0.65
                            if stage is SelfTrainingStage.OFFLINE_TRAIN
                            else 0.0
                        ),
                        deterministic_policy=(
                            stage is SelfTrainingStage.OFFLINE_EVAL
                        ),
                        max_steps=MAX_STEPS,
                        should_stop=lambda: (
                            self._stop_event.is_set()
                            or not self._task_status_active()
                        ),
                    )
                except EpisodeInterrupted:
                    self._publish_zero()
                    return
                metric = self._offline_metric(summary)
                if stage is SelfTrainingStage.OFFLINE_TRAIN:
                    learner.add_offline_episode(transitions)
                    try:
                        learner.update(config.updates_per_offline_episode)
                    except Exception:
                        self._record_self_training_metric(
                            self_logger, cursor, metric, learner
                        )
                        cursor = cursor.advance(config)
                        self._persist_self_training(
                            store, cursor, metadata, trainer, learner
                        )
                        raise
                else:
                    metadata["offline_evaluation"].append(metric)
                self._record_self_training_metric(
                    self_logger, cursor, metric, learner
                )
                cursor = cursor.advance(config)
                self.global_data.updateAlgorithmOutput(
                    cursor.completed_training_episodes,
                    int(metric["total_steps"]),
                    int(round(float(metric["total_reward"]))),
                    float(learner.latest_metrics.get("critic_loss", 0.0)),
                    cursor.target_training_episodes,
                    2,
                )
                self._persist_self_training(
                    store, cursor, metadata, trainer, learner
                )
                continue

            route = self._prepare_unity_route()
            unity_training = stage is SelfTrainingStage.UNITY_TRAIN
            baseline = stage is SelfTrainingStage.BASELINE_UNITY
            legacy_episode = (
                cursor.generation * 100_000
                + {
                    SelfTrainingStage.BASELINE_UNITY: 0,
                    SelfTrainingStage.UNITY_TRAIN: 10_000,
                    SelfTrainingStage.UNITY_VALIDATION: 20_000,
                }[stage]
                + cursor.stage_index
            )
            self.last_episode_metrics = None
            try:
                completed = self._run_episode(
                    route,
                    legacy_episode,
                    policy_override=(None if baseline else learner.sac),
                    deterministic_policy=not unity_training,
                    collect_transitions=True,
                    display_episode=cursor.completed_training_episodes,
                    display_max_epoch=cursor.target_training_episodes,
                    latest_critic_loss=float(
                        learner.latest_metrics.get("critic_loss", 0.0)
                    ),
                )
            finally:
                self._record_episode_report(
                    legacy_logger,
                    legacy_episode,
                )
            del completed
            metrics = self.last_episode_metrics or {}
            if not metrics.get("episode_committed", False):
                self._publish_zero()
                return
            metric = self._unity_metric(metrics)
            if stage is SelfTrainingStage.BASELINE_UNITY:
                metadata["baseline_unity"].append(metric)
            elif stage is SelfTrainingStage.UNITY_TRAIN:
                if self.last_episode_transitions:
                    learner.add_unity_episode(self.last_episode_transitions)
                if int(metric["collisions"]) > 0:
                    cursor = cursor.mark_collision()
                if cursor.stage_index + 1 == config.unity_training_episodes:
                    try:
                        learner.update(config.updates_after_unity_block)
                    except Exception:
                        self._record_self_training_metric(
                            self_logger, cursor, metric, learner
                        )
                        cursor = cursor.advance(config)
                        self._persist_self_training(
                            store, cursor, metadata, trainer, learner
                        )
                        raise
            else:
                metadata["unity_validation"].append(metric)
            self._record_self_training_metric(
                self_logger, cursor, metric, learner
            )
            if (
                stage is SelfTrainingStage.UNITY_VALIDATION
                and cursor.stage_index + 1
                == config.unity_validation_episodes
            ):
                cursor = self._finalize_self_training_generation(
                    config=config,
                    cursor=cursor,
                    metadata=metadata,
                    learner=learner,
                    report_logger=self_logger,
                )
            else:
                cursor = cursor.advance(config)
            self.global_data.updateAlgorithmOutput(
                cursor.completed_training_episodes,
                int(metric["total_steps"]),
                int(round(float(metric["total_reward"]))),
                float(learner.latest_metrics.get("critic_loss", 0.0)),
                cursor.target_training_episodes,
                2,
            )
            self._persist_self_training(
                store, cursor, metadata, trainer, learner
            )

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

                if self.self_training:
                    self._run_self_training_click()
                    continue

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
                if self.self_training:
                    while (
                        self._task_status_active()
                        and not self._stop_event.is_set()
                    ):
                        self._publish_zero()
                        time.sleep(0.1)
            finally:
                self._publish_zero()
                time.sleep(0.02)


__all__ = [
    "FixedMapNavigationService",
    "policy_loader_for_mode",
    "preflight_assets",
]
