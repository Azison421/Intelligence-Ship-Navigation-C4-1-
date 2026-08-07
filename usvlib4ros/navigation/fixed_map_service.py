"""ROS lifecycle adapter for the planning-free National_Test controller."""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Optional

from usvlib4ros.navigation.fixed_corridor import DEFAULT_CORRIDOR_PATH
from usvlib4ros.navigation.fixed_map_runtime import (
    DEFAULT_CHECKPOINT,
    FixedMapControllerCore,
    LiveInputAdapter,
    RuntimeDecision,
    build_fixed_route_context,
    load_policy,
)
from usvlib4ros.planning.fixed_route import LIVE_PROFILE_PATH, SIDECAR_PATH
from usvlib4ros.policy.checkpoint_promotion import PolicyMode
from usvlib4ros.policy.self_training import (
    MAX_TRAINING_EPISODES,
    SelfTrainingStage,
    TrainingStateStore,
    UnityTrainingGate,
    UnityTransitionRecorder,
    validate_checkpoint_manifest,
)
from usvlib4ros.usvRosUtil import LogUtil


CONTROL_PERIOD_S = 0.1
MAX_EPISODE_SECONDS = 600.0
TELEMETRY_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "logs"
TRAINING_STATE_PATH = (
    DEFAULT_CHECKPOINT.parent / "national_test_self_training_v6.pt"
)
UNITY_RESET_TIMEOUT_S = 30.0
ENVIRONMENT_STOP_REASONS = frozenset(
    {
        "DEVICE_STALE",
        "INPUT_INVALID",
        "NOT_IN_AUTO_MODE",
        "POSE_STALE",
        "SCAN_STALE",
    }
)
POLICY_FAILURE_REASONS = frozenset(
    {
        "DYNAMICS_INVALID",
        "LASER_EMERGENCY_STOP",
        "MAP_INVALID",
        "MOTION_STALLED",
        "NO_SAFE_ACTION_TRUNCATED",
        "POLICY_NO_ACTION",
    }
)


def preflight_assets(checkpoint_path: Optional[Path] = None) -> None:
    """Reject missing or non-V6 assets before opening a ROS connection."""

    checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT)
    manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    required = (
        checkpoint,
        manifest_path,
        SIDECAR_PATH,
        LIVE_PROFILE_PATH,
        DEFAULT_CORRIDOR_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required National_Test assets are missing: " + ", ".join(missing))
    validate_checkpoint_manifest(checkpoint)


class FixedMapNavigationService:
    """Own one control loop and write only the NavigationStatus payload."""

    def __init__(
        self,
        ros_ctrl,
        global_data,
        *,
        policy_mode: PolicyMode = PolicyMode.LIVE,
        checkpoint_path: Optional[Path] = None,
        single_episode: bool = True,
        self_training: bool = False,
        validate_only: bool = False,
    ) -> None:
        if type(single_episode) is not bool:
            raise ValueError("single_episode must be boolean")
        if type(self_training) is not bool or type(validate_only) is not bool:
            raise ValueError("training flags must be boolean")
        if self_training and validate_only:
            raise ValueError("self-training and validation are mutually exclusive")
        self.ros_ctrl = ros_ctrl
        self.global_data = global_data
        self.policy_mode = PolicyMode(policy_mode)
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.single_episode = single_episode
        self.self_training = self_training
        self.validate_only = validate_only
        self._stop = threading.Event()
        self._telemetry_path: Optional[Path] = None

    def request_stop(self) -> None:
        self._stop.set()
        self._publish_zero()

    def _publish_zero(self, *, point_index: int = 0, distance: float = 0.0) -> None:
        self.global_data.updateThrottleRudderOutput(
            0,
            0,
            0.0,
            point_index,
            distance if math.isfinite(distance) else 0.0,
        )

    def _publish_decision(self, decision: RuntimeDecision) -> tuple[int, int]:
        if decision.stop:
            self._publish_zero(
                point_index=decision.mission_index,
                distance=decision.distance_to_goal_m,
            )
            return 0, 0
        assert decision.control is not None
        throttle = int(round(decision.control.throttle * 100.0))
        rudder = int(round(decision.control.rudder * 100.0))
        self.global_data.updateThrottleRudderOutput(
            throttle,
            rudder,
            decision.advised_heading_deg,
            decision.mission_index,
            decision.distance_to_goal_m,
        )
        return throttle, rudder

    def _request_unity_reset(
        self,
        *,
        timeout_s: float = UNITY_RESET_TIMEOUT_S,
    ) -> bool:
        initial_request_time = float(
            getattr(
                self.global_data.device_data,
                "reset_request_time",
                0.0,
            )
            or 0.0
        )
        self._publish_zero()
        if not self.ros_ctrl.reset_unity():
            return False
        deadline = time.monotonic() + timeout_s
        observed_request = False
        while not self._stop.is_set() and time.monotonic() < deadline:
            device = self.global_data.device_data
            reset_status = int(getattr(device, "reset_status", 0) or 0)
            request_time = float(
                getattr(device, "reset_request_time", 0.0) or 0.0
            )
            if reset_status == 1 or request_time != initial_request_time:
                observed_request = True
            if observed_request and reset_status == 2:
                return True
            self._publish_zero()
            self._stop.wait(0.05)
        return False

    def _request_unity_episode_start(
        self,
        *,
        timeout_s: float = UNITY_RESET_TIMEOUT_S,
    ) -> bool:
        self._publish_zero()
        device = self.global_data.device_data
        if int(getattr(device, "work_model", 0) or 0) != 2:
            if not self.ros_ctrl.set_auto_work():
                return False
        deadline = time.monotonic() + timeout_s
        while not self._stop.is_set() and time.monotonic() < deadline:
            device = self.global_data.device_data
            if int(getattr(device, "work_model", 0) or 0) == 2:
                break
            self._publish_zero()
            self._stop.wait(0.05)
        else:
            return False

        if int(getattr(self.global_data.device_data, "task_status", 0) or 0) != 0:
            return True

        self.ros_ctrl.set_task()
        while not self._stop.is_set() and time.monotonic() < deadline:
            device = self.global_data.device_data
            if (
                int(getattr(device, "work_model", 0) or 0) == 2
                and int(getattr(device, "task_status", 0) or 0) != 0
            ):
                return True
            self._publish_zero()
            self._stop.wait(0.05)
        return False

    def _algorithm_status(self, *, episode: int, step: int, running: bool) -> None:
        self.global_data.updateAlgorithmOutput(
            episode,
            step,
            0,
            0.0,
            MAX_TRAINING_EPISODES,
            2 if running else 0,
        )

    def _wait_for_pose(self):
        while not self._stop.is_set():
            if int(getattr(self.global_data.device_data, "task_status", 0) or 0) == 0:
                self._publish_zero()
                self._stop.wait(0.05)
                continue
            pose = getattr(self.global_data.scada_data, "pose", None)
            if pose is not None:
                latitude = float(getattr(pose, "lat", 0.0) or 0.0)
                longitude = float(getattr(pose, "lng", 0.0) or 0.0)
                if math.isfinite(latitude) and math.isfinite(longitude) and (
                    abs(latitude) > 1e-9 and abs(longitude) > 1e-9
                ):
                    return pose
            self._publish_zero()
            self._stop.wait(0.05)
        return None

    @staticmethod
    def _feedback(global_data) -> dict[str, object]:
        device = global_data.device_data
        return {
            "work_model": int(getattr(device, "work_model", 0) or 0),
            "task_status": int(getattr(device, "task_status", 0) or 0),
            "throttle_percent": int(
                round(float(getattr(device, "throttle_percent", 0.0) or 0.0))
            ),
            "rudder_percent": int(
                round(float(getattr(device, "rudder_percent", 0.0) or 0.0))
            ),
        }

    def _open_telemetry(self) -> Path:
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = TELEMETRY_DIR / f"national-test-runtime-{stamp}.jsonl"
        if path.exists():
            raise FileExistsError(path)
        path.touch()
        self._telemetry_path = path
        return path

    def _record(
        self,
        path: Path,
        *,
        event: str,
        episode: int,
        step: int,
        cycle_ms: float,
        decision: Optional[RuntimeDecision] = None,
        command: tuple[int, int] = (0, 0),
        detail: Optional[str] = None,
        progress: Optional[dict[str, object]] = None,
    ) -> None:
        row: dict[str, object] = {
            "schema_version": "national-test-runtime-telemetry-v2",
            "event": event,
            "wall_time_s": time.time(),
            "episode": episode,
            "step": step,
            "cycle_ms": cycle_ms,
            "command": {
                "navigation_throttle_percent": command[0],
                "navigation_rudder_percent": command[1],
            },
            "device_feedback": self._feedback(self.global_data),
        }
        if detail is not None:
            row["detail"] = detail
        if progress is not None:
            row["progress"] = progress
        if decision is not None:
            row.update(
                {
                    "reason": decision.reason,
                    "mission_index": decision.mission_index,
                    "distance_to_goal_m": decision.distance_to_goal_m,
                    "policy_action": decision.policy_action,
                    "executed_action": decision.action,
                    "safe_action_mask": list(decision.safe_mask),
                    "reachability_mask": list(decision.reachability_mask),
                    "candidate_reasons": list(decision.candidate_reasons),
                    "candidate_clearances_m": list(
                        decision.candidate_clearances_m
                    ),
                    "safety_intervened": decision.safety_intervened,
                    "safety_truncated": decision.safety_truncated,
                    "completed": decision.completed,
                }
            )
            if decision.observation is not None:
                observation = decision.observation
                row["observation"] = {
                    "speed_mps": observation.speed_mps,
                    "yaw_rate_rad_s": observation.yaw_rate_rad_s,
                    "actual_throttle": observation.actual_throttle,
                    "actual_rudder": observation.actual_rudder,
                    "corridor_cross_track_m": (
                        observation.corridor_cross_track_m
                    ),
                    "corridor_heading_error_rad": (
                        observation.corridor_heading_error_rad
                    ),
                    "corridor_progress": observation.corridor_progress,
                    "map_clearance_m": observation.map_clearance_m,
                }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _run_episode(
        self,
        episode: int,
        telemetry: Path,
        *,
        policy=None,
        deterministic_policy: Optional[bool] = None,
    ) -> tuple[str, tuple]:
        if self._wait_for_pose() is None:
            return "SERVICE_STOPPED", ()
        context = build_fixed_route_context(
            session_id=f"national-test-live-{episode}-{time.time_ns()}",
        )
        active_policy = policy or load_policy(
            self.checkpoint_path,
            context,
            self.policy_mode,
        )
        deterministic = (
            self.validate_only or not self.self_training
            if deterministic_policy is None
            else deterministic_policy
        )
        if type(deterministic) is not bool:
            raise ValueError("deterministic policy flag must be boolean")
        core = FixedMapControllerCore(
            context,
            active_policy,
            deterministic_policy=deterministic,
        )
        adapter = LiveInputAdapter(self.global_data, context)
        recorder = UnityTransitionRecorder()
        episode_started = time.monotonic()
        step = 0
        inactive_task_cycles = 0
        while not self._stop.is_set():
            cycle_started = time.perf_counter()
            if time.monotonic() - episode_started >= MAX_EPISODE_SECONDS:
                recorder.truncate(
                    core.mission_index,
                    reason="TIME_LIMIT",
                )
                self._publish_zero(point_index=core.mission_index)
                return "TIME_LIMIT", recorder.transitions
            sample = adapter.build()
            if sample.task_status == 0:
                inactive_task_cycles += 1
                self._publish_zero(
                    point_index=core.mission_index,
                    distance=core._distance(sample.vessel_state),
                )
                if inactive_task_cycles < 20:
                    remaining = CONTROL_PERIOD_S - (
                        time.perf_counter() - cycle_started
                    )
                    if remaining > 0.0:
                        self._stop.wait(remaining)
                    continue
                recorder.operator_stop(core.mission_index)
                self._record(
                    telemetry,
                    event="operator_truncated",
                    episode=episode,
                    step=step,
                    cycle_ms=(time.perf_counter() - cycle_started) * 1000.0,
                )
                return "OPERATOR_TRUNCATED", recorder.transitions
            inactive_task_cycles = 0
            try:
                decision = core.step(sample)
            except Exception as exc:
                self._publish_zero(point_index=core.mission_index)
                self._record(
                    telemetry,
                    event="controller_exception",
                    episode=episode,
                    step=step,
                    cycle_ms=(time.perf_counter() - cycle_started) * 1000.0,
                    detail=f"{type(exc).__name__}:{exc}",
                )
                return "CONTROLLER_EXCEPTION", ()
            recorder.observe(decision)
            if decision.reason in ENVIRONMENT_STOP_REASONS:
                recorder.truncate(
                    core.mission_index,
                    reason="INPUT_STALE",
                )
                self._publish_zero(point_index=core.mission_index)
                return "INPUT_STALE", recorder.transitions
            command = self._publish_decision(decision)
            elapsed_ms = (time.perf_counter() - cycle_started) * 1000.0
            self._record(
                telemetry,
                event="control_cycle",
                episode=episode,
                step=step,
                cycle_ms=elapsed_ms,
                decision=decision,
                command=command,
            )
            self._algorithm_status(episode=episode, step=step, running=True)
            step += 1
            if elapsed_ms > CONTROL_PERIOD_S * 1_000.0:
                self._publish_zero(point_index=core.mission_index)
                return "CONTROL_DEADLINE_MISSED", ()
            if decision.completed:
                self._publish_zero(point_index=13)
                return "MISSION_COMPLETE", recorder.transitions
            if decision.reason in POLICY_FAILURE_REASONS:
                self._publish_zero(point_index=decision.mission_index)
                return decision.reason, recorder.transitions
            remaining = CONTROL_PERIOD_S - (time.perf_counter() - cycle_started)
            if remaining > 0.0:
                self._stop.wait(remaining)
        recorder.operator_stop(core.mission_index)
        return "SERVICE_STOPPED", recorder.transitions

    def run(self) -> None:
        telemetry = self._open_telemetry()
        episode = 0
        unity_gate = None
        try:
            if self.self_training:
                manifest_path = self.checkpoint_path.with_suffix(
                    self.checkpoint_path.suffix + ".json"
                )
                checkpoint_stage = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ).get("stage")
                if checkpoint_stage == SelfTrainingStage.PROMOTED.value:
                    self.self_training = False
                    self.validate_only = True
                else:
                    unity_gate = UnityTrainingGate(
                        state_store=TrainingStateStore(TRAINING_STATE_PATH),
                        checkpoint_dir=DEFAULT_CHECKPOINT.parent,
                    )
            while not self._stop.is_set():
                if unity_gate is not None and unity_gate.cursor.stage not in {
                    SelfTrainingStage.UNITY_ADAPT,
                    SelfTrainingStage.UNITY_VALIDATION,
                }:
                    return
                self._publish_zero()
                if int(
                    getattr(self.global_data.device_data, "task_status", 0) or 0
                ) == 0:
                    self._algorithm_status(episode=episode, step=0, running=False)
                    self._stop.wait(0.05)
                    continue
                episode += 1
                outcome, transitions = self._run_episode(
                    episode,
                    telemetry,
                    policy=(
                        unity_gate.trainer.sac
                        if unity_gate is not None
                        else None
                    ),
                    deterministic_policy=(
                        unity_gate.deterministic
                        if unity_gate is not None
                        else None
                    ),
                )
                stage_detail = outcome
                reset_required = False
                progress = None
                if unity_gate is not None:
                    episode_stage = unity_gate.cursor.stage
                    adapt_episodes = unity_gate.cursor.unity_adapt_episodes
                    validation_episodes = (
                        unity_gate.cursor.unity_validation_episodes
                    )
                    validation_passes = (
                        unity_gate.cursor.unity_validation_passes
                    )
                    counted = outcome not in {
                        "CONTROLLER_EXCEPTION",
                        "CONTROL_DEADLINE_MISSED",
                        "INPUT_STALE",
                        "OPERATOR_TRUNCATED",
                        "SERVICE_STOPPED",
                    }
                    if counted and episode_stage is SelfTrainingStage.UNITY_ADAPT:
                        adapt_episodes += 1
                    elif (
                        counted
                        and episode_stage is SelfTrainingStage.UNITY_VALIDATION
                    ):
                        validation_episodes += 1
                        validation_passes += int(outcome == "MISSION_COMPLETE")
                    cursor = unity_gate.finish_episode(
                        transitions,
                        counted=counted,
                        passed=outcome == "MISSION_COMPLETE",
                        operator_truncated=(
                            outcome == "OPERATOR_TRUNCATED"
                        ),
                    )
                    reset_required = counted and (
                        cursor.stage is not SelfTrainingStage.PROMOTED
                    )
                    if cursor.active_checkpoint is not None:
                        self.checkpoint_path = (
                            DEFAULT_CHECKPOINT.parent
                            / cursor.active_checkpoint
                        )
                    stage_detail = f"{outcome}:{episode_stage.value}"
                    progress = {
                        "outcome": outcome,
                        "stage": episode_stage.value,
                        "next_stage": cursor.stage.value,
                        "unity_adapt_episodes": adapt_episodes,
                        "unity_validation_episodes": validation_episodes,
                        "unity_validation_passes": validation_passes,
                    }
                    diagnostics = unity_gate.last_training_diagnostics
                    if diagnostics is not None:
                        progress["training"] = {
                            "attempted_updates": (
                                diagnostics.attempted_updates
                            ),
                            "applied_updates": diagnostics.applied_updates,
                            "critic_loss": diagnostics.critic_loss,
                            "actor_loss": diagnostics.actor_loss,
                            "alpha": diagnostics.alpha,
                            "entropy": diagnostics.entropy,
                        }
                    print(
                        json.dumps(
                            progress,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                self._record(
                    telemetry,
                    event="episode_end",
                    episode=episode,
                    step=0,
                    cycle_ms=0.0,
                    detail=stage_detail,
                    progress=progress,
                )
                self._publish_zero()
                episode_ready = False
                if reset_required and int(
                    getattr(self.global_data.device_data, "task_status", 0) or 0
                ) == 0:
                    reset_required = False
                if reset_required:
                    reset_ok = self._request_unity_reset()
                    self._record(
                        telemetry,
                        event=(
                            "unity_reset_complete"
                            if reset_ok
                            else "unity_reset_failed"
                        ),
                        episode=episode,
                        step=0,
                        cycle_ms=0.0,
                    )
                    if not reset_ok:
                        return
                    episode_ready = self._request_unity_episode_start()
                    self._record(
                        telemetry,
                        event=(
                            "unity_episode_ready"
                            if episode_ready
                            else "unity_retrigger_failed"
                        ),
                        episode=episode,
                        step=0,
                        cycle_ms=0.0,
                    )
                    if not episode_ready:
                        return
                if self.single_episode and not episode_ready:
                    while not self._stop.is_set() and int(
                        getattr(self.global_data.device_data, "task_status", 0) or 0
                    ) != 0:
                        self._publish_zero()
                        self._stop.wait(0.05)
        except Exception as exc:
            self._publish_zero()
            self._record(
                telemetry,
                event="service_exception",
                episode=episode,
                step=0,
                cycle_ms=0.0,
                detail=f"{type(exc).__name__}:{exc}",
            )
            LogUtil.error(f"National_Test service stopped: {type(exc).__name__}: {exc}")
        finally:
            self._publish_zero()
            self._algorithm_status(episode=episode, step=0, running=False)


__all__ = [
    "CONTROL_PERIOD_S",
    "FixedMapNavigationService",
    "preflight_assets",
]
