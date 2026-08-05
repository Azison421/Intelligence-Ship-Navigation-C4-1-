"""ROS lifecycle adapter for the planning-free National_Test controller."""

from __future__ import annotations

import hashlib
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
from usvlib4ros.usvRosUtil import LogUtil


CONTROL_PERIOD_S = 0.1
MAX_TRAINING_EPISODES = 1_000
TELEMETRY_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "logs"


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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "national-test-sac-checkpoint-v6":
        raise ValueError("only national-test-sac-checkpoint-v6 is supported")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if manifest.get("checkpoint_sha256") != digest:
        raise ValueError("V6 checkpoint hash is invalid")


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
    ) -> None:
        row: dict[str, object] = {
            "schema_version": "national-test-runtime-telemetry-v1",
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
                    "safety_intervened": decision.safety_intervened,
                    "safety_truncated": decision.safety_truncated,
                    "completed": decision.completed,
                }
            )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _run_episode(self, episode: int, telemetry: Path) -> str:
        if self._wait_for_pose() is None:
            return "SERVICE_STOPPED"
        context = build_fixed_route_context(
            session_id=f"national-test-live-{episode}-{time.time_ns()}",
        )
        policy = load_policy(self.checkpoint_path, context, self.policy_mode)
        core = FixedMapControllerCore(
            context,
            policy,
            deterministic_policy=self.validate_only or not self.self_training,
        )
        adapter = LiveInputAdapter(self.global_data, context)
        step = 0
        while not self._stop.is_set():
            cycle_started = time.perf_counter()
            sample = adapter.build()
            if sample.task_status == 0:
                self._publish_zero(
                    point_index=core.mission_index,
                    distance=core._distance(sample.vessel_state),
                )
                self._record(
                    telemetry,
                    event="operator_truncated",
                    episode=episode,
                    step=step,
                    cycle_ms=(time.perf_counter() - cycle_started) * 1000.0,
                )
                return "OPERATOR_TRUNCATED"
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
                return "CONTROLLER_EXCEPTION"
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
            if decision.completed:
                self._publish_zero(point_index=13)
                return "MISSION_COMPLETE"
            if decision.safety_truncated:
                self._publish_zero(point_index=decision.mission_index)
                return "NO_SAFE_ACTION_TRUNCATED"
            remaining = CONTROL_PERIOD_S - (time.perf_counter() - cycle_started)
            if remaining > 0.0:
                self._stop.wait(remaining)
        return "SERVICE_STOPPED"

    def run(self) -> None:
        telemetry = self._open_telemetry()
        episode = 0
        try:
            while not self._stop.is_set():
                self._publish_zero()
                if int(
                    getattr(self.global_data.device_data, "task_status", 0) or 0
                ) == 0:
                    self._algorithm_status(episode=episode, step=0, running=False)
                    self._stop.wait(0.05)
                    continue
                episode += 1
                outcome = self._run_episode(episode, telemetry)
                self._record(
                    telemetry,
                    event="episode_end",
                    episode=episode,
                    step=0,
                    cycle_ms=0.0,
                    detail=outcome,
                )
                self._publish_zero()
                if self.single_episode:
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
