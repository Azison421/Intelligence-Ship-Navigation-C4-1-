"""Planning-free runtime for the fixed National_Test route."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from usvlib4ros.mapping import (
    CompiledSidecarMap,
    GpsProjector,
    compass_yaw_deg_to_math_yaw_rad,
    compass_yaw_rate_degs_to_math_rad_s,
    enu_to_grid,
    load_sidecar_artifact,
    math_yaw_rad_to_compass_deg,
)
from usvlib4ros.navigation.fixed_corridor import (
    DEFAULT_CORRIDOR_PATH,
    FrozenRouteCorridor,
)
from usvlib4ros.navigation.waypoint_control import (
    ACTION_SCHEMA_V3,
    CHECKPOINT_SCHEMA_V6,
    OBSERVATION_SCHEMA_V3,
    ActuatorTransitionGuard,
    NoSafeActionWindow,
    combine_action_masks,
)
from usvlib4ros.planning import Control, PrototypeReducedDynamics, VesselState
from usvlib4ros.planning.fixed_route import (
    SIDECAR_PATH,
    compile_offline_national_map,
    fixed_route_waypoint_reached,
)
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    action_protocol_hash,
    forward_control_profile_from_dict,
    reduced_dynamics_from_profile,
)
from usvlib4ros.policy.checkpoint_promotion import PolicyMode
from usvlib4ros.policy.fixed_map_features import front_arc_laser_features
from usvlib4ros.policy.recurrent_sac import (
    LocalWaypointObservationV3,
    RecurrentDiscreteSAC,
    RecurrentHiddenState,
)
from usvlib4ros.policy.safety_supervisor import (
    CandidateControl,
    FIXED_MAP_PREDICTION_HORIZON_S,
    PredictiveSafetySupervisor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "national_test_sac_checkpoint_v6.pt"
)
DEFAULT_UNITY_TEST_CHECKPOINT = DEFAULT_CHECKPOINT
POSE_MAX_AGE_S = 0.5
SCAN_MAX_AGE_S = 1.0
DEVICE_MAX_AGE_S = 1.0
REQUIRED_MAP_CLEARANCE_M = 0.0
LASER_EMERGENCY_DISTANCE_M = 0.0
MOTION_STALL_CYCLES = 30
MOTION_STALL_DISTANCE_M = 0.05
MOTION_STALL_SPEED_MPS = 0.03


def laser_emergency_distance_m(snapshot, state: VesselState) -> float:
    """Stop only on an explicit zero-range laser contact."""

    del snapshot, state
    return LASER_EMERGENCY_DISTANCE_M


@dataclass(frozen=True)
class FixedRouteContext:
    compiled_map: CompiledSidecarMap
    projector: GpsProjector
    start_index: int
    corridor: FrozenRouteCorridor


@dataclass(frozen=True)
class RuntimeInput:
    vessel_state: VesselState
    laser_ranges: tuple[float, ...]
    laser_valid_mask: tuple[bool, ...]
    pose_age_s: float
    scan_age_s: float
    device_age_s: float
    work_model: int
    task_status: int


@dataclass(frozen=True)
class RuntimeTrainingTrace:
    observation: LocalWaypointObservationV3
    policy_action: int
    executed_action: int
    safe_action_mask: tuple[bool, ...]
    reachability_mask: tuple[bool, ...]
    final_control: Control
    mission_index: int
    distance_to_goal_m: float
    cross_track_error_m: float
    map_clearance_m: float
    safety_intervened: bool


@dataclass(frozen=True)
class RuntimeDecision:
    reason: str
    control: Optional[Control]
    action: Optional[int]
    policy_action: Optional[int]
    mission_index: int
    distance_to_goal_m: float
    advised_heading_deg: float
    safe_mask: tuple[bool, ...]
    reachability_mask: tuple[bool, ...]
    completed: bool
    safety_intervened: bool
    safety_truncated: bool
    observation: Optional[LocalWaypointObservationV3] = None
    training_trace: Optional[RuntimeTrainingTrace] = None
    candidate_reasons: tuple[str, ...] = ("UNAVAILABLE",) * 5
    candidate_clearances_m: tuple[float, ...] = (0.0,) * 5

    @property
    def stop(self) -> bool:
        return self.control is None


def build_fixed_route_context(
    *,
    session_id: str,
    start_index: int = 0,
    stamp_sim: float = 0.0,
) -> FixedRouteContext:
    """Build the sole hash-bound route context for National_Test."""

    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or not 0 <= start_index < 13
    ):
        raise ValueError("fixed route start index is invalid")
    compiled = compile_offline_national_map(
        session_id=session_id,
        stamp_sim=stamp_sim,
        required_clearance_m=REQUIRED_MAP_CLEARANCE_M,
    )
    artifact, _ = load_sidecar_artifact(SIDECAR_PATH)
    anchors = artifact["gps_anchors"]
    return FixedRouteContext(
        compiled_map=compiled,
        projector=GpsProjector(
            float(anchors["latitude1"]),
            float(anchors["longitude1"]),
        ),
        start_index=start_index,
        corridor=FrozenRouteCorridor.load(DEFAULT_CORRIDOR_PATH, compiled),
    )


def load_policy(
    checkpoint_path: Path,
    context: FixedRouteContext,
    policy_mode: PolicyMode,
) -> RecurrentDiscreteSAC:
    """Load only the current V6 checkpoint; older schemas are invalid."""

    checkpoint = Path(checkpoint_path)
    manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("V6 SAC checkpoint or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_V6:
        raise ValueError("only national-test-sac-checkpoint-v6 is supported")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest.get(
        "checkpoint_sha256"
    ):
        raise ValueError("V6 checkpoint hash is invalid")
    expected = {
        "observation_schema": OBSERVATION_SCHEMA_V3,
        "observation_dim": 166,
        "action_schema": ACTION_SCHEMA_V3,
        "action_dim": 5,
        "replay_schema": "national-test-replay-v3",
        "route_id": context.compiled_map.manifest.route_id,
        "map_payload_hash": context.compiled_map.snapshot.payload_content_hash,
        "corridor_sha256": context.corridor.corridor_hash,
        "required_clearance_m": REQUIRED_MAP_CLEARANCE_M,
        "laser_emergency_distance_m": LASER_EMERGENCY_DISTANCE_M,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"V6 checkpoint {key} is incompatible")
    initialization = manifest.get("initialization")
    if (
        not isinstance(initialization, dict)
        or set(initialization) != {"type", "seed", "inherited_checkpoint"}
        or initialization.get("type") != "random"
        or initialization.get("inherited_checkpoint") is not None
        or isinstance(initialization.get("seed"), bool)
        or not isinstance(initialization.get("seed"), int)
    ):
        raise ValueError("V6 checkpoint must start from random initialization")
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("status") != "verified":
        raise ValueError("two-sided Unity control calibration is not verified")
    stage = manifest.get("stage")
    mode = PolicyMode(policy_mode)
    allowed = {
        PolicyMode.LIVE: {"PROMOTED"},
        PolicyMode.OFFLINE_VALIDATION: {
            "OFFLINE_EVAL",
            "UNITY_ADAPT",
            "UNITY_VALIDATION",
            "PROMOTED",
        },
        PolicyMode.UNITY_TEST: {
            "UNITY_DIAGNOSTIC",
            "UNITY_ADAPT",
            "UNITY_VALIDATION",
            "PROMOTED",
        },
    }
    if stage not in allowed[mode]:
        raise ValueError(f"V6 checkpoint stage {stage!r} is not valid for {mode.value}")
    control_profile = manifest.get("forward_control_profile")
    if not isinstance(control_profile, dict):
        raise ValueError("V6 checkpoint control profile is missing")
    profile = forward_control_profile_from_dict(control_profile)
    if (
        calibration.get("calibration_hash") != profile.calibration_hash
        or calibration.get("action_protocol_hash") != action_protocol_hash(profile)
    ):
        raise ValueError("V6 checkpoint calibration identity is invalid")
    hidden_dim = manifest.get("hidden_dim")
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("V6 checkpoint hidden dimension is invalid")
    policy = RecurrentDiscreteSAC(
        hidden_dim=hidden_dim,
        seed=initialization["seed"],
    )
    policy.load_checkpoint(checkpoint)
    policy.forward_control_profile = profile
    policy.reduced_dynamics = reduced_dynamics_from_profile(profile)
    return policy


class LiveInputAdapter:
    """Convert replaced GlobalData objects into one fresh runtime sample."""

    def __init__(self, global_data, context: FixedRouteContext) -> None:
        self._data = global_data
        self._context = context
        self._started = time.monotonic()
        self._pose_object = None
        self._laser_object = None
        self._device_object = None
        self._pose_changed = 0.0
        self._laser_changed = 0.0
        self._device_changed = 0.0

    @staticmethod
    def _age(changed: float, now: float) -> float:
        return float("inf") if changed <= 0.0 else now - changed

    def build(self) -> RuntimeInput:
        now = time.monotonic()
        scada = self._data.scada_data
        laser = self._data.laser_data
        device = self._data.device_data
        if scada is not self._pose_object:
            self._pose_object = scada
            self._pose_changed = now
        if laser is not self._laser_object:
            self._laser_object = laser
            self._laser_changed = now
        if device is not self._device_object:
            self._device_object = device
            self._device_changed = now

        pose = getattr(scada, "pose", None)
        if pose is None:
            state = VesselState(
                x=float("nan"),
                y=float("nan"),
                yaw=float("nan"),
                speed=float("nan"),
                yaw_rate=float("nan"),
                stamp_sim=now - self._started,
                health="no-pose",
            )
        else:
            x_enu, y_enu = self._context.projector.gps_to_enu(
                float(getattr(pose, "lat", 0.0) or 0.0),
                float(getattr(pose, "lng", 0.0) or 0.0),
            )
            x, y = enu_to_grid(
                self._context.compiled_map.manifest,
                x_enu,
                y_enu,
            )
            state = VesselState(
                x=x,
                y=y,
                yaw=compass_yaw_deg_to_math_yaw_rad(
                    float(getattr(pose, "yaw", 0.0) or 0.0)
                ),
                speed=float(getattr(pose, "speed", 0.0) or 0.0),
                yaw_rate=compass_yaw_rate_degs_to_math_rad_s(
                    float(getattr(pose, "rotate_speed", 0.0) or 0.0)
                ),
                throttle_state=max(
                    -1.0,
                    min(
                        1.0,
                        float(getattr(device, "throttle_percent", 0.0) or 0.0)
                        / 100.0,
                    ),
                ),
                rudder_state=max(
                    -1.0,
                    min(
                        1.0,
                        float(getattr(device, "rudder_percent", 0.0) or 0.0)
                        / 100.0,
                    ),
                ),
                stamp_sim=now - self._started,
            )
        ranges, valid = front_arc_laser_features(
            getattr(laser, "ranges", ()) or ()
        )
        return RuntimeInput(
            vessel_state=state,
            laser_ranges=ranges,
            laser_valid_mask=valid,
            pose_age_s=self._age(self._pose_changed, now),
            scan_age_s=self._age(self._laser_changed, now),
            device_age_s=self._age(self._device_changed, now),
            work_model=int(getattr(device, "work_model", 0) or 0),
            task_status=int(getattr(device, "task_status", 0) or 0),
        )


class FixedMapControllerCore:
    """The one runtime/training state transition engine."""

    def __init__(
        self,
        context: FixedRouteContext,
        policy: RecurrentDiscreteSAC,
        *,
        dynamics: Optional[PrototypeReducedDynamics] = None,
        deterministic_policy: bool = True,
    ) -> None:
        if not isinstance(context, FixedRouteContext):
            raise ValueError("controller requires a National_Test context")
        if type(deterministic_policy) is not bool:
            raise ValueError("deterministic_policy must be boolean")
        profile = getattr(policy, "forward_control_profile", None)
        if not isinstance(profile, ForwardControlProfile):
            raise ValueError("SAC policy is missing its calibrated control profile")
        if getattr(policy, "action_schema", None) != ACTION_SCHEMA_V3:
            raise ValueError("SAC policy action schema is incompatible")
        self.context = context
        self.policy = policy
        self.deterministic_policy = deterministic_policy
        self.dynamics = dynamics or reduced_dynamics_from_profile(profile)
        self.controls = profile.action_controls
        self.candidates = tuple(
            CandidateControl(action=index, control=control)
            for index, control in enumerate(self.controls)
        )
        self.supervisor = PredictiveSafetySupervisor(
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
            max_state_age_s=1.0,
        )
        self.mission_index = max(0, min(context.start_index, 12))
        self.hidden: Optional[RecurrentHiddenState] = None
        self.hidden_reset = True
        self.corridor_progress = 0.0
        self.transition_guard = ActuatorTransitionGuard()
        self.no_safe_actions = NoSafeActionWindow(limit=10)
        self._last_commanded_throttle = 0.0
        self._stall_origin_xy: Optional[tuple[float, float]] = None
        self._stall_cycles = 0

    def _goal_xy(self) -> tuple[float, float]:
        if self.mission_index >= 13:
            return self.context.corridor.task_points[-1]
        return self.context.corridor.task_points[self.mission_index]

    def _distance(self, state: VesselState) -> float:
        if not state.is_finite():
            return float("inf")
        return math.dist((state.x, state.y), self._goal_xy())

    def _stop(
        self,
        reason: str,
        state: VesselState,
        *,
        completed: bool = False,
        safe_mask: tuple[bool, ...] = (False,) * 5,
        reachability_mask: tuple[bool, ...] = (False,) * 5,
        policy_action: Optional[int] = None,
        safety_intervened: bool = False,
        safety_truncated: bool = False,
        observation: Optional[LocalWaypointObservationV3] = None,
        candidate_reasons: tuple[str, ...] = ("UNAVAILABLE",) * 5,
        candidate_clearances_m: tuple[float, ...] = (0.0,) * 5,
    ) -> RuntimeDecision:
        self._last_commanded_throttle = 0.0
        return RuntimeDecision(
            reason=reason,
            control=None,
            action=None,
            policy_action=policy_action,
            mission_index=self.mission_index,
            distance_to_goal_m=self._distance(state),
            advised_heading_deg=(
                math_yaw_rad_to_compass_deg(state.yaw)
                if math.isfinite(state.yaw)
                else 0.0
            ),
            safe_mask=safe_mask,
            reachability_mask=reachability_mask,
            completed=completed,
            safety_intervened=safety_intervened,
            safety_truncated=safety_truncated,
            observation=observation,
            candidate_reasons=candidate_reasons,
            candidate_clearances_m=candidate_clearances_m,
        )

    def _motion_stalled(self, state: VesselState) -> bool:
        if (
            self._last_commanded_throttle <= 0.0
            or state.speed >= MOTION_STALL_SPEED_MPS
        ):
            self._stall_origin_xy = None
            self._stall_cycles = 0
            return False
        current = (state.x, state.y)
        if self._stall_origin_xy is None:
            self._stall_origin_xy = current
            self._stall_cycles = 1
            return False
        if math.dist(current, self._stall_origin_xy) >= MOTION_STALL_DISTANCE_M:
            self._stall_origin_xy = current
            self._stall_cycles = 0
            return False
        self._stall_cycles += 1
        return self._stall_cycles >= MOTION_STALL_CYCLES

    @staticmethod
    def _body_coordinates(
        state: VesselState,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        dx = point[0] - state.x
        dy = point[1] - state.y
        cosine = math.cos(state.yaw)
        sine = math.sin(state.yaw)
        return cosine * dx + sine * dy, -sine * dx + cosine * dy

    def _observation(
        self,
        sample: RuntimeInput,
        safe_mask: tuple[bool, ...],
        cross_track_error_m: float,
        heading_error_rad: float,
    ) -> LocalWaypointObservationV3:
        state = sample.vessel_state
        current = self._goal_xy()
        next_valid = self.mission_index < 12
        next_point = (
            self.context.corridor.task_points[self.mission_index + 1]
            if next_valid
            else current
        )
        return LocalWaypointObservationV3(
            laser_ranges=sample.laser_ranges,
            laser_valid_mask=sample.laser_valid_mask,
            scan_age_s=sample.scan_age_s,
            pose_age_s=sample.pose_age_s,
            device_age_s=sample.device_age_s,
            speed_mps=state.speed,
            yaw_rate_rad_s=state.yaw_rate,
            actual_throttle=state.throttle_state,
            actual_rudder=state.rudder_state,
            current_waypoint_body_xy=self._body_coordinates(state, current),
            next_waypoint_body_xy=self._body_coordinates(state, next_point),
            next_waypoint_valid=next_valid,
            mission_progress=min(1.0, self.mission_index / 12.0),
            corridor_cross_track_m=cross_track_error_m,
            corridor_heading_error_rad=heading_error_rad,
            corridor_progress=self.corridor_progress,
            map_clearance_m=self.context.compiled_map.snapshot.clearance_at(state),
            safe_action_mask=safe_mask,
            session_id=self.context.compiled_map.snapshot.session_id,
            stamp_sim=state.stamp_sim,
            hidden_reset=self.hidden_reset,
        )

    def step(self, sample: RuntimeInput) -> RuntimeDecision:
        if not isinstance(sample, RuntimeInput):
            raise ValueError("runtime sample type is invalid")
        state = sample.vessel_state
        if (
            not isinstance(state, VesselState)
            or len(sample.laser_ranges) != 72
            or len(sample.laser_valid_mask) != 72
            or any(type(value) is not bool for value in sample.laser_valid_mask)
        ):
            self.no_safe_actions.reset()
            return self._stop("INPUT_INVALID", state)
        if sample.task_status == 0:
            self.no_safe_actions.reset()
            return self._stop("TASK_INACTIVE", state)
        if sample.work_model != 2:
            self.no_safe_actions.reset()
            return self._stop("NOT_IN_AUTO_MODE", state)
        for age, limit, reason in (
            (sample.pose_age_s, POSE_MAX_AGE_S, "POSE_STALE"),
            (sample.scan_age_s, SCAN_MAX_AGE_S, "SCAN_STALE"),
            (sample.device_age_s, DEVICE_MAX_AGE_S, "DEVICE_STALE"),
        ):
            if not math.isfinite(age) or age > limit:
                self.no_safe_actions.reset()
                return self._stop(reason, state)
        if not self.dynamics.is_state_valid(state):
            self.no_safe_actions.reset()
            return self._stop("DYNAMICS_INVALID", state)
        snapshot = self.context.compiled_map.snapshot
        if not snapshot.is_state_valid(state):
            self.no_safe_actions.reset()
            return self._stop("MAP_INVALID", state)
        laser_stop_distance = laser_emergency_distance_m(snapshot, state)
        if any(
            valid
            and (
                not math.isfinite(float(distance))
                or float(distance) <= laser_stop_distance
            )
            for distance, valid in zip(
                sample.laser_ranges,
                sample.laser_valid_mask,
            )
        ):
            self.no_safe_actions.reset()
            return self._stop("LASER_EMERGENCY_STOP", state)

        while self.mission_index < 13 and fixed_route_waypoint_reached(
            self.context.compiled_map,
            self.mission_index,
            state,
        ):
            self.mission_index += 1
        if self.mission_index == 13:
            self.no_safe_actions.reset()
            projection = self.context.corridor.project(
                state,
                self.corridor_progress,
                12,
            )
            self.corridor_progress = projection.route_progress
            observation = self._observation(
                sample,
                (False,) * 5,
                projection.cross_track_error_m,
                projection.heading_error_rad,
            )
            return self._stop(
                "MISSION_COMPLETE",
                state,
                completed=True,
                observation=observation,
            )

        projection = self.context.corridor.project(
            state,
            self.corridor_progress,
            self.mission_index,
        )
        self.corridor_progress = projection.route_progress
        if self._motion_stalled(state):
            observation = self._observation(
                sample,
                (False,) * 5,
                projection.cross_track_error_m,
                projection.heading_error_rad,
            )
            return self._stop(
                "MOTION_STALLED",
                state,
                safety_intervened=True,
                safety_truncated=True,
                observation=observation,
            )
        reachability = self.transition_guard.reachability_mask()
        predictive_mask, reasons, clearances = self.supervisor.precheck(
            state,
            self.candidates,
            snapshot,
            self.dynamics,
            now_sim=state.stamp_sim,
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
        )
        safe_mask = combine_action_masks(reachability, predictive_mask)
        if not any(safe_mask):
            observation = self._observation(
                sample,
                safe_mask,
                projection.cross_track_error_m,
                projection.heading_error_rad,
            )
            truncated = self.no_safe_actions.observe(
                fresh_inputs=True,
                has_safe_action=False,
            )
            return self._stop(
                "NO_SAFE_ACTION_TRUNCATED" if truncated else "NO_SAFE_ACTION",
                state,
                safe_mask=safe_mask,
                reachability_mask=reachability,
                safety_intervened=True,
                safety_truncated=truncated,
                observation=observation,
                candidate_reasons=reasons,
                candidate_clearances_m=clearances,
            )
        self.no_safe_actions.observe(fresh_inputs=True, has_safe_action=True)
        observation = self._observation(
            sample,
            safe_mask,
            projection.cross_track_error_m,
            projection.heading_error_rad,
        )
        proposal, next_hidden = self.policy.act(
            observation,
            safe_mask,
            hidden=self.hidden,
            deterministic=self.deterministic_policy,
        )
        if proposal.action is None:
            self.no_safe_actions.reset()
            return self._stop(
                "POLICY_NO_ACTION",
                state,
                safe_mask=safe_mask,
                reachability_mask=reachability,
                safety_intervened=True,
                observation=observation,
            )
        final = self.supervisor.finalize(
            policy_action=proposal.action,
            candidate_mask=safe_mask,
            candidates=self.candidates,
            snapshot_id=snapshot.snapshot_id,
            current_snapshot_id=snapshot.snapshot_id,
            reasons=reasons,
            clearances=clearances,
            current_state=state,
            current_map_snapshot=snapshot,
            dynamics=self.dynamics,
            now_sim=state.stamp_sim,
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
        )
        if final.stop or final.final_action is None:
            stopped_observation = self._observation(
                sample,
                tuple(final.candidate_mask),
                projection.cross_track_error_m,
                projection.heading_error_rad,
            )
            truncated = self.no_safe_actions.observe(
                fresh_inputs=True,
                has_safe_action=False,
            )
            return self._stop(
                "NO_SAFE_ACTION_TRUNCATED" if truncated else "NO_SAFE_ACTION",
                state,
                safe_mask=tuple(final.candidate_mask),
                reachability_mask=reachability,
                policy_action=proposal.action,
                safety_intervened=True,
                safety_truncated=truncated,
                observation=stopped_observation,
                candidate_reasons=tuple(final.reasons),
                candidate_clearances_m=clearances,
            )

        self.transition_guard.record_executed(final.final_action)
        self._last_commanded_throttle = final.control.throttle
        self.hidden = next_hidden
        self.hidden_reset = False
        desired_yaw = state.yaw + projection.heading_error_rad
        intervened = bool(final.overridden) or proposal.action != final.final_action
        trace = RuntimeTrainingTrace(
            observation=observation,
            policy_action=proposal.action,
            executed_action=final.final_action,
            safe_action_mask=tuple(final.candidate_mask),
            reachability_mask=reachability,
            final_control=final.control,
            mission_index=self.mission_index,
            distance_to_goal_m=self._distance(state),
            cross_track_error_m=projection.cross_track_error_m,
            map_clearance_m=snapshot.clearance_at(state),
            safety_intervened=intervened,
        )
        return RuntimeDecision(
            reason=final.reason,
            control=final.control,
            action=final.final_action,
            policy_action=proposal.action,
            mission_index=self.mission_index,
            distance_to_goal_m=self._distance(state),
            advised_heading_deg=math_yaw_rad_to_compass_deg(desired_yaw),
            safe_mask=tuple(final.candidate_mask),
            reachability_mask=reachability,
            completed=False,
            safety_intervened=intervened,
            safety_truncated=False,
            observation=observation,
            training_trace=trace,
            candidate_reasons=tuple(final.reasons),
            candidate_clearances_m=clearances,
        )


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_UNITY_TEST_CHECKPOINT",
    "FixedRouteContext",
    "FixedMapControllerCore",
    "LASER_EMERGENCY_DISTANCE_M",
    "laser_emergency_distance_m",
    "MOTION_STALL_CYCLES",
    "LiveInputAdapter",
    "REQUIRED_MAP_CLEARANCE_M",
    "RuntimeDecision",
    "RuntimeInput",
    "RuntimeTrainingTrace",
    "build_fixed_route_context",
    "load_policy",
]
