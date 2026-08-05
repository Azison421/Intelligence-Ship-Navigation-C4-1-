"""Offline National_Test environment built on the live transition core."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from random import Random
from typing import Optional

import numpy as np

from usvlib4ros.navigation.fixed_map_runtime import (
    FixedMapControllerCore,
    LASER_EMERGENCY_DISTANCE_M,
    RuntimeDecision,
    RuntimeInput,
    RuntimeTrainingTrace,
    build_fixed_route_context,
)
from usvlib4ros.planning import Control, PrototypeReducedDynamics, VesselState
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    reduced_dynamics_from_profile,
)

from .recurrent_sac import (
    LocalWaypointObservationV3,
    RecurrentDiscreteSAC,
    SequenceReplay,
    SequenceTransition,
)
from .safety_supervisor import (
    FIXED_MAP_PREDICTION_HORIZON_S,
    CandidateControl,
    PredictiveSafetySupervisor,
)


CONTROL_PERIOD_S = 0.1
DEFAULT_MAX_STEPS = 2_500
LASER_MAX_RANGE_M = 20.0
LASER_RAYCAST_RANGE_M = 10.0
TERMINAL_REASONS = frozenset(
    {
        "DYNAMICS_INVALID",
        "MAP_INVALID",
        "LASER_EMERGENCY_STOP",
        "NO_SAFE_ACTION_TRUNCATED",
        "POLICY_NO_ACTION",
        "CONTROLLER_EXCEPTION",
    }
)


def control_transition_reward(
    observation: LocalWaypointObservationV3,
    next_observation: LocalWaypointObservationV3,
    control: Control,
    previous_control: Control,
    *,
    mission_delta: int,
    completed: bool,
    terminated: bool,
    truncated: bool,
    reason: str,
) -> float:
    """Reward actual control/progress, never the integer action label."""

    corridor_delta = (
        next_observation.corridor_progress - observation.corridor_progress
    )
    reward = 40.0 * corridor_delta + 8.0 * max(0, mission_delta)
    reward -= 0.03 * abs(control.throttle)
    reward -= 0.05 * abs(control.rudder)
    reward -= 0.08 * abs(control.throttle - previous_control.throttle)
    reward -= 0.12 * abs(control.rudder - previous_control.rudder)
    reward -= 0.1 * abs(next_observation.corridor_cross_track_m)
    if next_observation.map_clearance_m < 0.6:
        reward -= 0.5 * (0.6 - next_observation.map_clearance_m)
    if completed:
        reward += 250.0
    elif terminated:
        reward -= 150.0 if reason == "NO_SAFE_ACTION_TRUNCATED" else 250.0
    elif truncated and reason not in {"INPUT_STALE", "OPERATOR_TRUNCATED"}:
        reward -= 25.0
    return reward


@dataclass(frozen=True)
class EpisodeSummary:
    session_id: str
    total_reward: float
    steps: int
    completed_waypoints: int
    completed: bool
    collision: bool
    timed_out: bool
    no_safe_action: bool
    safety_interventions: int
    maximum_cross_track_m: float
    minimum_clearance_m: float
    end_reason: str

    @property
    def passed(self) -> bool:
        return (
            self.completed
            and not self.collision
            and not self.timed_out
            and not self.no_safe_action
            and self.completed_waypoints == 13
        )


@dataclass(frozen=True)
class OfflineEpisode:
    transitions: tuple[SequenceTransition, ...]
    summary: EpisodeSummary


@dataclass(frozen=True)
class _PendingAction:
    trace: RuntimeTrainingTrace
    previous_control: Control


class _StaticLaser:
    """Vectorized front-arc raycast over the immutable occupancy grid."""

    def __init__(self, snapshot) -> None:
        self._resolution = float(snapshot.resolution)
        self._blocked = np.asarray(
            [[cell in "#?" for cell in row] for row in snapshot.rows],
            dtype=bool,
        )
        self._distances = np.arange(
            self._resolution,
            LASER_RAYCAST_RANGE_M + self._resolution,
            self._resolution,
            dtype=np.float64,
        )
        self._angles = np.linspace(
            -math.pi / 2.0,
            math.pi / 2.0,
            72,
            dtype=np.float64,
        )

    def scan(self, state: VesselState) -> tuple[float, ...]:
        headings = state.yaw + self._angles
        x = state.x + np.cos(headings)[:, None] * self._distances[None, :]
        y = state.y + np.sin(headings)[:, None] * self._distances[None, :]
        x_index = np.floor(x / self._resolution).astype(np.int64)
        y_index = np.floor(y / self._resolution).astype(np.int64)
        in_bounds = (
            (x_index >= 0)
            & (x_index < self._blocked.shape[1])
            & (y_index >= 0)
            & (y_index < self._blocked.shape[0])
        )
        blocked = ~in_bounds
        valid_rows, valid_columns = np.nonzero(in_bounds)
        blocked[valid_rows, valid_columns] = self._blocked[
            y_index[valid_rows, valid_columns],
            x_index[valid_rows, valid_columns],
        ]
        has_hit = blocked.any(axis=1)
        first_hit = np.argmax(blocked, axis=1)
        ranges = np.full(72, LASER_MAX_RANGE_M, dtype=np.float64)
        ranges[has_hit] = self._distances[first_hit[has_hit]]
        return tuple(float(value) for value in ranges)


class FixedMapSACTrainer:
    """Run disturbed corridor episodes and train one V3 recurrent SAC."""

    def __init__(
        self,
        forward_profile: ForwardControlProfile,
        *,
        seed: int,
        hidden_dim: int = 64,
        sac: Optional[RecurrentDiscreteSAC] = None,
        replay_capacity: int = 64,
    ) -> None:
        if not isinstance(forward_profile, ForwardControlProfile):
            raise ValueError("trainer requires a verified forward profile")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("trainer seed must be an integer")
        self.forward_profile = forward_profile
        self.seed = seed
        self.rng = Random(seed)
        self.sac = sac or RecurrentDiscreteSAC(hidden_dim=hidden_dim, seed=seed)
        self.sac.forward_control_profile = forward_profile
        self.sac.reduced_dynamics = reduced_dynamics_from_profile(forward_profile)
        self.replay = SequenceReplay(capacity=replay_capacity, seed=seed)
        self.completed_training_episodes = 0
        self._context = build_fixed_route_context(
            session_id=f"offline-training-{seed}",
        )
        self._laser = _StaticLaser(self._context.compiled_map.snapshot)

    def _randomized_dynamics(self, episode: int) -> PrototypeReducedDynamics:
        base = reduced_dynamics_from_profile(self.forward_profile)
        speed_scale = self.rng.uniform(0.85, 1.15)
        positive_scale = self.rng.uniform(0.85, 1.15)
        negative_scale = self.rng.uniform(0.85, 1.15)
        return replace(
            base,
            version=f"national-test-randomized-{self.seed}-{episode}",
            throttle_speed_gain=base.throttle_speed_gain * speed_scale,
            rudder_yaw_rate_gain=base.rudder_yaw_rate_gain
            * min(positive_scale, negative_scale),
            positive_rudder_yaw_rate_gain=(
                base.positive_rudder_yaw_rate_gain * positive_scale
            ),
            negative_rudder_yaw_rate_gain=(
                base.negative_rudder_yaw_rate_gain * negative_scale
            ),
            speed_response=base.speed_response * self.rng.uniform(0.9, 1.1),
            yaw_response=base.yaw_response * self.rng.uniform(0.9, 1.1),
        )

    @staticmethod
    def _segment_progress(corridor, segment_index: int, fraction: float) -> float:
        prefix = sum(
            math.dist(start, end)
            for start, end in zip(
                corridor.polyline[:segment_index],
                corridor.polyline[1 : segment_index + 1],
            )
        )
        segment_length = math.dist(
            corridor.polyline[segment_index],
            corridor.polyline[segment_index + 1],
        )
        return (prefix + fraction * segment_length) / corridor.total_length_m

    @staticmethod
    def _mission_index(corridor, segment_index: int) -> int:
        target_polyline_index = segment_index + 1
        for mission_index, anchor_index in enumerate(
            corridor.anchor_polyline_indices
        ):
            if anchor_index >= target_polyline_index:
                return mission_index
        return 12

    def _initial_state(
        self,
        context,
        *,
        full_route: bool,
        dynamics: PrototypeReducedDynamics,
        laser: _StaticLaser,
    ) -> tuple[VesselState, float]:
        corridor = context.corridor
        candidates = tuple(
            CandidateControl(action=index, control=control)
            for index, control in enumerate(self.forward_profile.action_controls)
        )
        supervisor = PredictiveSafetySupervisor(
            prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
            max_state_age_s=1.0,
        )
        for _ in range(100):
            if full_route:
                segment_index = 0
                fraction = 0.0
            else:
                segment_index = self.rng.randrange(len(corridor.polyline) - 1)
                fraction = self.rng.uniform(0.05, 0.95)
            start = corridor.polyline[segment_index]
            end = corridor.polyline[segment_index + 1]
            heading = math.atan2(end[1] - start[1], end[0] - start[0])
            lateral = self.rng.uniform(-0.2, 0.2)
            x = start[0] + fraction * (end[0] - start[0]) - math.sin(heading) * lateral
            y = start[1] + fraction * (end[1] - start[1]) + math.cos(heading) * lateral
            state = VesselState(
                x=x,
                y=y,
                yaw=heading + self.rng.uniform(-0.35, 0.35),
                speed=self.rng.uniform(0.0, 0.25),
                yaw_rate=self.rng.uniform(-0.1, 0.1),
                throttle_state=0.0,
                rudder_state=0.0,
                stamp_sim=0.0,
            )
            if not context.compiled_map.snapshot.is_state_valid(state):
                continue
            ranges = laser.scan(state)
            if min(ranges) <= LASER_EMERGENCY_DISTANCE_M:
                continue
            safe_mask, _, _ = supervisor.precheck(
                state,
                candidates,
                context.compiled_map.snapshot,
                dynamics,
                now_sim=state.stamp_sim,
                prediction_horizon_s=FIXED_MAP_PREDICTION_HORIZON_S,
            )
            if any(safe_mask):
                return state, self._segment_progress(
                    corridor,
                    segment_index,
                    fraction,
                )
        raise RuntimeError("unable to sample a valid corridor start")

    @staticmethod
    def _runtime_input(state: VesselState, laser: _StaticLaser) -> RuntimeInput:
        ranges = laser.scan(state)
        return RuntimeInput(
            vessel_state=state,
            laser_ranges=ranges,
            laser_valid_mask=(True,) * 72,
            pose_age_s=0.0,
            scan_age_s=0.0,
            device_age_s=0.0,
            work_model=2,
            task_status=1,
        )

    @staticmethod
    def _terminal_observation(
        pending: _PendingAction,
        decision: RuntimeDecision,
    ) -> LocalWaypointObservationV3:
        return decision.observation or pending.trace.observation

    def _append_transition(
        self,
        transitions: list[SequenceTransition],
        pending: _PendingAction,
        decision: RuntimeDecision,
        *,
        terminated: bool,
        truncated: bool,
        reason: str,
    ) -> None:
        next_observation = self._terminal_observation(pending, decision)
        transitions.append(
            SequenceTransition(
                observation=pending.trace.observation,
                next_observation=next_observation,
                executed_action=pending.trace.executed_action,
                reward=control_transition_reward(
                    pending.trace.observation,
                    next_observation,
                    pending.trace.final_control,
                    pending.previous_control,
                    mission_delta=(
                        decision.mission_index - pending.trace.mission_index
                    ),
                    completed=decision.completed,
                    terminated=terminated,
                    truncated=truncated,
                    reason=reason,
                ),
                terminated=terminated,
                truncated=truncated,
                reason=reason,
            )
        )

    def run_episode(
        self,
        *,
        episode: int,
        training: bool,
        deterministic: bool,
        full_route: bool,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> OfflineEpisode:
        if type(training) is not bool or type(deterministic) is not bool:
            raise ValueError("episode mode flags must be boolean")
        if type(full_route) is not bool:
            raise ValueError("full_route must be boolean")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be positive")
        session_id = f"offline-{self.seed}-{episode}"
        provisional = self._context
        dynamics = self._randomized_dynamics(episode)
        state, corridor_progress = self._initial_state(
            provisional,
            full_route=full_route,
            dynamics=dynamics,
            laser=self._laser,
        )
        segment_index = 0
        if not full_route:
            cumulative = 0.0
            target = corridor_progress * provisional.corridor.total_length_m
            for index, (start, end) in enumerate(
                zip(
                    provisional.corridor.polyline,
                    provisional.corridor.polyline[1:],
                )
            ):
                cumulative += math.dist(start, end)
                if cumulative + 1e-9 >= target:
                    segment_index = index
                    break
        mission_index = 0 if full_route else self._mission_index(
            provisional.corridor,
            segment_index,
        )
        context = replace(provisional, start_index=mission_index)
        self.sac.forward_control_profile = self.forward_profile
        core = FixedMapControllerCore(
            context,
            self.sac,
            dynamics=dynamics,
            deterministic_policy=deterministic,
        )
        core.corridor_progress = corridor_progress
        transitions: list[SequenceTransition] = []
        pending: Optional[_PendingAction] = None
        previous_control = Control(0.0, 0.0)
        executed_steps = 0
        interventions = 0
        maximum_cross_track = 0.0
        minimum_clearance = float("inf")
        end_reason = "TIME_LIMIT"
        completed = False
        collision = False
        no_safe = False

        while True:
            sample = self._runtime_input(state, self._laser)
            try:
                decision = core.step(sample)
            except Exception:
                decision = RuntimeDecision(
                    reason="CONTROLLER_EXCEPTION",
                    control=None,
                    action=None,
                    policy_action=None,
                    mission_index=core.mission_index,
                    distance_to_goal_m=float("inf"),
                    advised_heading_deg=0.0,
                    safe_mask=(False,) * 5,
                    reachability_mask=(False,) * 5,
                    completed=False,
                    safety_intervened=True,
                    safety_truncated=False,
                )
            interventions += int(decision.safety_intervened)
            if decision.observation is not None:
                maximum_cross_track = max(
                    maximum_cross_track,
                    abs(decision.observation.corridor_cross_track_m),
                )
                minimum_clearance = min(
                    minimum_clearance,
                    decision.observation.map_clearance_m,
                )

            terminal = decision.completed or decision.reason in TERMINAL_REASONS
            timed_out = executed_steps >= max_steps
            if pending is not None and (decision.control is not None or terminal or timed_out):
                reason = (
                    decision.reason
                    if terminal
                    else "TIME_LIMIT"
                    if timed_out
                    else "STEP"
                )
                self._append_transition(
                    transitions,
                    pending,
                    decision,
                    terminated=terminal,
                    truncated=timed_out and not terminal,
                    reason=reason,
                )
                pending = None

            if terminal or timed_out:
                end_reason = decision.reason if terminal else "TIME_LIMIT"
                completed = decision.completed
                collision = decision.reason in {
                    "MAP_INVALID",
                    "DYNAMICS_INVALID",
                    "LASER_EMERGENCY_STOP",
                }
                no_safe = decision.reason == "NO_SAFE_ACTION_TRUNCATED"
                break
            if decision.control is None:
                state = dynamics.propagate(
                    state,
                    Control(0.0, 0.0),
                    CONTROL_PERIOD_S,
                )[-1]
                continue
            if decision.training_trace is None:
                raise RuntimeError("executed control has no training trace")
            pending = _PendingAction(
                trace=decision.training_trace,
                previous_control=previous_control,
            )
            previous_control = decision.control
            rollout = dynamics.propagate(state, decision.control, CONTROL_PERIOD_S)
            state = rollout[-1]
            executed_steps += 1

        if transitions:
            total_reward = sum(item.reward for item in transitions)
        else:
            total_reward = -250.0
        summary = EpisodeSummary(
            session_id=session_id,
            total_reward=total_reward,
            steps=executed_steps,
            completed_waypoints=core.mission_index,
            completed=completed,
            collision=collision,
            timed_out=end_reason == "TIME_LIMIT",
            no_safe_action=no_safe,
            safety_interventions=interventions,
            maximum_cross_track_m=maximum_cross_track,
            minimum_clearance_m=(
                minimum_clearance if math.isfinite(minimum_clearance) else 0.0
            ),
            end_reason=end_reason,
        )
        if training and transitions:
            self.replay.add_episode(transitions)
            self.completed_training_episodes += 1
            updates = min(32, max(1, len(transitions) // 16))
            for _ in range(updates):
                batch = self.replay.sample(batch_size=8, burn_in=8, unroll=16)
                self.sac.update(batch)
        return OfflineEpisode(tuple(transitions), summary)


__all__ = [
    "DEFAULT_MAX_STEPS",
    "EpisodeSummary",
    "FixedMapSACTrainer",
    "OfflineEpisode",
    "control_transition_reward",
]
