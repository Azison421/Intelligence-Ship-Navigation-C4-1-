"""Offline National_Test environment built on the live transition core."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from random import Random
from typing import Optional

import numpy as np
import torch

from usvlib4ros.navigation.fixed_map_runtime import (
    FixedMapControllerCore,
    RuntimeDecision,
    RuntimeInput,
    RuntimeTrainingTrace,
    build_fixed_route_context,
    laser_emergency_distance_m,
)
from usvlib4ros.planning import Control, PrototypeReducedDynamics, VesselState
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    reduced_dynamics_from_profile,
)
from usvlib4ros.planning.fixed_route import fixed_route_waypoint_reached

from .recurrent_sac import (
    LocalWaypointObservationV3,
    RecurrentDiscreteSAC,
    ReplaySequenceBatch,
    SequenceReplay,
    SequenceTransition,
)
from .safety_supervisor import (
    FIXED_MAP_PREDICTION_HORIZON_S,
    CandidateControl,
    PredictiveSafetySupervisor,
)


CONTROL_PERIOD_S = 0.1
DEFAULT_MAX_STEPS = 6_000
LASER_MAX_RANGE_M = 20.0
LASER_RAYCAST_RANGE_M = 10.0
GUIDED_REPLAY_EPISODES = 100
TERMINAL_REASONS = frozenset(
    {
        "DYNAMICS_INVALID",
        "MAP_INVALID",
        "LASER_EMERGENCY_STOP",
        "MOTION_STALLED",
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

    current_distance = math.hypot(*observation.current_waypoint_body_xy)
    next_distance = math.hypot(*next_observation.current_waypoint_body_xy)
    waypoint_progress = (
        0.0
        if mission_delta > 0
        else max(-1.0, min(1.0, current_distance - next_distance))
    )
    corridor_delta = (
        next_observation.corridor_progress - observation.corridor_progress
    )
    heading_improvement = (
        abs(observation.corridor_heading_error_rad)
        - abs(next_observation.corridor_heading_error_rad)
    )
    cross_track_improvement = (
        abs(observation.corridor_cross_track_m)
        - abs(next_observation.corridor_cross_track_m)
    )
    current_safe = sum(observation.safe_action_mask)
    next_safe = sum(next_observation.safe_action_mask)
    reward = 4.0 * waypoint_progress
    reward += 20.0 * corridor_delta
    reward += 25.0 * max(0, mission_delta)
    reward += 5.0 * heading_improvement
    reward += 2.0 * cross_track_improvement
    reward += 0.1 * (next_safe - current_safe)
    reward -= 0.08 * abs(next_observation.corridor_heading_error_rad)
    reward -= 0.2 * abs(next_observation.corridor_cross_track_m)
    reward -= (
        0.5
        * next_observation.speed_mps
        * max(0.0, min(1.0, (3.0 - next_distance) / 3.0))
    )
    reward -= 0.03 * abs(control.throttle)
    reward -= 0.05 * abs(control.rudder)
    reward -= 0.08 * abs(control.throttle - previous_control.throttle)
    reward -= 0.12 * abs(control.rudder - previous_control.rudder)
    if completed:
        reward += 100.0
    elif terminated:
        reward -= 50.0 if reason == "NO_SAFE_ACTION_TRUNCATED" else 75.0
    elif truncated and reason not in {"INPUT_STALE", "OPERATOR_TRUNCATED"}:
        reward -= 20.0
    return reward


@dataclass(frozen=True)
class EpisodeSummary:
    session_id: str
    total_reward: float
    steps: int
    start_mission_index: int
    ending_mission_index: int
    waypoints_completed: int
    full_route: bool
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
            self.full_route
            and self.completed
            and not self.collision
            and not self.timed_out
            and not self.no_safe_action
            and self.start_mission_index == 0
            and self.ending_mission_index == 13
            and self.waypoints_completed == 13
        )


@dataclass(frozen=True)
class TrainingDiagnostics:
    attempted_updates: int
    applied_updates: int
    critic_loss: Optional[float]
    actor_loss: Optional[float]
    actor_objective: str
    behavior_clone_loss: Optional[float]
    alpha: Optional[float]
    entropy: Optional[float]


@dataclass(frozen=True)
class OfflineEpisode:
    transitions: tuple[SequenceTransition, ...]
    summary: EpisodeSummary
    training: Optional[TrainingDiagnostics]


@dataclass(frozen=True)
class _PendingAction:
    trace: RuntimeTrainingTrace
    previous_control: Control


def _guided_corridor_action(
    cross_track_m: float,
    heading_error_rad: float,
    safe_action_mask: tuple[bool, ...],
) -> int:
    error = heading_error_rad - math.atan(0.5 * cross_track_m)
    if error > 0.3:
        preference = (0, 1, 2, 3, 4)
    elif error > 0.06:
        preference = (1, 0, 2, 3, 4)
    elif error < -0.3:
        preference = (4, 3, 2, 1, 0)
    elif error < -0.06:
        preference = (3, 4, 2, 1, 0)
    else:
        preference = (2, 1, 3, 0, 4)
    return next(action for action in preference if safe_action_mask[action])


class _GuidedExplorationPolicy:
    """Use the frozen corridor to seed early off-policy exploration."""

    def __init__(
        self,
        sac: RecurrentDiscreteSAC,
    ) -> None:
        self.sac = sac
        self.forward_control_profile = sac.forward_control_profile
        self.action_schema = sac.action_schema

    @staticmethod
    def _guided_action(
        observation: LocalWaypointObservationV3,
        safe_action_mask: tuple[bool, ...],
    ) -> int:
        return _guided_corridor_action(
            observation.corridor_cross_track_m,
            observation.corridor_heading_error_rad,
            safe_action_mask,
        )

    def act(
        self,
        observation: LocalWaypointObservationV3,
        safe_action_mask,
        *,
        hidden=None,
        deterministic: bool = False,
    ):
        proposal, next_hidden = self.sac.act(
            observation,
            safe_action_mask,
            hidden=hidden,
            deterministic=deterministic,
        )
        mask = tuple(safe_action_mask)
        if not deterministic:
            proposal = replace(
                proposal,
                action=self._guided_action(observation, mask),
            )
        return proposal, next_hidden


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
        self._circles = np.asarray(
            [
                (obstacle.x, obstacle.y, obstacle.radius)
                for obstacle in snapshot.circular_obstacles
            ],
            dtype=np.float64,
        ).reshape((-1, 3))

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
        if self._circles.size:
            directions_x = np.cos(headings)[:, None]
            directions_y = np.sin(headings)[:, None]
            offset_x = self._circles[None, :, 0] - state.x
            offset_y = self._circles[None, :, 1] - state.y
            projection = offset_x * directions_x + offset_y * directions_y
            perpendicular_sq = (
                offset_x * offset_x
                + offset_y * offset_y
                - projection * projection
            )
            radius_sq = self._circles[None, :, 2] ** 2
            intersects = (projection >= 0.0) & (perpendicular_sq <= radius_sq)
            entry = projection - np.sqrt(
                np.maximum(0.0, radius_sq - perpendicular_sq)
            )
            entry = np.where(intersects & (entry >= 0.0), entry, np.inf)
            ranges = np.minimum(ranges, np.min(entry, axis=1))
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

    @staticmethod
    def _expert_actions(batch: ReplaySequenceBatch) -> torch.Tensor:
        expert_actions = batch.actions.clone()
        valid = ~batch.padding_mask & batch.safe_action_mask.any(dim=-1)
        for row, column in torch.nonzero(valid, as_tuple=False).tolist():
            observation = batch.observations[row, column]
            safe_mask = tuple(
                bool(value)
                for value in batch.safe_action_mask[row, column].tolist()
            )
            expert_actions[row, column] = _guided_corridor_action(
                float(observation[157].item()),
                float(observation[158].item()),
                safe_mask,
            )
        return expert_actions

    def learn_from_episode(
        self,
        transitions: tuple[SequenceTransition, ...],
        *,
        maximum_updates: int,
        transitions_per_update: int,
        demonstration: bool,
    ) -> Optional[TrainingDiagnostics]:
        if not transitions:
            return None
        if type(demonstration) is not bool:
            raise ValueError("demonstration must be boolean")
        self.replay.add_episode(transitions)
        attempted = min(
            maximum_updates,
            max(1, len(transitions) // transitions_per_update),
        )
        results = []
        for _ in range(attempted):
            batch = self.replay.sample(batch_size=4, burn_in=32, unroll=64)
            result = self.sac.update(
                batch,
                demonstration=demonstration,
                expert_actions=(
                    None
                    if demonstration
                    else self._expert_actions(batch)
                ),
            )
            if result.get("updated") is True:
                results.append(result)
        if not results:
            return TrainingDiagnostics(
                attempted_updates=attempted,
                applied_updates=0,
                critic_loss=None,
                actor_loss=None,
                actor_objective=(
                    "BEHAVIOR_CLONING"
                    if demonstration
                    else "SAC_WITH_DAGGER"
                ),
                behavior_clone_loss=None,
                alpha=None,
                entropy=None,
            )

        def average(name: str) -> float:
            return sum(float(result[name]) for result in results) / len(results)

        return TrainingDiagnostics(
            attempted_updates=attempted,
            applied_updates=len(results),
            critic_loss=average("critic_loss"),
            actor_loss=average("actor_loss"),
            actor_objective=str(results[-1]["actor_objective"]),
            behavior_clone_loss=average("behavior_clone_loss"),
            alpha=float(results[-1]["alpha"]),
            entropy=average("entropy"),
        )

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
            mission_index = (
                0
                if full_route
                else self._mission_index(corridor, segment_index)
            )
            if fixed_route_waypoint_reached(
                context.compiled_map,
                mission_index,
                state,
            ):
                continue
            ranges = laser.scan(state)
            if min(ranges) <= laser_emergency_distance_m(
                context.compiled_map.snapshot,
                state,
            ):
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
        guided_replay = (
            training
            and self.completed_training_episodes < GUIDED_REPLAY_EPISODES
        )
        policy = (
            _GuidedExplorationPolicy(self.sac)
            if guided_replay
            else self.sac
        )
        core = FixedMapControllerCore(
            context,
            policy,
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
                    "MOTION_STALLED",
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
            start_mission_index=mission_index,
            ending_mission_index=core.mission_index,
            waypoints_completed=max(0, core.mission_index - mission_index),
            full_route=full_route,
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
        training_diagnostics = None
        if training and transitions:
            self.completed_training_episodes += 1
            training_diagnostics = self.learn_from_episode(
                tuple(transitions),
                maximum_updates=32,
                transitions_per_update=16,
                demonstration=guided_replay,
            )
        return OfflineEpisode(
            tuple(transitions),
            summary,
            training_diagnostics,
        )


__all__ = [
    "DEFAULT_MAX_STEPS",
    "EpisodeSummary",
    "FixedMapSACTrainer",
    "OfflineEpisode",
    "TrainingDiagnostics",
    "control_transition_reward",
]
