"""Persistent hybrid self-training contracts for the fixed National_Test map.

This module contains only deterministic state-machine, replay, promotion and
atomic persistence mechanics.  ROS/Unity orchestration remains in the fixed
map service so these invariants can be tested without starting Unity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from statistics import median
from tempfile import NamedTemporaryFile
from typing import Iterator, Optional
from uuid import uuid4

import torch

from usvlib4ros.planning import Control

from .recurrent_sac import (
    RecurrentDiscreteSAC,
    ReplaySequenceBatch,
    SequenceReplay,
    SequenceTransition,
)


SELF_TRAINING_STATE_SCHEMA = "national-test-self-training-state-v1"
ACTIVE_CHECKPOINT_SCHEMA = "national-test-sac-active-v1"
V5_CHECKPOINT_SCHEMA = "national-test-sac-checkpoint-v5"
SAFE_MASK_POLICY_GATE_VERSION = "sac-predictive-safe-mask-v1"


class SelfTrainingStage(str, Enum):
    BASELINE_UNITY = "baseline_unity"
    OFFLINE_TRAIN = "offline_train"
    UNITY_TRAIN = "unity_train"
    OFFLINE_EVAL = "offline_eval"
    UNITY_VALIDATION = "unity_validation"
    WAITING = "waiting"


@dataclass(frozen=True)
class SelfTrainingConfig:
    baseline_unity_episodes: int = 5
    offline_training_episodes: int = 95
    unity_training_episodes: int = 5
    offline_evaluation_episodes: int = 20
    unity_validation_episodes: int = 5
    initial_target_episodes: int = 1000
    target_increment_episodes: int = 1000
    updates_per_offline_episode: int = 16
    updates_after_unity_block: int = 80
    batch_size: int = 8
    offline_batch_size: int = 6
    unity_batch_size: int = 2
    offline_replay_capacity: int = 32
    unity_replay_capacity: int = 20
    burn_in: int = 2
    unroll: int = 8
    cpu_threads: int = 8
    minimum_step_improvement: float = 0.02

    def __post_init__(self) -> None:
        integer_fields = (
            "baseline_unity_episodes",
            "offline_training_episodes",
            "unity_training_episodes",
            "offline_evaluation_episodes",
            "unity_validation_episodes",
            "initial_target_episodes",
            "target_increment_episodes",
            "updates_per_offline_episode",
            "updates_after_unity_block",
            "batch_size",
            "offline_batch_size",
            "unity_batch_size",
            "offline_replay_capacity",
            "unity_replay_capacity",
            "unroll",
            "cpu_threads",
        )
        if any(type(getattr(self, name)) is not int or getattr(self, name) <= 0 for name in integer_fields):
            raise ValueError("self-training integer settings must be positive")
        if type(self.burn_in) is not int or self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if self.offline_batch_size + self.unity_batch_size != self.batch_size:
            raise ValueError("mixed replay counts must equal batch_size")
        if not 0.0 < self.minimum_step_improvement < 1.0:
            raise ValueError("minimum_step_improvement must be between zero and one")
        generation_size = self.offline_training_episodes + self.unity_training_episodes
        if self.initial_target_episodes % generation_size != 0:
            raise ValueError("initial target must end at a generation boundary")
        if self.target_increment_episodes % generation_size != 0:
            raise ValueError("target increment must end at a generation boundary")


@dataclass(frozen=True)
class SelfTrainingCursor:
    schema_version: str
    session_id: str
    generation: int
    stage: SelfTrainingStage
    stage_index: int
    completed_training_episodes: int
    target_training_episodes: int
    champion_path: str
    champion_sha256: str
    generation_collision: bool = False

    @classmethod
    def new(
        cls,
        config: SelfTrainingConfig,
        *,
        champion_path: str,
        champion_sha256: str,
    ) -> "SelfTrainingCursor":
        if len(champion_sha256) != 64:
            raise ValueError("champion SHA-256 is invalid")
        return cls(
            schema_version=SELF_TRAINING_STATE_SCHEMA,
            session_id=f"self-training-{uuid4().hex}",
            generation=0,
            stage=SelfTrainingStage.BASELINE_UNITY,
            stage_index=0,
            completed_training_episodes=0,
            target_training_episodes=config.initial_target_episodes,
            champion_path=str(champion_path),
            champion_sha256=champion_sha256.lower(),
        )

    def __post_init__(self) -> None:
        if self.schema_version != SELF_TRAINING_STATE_SCHEMA:
            raise ValueError("self-training state schema is incompatible")
        if not self.session_id:
            raise ValueError("self-training session id is required")
        if self.generation < 0 or self.stage_index < 0 or self.completed_training_episodes < 0:
            raise ValueError("self-training counters cannot be negative")
        if self.target_training_episodes <= 0 or self.completed_training_episodes > self.target_training_episodes:
            raise ValueError("self-training target is invalid")
        if len(self.champion_sha256) != 64:
            raise ValueError("champion SHA-256 is invalid")

    def advance(self, config: SelfTrainingConfig) -> "SelfTrainingCursor":
        """Commit exactly one completed episode and move to the next stage."""
        if self.stage is SelfTrainingStage.WAITING:
            raise ValueError("waiting state must be extended before it can advance")
        limit_by_stage = {
            SelfTrainingStage.BASELINE_UNITY: config.baseline_unity_episodes,
            SelfTrainingStage.OFFLINE_TRAIN: config.offline_training_episodes,
            SelfTrainingStage.UNITY_TRAIN: config.unity_training_episodes,
            SelfTrainingStage.OFFLINE_EVAL: config.offline_evaluation_episodes,
            SelfTrainingStage.UNITY_VALIDATION: config.unity_validation_episodes,
        }
        training_stage = self.stage in (SelfTrainingStage.OFFLINE_TRAIN, SelfTrainingStage.UNITY_TRAIN)
        completed = self.completed_training_episodes + (1 if training_stage else 0)
        next_index = self.stage_index + 1
        if next_index < limit_by_stage[self.stage]:
            return self._replace(stage_index=next_index, completed_training_episodes=completed)
        if self.stage is SelfTrainingStage.BASELINE_UNITY:
            return self._replace(
                generation=1,
                stage=SelfTrainingStage.OFFLINE_TRAIN,
                stage_index=0,
                completed_training_episodes=completed,
            )
        if self.stage is SelfTrainingStage.OFFLINE_TRAIN:
            return self._replace(
                stage=SelfTrainingStage.UNITY_TRAIN,
                stage_index=0,
                completed_training_episodes=completed,
            )
        if self.stage is SelfTrainingStage.UNITY_TRAIN:
            return self._replace(
                stage=SelfTrainingStage.OFFLINE_EVAL,
                stage_index=0,
                completed_training_episodes=completed,
            )
        if self.stage is SelfTrainingStage.OFFLINE_EVAL:
            return self._replace(stage=SelfTrainingStage.UNITY_VALIDATION, stage_index=0)
        if completed >= self.target_training_episodes:
            return self._replace(stage=SelfTrainingStage.WAITING, stage_index=0)
        return self._replace(
            generation=self.generation + 1,
            stage=SelfTrainingStage.OFFLINE_TRAIN,
            stage_index=0,
            generation_collision=False,
        )

    def extend_target(self, config: SelfTrainingConfig) -> "SelfTrainingCursor":
        if self.stage is not SelfTrainingStage.WAITING:
            raise ValueError("only a completed target can be extended")
        return self._replace(
            generation=self.generation + 1,
            stage=SelfTrainingStage.OFFLINE_TRAIN,
            stage_index=0,
            target_training_episodes=self.target_training_episodes + config.target_increment_episodes,
            generation_collision=False,
        )

    def mark_collision(self) -> "SelfTrainingCursor":
        return self._replace(generation_collision=True)

    def with_champion(self, path: str | Path, digest: str) -> "SelfTrainingCursor":
        if len(digest) != 64:
            raise ValueError("champion SHA-256 is invalid")
        return self._replace(
            champion_path=str(Path(path).resolve()),
            champion_sha256=digest.lower(),
        )

    def _replace(self, **changes: object) -> "SelfTrainingCursor":
        values = asdict(self)
        values.update(changes)
        values["stage"] = SelfTrainingStage(values["stage"])
        return SelfTrainingCursor(**values)


@dataclass(frozen=True)
class EvaluationSummary:
    completed: int
    attempted: int
    total_steps: tuple[int, ...]
    collisions: int = 0
    laser_stops: int = 0
    safety_stops: int = 0
    timeouts: int = 0
    unrecovered_unsafe_events: int = 0

    def __post_init__(self) -> None:
        counters = (
            self.completed,
            self.attempted,
            self.collisions,
            self.laser_stops,
            self.safety_stops,
            self.timeouts,
            self.unrecovered_unsafe_events,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("evaluation counters must be non-negative integers")
        if self.completed > self.attempted:
            raise ValueError("completed episodes cannot exceed attempted episodes")
        steps = tuple(self.total_steps)
        object.__setattr__(self, "total_steps", steps)
        if len(steps) != self.completed or any(type(value) is not int or value <= 0 for value in steps):
            raise ValueError("total_steps must contain one positive value per completion")

    @property
    def completion_rate(self) -> float:
        return self.completed / self.attempted if self.attempted else 0.0

    @property
    def median_steps(self) -> Optional[float]:
        return float(median(self.total_steps)) if self.total_steps else None

    @property
    def safety_clean(self) -> bool:
        return not any(
            (
                self.collisions,
                self.laser_stops,
                self.safety_stops,
                self.timeouts,
                self.unrecovered_unsafe_events,
            )
        )


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    step_improvement: Optional[float] = None


@dataclass(frozen=True)
class SelfTrainingOperationalProfile:
    required_clearance_m: float
    laser_emergency_distance_m: float
    point3_to_4_throttle_cap: float
    point3_to_4_rudder_cap: float
    point4_to_5_throttle_cap: float
    point4_to_5_rudder_cap: float
    turn_max_edges: int
    turn_entry_speed_limit_mps: float

    def __post_init__(self) -> None:
        numeric = (
            self.required_clearance_m,
            self.laser_emergency_distance_m,
            self.point3_to_4_throttle_cap,
            self.point3_to_4_rudder_cap,
            self.point4_to_5_throttle_cap,
            self.point4_to_5_rudder_cap,
            self.turn_entry_speed_limit_mps,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        ):
            raise ValueError("self-training operational profile is invalid")
        if self.required_clearance_m < 0.0 or self.laser_emergency_distance_m < 0.0:
            raise ValueError("self-training clearance distances cannot be negative")
        if any(
            not 0.0 < value <= 1.0
            for value in (
                self.point3_to_4_throttle_cap,
                self.point3_to_4_rudder_cap,
                self.point4_to_5_throttle_cap,
                self.point4_to_5_rudder_cap,
            )
        ):
            raise ValueError("self-training maneuver limits are invalid")
        if type(self.turn_max_edges) is not int or self.turn_max_edges <= 0:
            raise ValueError("self-training turn edge limit is invalid")
        if self.turn_entry_speed_limit_mps <= 0.0:
            raise ValueError("self-training turn entry speed is invalid")

    @property
    def turn_control(self) -> Control:
        return Control(self.point4_to_5_throttle_cap, self.point4_to_5_rudder_cap)


def operational_profile_from_manifest(manifest: dict[str, object]) -> SelfTrainingOperationalProfile:
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be an object")
    safety = manifest.get("safety_profile")
    maneuver = manifest.get("clearance_maneuver_profile")
    if not isinstance(safety, dict) or not isinstance(maneuver, dict):
        raise ValueError("self-training checkpoint profiles are missing")
    try:
        return SelfTrainingOperationalProfile(
            required_clearance_m=float(safety["required_clearance_m"]),
            laser_emergency_distance_m=float(safety["laser_emergency_distance_m"]),
            point3_to_4_throttle_cap=float(maneuver["approach_throttle_cap"]),
            point3_to_4_rudder_cap=float(maneuver["approach_rudder_cap"]),
            point4_to_5_throttle_cap=float(maneuver["turn_throttle"]),
            point4_to_5_rudder_cap=abs(float(maneuver["turn_rudder"])),
            turn_max_edges=int(maneuver["turn_max_edges"]),
            turn_entry_speed_limit_mps=float(maneuver["turn_entry_speed_limit_mps"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("self-training checkpoint profiles are invalid") from exc


def promotion_decision(
    champion_unity: EvaluationSummary,
    candidate_offline: EvaluationSummary,
    candidate_unity: EvaluationSummary,
    *,
    generation_collision: bool = False,
    minimum_step_improvement: float = 0.02,
) -> PromotionDecision:
    if (
        candidate_offline.attempted != 20
        or candidate_offline.completed != 20
        or not candidate_offline.safety_clean
    ):
        return PromotionDecision(False, "OFFLINE_GATE_FAILED")
    if (
        generation_collision
        or candidate_unity.attempted != 5
        or candidate_unity.completed != 5
        or not candidate_unity.safety_clean
    ):
        return PromotionDecision(False, "UNITY_SAFETY_GATE_FAILED")
    if candidate_unity.completion_rate > champion_unity.completion_rate:
        return PromotionDecision(True, "COMPLETION_RATE_IMPROVED")
    if candidate_unity.completion_rate < champion_unity.completion_rate:
        return PromotionDecision(False, "COMPLETION_RATE_REGRESSED")
    champion_steps = champion_unity.median_steps
    candidate_steps = candidate_unity.median_steps
    if champion_steps is None or candidate_steps is None or champion_steps <= 0.0:
        return PromotionDecision(False, "MISSING_UNITY_STEP_EVIDENCE")
    improvement = (champion_steps - candidate_steps) / champion_steps
    if improvement + 1e-12 >= minimum_step_improvement:
        return PromotionDecision(True, "UNITY_MEDIAN_STEPS_IMPROVED", improvement)
    return PromotionDecision(False, "UNITY_MEDIAN_STEP_GAIN_BELOW_2_PERCENT", improvement)


def fixed_map_reward(
    *,
    progress_m: float,
    cross_track_error_m: float,
    executed_action: int,
    minimum_clearance_m: float,
    task_point_advanced: bool,
    terminated: bool,
    collision: bool = False,
    timeout: bool = False,
) -> float:
    """One reward definition shared by offline and live self-training."""
    if collision:
        return -25.0
    reward = (
        2.0 * float(progress_m)
        - 0.1 * float(cross_track_error_m)
        - 0.03 * abs(int(executed_action) - 2)
        + 0.02 * min(float(minimum_clearance_m), 2.0)
        + (5.0 if task_point_advanced else 0.0)
        + (20.0 if terminated else 0.0)
    )
    return reward - (10.0 if timeout else 0.0)


class UnityTransitionRecorder:
    """Convert fresh consecutive Unity policy traces into one replay episode."""

    def __init__(self) -> None:
        self._pending = None
        self._transitions: list[SequenceTransition] = []
        self._last_stamp: Optional[float] = None
        self._session_id: Optional[str] = None
        self._minimum_clearance = float("inf")

    @property
    def transitions(self) -> tuple[SequenceTransition, ...]:
        return tuple(self._transitions)

    def observe(self, trace) -> bool:
        observation = getattr(trace, "observation", None)
        if not hasattr(observation, "stamp_sim") or not hasattr(observation, "session_id"):
            raise ValueError("Unity training trace is invalid")
        stamp = float(observation.stamp_sim)
        if self._last_stamp is not None and stamp <= self._last_stamp:
            return False
        if self._session_id is None:
            self._session_id = observation.session_id
        elif observation.session_id != self._session_id:
            raise ValueError("Unity transition cannot cross reset sessions")
        self._minimum_clearance = min(
            self._minimum_clearance,
            float(getattr(trace, "map_clearance_m")),
        )
        if self._pending is not None:
            previous = self._pending
            advanced = trace.mission_index > previous.mission_index
            progress = (
                0.5
                if advanced
                else previous.distance_to_goal_m - trace.distance_to_goal_m
            )
            self._transitions.append(
                SequenceTransition(
                    observation=previous.observation,
                    next_observation=trace.observation,
                    executed_action=previous.executed_action,
                    reward=fixed_map_reward(
                        progress_m=progress,
                        cross_track_error_m=trace.cross_track_error_m,
                        executed_action=previous.executed_action,
                        minimum_clearance_m=self._minimum_clearance,
                        task_point_advanced=advanced,
                        terminated=False,
                    ),
                    terminated=False,
                    timeout=False,
                    safety_truncation=False,
                    safe_action_mask=previous.safe_action_mask,
                    hidden_reset=previous.observation.hidden_reset,
                    next_safe_action_mask=trace.safe_action_mask,
                )
            )
        self._pending = trace
        self._last_stamp = stamp
        return True

    def finish(
        self,
        *,
        completed: bool = False,
        collision: bool = False,
        timeout: bool = False,
    ) -> tuple[SequenceTransition, ...]:
        if sum((bool(completed), bool(collision), bool(timeout))) > 1:
            raise ValueError("Unity episode can have only one terminal outcome")
        if self._pending is None:
            return tuple(self._transitions)
        previous = self._pending
        next_observation = replace(
            previous.observation,
            stamp_sim=previous.observation.stamp_sim + 1e-6,
            hidden_reset=False,
        )
        self._transitions.append(
            SequenceTransition(
                observation=previous.observation,
                next_observation=next_observation,
                executed_action=previous.executed_action,
                reward=fixed_map_reward(
                    progress_m=0.5 if completed else 0.0,
                    cross_track_error_m=previous.cross_track_error_m,
                    executed_action=previous.executed_action,
                    minimum_clearance_m=self._minimum_clearance,
                    task_point_advanced=completed,
                    terminated=completed or collision,
                    collision=collision,
                    timeout=timeout,
                ),
                terminated=completed or collision,
                timeout=timeout,
                safety_truncation=False,
                safe_action_mask=previous.safe_action_mask,
                hidden_reset=previous.observation.hidden_reset,
                next_safe_action_mask=(False,) * 5,
            )
        )
        self._pending = None
        return tuple(self._transitions)

    def discard_partial(self) -> None:
        self._pending = None
        self._transitions.clear()
        self._last_stamp = None
        self._session_id = None
        self._minimum_clearance = float("inf")


def concatenate_replay_batches(*batches: ReplaySequenceBatch) -> ReplaySequenceBatch:
    if not batches:
        raise ValueError("at least one replay batch is required")
    first = batches[0]
    if any(
        batch.observation_dim != first.observation_dim
        or batch.action_dim != first.action_dim
        or batch.observations.shape[1:] != first.observations.shape[1:]
        for batch in batches[1:]
    ):
        raise ValueError("replay batch schemas do not match")
    tensor_fields = (
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "terminated",
        "timeout",
        "safety_truncation",
        "safe_action_mask",
        "next_safe_action_mask",
        "learning_mask",
        "padding_mask",
        "hidden_reset",
        "next_hidden_reset",
    )
    values = {
        name: torch.cat(tuple(getattr(batch, name) for batch in batches), dim=0)
        for name in tensor_fields
    }
    values["session_ids"] = tuple(value for batch in batches for value in batch.session_ids)
    values["episode_ids"] = tuple(value for batch in batches for value in batch.episode_ids)
    return ReplaySequenceBatch(**values)


class SelfTrainingLearner:
    """Own two episode replays and apply finite, recoverable SAC updates."""

    def __init__(
        self,
        sac: RecurrentDiscreteSAC,
        config: SelfTrainingConfig,
        *,
        offline_replay: Optional[SequenceReplay] = None,
        unity_replay: Optional[SequenceReplay] = None,
        on_fault=None,
    ) -> None:
        if not isinstance(sac, RecurrentDiscreteSAC):
            raise ValueError("self-training requires RecurrentDiscreteSAC")
        self.sac = sac
        self.config = config
        self.offline_replay = offline_replay or SequenceReplay(
            capacity=config.offline_replay_capacity,
            seed=31,
        )
        self.unity_replay = unity_replay or SequenceReplay(
            capacity=config.unity_replay_capacity,
            seed=37,
        )
        if self.offline_replay.capacity != config.offline_replay_capacity:
            raise ValueError("offline replay capacity is incompatible")
        if self.unity_replay.capacity != config.unity_replay_capacity:
            raise ValueError("Unity replay capacity is incompatible")
        self.on_fault = on_fault
        self.latest_metrics: dict[str, float | bool | int | str] = {
            "updated": False,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
        }
        self._champion_state = sac.training_state_dict()
        torch.set_num_threads(config.cpu_threads)

    def add_offline_episode(self, transitions: tuple[SequenceTransition, ...] | list[SequenceTransition]) -> None:
        self.offline_replay.add_episode(transitions)

    def add_unity_episode(self, transitions: tuple[SequenceTransition, ...] | list[SequenceTransition]) -> None:
        self.unity_replay.add_episode(transitions)

    def sample_batch(self) -> ReplaySequenceBatch:
        if not len(self.offline_replay):
            raise ValueError("offline replay is empty")
        if not len(self.unity_replay):
            return self.offline_replay.sample(
                batch_size=self.config.batch_size,
                burn_in=self.config.burn_in,
                unroll=self.config.unroll,
            )
        return concatenate_replay_batches(
            self.offline_replay.sample(
                batch_size=self.config.offline_batch_size,
                burn_in=self.config.burn_in,
                unroll=self.config.unroll,
            ),
            self.unity_replay.sample(
                batch_size=self.config.unity_batch_size,
                burn_in=self.config.burn_in,
                unroll=self.config.unroll,
            ),
        )

    def update(self, count: int) -> dict[str, float | bool | int | str]:
        if type(count) is not int or count < 0:
            raise ValueError("SAC update count is invalid")
        try:
            for _ in range(count):
                metrics = self.sac.update(self.sample_batch())
                for name, value in metrics.items():
                    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)).item():
                        raise ValueError(f"non-finite SAC metric: {name}")
                self.latest_metrics = dict(metrics)
        except Exception:
            self.sac.load_training_state_dict(self._champion_state)
            if self.on_fault is not None:
                self.on_fault()
            raise
        return self.latest_metrics

    def mark_champion(self) -> None:
        self._champion_state = self.sac.training_state_dict()

    def set_champion_training_state(self, state: dict[str, object]) -> None:
        current = self.sac.training_state_dict()
        self.sac.load_training_state_dict(state)
        self._champion_state = self.sac.training_state_dict()
        self.sac.load_training_state_dict(current)

    def rollback_to_champion(self) -> None:
        self.sac.load_training_state_dict(self._champion_state)

    @property
    def champion_training_state(self) -> dict[str, object]:
        return {key: value for key, value in self._champion_state.items()}


@dataclass(frozen=True)
class GenerationEvidence:
    offline: EvaluationSummary
    unity: EvaluationSummary
    champion_unity: EvaluationSummary


def save_generation_checkpoint(
    sac: RecurrentDiscreteSAC,
    target_path: str | Path,
    *,
    parent_checkpoint: str | Path,
    cursor: SelfTrainingCursor,
    config: SelfTrainingConfig,
    evidence: GenerationEvidence,
    decision: PromotionDecision,
) -> tuple[Path, Path]:
    """Write one immutable v5 generation and its complete promotion evidence."""
    target = Path(target_path)
    manifest_path = target.with_suffix(target.suffix + ".json")
    if target.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite generation checkpoint: {target}")
    parent = Path(parent_checkpoint)
    parent_manifest_path = parent.with_suffix(parent.suffix + ".json")
    if not parent.is_file() or not parent_manifest_path.is_file():
        raise FileNotFoundError("parent checkpoint or manifest is missing")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_digest = _sha256_file(parent)
    if parent_manifest.get("checkpoint_sha256") != parent_digest:
        raise ValueError("parent checkpoint hash is invalid")
    expected = promotion_decision(
        evidence.champion_unity,
        evidence.offline,
        evidence.unity,
        generation_collision=cursor.generation_collision,
        minimum_step_improvement=config.minimum_step_improvement,
    )
    if decision != expected:
        raise ValueError("generation promotion decision does not match its evidence")
    checkpoint = sac.save_checkpoint(target)
    checkpoint_digest = _sha256_file(checkpoint)
    manifest = json.loads(json.dumps(parent_manifest))
    manifest.update(
        {
            "schema_version": V5_CHECKPOINT_SCHEMA,
            "checkpoint_sha256": checkpoint_digest,
            "parent_model_sha256": parent_digest,
            "source_checkpoint": str(parent.resolve()),
            "policy_gate_version": SAFE_MASK_POLICY_GATE_VERSION,
            "training_step": sac.training_step,
            "training_lineage": {
                "session_id": cursor.session_id,
                "generation": cursor.generation,
                "completed_training_episodes": cursor.completed_training_episodes,
                "target_training_episodes": cursor.target_training_episodes,
                "generation_collision": cursor.generation_collision,
                "config": asdict(config),
            },
            "evaluation_evidence": {
                "offline": asdict(evidence.offline),
                "unity": asdict(evidence.unity),
                "champion_unity": asdict(evidence.champion_unity),
            },
            "promotion_decision": asdict(decision),
            "offline_ready": decision.promote,
            "live_ready": decision.promote,
            # v5 validates the embedded 20/5 evidence above.  Do not invent
            # v4-style external log hashes when no immutable log artifact
            # exists.
            "unity_validation_log_hashes": [],
        }
    )
    for profile_name in ("safety_profile", "clearance_maneuver_profile"):
        profile = manifest.get(profile_name)
        if not isinstance(profile, dict):
            checkpoint.unlink(missing_ok=True)
            raise ValueError("generation checkpoint profile is missing")
        profile["unity_test_only"] = not decision.promote
    try:
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        checkpoint.unlink(missing_ok=True)
        raise
    return checkpoint, manifest_path


def bounded_safe_action_controls(
    nominal: Control,
    *,
    throttle_cap: float,
    rudder_cap: float,
) -> tuple[Control, ...]:
    """Build five distinct low-energy controls around a special-leg nominal."""
    if not isinstance(nominal, Control):
        raise ValueError("nominal must be a Control")
    numeric = (throttle_cap, rudder_cap)
    if (
        any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric
        )
        or not 0.0 < float(throttle_cap) <= 1.0
        or not 0.0 < float(rudder_cap) <= 1.0
    ):
        raise ValueError("special-leg hard limits are invalid")
    throttle = max(1e-6, min(float(throttle_cap), max(0.0, nominal.throttle)))
    rudder = max(-float(rudder_cap), min(float(rudder_cap), nominal.rudder))
    ratios = (0.5, 0.75, 1.0, 0.75, 0.5)
    rudder_step = min(0.01, float(rudder_cap) / 4.0)
    controls = tuple(
        Control(
            throttle * ratio,
            max(
                -float(rudder_cap),
                min(float(rudder_cap), rudder + offset * rudder_step),
            ),
        )
        for ratio, offset in zip(ratios, (-2, -1, 0, 1, 2))
    )
    if len({(control.throttle, control.rudder) for control in controls}) != 5:
        raise ValueError("special-leg controls could not be made distinct")
    return controls


class ActiveCheckpointRegistry:
    """Atomic pointer to the most recently promoted v5 champion."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def promote(self, checkpoint_path: str | Path) -> Path:
        checkpoint = Path(checkpoint_path).resolve()
        manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
        if not checkpoint.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("checkpoint or manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = _sha256_file(checkpoint)
        if manifest.get("schema_version") != V5_CHECKPOINT_SCHEMA:
            raise ValueError("only a v5 checkpoint can become active")
        if manifest.get("checkpoint_sha256") != digest:
            raise ValueError("active checkpoint hash is invalid")
        if manifest.get("offline_ready") is not True or manifest.get("live_ready") is not True:
            raise ValueError("active checkpoint has not passed both gates")
        for profile_name in ("safety_profile", "clearance_maneuver_profile"):
            profile = manifest.get(profile_name)
            if not isinstance(profile, dict) or profile.get("unity_test_only") is not False:
                raise ValueError("active checkpoint profiles are still restricted")
        pointer = {
            "schema_version": ACTIVE_CHECKPOINT_SCHEMA,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest,
            "manifest_sha256": _sha256_file(manifest_path),
        }
        _atomic_write_text(
            self.path,
            json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return checkpoint

    def resolve(
        self,
        fallback: str | Path,
        *,
        explicit: Optional[str | Path] = None,
    ) -> Path:
        if explicit is not None:
            return Path(explicit)
        if not self.path.is_file():
            return Path(fallback)
        try:
            pointer = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("active checkpoint pointer is invalid") from exc
        if not isinstance(pointer, dict) or pointer.get("schema_version") != ACTIVE_CHECKPOINT_SCHEMA:
            raise ValueError("active checkpoint pointer schema is incompatible")
        checkpoint = Path(str(pointer.get("checkpoint_path", "")))
        manifest = checkpoint.with_suffix(checkpoint.suffix + ".json")
        if not checkpoint.is_file() or not manifest.is_file():
            raise ValueError("active checkpoint target is missing")
        if _sha256_file(checkpoint) != pointer.get("checkpoint_sha256"):
            raise ValueError("active checkpoint hash is invalid")
        if _sha256_file(manifest) != pointer.get("manifest_sha256"):
            raise ValueError("active checkpoint manifest hash is invalid")
        return checkpoint.resolve()


@dataclass(frozen=True)
class SelfTrainingSnapshot:
    cursor: SelfTrainingCursor
    offline_replay: SequenceReplay
    unity_replay: SequenceReplay
    training_state: Optional[dict[str, object]]
    python_random_state: object
    torch_random_state: torch.Tensor
    metadata: dict[str, object]

    def __iter__(self) -> Iterator[object]:
        yield self.cursor
        yield self.offline_replay
        yield self.unity_replay


class SelfTrainingStateStore:
    """Content-addressed state snapshots with an atomic SHA pointer."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @property
    def pointer_path(self) -> Path:
        return self.directory / "state.sha256"

    def save(
        self,
        cursor: SelfTrainingCursor,
        *,
        offline_replay: SequenceReplay,
        unity_replay: SequenceReplay,
        training_state: Optional[dict[str, object]] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        states = self.directory / "states"
        states.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SELF_TRAINING_STATE_SCHEMA,
            "cursor": {**asdict(cursor), "stage": cursor.stage.value},
            "offline_replay": offline_replay.state_dict(),
            "unity_replay": unity_replay.state_dict(),
            "training_state": training_state,
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "metadata": dict(metadata or {}),
        }
        with NamedTemporaryFile(mode="wb", dir=states, prefix="state-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
        digest = _sha256_file(temporary)
        target = states / f"{digest}.pt"
        os.replace(temporary, target)
        previous = self.pointer_path.read_text(encoding="ascii").strip() if self.pointer_path.is_file() else None
        _atomic_write_text(self.pointer_path, digest)
        if previous and previous != digest:
            old_target = states / f"{previous}.pt"
            try:
                old_target.unlink()
            except FileNotFoundError:
                pass
        return target

    def load(self) -> SelfTrainingSnapshot:
        if not self.pointer_path.is_file():
            raise FileNotFoundError(self.pointer_path)
        digest = self.pointer_path.read_text(encoding="ascii").strip().lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("self-training state integrity pointer is invalid")
        source = self.directory / "states" / f"{digest}.pt"
        if not source.is_file() or _sha256_file(source) != digest:
            raise ValueError("self-training state integrity check failed")
        try:
            payload = torch.load(source, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError("self-training state cannot be loaded") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != SELF_TRAINING_STATE_SCHEMA:
            raise ValueError("self-training state schema is incompatible")
        raw_cursor = payload.get("cursor")
        if not isinstance(raw_cursor, dict):
            raise ValueError("self-training cursor is missing")
        try:
            cursor = SelfTrainingCursor(**{**raw_cursor, "stage": SelfTrainingStage(raw_cursor["stage"])})
            offline = SequenceReplay.from_state_dict(payload["offline_replay"])
            unity = SequenceReplay.from_state_dict(payload["unity_replay"])
            torch_state = payload["torch_random_state"]
            if not isinstance(torch_state, torch.Tensor):
                raise ValueError("torch random state is invalid")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("self-training state is invalid") from exc
        training_state = payload.get("training_state")
        if training_state is not None and not isinstance(training_state, dict):
            raise ValueError("self-training network state is invalid")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("self-training metadata is invalid")
        return SelfTrainingSnapshot(
            cursor=cursor,
            offline_replay=offline,
            unity_replay=unity,
            training_state=training_state,
            python_random_state=payload.get("python_random_state"),
            torch_random_state=torch_state,
            metadata=metadata,
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, value: str, *, encoding: str = "ascii") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding=encoding, dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


__all__ = [
    "ACTIVE_CHECKPOINT_SCHEMA",
    "ActiveCheckpointRegistry",
    "EvaluationSummary",
    "GenerationEvidence",
    "PromotionDecision",
    "SELF_TRAINING_STATE_SCHEMA",
    "SelfTrainingConfig",
    "SelfTrainingCursor",
    "SelfTrainingSnapshot",
    "SelfTrainingLearner",
    "SelfTrainingOperationalProfile",
    "SelfTrainingStage",
    "SelfTrainingStateStore",
    "SAFE_MASK_POLICY_GATE_VERSION",
    "V5_CHECKPOINT_SCHEMA",
    "bounded_safe_action_controls",
    "concatenate_replay_batches",
    "operational_profile_from_manifest",
    "fixed_map_reward",
    "promotion_decision",
    "save_generation_checkpoint",
    "UnityTransitionRecorder",
]
