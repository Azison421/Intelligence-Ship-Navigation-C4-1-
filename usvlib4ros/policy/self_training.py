"""Strict offline-to-Unity training gates for National_Test V6."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Optional

import torch

from usvlib4ros.navigation.fixed_map_runtime import (
    DEFAULT_CHECKPOINT,
    LASER_EMERGENCY_DISTANCE_M,
    REQUIRED_MAP_CLEARANCE_M,
    RuntimeDecision,
    RuntimeTrainingTrace,
    build_fixed_route_context,
)
from usvlib4ros.planning import Control
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    action_protocol_hash,
    forward_control_profile_from_dict,
    forward_control_profile_to_dict,
)

from .fixed_map_trainer import (
    EpisodeSummary,
    FixedMapSACTrainer,
    control_transition_reward,
)
from .recurrent_sac import (
    ACTION_SCHEMA,
    LOCAL_WAYPOINT_OBSERVATION_SCHEMA_V3,
    OBSERVATION_DIM,
    REPLAY_SCHEMA_V3,
    SequenceReplay,
    SequenceTransition,
)


STATE_SCHEMA = "national-test-self-training-state-v2"
ACTIVE_SCHEMA = "national-test-active-checkpoint-v2"
CHECKPOINT_SCHEMA = "national-test-sac-checkpoint-v6"
CALIBRATION_SCHEMA = "national-test-forward-calibration-v2"
OFFLINE_BLOCK_EPISODES = 100
OFFLINE_EVALUATION_EPISODES = 20
UNITY_ADAPT_EPISODES = 5
UNITY_VALIDATION_EPISODES = 5
MAX_TRAINING_EPISODES = 1_000


class SelfTrainingStage(str, Enum):
    OFFLINE_TRAIN = "OFFLINE_TRAIN"
    OFFLINE_EVAL = "OFFLINE_EVAL"
    UNITY_ADAPT = "UNITY_ADAPT"
    UNITY_VALIDATION = "UNITY_VALIDATION"
    PROMOTED = "PROMOTED"
    TRAINING_GATE_FAILED = "TRAINING_GATE_FAILED"


@dataclass(frozen=True)
class TrainingCursor:
    seed: int
    stage: SelfTrainingStage = SelfTrainingStage.OFFLINE_TRAIN
    generation: int = 1
    completed_training_episodes: int = 0
    offline_block_progress: int = 0
    offline_evaluations: int = 0
    offline_evaluation_passes: int = 0
    unity_adapt_episodes: int = 0
    unity_validation_episodes: int = 0
    unity_validation_passes: int = 0
    operator_truncated_episodes: int = 0
    active_checkpoint: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("training seed must be an integer")
        if not isinstance(self.stage, SelfTrainingStage):
            object.__setattr__(self, "stage", SelfTrainingStage(self.stage))
        for name in (
            "generation",
            "completed_training_episodes",
            "offline_block_progress",
            "offline_evaluations",
            "offline_evaluation_passes",
            "unity_adapt_episodes",
            "unity_validation_episodes",
            "unity_validation_passes",
            "operator_truncated_episodes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.generation < 1:
            raise ValueError("generation must be positive")
        if self.completed_training_episodes > MAX_TRAINING_EPISODES:
            raise ValueError("training episode budget exceeded")
        if self.offline_block_progress > OFFLINE_BLOCK_EPISODES:
            raise ValueError("offline block progress is invalid")
        if not 0 <= self.offline_evaluation_passes <= self.offline_evaluations <= 20:
            raise ValueError("offline evaluation counters are invalid")
        if self.unity_adapt_episodes > 5 or self.unity_validation_episodes > 5:
            raise ValueError("Unity episode counters are invalid")
        if self.unity_validation_passes > self.unity_validation_episodes:
            raise ValueError("Unity validation counters are invalid")
        if self.active_checkpoint is not None and (
            not isinstance(self.active_checkpoint, str)
            or not self.active_checkpoint.strip()
        ):
            raise ValueError("active checkpoint path is invalid")

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TrainingCursor":
        required = {
            "seed",
            "stage",
            "generation",
            "completed_training_episodes",
            "offline_block_progress",
            "offline_evaluations",
            "offline_evaluation_passes",
            "unity_adapt_episodes",
            "unity_validation_episodes",
            "unity_validation_passes",
            "operator_truncated_episodes",
            "active_checkpoint",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("training cursor schema is incompatible")
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise ValueError("training cursor is invalid") from exc


@dataclass(frozen=True)
class TrainingSnapshot:
    cursor: TrainingCursor
    profile: ForwardControlProfile
    training_state: dict[str, object]
    replay: SequenceReplay


class TrainingStateStore:
    """Persist only the current V2 cursor, V3 replay, and V2 SAC state."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save(
        self,
        cursor: TrainingCursor,
        trainer: FixedMapSACTrainer,
    ) -> Path:
        payload = {
            "schema_version": STATE_SCHEMA,
            "cursor": cursor.to_payload(),
            "profile": forward_control_profile_to_dict(
                trainer.forward_profile
            ),
            "training_state": trainer.sac.training_state_dict(),
            "replay_state": trainer.replay.state_dict(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(payload, temporary)
        for attempt in range(20):
            try:
                temporary.replace(self.path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
        return self.path

    def load(self) -> TrainingSnapshot:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("training state cannot be safely loaded") from exc
        required = {
            "schema_version",
            "cursor",
            "profile",
            "training_state",
            "replay_state",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("training state schema is incompatible")
        if payload.get("schema_version") != STATE_SCHEMA:
            raise ValueError("only national-test-self-training-state-v2 is supported")
        profile = forward_control_profile_from_dict(payload["profile"])
        replay = SequenceReplay.from_state_dict(payload["replay_state"])
        return TrainingSnapshot(
            cursor=TrainingCursor.from_payload(payload["cursor"]),
            profile=profile,
            training_state=dict(payload["training_state"]),
            replay=replay,
        )

    def restore_trainer(self) -> tuple[TrainingCursor, FixedMapSACTrainer]:
        snapshot = self.load()
        trainer = FixedMapSACTrainer(
            snapshot.profile,
            seed=snapshot.cursor.seed,
        )
        trainer.sac.load_training_state_dict(snapshot.training_state)
        trainer.replay = snapshot.replay
        trainer.completed_training_episodes = (
            snapshot.cursor.completed_training_episodes
        )
        return snapshot.cursor, trainer


class ActiveCheckpointRegistry:
    """Resolve an explicitly registered current checkpoint; never fall back."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def resolve(self, fallback: Path) -> Path:
        del fallback
        if not self.path.is_file():
            raise FileNotFoundError("active V6 checkpoint registry is missing")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        required = {"schema_version", "checkpoint", "checkpoint_sha256", "stage"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("active checkpoint registry schema is incompatible")
        if payload.get("schema_version") != ACTIVE_SCHEMA:
            raise ValueError("active checkpoint registry schema is incompatible")
        stage = SelfTrainingStage(payload["stage"])
        if stage not in {
            SelfTrainingStage.UNITY_ADAPT,
            SelfTrainingStage.UNITY_VALIDATION,
            SelfTrainingStage.PROMOTED,
        }:
            raise ValueError(f"active checkpoint is unavailable during {stage.value}")
        checkpoint = self.path.parent / str(payload["checkpoint"])
        if not checkpoint.is_file() or _digest(checkpoint) != payload.get(
            "checkpoint_sha256"
        ):
            raise ValueError("active checkpoint hash is invalid")
        return checkpoint

    def write(self, checkpoint: Path, stage: SelfTrainingStage) -> Path:
        checkpoint = Path(checkpoint)
        if checkpoint.parent.resolve() != self.path.parent.resolve():
            raise ValueError("active checkpoint must share the registry directory")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = {
            "schema_version": ACTIVE_SCHEMA,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": _digest(checkpoint),
            "stage": SelfTrainingStage(stage).value,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
        return self.path


def _digest(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_calibration(path: Path) -> ForwardControlProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CALIBRATION_SCHEMA
        or payload.get("verdict") != "calibrated"
    ):
        raise ValueError("verified V2 forward calibration is required")
    profile = forward_control_profile_from_dict(payload.get("profile"))
    if payload.get("action_protocol_hash") != action_protocol_hash(profile):
        raise ValueError("calibration action protocol hash is invalid")
    return profile


def _manifest(
    checkpoint: Path,
    trainer: FixedMapSACTrainer,
    cursor: TrainingCursor,
) -> dict[str, object]:
    context = build_fixed_route_context(session_id="checkpoint-manifest-v6")
    profile = trainer.forward_profile
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "checkpoint_sha256": _digest(checkpoint),
        "observation_schema": LOCAL_WAYPOINT_OBSERVATION_SCHEMA_V3,
        "observation_dim": OBSERVATION_DIM,
        "action_schema": ACTION_SCHEMA,
        "action_dim": 5,
        "replay_schema": REPLAY_SCHEMA_V3,
        "route_id": context.compiled_map.manifest.route_id,
        "map_payload_hash": context.compiled_map.snapshot.payload_content_hash,
        "corridor_sha256": context.corridor.corridor_hash,
        "required_clearance_m": REQUIRED_MAP_CLEARANCE_M,
        "laser_emergency_distance_m": LASER_EMERGENCY_DISTANCE_M,
        "hidden_dim": trainer.sac.hidden_dim,
        "initialization": {
            "type": "random",
            "seed": cursor.seed,
            "inherited_checkpoint": None,
        },
        "calibration": {
            "status": "verified",
            "calibration_hash": profile.calibration_hash,
            "action_protocol_hash": action_protocol_hash(profile),
        },
        "forward_control_profile": forward_control_profile_to_dict(profile),
        "stage": cursor.stage.value,
        "gate_evidence": {
            "completed_training_episodes": cursor.completed_training_episodes,
            "offline_evaluations": cursor.offline_evaluations,
            "offline_evaluation_passes": cursor.offline_evaluation_passes,
            "unity_adapt_episodes": cursor.unity_adapt_episodes,
            "unity_validation_episodes": cursor.unity_validation_episodes,
            "unity_validation_passes": cursor.unity_validation_passes,
        },
    }


def save_stage_checkpoint(
    directory: Path,
    trainer: FixedMapSACTrainer,
    cursor: TrainingCursor,
) -> tuple[Path, TrainingCursor]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if cursor.stage is SelfTrainingStage.PROMOTED:
        name = DEFAULT_CHECKPOINT.name
    else:
        name = (
            f"national_test_sac_v6_seed{cursor.seed}_"
            f"g{cursor.generation}_{cursor.stage.value.lower()}_"
            f"a{cursor.unity_adapt_episodes}_"
            f"v{cursor.unity_validation_episodes}.pt"
        )
    checkpoint = directory / name
    trainer.sac.save_checkpoint(checkpoint)
    manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    manifest_path.write_text(
        json.dumps(
            _manifest(checkpoint, trainer, cursor),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, replace(cursor, active_checkpoint=checkpoint.name)


def update_checkpoint_stage(
    checkpoint: Path,
    cursor: TrainingCursor,
) -> None:
    path = Path(checkpoint).with_suffix(Path(checkpoint).suffix + ".json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint manifest schema is incompatible")
    if manifest.get("checkpoint_sha256") != _digest(Path(checkpoint)):
        raise ValueError("checkpoint hash changed")
    manifest["stage"] = cursor.stage.value
    manifest["gate_evidence"] = {
        "completed_training_episodes": cursor.completed_training_episodes,
        "offline_evaluations": cursor.offline_evaluations,
        "offline_evaluation_passes": cursor.offline_evaluation_passes,
        "unity_adapt_episodes": cursor.unity_adapt_episodes,
        "unity_validation_episodes": cursor.unity_validation_episodes,
        "unity_validation_passes": cursor.unity_validation_passes,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class OfflineTrainingGate:
    """Run exact 100-train/20-evaluate blocks until Unity admission."""

    def __init__(
        self,
        *,
        profile: ForwardControlProfile,
        seed: int,
        state_store: TrainingStateStore,
        checkpoint_dir: Path,
        progress: Optional[Callable[[TrainingCursor, EpisodeSummary], None]] = None,
    ) -> None:
        self.profile = profile
        self.seed = seed
        self.state_store = state_store
        self.checkpoint_dir = Path(checkpoint_dir)
        self.progress = progress

    def _load_or_create(self) -> tuple[TrainingCursor, FixedMapSACTrainer]:
        if not self.state_store.path.exists():
            return TrainingCursor(seed=self.seed), FixedMapSACTrainer(
                self.profile,
                seed=self.seed,
            )
        cursor, trainer = self.state_store.restore_trainer()
        if cursor.seed != self.seed:
            raise ValueError("training state seed differs from requested seed")
        if trainer.forward_profile.calibration_hash != self.profile.calibration_hash:
            raise ValueError("training state calibration differs from requested profile")
        return cursor, trainer

    def _save(
        self,
        cursor: TrainingCursor,
        trainer: FixedMapSACTrainer,
        summary: EpisodeSummary,
    ) -> None:
        self.state_store.save(cursor, trainer)
        if self.progress is not None:
            self.progress(cursor, summary)

    def run(self) -> tuple[TrainingCursor, FixedMapSACTrainer]:
        cursor, trainer = self._load_or_create()
        if cursor.stage not in {
            SelfTrainingStage.OFFLINE_TRAIN,
            SelfTrainingStage.OFFLINE_EVAL,
        }:
            return cursor, trainer
        while cursor.completed_training_episodes < MAX_TRAINING_EPISODES:
            if cursor.stage is SelfTrainingStage.OFFLINE_TRAIN:
                while cursor.offline_block_progress < OFFLINE_BLOCK_EPISODES:
                    episode_id = cursor.completed_training_episodes + 1
                    result = trainer.run_episode(
                        episode=episode_id,
                        training=True,
                        deterministic=False,
                        full_route=(episode_id % 4 == 0),
                    )
                    cursor = replace(
                        cursor,
                        completed_training_episodes=episode_id,
                        offline_block_progress=cursor.offline_block_progress + 1,
                    )
                    self._save(cursor, trainer, result.summary)
                    if cursor.completed_training_episodes >= MAX_TRAINING_EPISODES:
                        break
                if cursor.offline_block_progress < OFFLINE_BLOCK_EPISODES:
                    break
                cursor = replace(
                    cursor,
                    stage=SelfTrainingStage.OFFLINE_EVAL,
                    offline_evaluations=0,
                    offline_evaluation_passes=0,
                )
                self.state_store.save(cursor, trainer)

            while (
                cursor.stage is SelfTrainingStage.OFFLINE_EVAL
                and cursor.offline_evaluations < OFFLINE_EVALUATION_EPISODES
            ):
                evaluation_id = (
                    cursor.completed_training_episodes * 100
                    + cursor.offline_evaluations
                    + 1
                )
                result = trainer.run_episode(
                    episode=evaluation_id,
                    training=False,
                    deterministic=True,
                    full_route=True,
                )
                passed = result.summary.passed
                if not passed and result.transitions:
                    trainer.replay.add_episode(result.transitions)
                cursor = replace(
                    cursor,
                    offline_evaluations=cursor.offline_evaluations + 1,
                    offline_evaluation_passes=(
                        cursor.offline_evaluation_passes + int(passed)
                    ),
                )
                self._save(cursor, trainer, result.summary)

            if (
                cursor.offline_evaluations == OFFLINE_EVALUATION_EPISODES
                and cursor.offline_evaluation_passes
                == OFFLINE_EVALUATION_EPISODES
            ):
                cursor = replace(cursor, stage=SelfTrainingStage.UNITY_ADAPT)
                checkpoint, cursor = save_stage_checkpoint(
                    self.checkpoint_dir,
                    trainer,
                    cursor,
                )
                ActiveCheckpointRegistry(
                    self.checkpoint_dir / "national_test_sac_active.json"
                ).write(checkpoint, cursor.stage)
                self.state_store.save(cursor, trainer)
                return cursor, trainer

            cursor = replace(
                cursor,
                stage=SelfTrainingStage.OFFLINE_TRAIN,
                generation=cursor.generation + 1,
                offline_block_progress=0,
                offline_evaluations=0,
                offline_evaluation_passes=0,
            )
            self.state_store.save(cursor, trainer)

        cursor = replace(cursor, stage=SelfTrainingStage.TRAINING_GATE_FAILED)
        self.state_store.save(cursor, trainer)
        registry = ActiveCheckpointRegistry(
            self.checkpoint_dir / "national_test_sac_active.json"
        )
        if cursor.active_checkpoint is not None:
            checkpoint = self.checkpoint_dir / cursor.active_checkpoint
            if checkpoint.is_file():
                registry.write(checkpoint, cursor.stage)
                update_checkpoint_stage(checkpoint, cursor)
        return cursor, trainer


class UnityTransitionRecorder:
    """Record fresh live decisions using Gymnasium-style episode boundaries."""

    def __init__(self) -> None:
        self._transitions: list[SequenceTransition] = []
        self._pending: Optional[RuntimeTrainingTrace] = None
        self._previous_control = Control(0.0, 0.0)

    @property
    def transitions(self) -> tuple[SequenceTransition, ...]:
        return tuple(self._transitions)

    def _finalize(
        self,
        decision: RuntimeDecision,
        *,
        terminated: bool,
        truncated: bool,
        reason: str,
        operator_truncated: bool = False,
    ) -> None:
        if self._pending is None:
            return
        next_observation = decision.observation or self._pending.observation
        trace = self._pending
        self._transitions.append(
            SequenceTransition(
                observation=trace.observation,
                next_observation=next_observation,
                executed_action=trace.executed_action,
                reward=control_transition_reward(
                    trace.observation,
                    next_observation,
                    trace.final_control,
                    self._previous_control,
                    mission_delta=decision.mission_index - trace.mission_index,
                    completed=decision.completed,
                    terminated=terminated,
                    truncated=truncated,
                    reason=reason,
                ),
                terminated=terminated,
                truncated=truncated,
                reason=reason,
                operator_truncated=operator_truncated,
            )
        )
        self._previous_control = trace.final_control
        self._pending = None

    def observe(self, decision: RuntimeDecision) -> None:
        terminal = decision.completed or decision.safety_truncated or decision.reason in {
            "MAP_INVALID",
            "DYNAMICS_INVALID",
            "LASER_EMERGENCY_STOP",
            "POLICY_NO_ACTION",
        }
        if self._pending is not None and (decision.control is not None or terminal):
            self._finalize(
                decision,
                terminated=terminal,
                truncated=False,
                reason=decision.reason if terminal else "STEP",
            )
        if not terminal and decision.training_trace is not None:
            self._pending = decision.training_trace

    def truncate(
        self,
        mission_index: int,
        *,
        reason: str,
        operator_truncated: bool = False,
    ) -> None:
        if self._pending is None:
            return
        decision = RuntimeDecision(
            reason=reason,
            control=None,
            action=None,
            policy_action=None,
            mission_index=mission_index,
            distance_to_goal_m=0.0,
            advised_heading_deg=0.0,
            safe_mask=(False,) * 5,
            reachability_mask=(False,) * 5,
            completed=False,
            safety_intervened=False,
            safety_truncated=False,
            observation=self._pending.observation,
        )
        self._finalize(
            decision,
            terminated=False,
            truncated=True,
            reason=reason,
            operator_truncated=operator_truncated,
        )

    def operator_stop(self, mission_index: int) -> None:
        self.truncate(
            mission_index,
            reason="OPERATOR_TRUNCATED",
            operator_truncated=True,
        )


def train_from_unity_episode(
    trainer: FixedMapSACTrainer,
    transitions: tuple[SequenceTransition, ...],
) -> None:
    if not transitions:
        return
    trainer.replay.add_episode(transitions)
    updates = min(64, max(1, len(transitions) // 8))
    for _ in range(updates):
        batch = trainer.replay.sample(batch_size=8, burn_in=8, unroll=16)
        trainer.sac.update(batch)


class UnityTrainingGate:
    """Advance exactly five adaptation and five frozen validation episodes."""

    def __init__(
        self,
        *,
        state_store: TrainingStateStore,
        checkpoint_dir: Path,
    ) -> None:
        self.state_store = state_store
        self.checkpoint_dir = Path(checkpoint_dir)
        self.registry = ActiveCheckpointRegistry(
            self.checkpoint_dir / "national_test_sac_active.json"
        )
        self.cursor, self.trainer = state_store.restore_trainer()
        if self.cursor.stage not in {
            SelfTrainingStage.UNITY_ADAPT,
            SelfTrainingStage.UNITY_VALIDATION,
        }:
            raise ValueError(
                f"Unity gate is unavailable during {self.cursor.stage.value}"
            )
        active = self.registry.resolve(DEFAULT_CHECKPOINT)
        if self.cursor.active_checkpoint != active.name:
            raise ValueError("training state and active checkpoint differ")

    @property
    def deterministic(self) -> bool:
        return self.cursor.stage is SelfTrainingStage.UNITY_VALIDATION

    def _save_new_active(self) -> None:
        checkpoint, self.cursor = save_stage_checkpoint(
            self.checkpoint_dir,
            self.trainer,
            self.cursor,
        )
        self.registry.write(checkpoint, self.cursor.stage)
        self.state_store.save(self.cursor, self.trainer)

    def finish_episode(
        self,
        transitions: tuple[SequenceTransition, ...],
        *,
        counted: bool,
        passed: bool,
        operator_truncated: bool,
    ) -> TrainingCursor:
        if type(counted) is not bool or type(passed) is not bool:
            raise ValueError("Unity gate flags must be boolean")
        if type(operator_truncated) is not bool:
            raise ValueError("operator_truncated must be boolean")
        if not counted:
            if transitions:
                self.trainer.replay.add_episode(transitions)
            self.cursor = replace(
                self.cursor,
                operator_truncated_episodes=(
                    self.cursor.operator_truncated_episodes
                    + int(operator_truncated)
                ),
            )
            self.state_store.save(self.cursor, self.trainer)
            return self.cursor

        if self.cursor.stage is SelfTrainingStage.UNITY_ADAPT:
            train_from_unity_episode(self.trainer, transitions)
            adapt_episodes = self.cursor.unity_adapt_episodes + 1
            self.cursor = replace(
                self.cursor,
                unity_adapt_episodes=adapt_episodes,
                stage=(
                    SelfTrainingStage.UNITY_VALIDATION
                    if adapt_episodes == UNITY_ADAPT_EPISODES
                    else SelfTrainingStage.UNITY_ADAPT
                ),
            )
            self._save_new_active()
            return self.cursor

        if not passed and transitions:
            self.trainer.replay.add_episode(transitions)
        validation_episodes = self.cursor.unity_validation_episodes + 1
        validation_passes = self.cursor.unity_validation_passes + int(passed)
        self.cursor = replace(
            self.cursor,
            unity_validation_episodes=validation_episodes,
            unity_validation_passes=validation_passes,
        )
        if validation_episodes < UNITY_VALIDATION_EPISODES:
            checkpoint = self.checkpoint_dir / str(
                self.cursor.active_checkpoint
            )
            update_checkpoint_stage(checkpoint, self.cursor)
            self.registry.write(checkpoint, self.cursor.stage)
            self.state_store.save(self.cursor, self.trainer)
            return self.cursor

        if validation_passes == UNITY_VALIDATION_EPISODES:
            self.cursor = replace(
                self.cursor,
                stage=SelfTrainingStage.PROMOTED,
            )
            self._save_new_active()
            return self.cursor

        failed_checkpoint = self.checkpoint_dir / str(
            self.cursor.active_checkpoint
        )
        failed_cursor = replace(
            self.cursor,
            stage=SelfTrainingStage.OFFLINE_TRAIN,
        )
        update_checkpoint_stage(failed_checkpoint, failed_cursor)
        self.registry.write(failed_checkpoint, failed_cursor.stage)
        self.cursor = replace(
            failed_cursor,
            generation=failed_cursor.generation + 1,
            offline_block_progress=0,
            offline_evaluations=0,
            offline_evaluation_passes=0,
            unity_adapt_episodes=0,
            unity_validation_episodes=0,
            unity_validation_passes=0,
            active_checkpoint=None,
        )
        self.state_store.save(self.cursor, self.trainer)
        return self.cursor


__all__ = [
    "ACTIVE_SCHEMA",
    "ActiveCheckpointRegistry",
    "CALIBRATION_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "MAX_TRAINING_EPISODES",
    "OfflineTrainingGate",
    "STATE_SCHEMA",
    "SelfTrainingStage",
    "TrainingCursor",
    "TrainingSnapshot",
    "TrainingStateStore",
    "UNITY_ADAPT_EPISODES",
    "UNITY_VALIDATION_EPISODES",
    "UnityTransitionRecorder",
    "UnityTrainingGate",
    "load_calibration",
    "save_stage_checkpoint",
    "train_from_unity_episode",
    "update_checkpoint_stage",
]
