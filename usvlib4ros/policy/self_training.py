"""Strict offline-to-Unity training gates for National_Test V6."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from random import Random
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
    OfflineEpisode,
    TrainingDiagnostics,
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


STATE_SCHEMA = "national-test-self-training-state-v6"
ACTIVE_SCHEMA = "national-test-active-checkpoint-v3"
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
    UNITY_DIAGNOSTIC = "UNITY_DIAGNOSTIC"
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
        if (
            self.unity_adapt_episodes > MAX_TRAINING_EPISODES
            or self.unity_validation_episodes > UNITY_VALIDATION_EPISODES
        ):
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
    map_payload_hash: str
    corridor_hash: str
    training_state: dict[str, object]
    replay: SequenceReplay
    trainer_rng_state: object


class TrainingStateStore:
    """Persist the current cursor, replay, SAC state, and training RNG."""

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
            "map_payload_hash": (
                trainer._context.compiled_map.snapshot.payload_content_hash
            ),
            "corridor_hash": trainer._context.corridor.corridor_hash,
            "training_state": trainer.sac.training_state_dict(),
            "replay_state": trainer.replay.state_dict(),
            "trainer_rng_state": trainer.rng.getstate(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(payload, temporary)
        for attempt in range(240):
            try:
                temporary.replace(self.path)
                break
            except PermissionError:
                if attempt == 239:
                    raise
                time.sleep(0.25)
        return self.path

    def load(self) -> TrainingSnapshot:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        try:
            payload = torch.load(
                BytesIO(self.path.read_bytes()),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError("training state cannot be safely loaded") from exc
        required = {
            "schema_version",
            "cursor",
            "profile",
            "map_payload_hash",
            "corridor_hash",
            "training_state",
            "replay_state",
            "trainer_rng_state",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("training state schema is incompatible")
        if payload.get("schema_version") != STATE_SCHEMA:
            raise ValueError("only national-test-self-training-state-v6 is supported")
        map_payload_hash = payload["map_payload_hash"]
        corridor_hash = payload["corridor_hash"]
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (map_payload_hash, corridor_hash)
        ):
            raise ValueError("training state map identity is invalid")
        profile = forward_control_profile_from_dict(payload["profile"])
        replay = SequenceReplay.from_state_dict(payload["replay_state"])
        trainer_rng_state = payload["trainer_rng_state"]
        try:
            Random().setstate(trainer_rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("trainer random state is invalid") from exc
        return TrainingSnapshot(
            cursor=TrainingCursor.from_payload(payload["cursor"]),
            profile=profile,
            map_payload_hash=map_payload_hash,
            corridor_hash=corridor_hash,
            training_state=dict(payload["training_state"]),
            replay=replay,
            trainer_rng_state=trainer_rng_state,
        )

    def restore_trainer(self) -> tuple[TrainingCursor, FixedMapSACTrainer]:
        snapshot = self.load()
        trainer = FixedMapSACTrainer(
            snapshot.profile,
            seed=snapshot.cursor.seed,
        )
        if (
            snapshot.map_payload_hash
            != trainer._context.compiled_map.snapshot.payload_content_hash
            or snapshot.corridor_hash != trainer._context.corridor.corridor_hash
        ):
            raise ValueError("training state map or corridor is incompatible")
        trainer.sac.load_training_state_dict(snapshot.training_state)
        trainer.replay = snapshot.replay
        trainer.rng.setstate(snapshot.trainer_rng_state)
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
        validate_checkpoint_manifest(checkpoint, expected_stage=stage)
        return checkpoint

    def write(self, checkpoint: Path, stage: SelfTrainingStage) -> Path:
        checkpoint = Path(checkpoint)
        stage = SelfTrainingStage(stage)
        if checkpoint.parent.resolve() != self.path.parent.resolve():
            raise ValueError("active checkpoint must share the registry directory")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        validate_checkpoint_manifest(checkpoint, expected_stage=stage)
        payload = {
            "schema_version": ACTIVE_SCHEMA,
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": _digest(checkpoint),
            "stage": stage.value,
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


def validate_checkpoint_manifest(
    checkpoint: Path,
    *,
    expected_stage: Optional[SelfTrainingStage] = None,
) -> dict[str, object]:
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or manifest.get("checkpoint_sha256") != _digest(checkpoint)
    ):
        raise ValueError("checkpoint manifest is incompatible")
    try:
        stage = SelfTrainingStage(manifest["stage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint stage is invalid") from exc
    if expected_stage is not None and stage is not SelfTrainingStage(expected_stage):
        raise ValueError("active registry and checkpoint stage differ")

    evidence = manifest.get("gate_evidence")
    required = {
        "completed_training_episodes",
        "offline_evaluations",
        "offline_evaluation_passes",
        "unity_adapt_episodes",
        "unity_validation_episodes",
        "unity_validation_passes",
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != required
        or any(
            isinstance(evidence[name], bool)
            or not isinstance(evidence[name], int)
            or evidence[name] < 0
            for name in required
        )
    ):
        raise ValueError("checkpoint gate evidence is invalid")
    if (
        evidence["completed_training_episodes"] > MAX_TRAINING_EPISODES
        or not (
            0
            <= evidence["offline_evaluation_passes"]
            <= evidence["offline_evaluations"]
            <= OFFLINE_EVALUATION_EPISODES
        )
        or evidence["unity_adapt_episodes"] > MAX_TRAINING_EPISODES
        or not (
            0
            <= evidence["unity_validation_passes"]
            <= evidence["unity_validation_episodes"]
            <= UNITY_VALIDATION_EPISODES
        )
    ):
        raise ValueError("checkpoint gate evidence is invalid")

    offline_passed = (
        evidence["offline_evaluations"] == OFFLINE_EVALUATION_EPISODES
        and evidence["offline_evaluation_passes"] == OFFLINE_EVALUATION_EPISODES
    )
    training_gate_complete = (
        OFFLINE_BLOCK_EPISODES
        <= evidence["completed_training_episodes"]
        <= MAX_TRAINING_EPISODES
        and evidence["completed_training_episodes"] % OFFLINE_BLOCK_EPISODES == 0
    )
    direct_unity_entry = (
        evidence["completed_training_episodes"] > 0
        and evidence["offline_evaluations"] == 0
        and evidence["offline_evaluation_passes"] == 0
    )
    if stage in {
        SelfTrainingStage.UNITY_ADAPT,
        SelfTrainingStage.UNITY_VALIDATION,
        SelfTrainingStage.PROMOTED,
    } and not (
        direct_unity_entry
        or (training_gate_complete and offline_passed)
    ):
        raise ValueError("promotion gate evidence is incomplete")
    if stage is SelfTrainingStage.UNITY_DIAGNOSTIC and (
        evidence["completed_training_episodes"] <= 0
        or evidence["offline_evaluations"] != 0
        or evidence["offline_evaluation_passes"] != 0
        or evidence["unity_adapt_episodes"] != 0
        or evidence["unity_validation_episodes"] != 0
        or evidence["unity_validation_passes"] != 0
    ):
        raise ValueError("diagnostic checkpoint evidence is invalid")
    if stage is SelfTrainingStage.UNITY_ADAPT and (
        evidence["unity_validation_episodes"] != 0
        or evidence["unity_validation_passes"] != 0
    ):
        raise ValueError("promotion gate evidence is incomplete")
    if stage in {
        SelfTrainingStage.UNITY_VALIDATION,
        SelfTrainingStage.PROMOTED,
    } and evidence["unity_adapt_episodes"] < UNITY_ADAPT_EPISODES:
        raise ValueError("promotion gate evidence is incomplete")
    if stage is SelfTrainingStage.UNITY_VALIDATION and (
        evidence["unity_validation_episodes"] >= UNITY_VALIDATION_EPISODES
    ):
        raise ValueError("promotion gate evidence is incomplete")
    if stage is SelfTrainingStage.PROMOTED and (
        evidence["unity_validation_episodes"] != UNITY_VALIDATION_EPISODES
        or evidence["unity_validation_passes"] != UNITY_VALIDATION_EPISODES
    ):
        raise ValueError("promotion gate evidence is incomplete")
    return manifest


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
            f"g{cursor.generation}_t{cursor.completed_training_episodes}_"
            f"{cursor.stage.value.lower()}_"
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
        stop_after_training_episodes: Optional[int] = None,
        progress: Optional[
            Callable[
                [TrainingCursor, EpisodeSummary, Optional[TrainingDiagnostics]],
                None,
            ]
        ] = None,
    ) -> None:
        self.profile = profile
        self.seed = seed
        self.state_store = state_store
        self.checkpoint_dir = Path(checkpoint_dir)
        if (
            isinstance(stop_after_training_episodes, bool)
            or (
                stop_after_training_episodes is not None
                and (
                    not isinstance(stop_after_training_episodes, int)
                    or not 1
                    <= stop_after_training_episodes
                    <= MAX_TRAINING_EPISODES
                )
            )
        ):
            raise ValueError("training stop must be between 1 and 1000")
        self.stop_after_training_episodes = stop_after_training_episodes
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
        result: OfflineEpisode,
    ) -> None:
        self.state_store.save(cursor, trainer)
        if self.progress is not None:
            self.progress(cursor, result.summary, result.training)

    def run(self) -> tuple[TrainingCursor, FixedMapSACTrainer]:
        cursor, trainer = self._load_or_create()
        if (
            self.stop_after_training_episodes is not None
            and cursor.completed_training_episodes
            >= self.stop_after_training_episodes
        ):
            return cursor, trainer
        if cursor.stage not in {
            SelfTrainingStage.OFFLINE_TRAIN,
            SelfTrainingStage.OFFLINE_EVAL,
        }:
            return cursor, trainer
        while cursor.completed_training_episodes < MAX_TRAINING_EPISODES:
            if cursor.stage is SelfTrainingStage.OFFLINE_TRAIN:
                while cursor.offline_block_progress < OFFLINE_BLOCK_EPISODES:
                    episode_id = cursor.completed_training_episodes + 1
                    dagger_rollout = (
                        episode_id > OFFLINE_BLOCK_EPISODES
                        and episode_id % 2 == 1
                    )
                    result = trainer.run_episode(
                        episode=episode_id,
                        training=True,
                        deterministic=dagger_rollout,
                        full_route=dagger_rollout or episode_id % 4 == 0,
                    )
                    cursor = replace(
                        cursor,
                        completed_training_episodes=episode_id,
                        offline_block_progress=cursor.offline_block_progress + 1,
                    )
                    self._save(cursor, trainer, result)
                    if (
                        self.stop_after_training_episodes is not None
                        and cursor.completed_training_episodes
                        >= self.stop_after_training_episodes
                    ):
                        return cursor, trainer
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
                and cursor.offline_evaluation_passes
                == cursor.offline_evaluations
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
                self._save(cursor, trainer, result)

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

        cursor = replace(
            cursor,
            stage=SelfTrainingStage.TRAINING_GATE_FAILED,
            active_checkpoint=None,
        )
        self.state_store.save(cursor, trainer)
        registry_path = self.checkpoint_dir / "national_test_sac_active.json"
        registry_path.unlink(missing_ok=True)
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
) -> Optional[TrainingDiagnostics]:
    if not transitions:
        return None
    return trainer.learn_from_episode(
        transitions,
        maximum_updates=64,
        transitions_per_update=2,
        demonstration=False,
    )


class UnityTrainingGate:
    """Adapt until a route pass, then run five frozen validation episodes."""

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
        self.last_training_diagnostics: Optional[TrainingDiagnostics] = None
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
            self.last_training_diagnostics = None
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
            self.last_training_diagnostics = train_from_unity_episode(
                self.trainer,
                transitions,
            )
            adapt_episodes = self.cursor.unity_adapt_episodes + 1
            self.cursor = replace(
                self.cursor,
                unity_adapt_episodes=adapt_episodes,
                stage=(
                    SelfTrainingStage.UNITY_VALIDATION
                    if adapt_episodes >= UNITY_ADAPT_EPISODES and passed
                    else SelfTrainingStage.UNITY_ADAPT
                ),
            )
            self._save_new_active()
            return self.cursor

        self.last_training_diagnostics = None
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
        self.registry.path.unlink(missing_ok=True)
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
    "validate_checkpoint_manifest",
]
