from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from usvlib4ros.planning import Control
from usvlib4ros.navigation.fixed_map_runtime import RuntimeTrainingTrace
from usvlib4ros.navigation.fixed_map_service import FixedMapNavigationService
from usvlib4ros.policy.recurrent_sac import (
    LocalObservationV2,
    SequenceReplay,
    SequenceTransition,
)
from usvlib4ros.policy.self_training import (
    ACTIVE_CHECKPOINT_SCHEMA,
    ActiveCheckpointRegistry,
    EvaluationSummary,
    GenerationEvidence,
    PromotionDecision,
    SelfTrainingConfig,
    SelfTrainingCursor,
    SelfTrainingStage,
    SelfTrainingStateStore,
    SelfTrainingLearner,
    UnityTransitionRecorder,
    bounded_safe_action_controls,
    concatenate_replay_batches,
    operational_profile_from_manifest,
    promotion_decision,
    save_generation_checkpoint,
)


def _observation(stamp: float, session: str) -> LocalObservationV2:
    return LocalObservationV2(
        laser_ranges=(1.0,) * 72,
        laser_valid_mask=(True,) * 72,
        scan_age_s=0.0,
        ego_features=(0.1, 0.0, 0.0, 0.0),
        path_features=(0.0,) * 8,
        target_features=(0.0,) * 6,
        target_mask=(False,),
        safety_features=(1.0, 0.0, 0.0),
        event_features=(0.0, 0.0, 0.0),
        session_id=session,
        stamp_sim=stamp,
    )


def _episode(session: str, length: int = 2) -> list[SequenceTransition]:
    transitions: list[SequenceTransition] = []
    for index in range(length):
        transitions.append(
            SequenceTransition(
                observation=_observation(float(index), session),
                next_observation=_observation(float(index + 1), session),
                executed_action=2,
                reward=1.0,
                terminated=index == length - 1,
                timeout=False,
                safety_truncation=False,
                safe_action_mask=(True,) * 5,
                hidden_reset=index == 0,
            )
        )
    return transitions


def test_default_self_training_schedule_is_exact_and_extends_in_1000_episode_blocks():
    config = SelfTrainingConfig()
    assert config.baseline_unity_episodes == 5
    assert config.offline_training_episodes == 95
    assert config.unity_training_episodes == 5
    assert config.offline_evaluation_episodes == 20
    assert config.unity_validation_episodes == 5
    assert config.updates_per_offline_episode == 16
    assert config.updates_after_unity_block == 80
    assert config.batch_size == 8
    assert config.offline_batch_size == 6
    assert config.unity_batch_size == 2
    assert config.cpu_threads == 8

    cursor = SelfTrainingCursor.new(config, champion_path="seed.pt", champion_sha256="a" * 64)
    assert cursor.stage is SelfTrainingStage.BASELINE_UNITY
    assert cursor.target_training_episodes == 1000

    cursor = replace(cursor, stage_index=4).advance(config)
    assert cursor.stage is SelfTrainingStage.OFFLINE_TRAIN
    assert cursor.generation == 1
    cursor = replace(cursor, stage_index=94, completed_training_episodes=94).advance(config)
    assert cursor.stage is SelfTrainingStage.UNITY_TRAIN
    assert cursor.completed_training_episodes == 95
    cursor = replace(cursor, stage_index=4, completed_training_episodes=99).advance(config)
    assert cursor.stage is SelfTrainingStage.OFFLINE_EVAL
    assert cursor.completed_training_episodes == 100
    cursor = replace(cursor, stage=SelfTrainingStage.UNITY_VALIDATION, stage_index=4).advance(config)
    assert cursor.stage is SelfTrainingStage.OFFLINE_TRAIN
    assert cursor.generation == 2

    complete = replace(
        cursor,
        stage=SelfTrainingStage.UNITY_VALIDATION,
        stage_index=4,
        generation=10,
        completed_training_episodes=1000,
    ).advance(config)
    assert complete.stage is SelfTrainingStage.WAITING
    extended = complete.extend_target(config)
    assert extended.target_training_episodes == 2000
    assert extended.generation == 11
    assert extended.stage is SelfTrainingStage.OFFLINE_TRAIN


def test_promotion_requires_both_strict_gates_and_two_percent_step_gain_on_tie():
    champion = EvaluationSummary(completed=5, attempted=5, total_steps=(1000, 1000, 1000, 1000, 1000))
    faster = EvaluationSummary(completed=5, attempted=5, total_steps=(970, 970, 970, 970, 970))
    too_close = EvaluationSummary(completed=5, attempted=5, total_steps=(985, 985, 985, 985, 985))
    perfect_offline = EvaluationSummary(completed=20, attempted=20, total_steps=(900,) * 20)

    assert promotion_decision(champion, perfect_offline, faster).promote is True
    decision = promotion_decision(champion, perfect_offline, too_close)
    assert decision.promote is False
    assert decision.reason == "UNITY_MEDIAN_STEP_GAIN_BELOW_2_PERCENT"

    collision = replace(faster, collisions=1)
    assert promotion_decision(champion, perfect_offline, collision).reason == "UNITY_SAFETY_GATE_FAILED"
    assert promotion_decision(
        champion,
        perfect_offline,
        faster,
        generation_collision=True,
    ).reason == "UNITY_SAFETY_GATE_FAILED"
    unsafe_offline = replace(perfect_offline, safety_stops=1)
    assert promotion_decision(champion, unsafe_offline, faster).reason == "OFFLINE_GATE_FAILED"


def test_mixed_replay_batch_is_six_offline_and_two_unity_when_unity_exists():
    offline = SequenceReplay(capacity=32, seed=3)
    unity = SequenceReplay(capacity=20, seed=5)
    offline.add_episode(_episode("offline-a"))
    unity.add_episode(_episode("unity-a"))

    batch = concatenate_replay_batches(
        offline.sample(batch_size=6, burn_in=0, unroll=1),
        unity.sample(batch_size=2, burn_in=0, unroll=1),
    )

    assert batch.observations.shape[0] == 8
    assert batch.session_ids.count("offline-a") == 6
    assert batch.session_ids.count("unity-a") == 2


def test_state_store_round_trips_only_complete_episode_state_and_rejects_corruption(tmp_path: Path):
    config = SelfTrainingConfig()
    store = SelfTrainingStateStore(tmp_path / "session")
    cursor = SelfTrainingCursor.new(config, champion_path="seed.pt", champion_sha256="b" * 64)
    replay = SequenceReplay(capacity=32, seed=7)
    replay.add_episode(_episode("offline-roundtrip"))

    store.save(cursor, offline_replay=replay, unity_replay=SequenceReplay(capacity=20, seed=11))
    restored_cursor, restored_offline, restored_unity = store.load()

    assert restored_cursor == cursor
    assert len(restored_offline) == 1
    assert len(restored_unity) == 0
    assert restored_offline.capacity == 32
    assert restored_unity.capacity == 20

    (store.directory / "state.sha256").write_text("0" * 64, encoding="ascii")
    with pytest.raises(ValueError, match="integrity"):
        store.load()


def test_special_segment_has_five_distinct_candidates_inside_hard_limits():
    candidates = bounded_safe_action_controls(
        Control(0.1, 0.12),
        throttle_cap=0.1,
        rudder_cap=0.12,
    )

    assert len(candidates) == 5
    assert len({(control.throttle, control.rudder) for control in candidates}) == 5
    assert all(0.0 < control.throttle <= 0.1 for control in candidates)
    assert all(abs(control.rudder) <= 0.12 for control in candidates)
    assert candidates[2] == Control(0.1, 0.12)

    with pytest.raises(ValueError, match="hard limits"):
        bounded_safe_action_controls(
            Control(0.1, 0.12),
            throttle_cap=float("nan"),
            rudder_cap=0.12,
        )


def test_active_checkpoint_pointer_is_atomic_hash_checked_and_explicit_path_is_pinned(tmp_path: Path):
    checkpoint = tmp_path / "generation-001.pt"
    checkpoint.write_bytes(b"generation-one")
    digest = __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "national-test-sac-checkpoint-v5",
        "checkpoint_sha256": digest,
        "offline_ready": True,
        "live_ready": True,
        "safety_profile": {"unity_test_only": False},
        "clearance_maneuver_profile": {"unity_test_only": False},
    }
    checkpoint.with_suffix(".pt.json").write_text(
        __import__("json").dumps(manifest),
        encoding="utf-8",
    )
    registry = ActiveCheckpointRegistry(tmp_path / "national_test_sac_active.json")

    registry.promote(checkpoint)

    pointer = __import__("json").loads(registry.path.read_text(encoding="utf-8"))
    assert pointer["schema_version"] == ACTIVE_CHECKPOINT_SCHEMA
    assert registry.resolve(tmp_path / "fallback.pt") == checkpoint.resolve()
    explicit = tmp_path / "old-v37.pt"
    assert registry.resolve(tmp_path / "fallback.pt", explicit=explicit) == explicit

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        registry.resolve(tmp_path / "fallback.pt")


def test_operational_profile_parses_seed_zero_clearance_and_conservative_limits():
    manifest_path = Path(
        "artifacts/checkpoints/"
        "national_test_sac_v37_zero_clearance_conservative_345_unity_test.pt.json"
    )
    profile = operational_profile_from_manifest(
        __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    )

    assert profile.required_clearance_m == 0.0
    assert profile.laser_emergency_distance_m == 0.0
    assert profile.point3_to_4_throttle_cap == 0.1
    assert profile.point3_to_4_rudder_cap == 0.1
    assert profile.point4_to_5_throttle_cap == 0.1
    assert profile.point4_to_5_rudder_cap == 0.12

    manifest = __import__("json").loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["safety_profile"]["required_clearance_m"] = float("nan")
    with pytest.raises(ValueError, match="checkpoint profiles are invalid"):
        operational_profile_from_manifest(manifest)


def test_learner_uses_pure_offline_then_six_two_mixed_batches_and_updates_weights(
    tmp_path: Path,
):
    observation = _observation(0.0, "schema")
    agent = __import__(
        "usvlib4ros.policy.recurrent_sac", fromlist=["RecurrentDiscreteSAC"]
    ).RecurrentDiscreteSAC(
        observation_dim=observation.feature_dim,
        hidden_dim=8,
        seed=17,
    )
    config = SelfTrainingConfig(
        updates_per_offline_episode=1,
        updates_after_unity_block=1,
        burn_in=0,
        unroll=1,
    )
    learner = SelfTrainingLearner(agent, config)
    learner.add_offline_episode(_episode("offline-learn"))
    before_checkpoint = agent.save_checkpoint(tmp_path / "before.pt")

    offline_batch = learner.sample_batch()
    assert offline_batch.observations.shape[0] == 8
    assert set(offline_batch.session_ids) == {"offline-learn"}
    before = {name: value.clone() for name, value in agent.actor.state_dict().items()}
    metrics = learner.update(1)
    assert metrics["updated"] is True
    assert agent.training_step == 1
    assert any(not torch.equal(value, before[name]) for name, value in agent.actor.state_dict().items())
    after_checkpoint = agent.save_checkpoint(tmp_path / "after.pt")
    assert __import__("hashlib").sha256(before_checkpoint.read_bytes()).digest() != (
        __import__("hashlib").sha256(after_checkpoint.read_bytes()).digest()
    )

    learner.add_unity_episode(_episode("unity-learn"))
    mixed = learner.sample_batch()
    assert mixed.session_ids.count("offline-learn") == 6
    assert mixed.session_ids.count("unity-learn") == 2


def test_generation_checkpoint_is_v5_immutable_and_records_lineage_and_evidence(tmp_path: Path):
    observation = _observation(0.0, "checkpoint")
    recurrent = __import__(
        "usvlib4ros.policy.recurrent_sac", fromlist=["RecurrentDiscreteSAC"]
    ).RecurrentDiscreteSAC(observation.feature_dim, hidden_dim=8, seed=23)
    parent = tmp_path / "parent.pt"
    recurrent.save_checkpoint(parent)
    parent_digest = __import__("hashlib").sha256(parent.read_bytes()).hexdigest()
    parent_manifest = {
        "schema_version": "national-test-sac-checkpoint-v4",
        "checkpoint_sha256": parent_digest,
        "safety_profile": {
            "id": "zero",
            "required_clearance_m": 0.0,
            "laser_emergency_distance_m": 0.0,
            "unity_test_only": True,
        },
        "clearance_maneuver_profile": {
            "id": "conservative",
            "approach_throttle_cap": 0.1,
            "approach_rudder_cap": 0.1,
            "turn_throttle": 0.1,
            "turn_rudder": 0.12,
            "turn_max_edges": 180,
            "turn_entry_speed_limit_mps": 0.15,
            "unity_test_only": True,
        },
    }
    parent.with_suffix(".pt.json").write_text(
        __import__("json").dumps(parent_manifest), encoding="utf-8"
    )
    config = SelfTrainingConfig()
    cursor = replace(
        SelfTrainingCursor.new(config, champion_path=str(parent), champion_sha256=parent_digest),
        generation=1,
        stage=SelfTrainingStage.UNITY_VALIDATION,
        completed_training_episodes=100,
    )
    evidence = GenerationEvidence(
        offline=EvaluationSummary(20, 20, (800,) * 20),
        unity=EvaluationSummary(5, 5, (900,) * 5),
        champion_unity=EvaluationSummary(5, 5, (1000,) * 5),
    )
    target = tmp_path / "generation-001.pt"

    checkpoint, manifest_path = save_generation_checkpoint(
        recurrent,
        target,
        parent_checkpoint=parent,
        cursor=cursor,
        config=config,
        evidence=evidence,
        decision=promotion_decision(evidence.champion_unity, evidence.offline, evidence.unity),
    )

    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    assert checkpoint == target
    assert manifest["schema_version"] == "national-test-sac-checkpoint-v5"
    assert manifest["parent_model_sha256"] == parent_digest
    assert manifest["training_lineage"]["generation"] == 1
    assert manifest["training_lineage"]["completed_training_episodes"] == 100
    assert manifest["policy_gate_version"] == "sac-predictive-safe-mask-v1"
    assert manifest["offline_ready"] is True
    assert manifest["live_ready"] is True
    assert manifest["safety_profile"]["unity_test_only"] is False
    with pytest.raises(FileExistsError):
        save_generation_checkpoint(
            recurrent,
            target,
            parent_checkpoint=parent,
            cursor=cursor,
            config=config,
            evidence=evidence,
            decision=PromotionDecision(True, "again"),
        )


def _trace(stamp: float, distance: float, mission_index: int = 3) -> RuntimeTrainingTrace:
    return RuntimeTrainingTrace(
        observation=_observation(stamp, "unity-continuous"),
        executed_action=2,
        safe_action_mask=(True,) * 5,
        mission_index=mission_index,
        distance_to_goal_m=distance,
        cross_track_error_m=0.1,
        map_clearance_m=0.05,
    )


def test_unity_transition_recorder_is_continuous_ignores_duplicate_samples_and_marks_collision_negative():
    recorder = UnityTransitionRecorder()
    assert recorder.observe(_trace(1.0, 2.0)) is True
    assert recorder.observe(_trace(1.0, 1.9)) is False
    assert recorder.observe(_trace(0.9, 1.8)) is False
    assert recorder.observe(_trace(1.1, 1.5)) is True

    episode = recorder.finish(collision=True)

    assert len(episode) == 2
    assert episode[0].observation.stamp_sim == 1.0
    assert episode[0].next_observation.stamp_sim == 1.1
    assert episode[-1].reward == -25.0
    assert episode[-1].terminated is True
    assert episode[-1].timeout is False
    assert episode[-1].executed_action == 2

    interrupted = UnityTransitionRecorder()
    interrupted.observe(_trace(2.0, 1.0))
    interrupted.discard_partial()
    assert interrupted.transitions == ()


def test_service_initializes_resumable_session_from_v37_without_unity(tmp_path: Path):
    source = (
        Path("artifacts/checkpoints")
        / "national_test_sac_v37_zero_clearance_conservative_345_unity_test.pt"
    )
    output = type(
        "Output",
        (),
        {
            "updateThrottleRudderOutput": lambda *args: None,
            "updateAlgorithmOutput": lambda *args: None,
        },
    )()
    service = FixedMapNavigationService(
        object(),
        output,
        checkpoint_path=source,
        policy_mode="unity_test",
        self_training=True,
        reports_dir=tmp_path,
    )

    config, store, cursor, metadata, trainer, learner = service._load_self_training_session()

    assert cursor.stage is SelfTrainingStage.BASELINE_UNITY
    assert cursor.target_training_episodes == 1000
    assert trainer.compiled_map.snapshot.required_clearance == 0.0
    assert learner.offline_replay.capacity == 32
    assert learner.unity_replay.capacity == 20
    assert metadata["config"]["offline_training_episodes"] == 95
    snapshot = store.load()
    assert snapshot.cursor == cursor
    assert snapshot.training_state is not None
