"""Focused offline-to-Unity gate contracts."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from usvlib4ros.navigation.fixed_map_service import FixedMapNavigationService
from usvlib4ros.planning import Control
from usvlib4ros.policy.fixed_map_trainer import (
    EpisodeSummary,
    control_transition_reward,
)
from usvlib4ros.policy import self_training
from usvlib4ros.policy.recurrent_sac import LocalWaypointObservationV3
from usvlib4ros.policy.self_training import (
    ActiveCheckpointRegistry,
    OFFLINE_BLOCK_EPISODES,
    OFFLINE_EVALUATION_EPISODES,
    SelfTrainingStage,
    TrainingCursor,
    UNITY_ADAPT_EPISODES,
    UNITY_VALIDATION_EPISODES,
    UnityTrainingGate,
)


class _Store:
    def __init__(self, cursor, trainer) -> None:
        self.cursor = cursor
        self.trainer = trainer
        self.saved = []

    def restore_trainer(self):
        return self.cursor, self.trainer

    def save(self, cursor, trainer):
        self.cursor = cursor
        self.trainer = trainer
        self.saved.append(cursor)


def _write_manifest(
    checkpoint: Path,
    stage: SelfTrainingStage,
    *,
    offline_passes: int,
    offline_evaluations: int = 20,
    training_episodes: int = 100,
    unity_adapt: int = 0,
    unity_validation: int = 0,
    unity_passes: int = 0,
) -> None:
    checkpoint.with_suffix(checkpoint.suffix + ".json").write_text(
        json.dumps(
            {
                "schema_version": "national-test-sac-checkpoint-v6",
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "stage": stage.value,
                "gate_evidence": {
                    "completed_training_episodes": training_episodes,
                    "offline_evaluations": offline_evaluations,
                    "offline_evaluation_passes": offline_passes,
                    "unity_adapt_episodes": unity_adapt,
                    "unity_validation_episodes": unity_validation,
                    "unity_validation_passes": unity_passes,
                },
            }
        ),
        encoding="utf-8",
    )


def test_training_schedule_is_exact_and_budgeted():
    cursor = TrainingCursor(seed=20260805)

    assert OFFLINE_BLOCK_EPISODES == 100
    assert OFFLINE_EVALUATION_EPISODES == 20
    assert UNITY_ADAPT_EPISODES == 5
    assert UNITY_VALIDATION_EPISODES == 5
    assert cursor.stage is SelfTrainingStage.OFFLINE_TRAIN
    assert cursor.completed_training_episodes == 0


def test_promoted_registry_accepts_direct_unity_but_rejects_fake_offline_gate(
    tmp_path: Path,
):
    checkpoint = tmp_path / "forced.pt"
    checkpoint.write_bytes(b"forced")
    _write_manifest(
        checkpoint,
        SelfTrainingStage.PROMOTED,
        offline_passes=20,
        training_episodes=68,
        unity_adapt=5,
        unity_validation=5,
        unity_passes=5,
    )

    with pytest.raises(ValueError, match="promotion gate evidence"):
        ActiveCheckpointRegistry(
            tmp_path / "national_test_sac_active.json"
        ).write(checkpoint, SelfTrainingStage.PROMOTED)

    direct = tmp_path / "direct-unity.pt"
    direct.write_bytes(b"direct-unity")
    _write_manifest(
        direct,
        SelfTrainingStage.PROMOTED,
        offline_evaluations=0,
        offline_passes=0,
        training_episodes=70,
        unity_adapt=5,
        unity_validation=5,
        unity_passes=5,
    )
    registry = ActiveCheckpointRegistry(
        tmp_path / "national_test_sac_active.json"
    )
    registry.write(direct, SelfTrainingStage.PROMOTED)
    assert registry.resolve(direct) == direct


def test_random_start_completion_is_not_a_full_route_pass():
    summary = EpisodeSummary(
        session_id="random-near-finish",
        total_reward=1.0,
        steps=4,
        start_mission_index=12,
        ending_mission_index=13,
        waypoints_completed=1,
        full_route=False,
        completed=True,
        collision=False,
        timed_out=False,
        no_safe_action=False,
        safety_interventions=0,
        maximum_cross_track_m=0.0,
        minimum_clearance_m=1.0,
        end_reason="MISSION_COMPLETE",
    )

    assert not summary.passed


def test_unity_reset_rearms_an_inactive_training_task():
    data = SimpleNamespace(
        device_data=SimpleNamespace(
            work_model=0,
            task_status=0,
            reset_status=0,
            reset_request_time=1.0,
        ),
        updateThrottleRudderOutput=lambda *_: None,
    )

    class _Ros:
        task_requests = 0

        def reset_unity(self):
            data.device_data.reset_status = 1
            data.device_data.reset_request_time = 2.0
            return True

        def set_auto_work(self):
            data.device_data.work_model = 2
            return True

        def set_task(self):
            self.task_requests += 1
            data.device_data.task_status = 2
            return {"code": 0}

    class _Stop:
        def is_set(self):
            return False

        def wait(self, _seconds):
            data.device_data.reset_status = 2

    service = FixedMapNavigationService.__new__(FixedMapNavigationService)
    service.ros_ctrl = _Ros()
    service.global_data = data
    service._stop = _Stop()

    assert service._request_unity_reset(timeout_s=0.1)
    assert service._request_unity_episode_start(timeout_s=0.1)
    assert service.ros_ctrl.task_requests == 1


def _reward_observation(
    distance_to_waypoint: float,
    safe_mask: tuple[bool, ...],
) -> LocalWaypointObservationV3:
    return LocalWaypointObservationV3(
        laser_ranges=(20.0,) * 72,
        laser_valid_mask=(True,) * 72,
        scan_age_s=0.0,
        pose_age_s=0.0,
        device_age_s=0.0,
        speed_mps=0.2,
        yaw_rate_rad_s=0.0,
        actual_throttle=0.1,
        actual_rudder=0.0,
        current_waypoint_body_xy=(distance_to_waypoint, 0.0),
        next_waypoint_body_xy=(distance_to_waypoint + 2.0, 0.0),
        next_waypoint_valid=True,
        mission_progress=0.0,
        corridor_cross_track_m=0.1,
        corridor_heading_error_rad=0.0,
        corridor_progress=0.1,
        map_clearance_m=1.0,
        safe_action_mask=safe_mask,
        session_id="reward-contract",
        stamp_sim=0.0,
        hidden_reset=False,
    )


def test_reward_prefers_waypoint_progress_and_preserved_safety_margin():
    observation = _reward_observation(2.0, (True,) * 5)
    safer = _reward_observation(1.5, (True,) * 5)
    riskier = _reward_observation(
        2.5,
        (True, False, False, False, False),
    )
    control = Control(0.1, 0.0)

    safer_reward = control_transition_reward(
        observation,
        safer,
        control,
        control,
        mission_delta=0,
        completed=False,
        terminated=False,
        truncated=False,
        reason="STEP",
    )
    riskier_reward = control_transition_reward(
        observation,
        riskier,
        control,
        control,
        mission_delta=0,
        completed=False,
        terminated=False,
        truncated=False,
        reason="STEP",
    )

    assert safer_reward > riskier_reward


def test_unity_gate_adapts_until_a_full_route_pass_then_freezes(
    tmp_path: Path,
    monkeypatch,
):
    active = tmp_path / "active.pt"
    active.write_bytes(b"candidate")
    _write_manifest(
        active,
        SelfTrainingStage.UNITY_ADAPT,
        offline_passes=20,
    )
    ActiveCheckpointRegistry(
        tmp_path / "national_test_sac_active.json"
    ).write(active, SelfTrainingStage.UNITY_ADAPT)
    cursor = TrainingCursor(
        seed=19,
        stage=SelfTrainingStage.UNITY_ADAPT,
        completed_training_episodes=100,
        offline_evaluations=20,
        offline_evaluation_passes=20,
        active_checkpoint=active.name,
    )
    trainer = SimpleNamespace(replay=SimpleNamespace(add_episode=lambda _: None))
    store = _Store(cursor, trainer)
    updates = []
    checkpoint_index = 0

    def train_once(_trainer, transitions):
        updates.append(transitions)

    def save_checkpoint(directory, _trainer, current):
        nonlocal checkpoint_index
        checkpoint_index += 1
        path = Path(directory) / f"stage-{checkpoint_index}.pt"
        path.write_bytes(f"stage-{checkpoint_index}".encode())
        _write_manifest(
            path,
            current.stage,
            offline_passes=current.offline_evaluation_passes,
            unity_adapt=current.unity_adapt_episodes,
            unity_validation=current.unity_validation_episodes,
            unity_passes=current.unity_validation_passes,
        )
        return path, replace(current, active_checkpoint=path.name)

    monkeypatch.setattr(self_training, "train_from_unity_episode", train_once)
    monkeypatch.setattr(self_training, "save_stage_checkpoint", save_checkpoint)
    monkeypatch.setattr(self_training, "update_checkpoint_stage", lambda *_: None)

    gate = UnityTrainingGate(state_store=store, checkpoint_dir=tmp_path)
    for _ in range(UNITY_ADAPT_EPISODES):
        gate.finish_episode(
            (),
            counted=True,
            passed=False,
            operator_truncated=False,
        )

    assert gate.cursor.stage is SelfTrainingStage.UNITY_ADAPT
    assert gate.cursor.unity_adapt_episodes == 5
    assert len(updates) == 5

    gate.finish_episode(
        (),
        counted=True,
        passed=True,
        operator_truncated=False,
    )

    assert gate.cursor.stage is SelfTrainingStage.UNITY_VALIDATION
    assert gate.cursor.unity_adapt_episodes == 6
    assert len(updates) == 6

    for _ in range(UNITY_VALIDATION_EPISODES):
        gate.finish_episode(
            (),
            counted=True,
            passed=True,
            operator_truncated=False,
        )

    assert gate.cursor.stage is SelfTrainingStage.PROMOTED
    assert gate.cursor.unity_validation_passes == 5
    assert len(updates) == 6
