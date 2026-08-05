"""Focused offline-to-Unity gate contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from usvlib4ros.policy import self_training
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


def test_training_schedule_is_exact_and_budgeted():
    cursor = TrainingCursor(seed=20260805)

    assert OFFLINE_BLOCK_EPISODES == 100
    assert OFFLINE_EVALUATION_EPISODES == 20
    assert UNITY_ADAPT_EPISODES == 5
    assert UNITY_VALIDATION_EPISODES == 5
    assert cursor.stage is SelfTrainingStage.OFFLINE_TRAIN
    assert cursor.completed_training_episodes == 0


def test_unity_gate_updates_only_five_adaptation_episodes_then_freezes(
    tmp_path: Path,
    monkeypatch,
):
    active = tmp_path / "active.pt"
    active.write_bytes(b"candidate")
    ActiveCheckpointRegistry(
        tmp_path / "national_test_sac_active.json"
    ).write(active, SelfTrainingStage.UNITY_ADAPT)
    cursor = TrainingCursor(
        seed=19,
        stage=SelfTrainingStage.UNITY_ADAPT,
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

    assert gate.cursor.stage is SelfTrainingStage.UNITY_VALIDATION
    assert gate.cursor.unity_adapt_episodes == 5
    assert len(updates) == 5

    for _ in range(UNITY_VALIDATION_EPISODES):
        gate.finish_episode(
            (),
            counted=True,
            passed=True,
            operator_truncated=False,
        )

    assert gate.cursor.stage is SelfTrainingStage.PROMOTED
    assert gate.cursor.unity_validation_passes == 5
    assert len(updates) == 5
