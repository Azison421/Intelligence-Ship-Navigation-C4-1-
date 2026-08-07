"""Continue the current model in counted Unity adaptation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from usvlib4ros.policy.self_training import (
    ActiveCheckpointRegistry,
    SelfTrainingStage,
    TrainingStateStore,
    save_stage_checkpoint,
    update_checkpoint_stage,
)


CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
STATE_PATH = CHECKPOINT_DIR / "national_test_self_training_v6.pt"


def main() -> int:
    state_store = TrainingStateStore(STATE_PATH)
    cursor, trainer = state_store.restore_trainer()
    if cursor.stage not in {
        SelfTrainingStage.OFFLINE_TRAIN,
        SelfTrainingStage.UNITY_VALIDATION,
    }:
        raise ValueError(
            "Unity adaptation requires OFFLINE_TRAIN or unused UNITY_VALIDATION, "
            f"got {cursor.stage.value}"
        )
    if cursor.completed_training_episodes <= 0:
        raise ValueError("Unity adaptation requires completed training episodes")
    if cursor.stage is SelfTrainingStage.UNITY_VALIDATION:
        if (
            cursor.unity_validation_episodes != 0
            or cursor.unity_validation_passes != 0
            or cursor.active_checkpoint is None
        ):
            raise ValueError("started Unity validation cannot return to adaptation")
        adapt = replace(cursor, stage=SelfTrainingStage.UNITY_ADAPT)
        checkpoint = CHECKPOINT_DIR / cursor.active_checkpoint
        update_checkpoint_stage(checkpoint, adapt)
    else:
        adapt = replace(
            cursor,
            stage=SelfTrainingStage.UNITY_ADAPT,
            offline_evaluations=0,
            offline_evaluation_passes=0,
            unity_adapt_episodes=0,
            unity_validation_episodes=0,
            unity_validation_passes=0,
            active_checkpoint=None,
        )
        checkpoint, adapt = save_stage_checkpoint(CHECKPOINT_DIR, trainer, adapt)
    ActiveCheckpointRegistry(
        CHECKPOINT_DIR / "national_test_sac_active.json"
    ).write(checkpoint, adapt.stage)
    state_store.save(adapt, trainer)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "stage": adapt.stage.value,
                "training_episodes": adapt.completed_training_episodes,
                "registered_active": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
