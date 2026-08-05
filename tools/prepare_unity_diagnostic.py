"""Export the current V6 training state for explicit, uncounted Unity diagnosis."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from usvlib4ros.policy.self_training import (
    SelfTrainingStage,
    TrainingStateStore,
    save_stage_checkpoint,
)


CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
STATE_PATH = CHECKPOINT_DIR / "national_test_self_training_v2.pt"


def main() -> int:
    cursor, trainer = TrainingStateStore(STATE_PATH).restore_trainer()
    if cursor.stage is not SelfTrainingStage.OFFLINE_TRAIN:
        raise ValueError(f"diagnostic export requires OFFLINE_TRAIN, got {cursor.stage.value}")
    if cursor.completed_training_episodes <= 0:
        raise ValueError("diagnostic export requires completed training episodes")
    diagnostic = replace(
        cursor,
        stage=SelfTrainingStage.OFFLINE_EVAL,
        offline_evaluations=0,
        offline_evaluation_passes=0,
        active_checkpoint=None,
    )
    checkpoint, _ = save_stage_checkpoint(CHECKPOINT_DIR, trainer, diagnostic)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "stage": diagnostic.stage.value,
                "training_episodes": diagnostic.completed_training_episodes,
                "registered_active": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
