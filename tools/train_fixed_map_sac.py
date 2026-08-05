"""Run the strict National_Test offline SAC admission gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from usvlib4ros.policy.self_training import (
    OfflineTrainingGate,
    SelfTrainingStage,
    TrainingStateStore,
    load_calibration,
)


CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
STATE_PATH = CHECKPOINT_DIR / "national_test_self_training_v2.pt"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-log", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    profile = load_calibration(args.calibration_log)

    def progress(cursor, summary) -> None:
        print(
            json.dumps(
                {
                    "stage": cursor.stage.value,
                    "generation": cursor.generation,
                    "training_episodes": cursor.completed_training_episodes,
                    "block_progress": cursor.offline_block_progress,
                    "evaluations": cursor.offline_evaluations,
                    "evaluation_passes": cursor.offline_evaluation_passes,
                    "episode_passed": summary.passed,
                    "completed_waypoints": summary.completed_waypoints,
                    "end_reason": summary.end_reason,
                    "reward": summary.total_reward,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    gate = OfflineTrainingGate(
        profile=profile,
        seed=args.seed,
        state_store=TrainingStateStore(STATE_PATH),
        checkpoint_dir=CHECKPOINT_DIR,
        progress=progress,
    )
    cursor, _ = gate.run()
    print(
        json.dumps(
            {
                "stage": cursor.stage.value,
                "training_episodes": cursor.completed_training_episodes,
                "active_checkpoint": cursor.active_checkpoint,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if cursor.stage is SelfTrainingStage.UNITY_ADAPT else 2


if __name__ == "__main__":
    raise SystemExit(main())
