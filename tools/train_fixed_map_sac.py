"""Run the strict National_Test offline SAC admission gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from usvlib4ros.policy.self_training import (
    OfflineTrainingGate,
    SelfTrainingStage,
    TrainingStateStore,
    load_calibration,
)


CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints"
STATE_PATH = CHECKPOINT_DIR / "national_test_self_training_v6.pt"
PROGRESS_PATH = PROJECT_ROOT / "artifacts" / "logs" / "national_test_training_progress.json"
RUN_LOG_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / f"national-test-offline-training-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-log", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--stop-after-training-episodes", type=int)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    profile = load_calibration(args.calibration_log)

    def progress(cursor, summary, training) -> None:
        payload = {
            "schema_version": "national-test-offline-training-v3",
            "stage": cursor.stage.value,
            "generation": cursor.generation,
            "training_episodes": cursor.completed_training_episodes,
            "block_progress": cursor.offline_block_progress,
            "evaluations": cursor.offline_evaluations,
            "evaluation_passes": cursor.offline_evaluation_passes,
            "full_route": summary.full_route,
            "gate_eligible": cursor.stage is SelfTrainingStage.OFFLINE_EVAL,
            "episode_passed": summary.passed,
            "steps": summary.steps,
            "start_mission_index": summary.start_mission_index,
            "ending_mission_index": summary.ending_mission_index,
            "waypoints_completed": summary.waypoints_completed,
            "end_reason": summary.end_reason,
            "reward": summary.total_reward,
            "collision": summary.collision,
            "timed_out": summary.timed_out,
            "no_safe_action": summary.no_safe_action,
            "safety_interventions": summary.safety_interventions,
            "maximum_cross_track_m": summary.maximum_cross_track_m,
            "minimum_clearance_m": summary.minimum_clearance_m,
            "attempted_updates": (
                training.attempted_updates if training is not None else 0
            ),
            "applied_updates": (
                training.applied_updates if training is not None else 0
            ),
            "critic_loss": (
                training.critic_loss if training is not None else None
            ),
            "actor_loss": (
                training.actor_loss if training is not None else None
            ),
            "actor_objective": (
                training.actor_objective if training is not None else None
            ),
            "behavior_clone_loss": (
                training.behavior_clone_loss if training is not None else None
            ),
            "alpha": training.alpha if training is not None else None,
            "entropy": training.entropy if training is not None else None,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        print(encoded, flush=True)
        PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROGRESS_PATH.write_text(encoded + "\n", encoding="utf-8")
        with RUN_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")

    gate = OfflineTrainingGate(
        profile=profile,
        seed=args.seed,
        state_store=TrainingStateStore(STATE_PATH),
        checkpoint_dir=CHECKPOINT_DIR,
        stop_after_training_episodes=args.stop_after_training_episodes,
        progress=progress,
    )
    cursor, _ = gate.run()
    paused = (
        args.stop_after_training_episodes is not None
        and cursor.completed_training_episodes
        >= args.stop_after_training_episodes
        and cursor.stage is SelfTrainingStage.OFFLINE_TRAIN
    )
    print(
        json.dumps(
            {
                "stage": cursor.stage.value,
                "training_episodes": cursor.completed_training_episodes,
                "active_checkpoint": cursor.active_checkpoint,
                "paused": paused,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if cursor.stage is SelfTrainingStage.UNITY_ADAPT or paused else 2


if __name__ == "__main__":
    raise SystemExit(main())
