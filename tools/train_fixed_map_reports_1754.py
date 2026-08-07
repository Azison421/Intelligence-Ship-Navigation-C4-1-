"""Train an isolated 1,754-episode model and export report-only evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.export_training_reports import export_offline
from usvlib4ros.policy.fixed_map_trainer import FixedMapSACTrainer
from usvlib4ros.policy.self_training import load_calibration


DEFAULT_EPISODES = 1_754
REPORT_DIR = PROJECT_ROOT / "reports" / "national_test_v6_offline_1754"
WORKING_LOG = REPORT_DIR / ".training.jsonl"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-log", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def _payload(episode: int, summary, training) -> dict[str, object]:
    return {
        "schema_version": "national-test-offline-training-v3",
        "stage": "OFFLINE_TRAIN",
        "generation": 1,
        "training_episodes": episode,
        "block_progress": ((episode - 1) % 100) + 1,
        "evaluations": 0,
        "evaluation_passes": 0,
        "full_route": summary.full_route,
        "gate_eligible": False,
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
        "attempted_updates": training.attempted_updates if training else 0,
        "applied_updates": training.applied_updates if training else 0,
        "critic_loss": training.critic_loss if training else None,
        "actor_loss": training.actor_loss if training else None,
        "actor_objective": training.actor_objective if training else None,
        "behavior_clone_loss": (
            training.behavior_clone_loss if training else None
        ),
        "alpha": training.alpha if training else None,
        "entropy": training.entropy if training else None,
    }


def main() -> int:
    args = _arguments()
    if isinstance(args.episodes, bool) or args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if REPORT_DIR.exists() and any(REPORT_DIR.iterdir()):
        raise SystemExit(f"refusing to overwrite report directory: {REPORT_DIR}")

    profile = load_calibration(args.calibration_log)
    trainer = FixedMapSACTrainer(profile, seed=args.seed)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with WORKING_LOG.open("x", encoding="utf-8") as stream:
        for episode in range(1, args.episodes + 1):
            result = trainer.run_episode(
                episode=episode,
                training=True,
                deterministic=False,
                full_route=(episode % 4 == 0),
            )
            stream.write(
                json.dumps(
                    _payload(episode, result.summary, result.training),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()

    export_offline([WORKING_LOG], REPORT_DIR)
    WORKING_LOG.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
