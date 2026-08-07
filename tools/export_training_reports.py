"""Project V6 offline and Unity JSONL evidence into local CSV/SVG reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "artifacts" / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"


def _rows(path: Path) -> list[dict[str, object]]:
    result = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        result.append(value)
    if not result:
        raise ValueError(f"{path} contains no evidence rows")
    return result


def _newest(pattern: str) -> Path:
    matches = sorted(LOG_DIR.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no log matches {pattern}")
    return matches[-1]


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _chart(
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    values: list[tuple[float, float]],
) -> None:
    if not values or any(not math.isfinite(x) or not math.isfinite(y) for x, y in values):
        raise ValueError(f"{title} has invalid values")
    width, height = 900, 520
    left, right, top, bottom = 85, 30, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = min(x for x, _ in values), max(x for x, _ in values)
    y_min, y_max = min(y for _, y in values), max(y for _, y in values)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    x_precision = 1 if x_max - x_min < 5.0 else 0
    padding = 0.05 * (y_max - y_min)
    y_min -= padding
    y_max += padding

    def sx(value: float) -> float:
        return left + (value - x_min) * plot_width / (x_max - x_min)

    def sy(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{html.escape(title)}</title>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Times New Roman, Noto Serif SC, serif" fill="#202020">',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#303030"/>',
    ]
    for index in range(6):
        ratio = index / 5
        x_value = x_min + ratio * (x_max - x_min)
        y_value = y_min + ratio * (y_max - y_min)
        x = sx(x_value)
        y = sy(y_value)
        lines.extend(
            (
                f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 5}" stroke="#303030"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-size="11">{x_value:.{x_precision}f}</text>',
                f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#303030"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{y_value:.2f}</text>',
            )
        )
    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
    lines.extend(
        (
            f'<polyline points="{points}" fill="none" stroke="#1f1f1f" stroke-width="1.8" stroke-linejoin="round"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 24 {top + plot_height / 2})">{html.escape(y_label)}</text>',
            "</g>",
            "</svg>",
        )
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rolling_mean(
    rows: list[dict[str, object]],
    field: str,
    *,
    window: int = 20,
) -> list[tuple[float, float]]:
    result = []
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        values.append(float(value))
        current = values[-window:]
        result.append(
            (
                float(row["training_episodes"]),
                sum(current) / len(current),
            )
        )
    return result


def export_offline(sources: list[Path], output: Path) -> list[Path]:
    raw = []
    seen: set[tuple[object, ...]] = set()
    for source in sources:
        for row in _rows(source):
            key = (
                row.get("stage"),
                row.get("generation"),
                row.get("training_episodes"),
                row.get("block_progress"),
                row.get("evaluations"),
            )
            if key not in seen:
                raw.append(dict(row, source_log=source.name))
                seen.add(key)
    raw.sort(
        key=lambda row: (
            int(row["training_episodes"]),
            int(row["generation"]),
            0 if row["stage"] == "OFFLINE_TRAIN" else 1,
            int(row["evaluations"]),
        )
    )
    if any(
        row.get("schema_version") != "national-test-offline-training-v3"
        for row in raw
    ):
        raise ValueError("only national-test-offline-training-v3 is supported")
    fields = (
        "record_index",
        "source_log",
        "schema_version",
        "stage",
        "generation",
        "training_episodes",
        "block_progress",
        "evaluations",
        "evaluation_passes",
        "full_route",
        "gate_eligible",
        "episode_passed",
        "steps",
        "start_mission_index",
        "ending_mission_index",
        "waypoints_completed",
        "reward",
        "collision",
        "timed_out",
        "no_safe_action",
        "safety_interventions",
        "maximum_cross_track_m",
        "minimum_clearance_m",
        "end_reason",
        "attempted_updates",
        "applied_updates",
        "critic_loss",
        "actor_loss",
        "actor_objective",
        "behavior_clone_loss",
        "alpha",
        "entropy",
    )
    rows = []
    for index, item in enumerate(raw, 1):
        row = dict(item, record_index=index)
        rows.append(row)
    missing = [field for field in fields[1:] if any(field not in row for row in rows)]
    if missing:
        raise ValueError("offline log fields are missing: " + ", ".join(missing))
    csv_path = output / "national_test_v6_offline_episodes.csv"
    reward_path = output / "national_test_v6_offline_reward.svg"
    steps_path = output / "national_test_v6_offline_steps.svg"
    waypoint_path = output / "national_test_v6_offline_waypoints.svg"
    completion_path = output / "national_test_v6_offline_completion_rate.svg"
    critic_path = output / "national_test_v6_critic_loss.svg"
    actor_path = output / "national_test_v6_actor_loss.svg"
    behavior_clone_path = output / "national_test_v6_behavior_clone_loss.svg"
    alpha_path = output / "national_test_v6_alpha.svg"
    updates_path = output / "national_test_v6_applied_updates.svg"
    pass_rate_path = output / "national_test_v6_eval_pass_rate.svg"
    pass_streak_path = output / "national_test_v6_eval_streak.svg"
    _write_csv(csv_path, fields, rows)
    full_route_training = [
        row
        for row in rows
        if row["stage"] == "OFFLINE_TRAIN" and row["full_route"] is True
    ]
    if not full_route_training:
        raise ValueError("offline log has no full-route training episodes")
    x = [float(row["training_episodes"]) for row in full_route_training]
    _chart(
        reward_path,
        "National_Test V6 Full-Route Training Reward",
        "Training episode",
        "Reward",
        list(
            zip(
                x,
                (float(row["reward"]) for row in full_route_training),
            )
        ),
    )
    _chart(
        steps_path,
        "National_Test V6 Full-Route Training Steps",
        "Training episode",
        "Steps",
        list(
            zip(
                x,
                (float(row["steps"]) for row in full_route_training),
            )
        ),
    )
    _chart(
        waypoint_path,
        "National_Test V6 Full-Route Waypoints Completed",
        "Training episode",
        "Waypoints",
        list(
            zip(
                x,
                (
                    float(row["waypoints_completed"])
                    for row in full_route_training
                ),
            )
        ),
    )
    completion_values = []
    completion_window: list[float] = []
    for row in full_route_training:
        completion_window.append(float(row["episode_passed"] is True))
        current = completion_window[-20:]
        completion_values.append(
            (
                float(row["training_episodes"]),
                100.0 * sum(current) / len(current),
            )
        )
    _chart(
        completion_path,
        "National_Test V6 Full-Route Completion Rate MA20",
        "Training episode",
        "Completion rate (%)",
        completion_values,
    )
    training = [row for row in rows if row["stage"] == "OFFLINE_TRAIN"]
    demonstrations = [
        row for row in training if row["actor_objective"] == "BEHAVIOR_CLONING"
    ]
    _chart(
        updates_path,
        "National_Test V6 Applied SAC Updates",
        "Training episode",
        "Applied updates",
        [
            (
                float(row["training_episodes"]),
                float(row["applied_updates"]),
            )
            for row in training
        ],
    )
    generated = [
        csv_path,
        reward_path,
        steps_path,
        waypoint_path,
        completion_path,
        updates_path,
    ]
    for path, title, label, values in (
        (
            critic_path,
            "National_Test V6 Critic Loss MA20",
            "Critic loss",
            _rolling_mean(training, "critic_loss"),
        ),
        (
            actor_path,
            "National_Test V6 Actor Loss MA20",
            "Actor loss",
            _rolling_mean(training, "actor_loss"),
        ),
        (
            behavior_clone_path,
            "National_Test V6 Demonstration Loss MA20",
            "Behavior clone loss",
            _rolling_mean(demonstrations, "behavior_clone_loss"),
        ),
        (
            alpha_path,
            "National_Test V6 Entropy Temperature",
            "Alpha",
            _rolling_mean(training, "alpha", window=1),
        ),
    ):
        if values:
            _chart(path, title, "Training episode", label, values)
            generated.append(path)
        else:
            path.unlink(missing_ok=True)
    evaluations = [row for row in rows if row["stage"] == "OFFLINE_EVAL"]
    if evaluations:
        pass_rate = []
        for generation in sorted(
            {int(row["generation"]) for row in evaluations}
        ):
            group = [
                row
                for row in evaluations
                if int(row["generation"]) == generation
            ]
            passed = sum(row["episode_passed"] is True for row in group)
            pass_rate.append(
                (float(generation), 100.0 * passed / len(group))
            )
        _chart(
            pass_rate_path,
            "National_Test V6 Deterministic Evaluation Pass Rate",
            "Generation",
            "Pass rate (%)",
            pass_rate,
        )
        _chart(
            pass_streak_path,
            "National_Test V6 Consecutive Deterministic Full-Route Passes",
            "Training episode",
            "Consecutive passes",
            [
                (
                    float(row["training_episodes"])
                    + float(row["evaluations"]) / 100.0,
                    float(row["evaluation_passes"]),
                )
                for row in evaluations
            ],
        )
        generated.extend((pass_rate_path, pass_streak_path))
    else:
        pass_rate_path.unlink(missing_ok=True)
        pass_streak_path.unlink(missing_ok=True)
    return generated


def export_runtime(source: Path, output: Path) -> list[Path]:
    raw = _rows(source)
    if any(
        row.get("schema_version") != "national-test-runtime-telemetry-v2"
        for row in raw
    ):
        raise ValueError("only national-test-runtime-telemetry-v2 is supported")
    cycles = [row for row in raw if row.get("event") == "control_cycle"]
    if not cycles:
        raise ValueError("runtime log contains no control cycles")
    cycle_fields = (
        "episode",
        "step",
        "cycle_ms",
        "reason",
        "mission_index",
        "distance_to_goal_m",
        "policy_action",
        "executed_action",
        "safety_intervened",
        "safety_truncated",
        "navigation_throttle_percent",
        "navigation_rudder_percent",
        "feedback_throttle_percent",
        "feedback_rudder_percent",
        "safe_action_mask",
        "reachability_mask",
        "candidate_reasons",
        "candidate_clearances_m",
        "speed_mps",
        "yaw_rate_rad_s",
        "actual_throttle",
        "actual_rudder",
        "corridor_cross_track_m",
        "corridor_heading_error_rad",
        "corridor_progress",
        "map_clearance_m",
    )
    flat = []
    for row in cycles:
        command = row.get("command") or {}
        feedback = row.get("device_feedback") or {}
        observation = row.get("observation") or {}
        flat.append(
            dict(
                row,
                navigation_throttle_percent=command.get("navigation_throttle_percent"),
                navigation_rudder_percent=command.get("navigation_rudder_percent"),
                feedback_throttle_percent=feedback.get("throttle_percent"),
                feedback_rudder_percent=feedback.get("rudder_percent"),
                safe_action_mask=json.dumps(
                    row.get("safe_action_mask") or (),
                    separators=(",", ":"),
                ),
                reachability_mask=json.dumps(
                    row.get("reachability_mask") or (),
                    separators=(",", ":"),
                ),
                candidate_reasons=json.dumps(
                    row.get("candidate_reasons") or (),
                    separators=(",", ":"),
                ),
                candidate_clearances_m=json.dumps(
                    row.get("candidate_clearances_m") or (),
                    separators=(",", ":"),
                ),
                speed_mps=observation.get("speed_mps"),
                yaw_rate_rad_s=observation.get("yaw_rate_rad_s"),
                actual_throttle=observation.get("actual_throttle"),
                actual_rudder=observation.get("actual_rudder"),
                corridor_cross_track_m=observation.get(
                    "corridor_cross_track_m"
                ),
                corridor_heading_error_rad=observation.get(
                    "corridor_heading_error_rad"
                ),
                corridor_progress=observation.get("corridor_progress"),
                map_clearance_m=observation.get("map_clearance_m"),
            )
        )
    csv_path = output / "national_test_v6_unity_cycles.csv"
    cycle_path = output / "national_test_v6_unity_cycle_ms.svg"
    mission_path = output / "national_test_v6_unity_mission_index.svg"
    _write_csv(csv_path, cycle_fields, flat)
    x = [float(index) for index in range(1, len(flat) + 1)]
    _chart(
        cycle_path,
        "National_Test V6 Unity Cycle Time",
        "Control cycle",
        "Milliseconds",
        list(zip(x, (float(row["cycle_ms"]) for row in flat))),
    )
    _chart(
        mission_path,
        "National_Test V6 Unity Mission Index",
        "Control cycle",
        "Mission index",
        list(zip(x, (float(row["mission_index"]) for row in flat))),
    )
    generated = [csv_path, cycle_path, mission_path]
    episode_rows = [row for row in raw if row.get("event") == "episode_end"]
    if episode_rows:
        episode_fields = (
            "episode",
            "outcome",
            "stage",
            "next_stage",
            "unity_adapt_episodes",
            "unity_validation_episodes",
            "unity_validation_passes",
            "attempted_updates",
            "applied_updates",
            "critic_loss",
            "actor_loss",
            "alpha",
            "entropy",
        )
        episodes = []
        for row in episode_rows:
            progress = row.get("progress") or {}
            training = progress.get("training") or {}
            detail = str(row.get("detail") or "")
            episodes.append(
                {
                    "episode": row["episode"],
                    "outcome": progress.get("outcome") or detail.split(":")[0],
                    "stage": progress.get("stage")
                    or (detail.split(":", 1)[1] if ":" in detail else ""),
                    "next_stage": progress.get("next_stage"),
                    "unity_adapt_episodes": progress.get(
                        "unity_adapt_episodes"
                    ),
                    "unity_validation_episodes": progress.get(
                        "unity_validation_episodes"
                    ),
                    "unity_validation_passes": progress.get(
                        "unity_validation_passes"
                    ),
                    "attempted_updates": training.get("attempted_updates"),
                    "applied_updates": training.get("applied_updates"),
                    "critic_loss": training.get("critic_loss"),
                    "actor_loss": training.get("actor_loss"),
                    "alpha": training.get("alpha"),
                    "entropy": training.get("entropy"),
                }
            )
        episode_csv = output / "national_test_v6_unity_episodes.csv"
        success_path = output / "national_test_v6_unity_success.svg"
        validation_path = output / "national_test_v6_unity_validation.svg"
        actor_path = output / "national_test_v6_unity_actor_loss.svg"
        critic_path = output / "national_test_v6_unity_critic_loss.svg"
        _write_csv(episode_csv, episode_fields, episodes)
        _chart(
            success_path,
            "National_Test V6 Unity Episode Success",
            "Unity episode",
            "Completed all 13 points",
            [
                (
                    float(row["episode"]),
                    float(row["outcome"] == "MISSION_COMPLETE"),
                )
                for row in episodes
            ],
        )
        validation = [
            (
                float(row["episode"]),
                float(row["unity_validation_passes"]),
            )
            for row in episodes
            if row["unity_validation_passes"] is not None
        ]
        if validation:
            _chart(
                validation_path,
                "National_Test V6 Unity Validation Passes",
                "Unity episode",
                "Validation passes",
                validation,
            )
            generated.append(validation_path)
        for path, title, field in (
            (actor_path, "National_Test V6 Unity Actor Loss", "actor_loss"),
            (critic_path, "National_Test V6 Unity Critic Loss", "critic_loss"),
        ):
            values = [
                (float(row["episode"]), float(row[field]))
                for row in episodes
                if row[field] is not None
            ]
            if values:
                _chart(path, title, "Unity episode", title.rsplit(" ", 1)[-1], values)
                generated.append(path)
        generated.extend((episode_csv, success_path))
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-log", type=Path, action="append")
    parser.add_argument("--runtime-log", type=Path)
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    if args.runtime_only and args.runtime_log is None:
        parser.error("--runtime-only requires --runtime-log")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    if not args.runtime_only:
        offline = args.offline_log or [
            _newest("national-test-offline-training-*.jsonl")
        ]
        generated.extend(export_offline(offline, output))
    if args.runtime_log is not None:
        generated.extend(export_runtime(args.runtime_log, output))
    print(json.dumps([str(path) for path in generated], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
