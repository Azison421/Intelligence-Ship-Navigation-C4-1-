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


def _chart(path: Path, title: str, y_label: str, values: list[tuple[float, float]]) -> None:
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
                f'<text x="{x:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-size="11">{x_value:.0f}</text>',
                f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#303030"/>',
                f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{y_value:.2f}</text>',
            )
        )
    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
    lines.extend(
        (
            f'<polyline points="{points}" fill="none" stroke="#1f1f1f" stroke-width="1.8" stroke-linejoin="round"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" font-size="14">Evidence row</text>',
            f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 24 {top + plot_height / 2})">{html.escape(y_label)}</text>',
            "</g>",
            "</svg>",
        )
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_offline(source: Path, output: Path) -> list[Path]:
    raw = _rows(source)
    fields = (
        "record_index",
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
        "completed_waypoints",
        "reward",
        "collision",
        "timed_out",
        "no_safe_action",
        "safety_interventions",
        "maximum_cross_track_m",
        "minimum_clearance_m",
        "end_reason",
    )
    rows = []
    for index, item in enumerate(raw, 1):
        row = dict(item, record_index=index)
        stage = str(row.get("stage", ""))
        training_episode = int(row.get("training_episodes", 0))
        row.setdefault(
            "full_route",
            stage == "OFFLINE_EVAL"
            or (stage == "OFFLINE_TRAIN" and training_episode % 4 == 0),
        )
        row.setdefault("gate_eligible", stage == "OFFLINE_EVAL")
        rows.append(row)
    missing = [field for field in fields[1:] if any(field not in row for row in rows)]
    if missing:
        raise ValueError("offline log fields are missing: " + ", ".join(missing))
    csv_path = output / "national_test_v6_offline_episodes.csv"
    reward_path = output / "national_test_v6_offline_reward.svg"
    steps_path = output / "national_test_v6_offline_steps.svg"
    waypoint_path = output / "national_test_v6_offline_waypoints.svg"
    _write_csv(csv_path, fields, rows)
    x = [float(row["record_index"]) for row in rows]
    _chart(reward_path, "National_Test V6 Offline Reward", "Reward", list(zip(x, (float(row["reward"]) for row in rows))))
    _chart(steps_path, "National_Test V6 Offline Steps", "Steps", list(zip(x, (float(row["steps"]) for row in rows))))
    _chart(waypoint_path, "National_Test V6 Completed Waypoints", "Waypoints", list(zip(x, (float(row["completed_waypoints"]) for row in rows))))
    return [csv_path, reward_path, steps_path, waypoint_path]


def export_runtime(source: Path, output: Path) -> list[Path]:
    raw = _rows(source)
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
    )
    flat = []
    for row in cycles:
        command = row.get("command") or {}
        feedback = row.get("device_feedback") or {}
        flat.append(
            dict(
                row,
                navigation_throttle_percent=command.get("navigation_throttle_percent"),
                navigation_rudder_percent=command.get("navigation_rudder_percent"),
                feedback_throttle_percent=feedback.get("throttle_percent"),
                feedback_rudder_percent=feedback.get("rudder_percent"),
            )
        )
    csv_path = output / "national_test_v6_unity_cycles.csv"
    cycle_path = output / "national_test_v6_unity_cycle_ms.svg"
    mission_path = output / "national_test_v6_unity_mission_index.svg"
    _write_csv(csv_path, cycle_fields, flat)
    x = [float(index) for index in range(1, len(flat) + 1)]
    _chart(cycle_path, "National_Test V6 Unity Cycle Time", "Milliseconds", list(zip(x, (float(row["cycle_ms"]) for row in flat))))
    _chart(mission_path, "National_Test V6 Unity Mission Index", "Mission index", list(zip(x, (float(row["mission_index"]) for row in flat))))
    return [csv_path, cycle_path, mission_path]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-log", type=Path)
    parser.add_argument("--runtime-log", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    offline = args.offline_log or _newest("national-test-offline-training-*.jsonl")
    generated = export_offline(offline, output)
    if args.runtime_log is not None:
        generated.extend(export_runtime(args.runtime_log, output))
    print(json.dumps([str(path) for path in generated], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
