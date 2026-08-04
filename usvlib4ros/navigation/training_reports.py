"""Append-only CSV logs and dependency-free SVG training curves."""

from __future__ import annotations

import csv
import html
import math
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
WAYPOINT_COUNT = 13

EPISODE_FIELDS = (
    "global_episode",
    "run_id",
    "source",
    "run_started_at",
    "recorded_at",
    "episode",
    "total_steps",
    "completed",
    "total_steps_to_goal",
    "completed_waypoints",
    "duration_s",
    "stop_reason",
    "replans",
    "collisions",
    "laser_emergency_stops",
    "unrecovered_unsafe_events",
)
WAYPOINT_FIELDS = (
    "global_episode",
    "run_id",
    "source",
    "episode",
    "waypoint",
    "reached",
    "cumulative_steps",
    "segment_steps",
    "minimum_distance_m",
)
RUN_FIELDS = (
    "run_id",
    "source",
    "started_at",
    "updated_at",
    "first_global_episode",
    "last_global_episode",
    "episode_count",
    "completed_episodes",
)
SELF_TRAINING_EPISODE_FIELDS = (
    "global_episode",
    "recorded_at",
    "session_id",
    "generation",
    "stage",
    "episode",
    "total_steps",
    "total_reward",
    "completed",
    "actor_loss",
    "critic_loss",
    "training_step",
    "collisions",
    "laser_emergency_stops",
    "unrecovered_unsafe_events",
    "stop_reason",
)
SELF_TRAINING_GENERATION_FIELDS = (
    "recorded_at",
    "session_id",
    "generation",
    "completed_training_episodes",
    "parent_sha256",
    "candidate_sha256",
    "promoted",
    "promotion_reason",
    "offline_completed",
    "unity_completed",
    "unity_median_steps",
)

_REPORT_LOCK = threading.Lock()


@dataclass(frozen=True)
class EpisodeReport:
    """One episode as observed by the Unity training-button loop."""

    episode: int
    total_steps: int
    completed: bool
    completed_waypoints: int
    duration_s: float
    waypoint_reached_steps: tuple[Optional[int], ...]
    waypoint_min_distances_m: tuple[float, ...] = ()
    stop_reason: str = ""
    replans: int = 0
    collisions: int = 0
    laser_emergency_stops: int = 0
    unrecovered_unsafe_events: int = 0

    def validate(self) -> None:
        if self.episode < 0:
            raise ValueError("episode must be non-negative")
        if self.total_steps < 0:
            raise ValueError("total_steps must be non-negative")
        if len(self.waypoint_reached_steps) != WAYPOINT_COUNT:
            raise ValueError("waypoint_reached_steps must contain 13 values")
        if self.waypoint_min_distances_m and (
            len(self.waypoint_min_distances_m) != WAYPOINT_COUNT
        ):
            raise ValueError(
                "waypoint_min_distances_m must be empty or contain 13 values"
            )
        if not math.isfinite(self.duration_s) or self.duration_s < 0.0:
            raise ValueError("duration_s must be finite and non-negative")
        if not 0 <= self.completed_waypoints <= WAYPOINT_COUNT:
            raise ValueError("completed_waypoints is outside the route")

        previous = 0
        missing_seen = False
        reached_count = 0
        for step in self.waypoint_reached_steps:
            if step is None:
                missing_seen = True
                continue
            if missing_seen:
                raise ValueError("waypoint steps must not resume after a gap")
            if isinstance(step, bool) or not isinstance(step, int):
                raise ValueError("waypoint steps must be integers or empty")
            if step <= previous:
                raise ValueError("waypoint steps must be strictly monotonic")
            if step > self.total_steps:
                raise ValueError("waypoint step exceeds total_steps")
            previous = step
            reached_count += 1

        if reached_count != self.completed_waypoints:
            raise ValueError(
                "completed_waypoints does not match waypoint step count"
            )
        if self.completed and reached_count != WAYPOINT_COUNT:
            raise ValueError("completed episode must reach all 13 waypoints")


@dataclass(frozen=True)
class SelfTrainingEpisodeReport:
    session_id: str
    generation: int
    stage: str
    episode: int
    total_steps: int
    total_reward: float
    completed: bool
    actor_loss: float
    critic_loss: float
    training_step: int
    collisions: int = 0
    laser_emergency_stops: int = 0
    unrecovered_unsafe_events: int = 0
    stop_reason: str = ""

    def validate(self) -> None:
        if not self.session_id or not self.stage:
            raise ValueError("self-training session and stage are required")
        integer_values = (
            self.generation,
            self.episode,
            self.total_steps,
            self.training_step,
            self.collisions,
            self.laser_emergency_stops,
            self.unrecovered_unsafe_events,
        )
        if any(type(value) is not int or value < 0 for value in integer_values):
            raise ValueError("self-training episode counters are invalid")
        if type(self.completed) is not bool:
            raise ValueError("self-training completion flag must be boolean")
        if not all(
            math.isfinite(float(value))
            for value in (self.total_reward, self.actor_loss, self.critic_loss)
        ):
            raise ValueError("self-training episode metrics must be finite")


@dataclass(frozen=True)
class SelfTrainingGenerationReport:
    session_id: str
    generation: int
    completed_training_episodes: int
    parent_sha256: str
    candidate_sha256: str
    promoted: bool
    promotion_reason: str
    offline_completed: int
    unity_completed: int
    unity_median_steps: Optional[float]

    def validate(self) -> None:
        if not self.session_id or not self.promotion_reason:
            raise ValueError("self-training generation identity is required")
        if len(self.parent_sha256) != 64 or len(self.candidate_sha256) != 64:
            raise ValueError("self-training generation hashes are invalid")
        for value in (
            self.generation,
            self.completed_training_episodes,
            self.offline_completed,
            self.unity_completed,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("self-training generation counters are invalid")
        if type(self.promoted) is not bool:
            raise ValueError("self-training promotion flag must be boolean")
        if self.unity_median_steps is not None and (
            not math.isfinite(float(self.unity_median_steps))
            or float(self.unity_median_steps) <= 0.0
        ):
            raise ValueError("self-training Unity median steps are invalid")


class SelfTrainingReportLogger:
    """Append hybrid-training evidence without changing the legacy reports."""

    def __init__(self, reports_dir: Path = DEFAULT_REPORTS_DIR) -> None:
        self.reports_dir = Path(reports_dir)

    def record_episode(self, report: SelfTrainingEpisodeReport) -> int:
        report.validate()
        with _REPORT_LOCK:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            path = self.reports_dir / "self_training_episodes.csv"
            rows = _read_csv(path, SELF_TRAINING_EPISODE_FIELDS)
            global_episode = 1 + max(
                (int(row["global_episode"]) for row in rows),
                default=0,
            )
            rows.append(
                {
                    "global_episode": str(global_episode),
                    "recorded_at": _utc_text(datetime.now(timezone.utc)),
                    "session_id": report.session_id,
                    "generation": str(report.generation),
                    "stage": report.stage,
                    "episode": str(report.episode),
                    "total_steps": str(report.total_steps),
                    "total_reward": _format_float(report.total_reward),
                    "completed": "1" if report.completed else "0",
                    "actor_loss": _format_float(report.actor_loss),
                    "critic_loss": _format_float(report.critic_loss),
                    "training_step": str(report.training_step),
                    "collisions": str(report.collisions),
                    "laser_emergency_stops": str(report.laser_emergency_stops),
                    "unrecovered_unsafe_events": str(report.unrecovered_unsafe_events),
                    "stop_reason": report.stop_reason,
                }
            )
            _write_csv(path, SELF_TRAINING_EPISODE_FIELDS, rows)
            _render_self_training_reports(self.reports_dir, rows)
            return global_episode

    def record_generation(self, report: SelfTrainingGenerationReport) -> None:
        report.validate()
        with _REPORT_LOCK:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            path = self.reports_dir / "self_training_generations.csv"
            rows = _read_csv(path, SELF_TRAINING_GENERATION_FIELDS)
            if any(
                row["session_id"] == report.session_id
                and int(row["generation"]) == report.generation
                for row in rows
            ):
                raise ValueError("self-training generation is already recorded")
            rows.append(
                {
                    "recorded_at": _utc_text(datetime.now(timezone.utc)),
                    "session_id": report.session_id,
                    "generation": str(report.generation),
                    "completed_training_episodes": str(report.completed_training_episodes),
                    "parent_sha256": report.parent_sha256,
                    "candidate_sha256": report.candidate_sha256,
                    "promoted": "1" if report.promoted else "0",
                    "promotion_reason": report.promotion_reason,
                    "offline_completed": str(report.offline_completed),
                    "unity_completed": str(report.unity_completed),
                    "unity_median_steps": (
                        "" if report.unity_median_steps is None else _format_float(report.unity_median_steps)
                    ),
                }
            )
            _write_csv(path, SELF_TRAINING_GENERATION_FIELDS, rows)


@dataclass(frozen=True)
class _Series:
    label: str
    values: tuple[tuple[float, Optional[float]], ...]
    stroke: str
    dash: str = ""


_SERIES_STYLES = (
    ("#1f1f1f", ""),
    ("#363636", "10 5"),
    ("#555555", "2 4"),
    ("#707070", "8 3 2 3"),
    ("#292929", "5 3"),
    ("#4a4a4a", "12 4 3 4"),
    ("#686868", "1 3"),
    ("#202020", "7 4"),
    ("#3f3f3f", "3 3"),
    ("#5c5c5c", "9 3 2 3"),
    ("#777777", "4 4"),
    ("#303030", "12 5"),
    ("#505050", "2 3 6 3"),
)


class TrainingReportLogger:
    """Persist every episode and redraw cumulative publication-style SVGs."""

    def __init__(
        self,
        reports_dir: Path,
        *,
        run_id: str,
        source: str,
        started_at: str,
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.run_id = str(run_id).strip()
        self.source = str(source).strip()
        self.started_at = str(started_at).strip()
        if not self.run_id or not self.source or not self.started_at:
            raise ValueError("run_id, source, and started_at are required")

    @classmethod
    def for_train_click(
        cls,
        reports_dir: Path = DEFAULT_REPORTS_DIR,
    ) -> "TrainingReportLogger":
        now = datetime.now(timezone.utc)
        started_at = _utc_text(now)
        run_id = (
            now.strftime("%Y%m%dT%H%M%S.%fZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        return cls(
            reports_dir,
            run_id=run_id,
            source="unity_train_button",
            started_at=started_at,
        )

    def record_episode(self, report: EpisodeReport) -> int:
        """Append one episode and atomically refresh all CSV/SVG outputs."""

        report.validate()
        with _REPORT_LOCK:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            episodes_path = self.reports_dir / "training_episodes.csv"
            waypoints_path = self.reports_dir / "waypoint_steps.csv"
            runs_path = self.reports_dir / "training_runs.csv"
            episode_rows = _read_csv(episodes_path, EPISODE_FIELDS)
            waypoint_rows = _read_csv(waypoints_path, WAYPOINT_FIELDS)
            run_rows = _read_csv(runs_path, RUN_FIELDS)

            if any(
                row["run_id"] == self.run_id
                and int(row["episode"]) == report.episode
                for row in episode_rows
            ):
                raise ValueError("episode is already present in this run")

            global_episode = 1 + max(
                (int(row["global_episode"]) for row in episode_rows),
                default=0,
            )
            recorded_at = _utc_text(datetime.now(timezone.utc))
            episode_rows.append(
                {
                    "global_episode": str(global_episode),
                    "run_id": self.run_id,
                    "source": self.source,
                    "run_started_at": self.started_at,
                    "recorded_at": recorded_at,
                    "episode": str(report.episode),
                    "total_steps": str(report.total_steps),
                    "completed": "1" if report.completed else "0",
                    "total_steps_to_goal": (
                        str(report.total_steps) if report.completed else ""
                    ),
                    "completed_waypoints": str(report.completed_waypoints),
                    "duration_s": _format_float(report.duration_s),
                    "stop_reason": report.stop_reason,
                    "replans": str(report.replans),
                    "collisions": str(report.collisions),
                    "laser_emergency_stops": str(
                        report.laser_emergency_stops
                    ),
                    "unrecovered_unsafe_events": str(
                        report.unrecovered_unsafe_events
                    ),
                }
            )
            waypoint_rows.extend(
                self._waypoint_rows(global_episode, report)
            )
            _update_run_rows(
                run_rows,
                run_id=self.run_id,
                source=self.source,
                started_at=self.started_at,
                updated_at=recorded_at,
                global_episode=global_episode,
                completed=report.completed,
            )

            _write_csv(episodes_path, EPISODE_FIELDS, episode_rows)
            _write_csv(waypoints_path, WAYPOINT_FIELDS, waypoint_rows)
            _write_csv(runs_path, RUN_FIELDS, run_rows)
            _render_reports(
                self.reports_dir,
                episode_rows,
                waypoint_rows,
            )
            return global_episode

    def _waypoint_rows(
        self,
        global_episode: int,
        report: EpisodeReport,
    ) -> list[dict[str, str]]:
        rows = []
        previous_step = 0
        distances = report.waypoint_min_distances_m or (
            (float("inf"),) * WAYPOINT_COUNT
        )
        for index, (step, distance) in enumerate(
            zip(report.waypoint_reached_steps, distances),
            start=1,
        ):
            segment_steps = None if step is None else step - previous_step
            if step is not None:
                previous_step = step
            rows.append(
                {
                    "global_episode": str(global_episode),
                    "run_id": self.run_id,
                    "source": self.source,
                    "episode": str(report.episode),
                    "waypoint": str(index),
                    "reached": "1" if step is not None else "0",
                    "cumulative_steps": "" if step is None else str(step),
                    "segment_steps": (
                        "" if segment_steps is None else str(segment_steps)
                    ),
                    "minimum_distance_m": _format_float(distance),
                }
            )
        return rows


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _format_float(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _read_csv(
    path: Path,
    fields: tuple[str, ...],
) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"CSV header is incompatible: {path}")
        return [dict(row) for row in reader]


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: Iterable[dict[str, str]],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _update_run_rows(
    rows: list[dict[str, str]],
    *,
    run_id: str,
    source: str,
    started_at: str,
    updated_at: str,
    global_episode: int,
    completed: bool,
) -> None:
    for row in rows:
        if row["run_id"] != run_id:
            continue
        if row["source"] != source or row["started_at"] != started_at:
            raise ValueError("run metadata changed during logging")
        row["updated_at"] = updated_at
        row["last_global_episode"] = str(global_episode)
        row["episode_count"] = str(int(row["episode_count"]) + 1)
        row["completed_episodes"] = str(
            int(row["completed_episodes"]) + int(completed)
        )
        return
    rows.append(
        {
            "run_id": run_id,
            "source": source,
            "started_at": started_at,
            "updated_at": updated_at,
            "first_global_episode": str(global_episode),
            "last_global_episode": str(global_episode),
            "episode_count": "1",
            "completed_episodes": "1" if completed else "0",
        }
    )


def _render_reports(
    reports_dir: Path,
    episode_rows: list[dict[str, str]],
    waypoint_rows: list[dict[str, str]],
) -> None:
    episode_rows = sorted(
        episode_rows,
        key=lambda row: int(row["global_episode"]),
    )
    episode_numbers = tuple(
        int(row["global_episode"]) for row in episode_rows
    )
    total_steps = _Series(
        label="Total steps",
        values=tuple(
            (float(row["global_episode"]), float(row["total_steps"]))
            for row in episode_rows
        ),
        stroke=_SERIES_STYLES[0][0],
    )
    _render_line_chart(
        reports_dir / "training_total_steps.svg",
        title="Training Episode Total Steps",
        y_label="Total steps",
        series=(total_steps,),
    )

    waypoint_lookup = {
        (int(row["global_episode"]), int(row["waypoint"])): row
        for row in waypoint_rows
    }
    cumulative_series = []
    segment_series = []
    for waypoint in range(1, WAYPOINT_COUNT + 1):
        stroke, dash = _SERIES_STYLES[waypoint - 1]
        cumulative_series.append(
            _Series(
                label=f"Target {waypoint:02d}",
                values=tuple(
                    (
                        float(episode),
                        _optional_number(
                            waypoint_lookup[(episode, waypoint)][
                                "cumulative_steps"
                            ]
                        ),
                    )
                    for episode in episode_numbers
                ),
                stroke=stroke,
                dash=dash,
            )
        )
        segment_series.append(
            _Series(
                label=f"Target {waypoint:02d}",
                values=tuple(
                    (
                        float(episode),
                        _optional_number(
                            waypoint_lookup[(episode, waypoint)][
                                "segment_steps"
                            ]
                        ),
                    )
                    for episode in episode_numbers
                ),
                stroke=stroke,
                dash=dash,
            )
        )

    _render_line_chart(
        reports_dir / "waypoint_cumulative_steps.svg",
        title="Cumulative Steps to Each Target",
        y_label="Cumulative steps",
        series=tuple(cumulative_series),
    )
    _render_line_chart(
        reports_dir / "waypoint_segment_steps.svg",
        title="Segment Steps Between Targets",
        y_label="Segment steps",
        series=tuple(segment_series),
    )


def _optional_number(value: str) -> Optional[float]:
    return None if value == "" else float(value)


def _render_self_training_reports(
    reports_dir: Path,
    rows: list[dict[str, str]],
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["global_episode"]))
    values = tuple(float(row["global_episode"]) for row in ordered)
    _render_line_chart(
        reports_dir / "self_training_reward.svg",
        title="Self-training Episode Reward",
        y_label="Reward",
        series=(
            _Series(
                "Reward",
                tuple((episode, float(row["total_reward"])) for episode, row in zip(values, ordered)),
                _SERIES_STYLES[0][0],
            ),
        ),
    )
    successes = 0
    success_values = []
    for index, (episode, row) in enumerate(zip(values, ordered), start=1):
        successes += int(row["completed"])
        success_values.append((episode, successes / index))
    _render_line_chart(
        reports_dir / "self_training_success_rate.svg",
        title="Self-training Cumulative Success Rate",
        y_label="Success rate",
        series=(_Series("Success rate", tuple(success_values), _SERIES_STYLES[0][0]),),
    )
    _render_line_chart(
        reports_dir / "self_training_total_steps.svg",
        title="Self-training Episode Total Steps",
        y_label="Total steps",
        series=(
            _Series(
                "Total steps",
                tuple((episode, float(row["total_steps"])) for episode, row in zip(values, ordered)),
                _SERIES_STYLES[0][0],
            ),
        ),
    )
    _render_line_chart(
        reports_dir / "self_training_losses.svg",
        title="Self-training SAC Loss",
        y_label="Loss",
        series=(
            _Series(
                "Actor loss",
                tuple((episode, float(row["actor_loss"])) for episode, row in zip(values, ordered)),
                _SERIES_STYLES[0][0],
            ),
            _Series(
                "Critic loss",
                tuple((episode, float(row["critic_loss"])) for episode, row in zip(values, ordered)),
                _SERIES_STYLES[1][0],
                _SERIES_STYLES[1][1],
            ),
        ),
    )


def _render_line_chart(
    path: Path,
    *,
    title: str,
    y_label: str,
    series: tuple[_Series, ...],
) -> None:
    width = 900
    height = 560
    left = 90
    right = 35
    bottom = 75
    legend_rows = max(1, math.ceil(len(series) / 7))
    top = 55 + 24 * legend_rows
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [
        value
        for item in series
        for _, value in item.values
        if value is not None and math.isfinite(value)
    ]
    all_x = [
        x
        for item in series
        for x, value in item.values
        if value is not None and math.isfinite(value)
    ]
    x_max = max(2.0, max(all_x, default=1.0))
    y_max = _nice_upper(max(all_values, default=1.0))
    y_min = (
        -_nice_upper(abs(min(all_values)))
        if all_values and min(all_values) < 0.0
        else 0.0
    )
    y_span = y_max - y_min

    def x_position(value: float) -> float:
        return left + plot_width * value / x_max

    def y_position(value: float) -> float:
        return top + plot_height * (1.0 - (value - y_min) / y_span)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="chart-title chart-desc">'
        ),
        f'<title id="chart-title">{html.escape(title)}</title>',
        (
            '<desc id="chart-desc">Grayscale training curve. '
            'Missing target values are rendered as gaps.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            '<g font-family="Times New Roman, Noto Serif SC, serif" '
            'fill="#202020">'
        ),
        (
            f'<text x="{width / 2:.1f}" y="25" text-anchor="middle" '
            f'font-size="18">{html.escape(title)}</text>'
        ),
    ]

    for index, item in enumerate(series):
        column = index % 7
        row = index // 7
        legend_x = left + column * 110
        legend_y = 48 + row * 23
        dash = (
            f' stroke-dasharray="{item.dash}"' if item.dash else ""
        )
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y}" '
            f'x2="{legend_x + 25}" y2="{legend_y}" '
            f'stroke="{item.stroke}" stroke-width="1.7"{dash}/>'
        )
        lines.append(
            f'<text x="{legend_x + 31}" y="{legend_y + 4}" '
            f'font-size="11">{html.escape(item.label)}</text>'
        )

    x_ticks = 5
    y_ticks = 5
    for index in range(x_ticks + 1):
        value = x_max * index / x_ticks
        x = x_position(value)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top + plot_height:.2f}" '
            f'x2="{x:.2f}" y2="{top + plot_height + 6:.2f}" '
            'stroke="#303030" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 23:.2f}" '
            f'text-anchor="middle" font-size="12">'
            f'{_axis_number(value)}</text>'
        )
    for index in range(y_ticks + 1):
        value = y_min + y_span * index / y_ticks
        y = y_position(value)
        lines.append(
            f'<line x1="{left - 6:.2f}" y1="{y:.2f}" '
            f'x2="{left:.2f}" y2="{y:.2f}" '
            'stroke="#303030" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 11:.2f}" y="{y + 4:.2f}" '
            f'text-anchor="end" font-size="12">'
            f'{_axis_number(value)}</text>'
        )

    lines.extend(
        [
            (
                f'<rect x="{left}" y="{top}" width="{plot_width}" '
                f'height="{plot_height}" fill="none" stroke="#303030" '
                'stroke-width="1.2"/>'
            ),
            (
                f'<text x="{left + plot_width / 2:.2f}" y="{height - 25}" '
                'text-anchor="middle" font-size="15">Episode</text>'
            ),
            (
                f'<text x="25" y="{top + plot_height / 2:.2f}" '
                'text-anchor="middle" font-size="15" '
                f'transform="rotate(-90 25 {top + plot_height / 2:.2f})">'
                f'{html.escape(y_label)}</text>'
            ),
        ]
    )

    for item in series:
        dash = (
            f' stroke-dasharray="{item.dash}"' if item.dash else ""
        )
        for segment in _continuous_segments(item.values):
            if len(segment) == 1:
                x, value = segment[0]
                lines.append(
                    f'<circle cx="{x_position(x):.2f}" '
                    f'cy="{y_position(value):.2f}" r="2.7" '
                    f'fill="{item.stroke}"/>'
                )
                continue
            points = " ".join(
                f"{x_position(x):.2f},{y_position(value):.2f}"
                for x, value in segment
            )
            lines.append(
                f'<polyline points="{points}" fill="none" '
                f'stroke="{item.stroke}" stroke-width="1.5" '
                f'stroke-linecap="round" stroke-linejoin="round"{dash}/>'
            )

    lines.extend(("</g>", "</svg>"))
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _continuous_segments(
    values: tuple[tuple[float, Optional[float]], ...],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    segments = []
    current = []
    for x, value in values:
        if value is None or not math.isfinite(value):
            if current:
                segments.append(tuple(current))
                current = []
            continue
        current.append((x, value))
    if current:
        segments.append(tuple(current))
    return tuple(segments)


def _nice_upper(value: float) -> float:
    raw = max(1.0, float(value) * 1.05)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    scaled = raw / magnitude
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if scaled <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def _axis_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


__all__ = [
    "DEFAULT_REPORTS_DIR",
    "EpisodeReport",
    "SelfTrainingEpisodeReport",
    "SelfTrainingGenerationReport",
    "SelfTrainingReportLogger",
    "TrainingReportLogger",
]
