import csv
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest

from usvlib4ros.navigation.fixed_map_service import (
    FixedMapNavigationService,
)
from usvlib4ros.navigation.training_reports import (
    EpisodeReport,
    SelfTrainingEpisodeReport,
    SelfTrainingGenerationReport,
    SelfTrainingReportLogger,
    TrainingReportLogger,
)


def _episode(
    *,
    episode: int,
    total_steps: int,
    reached_steps,
    completed: bool,
) -> EpisodeReport:
    reached = tuple(reached_steps)
    return EpisodeReport(
        episode=episode,
        total_steps=total_steps,
        completed=completed,
        completed_waypoints=sum(step is not None for step in reached),
        duration_s=total_steps / 10.0,
        waypoint_reached_steps=reached,
        waypoint_min_distances_m=tuple(
            0.1 if step is not None else float("inf")
            for step in reached
        ),
        stop_reason="MISSION_COMPLETED" if completed else "TIMEOUT",
        replans=3,
    )


def _csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_report_logger_appends_runs_and_renders_step_curves(tmp_path):
    first = TrainingReportLogger(
        tmp_path,
        run_id="run-001",
        source="unity_train_button",
        started_at="2026-08-04T10:00:00Z",
    )
    first.record_episode(
        _episode(
            episode=0,
            total_steps=130,
            reached_steps=tuple(range(10, 131, 10)),
            completed=True,
        )
    )
    second = TrainingReportLogger(
        tmp_path,
        run_id="run-002",
        source="unity_train_button",
        started_at="2026-08-04T11:00:00Z",
    )
    second.record_episode(
        _episode(
            episode=0,
            total_steps=80,
            reached_steps=(12, 25, *([None] * 11)),
            completed=False,
        )
    )

    episodes = _csv_rows(tmp_path / "training_episodes.csv")
    assert [row["global_episode"] for row in episodes] == ["1", "2"]
    assert [row["run_id"] for row in episodes] == ["run-001", "run-002"]
    assert [row["total_steps"] for row in episodes] == ["130", "80"]
    assert [row["total_steps_to_goal"] for row in episodes] == ["130", ""]

    waypoints = _csv_rows(tmp_path / "waypoint_steps.csv")
    assert len(waypoints) == 26
    assert waypoints[0]["cumulative_steps"] == "10"
    assert waypoints[0]["segment_steps"] == "10"
    assert waypoints[1]["segment_steps"] == "10"
    assert waypoints[14]["cumulative_steps"] == "25"
    assert waypoints[14]["segment_steps"] == "13"
    assert waypoints[15]["cumulative_steps"] == ""
    assert waypoints[15]["segment_steps"] == ""

    runs = _csv_rows(tmp_path / "training_runs.csv")
    assert [row["episode_count"] for row in runs] == ["1", "1"]
    assert [row["completed_episodes"] for row in runs] == ["1", "0"]

    expected_svgs = (
        "training_total_steps.svg",
        "waypoint_cumulative_steps.svg",
        "waypoint_segment_steps.svg",
    )
    for name in expected_svgs:
        svg_path = tmp_path / name
        svg = svg_path.read_text(encoding="utf-8")
        assert svg.startswith("<?xml")
        assert "<svg" in svg
        assert 'font-family="Times New Roman, Noto Serif SC, serif"' in svg
        assert 'stroke="#1f1f1f"' in svg
        assert ElementTree.parse(svg_path).getroot().tag.endswith("svg")
    waypoint_svg = (tmp_path / "waypoint_cumulative_steps.svg").read_text(
        encoding="utf-8"
    )
    assert "Target 01" in waypoint_svg
    assert "Target 13" in waypoint_svg
    assert "stroke-dasharray" in waypoint_svg


def test_report_logger_rejects_non_monotonic_waypoint_steps(tmp_path):
    logger = TrainingReportLogger(
        tmp_path,
        run_id="invalid-run",
        source="unity_train_button",
        started_at="2026-08-04T10:00:00Z",
    )

    with pytest.raises(ValueError, match="monotonic"):
        logger.record_episode(
            _episode(
                episode=0,
                total_steps=30,
                reached_steps=(10, 9, *([None] * 11)),
                completed=False,
            )
        )


def test_report_logger_records_5000_steps_with_automatic_svg_scale(tmp_path):
    logger = TrainingReportLogger(
        tmp_path,
        run_id="run-5000",
        source="unity_train_button",
        started_at="2026-08-04T12:00:00Z",
    )
    logger.record_episode(
        _episode(
            episode=0,
            total_steps=5_000,
            reached_steps=(
                300,
                600,
                900,
                1_200,
                1_500,
                1_800,
                2_100,
                2_400,
                2_700,
                3_000,
                3_300,
                3_600,
                5_000,
            ),
            completed=True,
        )
    )

    episodes = _csv_rows(tmp_path / "training_episodes.csv")
    svg = (tmp_path / "training_total_steps.svg").read_text(
        encoding="utf-8"
    )

    assert episodes[0]["total_steps"] == "5000"
    assert episodes[0]["total_steps_to_goal"] == "5000"
    assert ">10000</text>" in svg


def test_train_button_episode_is_written_to_reports(tmp_path, monkeypatch):
    class _StopService(Exception):
        pass

    class _RosCapture:
        def initParameterList(self):
            return None

        def reset_unity(self):
            return True

        def set_auto_work(self):
            return True

        def getRoute(self):
            return SimpleNamespace(
                name="National_Test",
                points=[object()] * 13,
                obstacles=[object()] * 16,
            )

    output = SimpleNamespace(
        device_data=SimpleNamespace(task_status=1, reset_request_time=0.0),
        scada_data=SimpleNamespace(),
        updateThrottleRudderOutput=lambda *_: None,
        updateAlgorithmOutput=lambda *_: None,
    )
    service = FixedMapNavigationService(
        _RosCapture(),
        output,
        reports_dir=tmp_path,
    )
    monkeypatch.setattr(service, "_wait_for_reset", lambda **_: True)
    monkeypatch.setattr(service, "_wait_for_auto", lambda **_: True)

    def finish_episode(*_, **__):
        service.last_episode_metrics = {
            "total_steps": 130,
            "completed": True,
            "duration_s": 13.0,
            "completed_waypoints": 13,
            "waypoint_reached_steps": tuple(range(10, 131, 10)),
            "waypoint_min_distances_m": (0.1,) * 13,
            "collisions": 0,
            "laser_emergency_stops": 0,
            "unrecovered_unsafe_events": 0,
            "replans": 3,
            "stop_reason": "MISSION_COMPLETED",
        }
        return True

    monkeypatch.setattr(service, "_run_episode", finish_episode)

    def advance(duration):
        if duration == 0.1:
            output.device_data.task_status = 0
        elif duration == 0.02:
            raise _StopService

    monkeypatch.setattr(
        "usvlib4ros.navigation.fixed_map_service.time.sleep",
        advance,
    )

    with pytest.raises(_StopService):
        service.run()

    episodes = _csv_rows(tmp_path / "training_episodes.csv")
    assert len(episodes) == 1
    assert episodes[0]["source"] == "unity_train_button"
    assert episodes[0]["episode"] == "0"
    assert episodes[0]["total_steps_to_goal"] == "130"
    assert (tmp_path / "waypoint_cumulative_steps.svg").is_file()


def test_self_training_reports_append_across_clicks_and_render_all_requested_curves(tmp_path):
    first = SelfTrainingReportLogger(tmp_path)
    first.record_episode(
        SelfTrainingEpisodeReport(
            session_id="session-a",
            generation=1,
            stage="offline_train",
            episode=1,
            total_steps=5000,
            total_reward=-12.5,
            completed=False,
            actor_loss=0.4,
            critic_loss=1.2,
            training_step=16,
        )
    )
    second = SelfTrainingReportLogger(tmp_path)
    second.record_episode(
        SelfTrainingEpisodeReport(
            session_id="session-a",
            generation=1,
            stage="unity_train",
            episode=2,
            total_steps=900,
            total_reward=38.0,
            completed=True,
            actor_loss=0.2,
            critic_loss=0.7,
            training_step=16,
        )
    )
    second.record_generation(
        SelfTrainingGenerationReport(
            session_id="session-a",
            generation=1,
            completed_training_episodes=100,
            parent_sha256="a" * 64,
            candidate_sha256="b" * 64,
            promoted=True,
            promotion_reason="UNITY_MEDIAN_STEPS_IMPROVED",
            offline_completed=20,
            unity_completed=5,
            unity_median_steps=900.0,
        )
    )

    episodes = _csv_rows(tmp_path / "self_training_episodes.csv")
    generations = _csv_rows(tmp_path / "self_training_generations.csv")
    assert [row["global_episode"] for row in episodes] == ["1", "2"]
    assert [row["stage"] for row in episodes] == ["offline_train", "unity_train"]
    assert episodes[0]["total_steps"] == "5000"
    assert generations[0]["promoted"] == "1"
    for name in (
        "self_training_reward.svg",
        "self_training_success_rate.svg",
        "self_training_total_steps.svg",
        "self_training_losses.svg",
    ):
        path = tmp_path / name
        assert path.is_file()
        assert ElementTree.parse(path).getroot().tag.endswith("svg")
