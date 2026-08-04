import hashlib
import inspect
import json
from types import SimpleNamespace

from usvlib4ros.navigation.reverse_control_calibration import (
    ReverseControlProfile,
)
from usvlib4ros.planning import Control
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
)
from usvlib4ros.planning.fixed_route import FIXED_ROUTE_TOLERANCE_M
from usvlib4ros.policy.fixed_map_trainer import (
    FixedMapSACTrainer,
    LIVE_RESET_SPAWN_X_M,
    LIVE_RESET_SPAWN_Y_M,
    LIVE_RESET_SPAWN_YAW_RAD,
    advance_training_maneuver,
)
from usvlib4ros.policy.fixed_map_features import TrajectoryPreview
from usvlib4ros.policy.self_training import (
    operational_profile_from_manifest,
)


def test_offline_evaluation_uses_competition_step_limit():
    for method in (
        FixedMapSACTrainer.run_episode,
        FixedMapSACTrainer.evaluate,
    ):
        assert (
            inspect.signature(method).parameters["max_steps"].default
            == 5_000
        )


def test_self_training_profile_builds_zero_clearance_offline_map_and_enables_safe_mask_authority():
    manifest_path = (
        __import__("pathlib").Path("artifacts/checkpoints")
        / "national_test_sac_v37_zero_clearance_conservative_345_unity_test.pt.json"
    )
    profile = operational_profile_from_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )

    trainer = FixedMapSACTrainer(
        operational_profile=profile,
        full_safe_action_authority=True,
        seed=131,
    )

    assert trainer.compiled_map.snapshot.required_clearance == 0.0
    assert trainer.operational_profile == profile
    assert trainer.full_safe_action_authority is True


def _forward_profile():
    return ForwardControlProfile(
        calibration_hash="0" * 64,
        minimum_steerage_throttle=0.1,
        cruise_throttle=0.4,
        action_controls=(
            Control(0.1, -0.1),
            Control(0.1, -0.05),
            Control(0.4, 0.0),
            Control(0.1, 0.05),
            Control(0.1, 0.1),
        ),
        throttle_speed_gain=1.2681317113395243,
        positive_rudder_yaw_rate_gain=2.0353030676101787,
        negative_rudder_yaw_rate_gain=2.0871446427732967,
    )


def _reverse_profile():
    return ReverseControlProfile(
        source_log_sha256="1" * 64,
        command_throttle=-0.4,
        command_signed_speed_mps=-0.12256225167642798,
        reverse_throttle_speed_gain=0.3064056291910699,
        max_reverse_speed_mps=0.2,
    )


def test_training_keeps_composite_state_when_narrow_target_is_reached():
    transition = advance_training_maneuver(
        mission_index=10,
        maneuver_phase="NORMAL",
        reached=True,
        route_point_count=13,
    )

    assert transition.mission_index == 11
    assert transition.maneuver_phase == "ESCAPE_PENDING"
    assert transition.task_point_advanced
    assert not transition.needs_new_plan
    assert not transition.completed

    escaped = advance_training_maneuver(
        mission_index=11,
        maneuver_phase="ESCAPE_PENDING",
        reached=True,
        route_point_count=13,
    )
    assert escaped.mission_index == 11
    assert escaped.maneuver_phase == "NORMAL"
    assert not escaped.task_point_advanced
    assert escaped.needs_new_plan
    assert not escaped.completed


def test_training_keeps_point_four_trajectory_until_clearance_handoff():
    reached = advance_training_maneuver(
        mission_index=3,
        maneuver_phase="NORMAL",
        reached=True,
        route_point_count=13,
    )

    assert reached.mission_index == 4
    assert reached.maneuver_phase == "CLEARANCE_PENDING"
    assert reached.task_point_advanced
    assert not reached.needs_new_plan

    handed_off = advance_training_maneuver(
        mission_index=4,
        maneuver_phase="CLEARANCE_PENDING",
        reached=True,
        route_point_count=13,
    )
    assert handed_off.mission_index == 4
    assert handed_off.maneuver_phase == "CLEARANCE_TURN_PENDING"
    assert not handed_off.task_point_advanced
    assert handed_off.needs_new_plan

    point_five = advance_training_maneuver(
        mission_index=4,
        maneuver_phase="CLEARANCE_TURN_PENDING",
        reached=True,
        route_point_count=13,
    )
    assert point_five.mission_index == 5
    assert point_five.maneuver_phase == "CLEARANCE_EXIT_PENDING"
    assert point_five.task_point_advanced
    assert point_five.needs_new_plan

    point_six = advance_training_maneuver(
        mission_index=5,
        maneuver_phase="CLEARANCE_EXIT_PENDING",
        reached=True,
        route_point_count=13,
    )
    assert point_six.mission_index == 6
    assert point_six.maneuver_phase == "NORMAL"
    assert point_six.task_point_advanced
    assert point_six.needs_new_plan


def test_nominal_future_starts_after_candidate_prefix():
    first = Control(0.4, 0.2)
    second = Control(0.4, 0.0)
    preview = TrajectoryPreview(
        state_index=0,
        nominal_control_index=0,
        cross_track_error_m=0.0,
        remaining_arc_length_m=1.0,
        progress=0.0,
        lookahead_x=1.0,
        lookahead_y=0.0,
        heading_error=0.0,
    )

    future = FixedMapSACTrainer._nominal_future_controls(
        SimpleNamespace(
            controls=(first, second),
            durations=(0.8, 0.8),
        ),
        preview,
    )

    assert future[:2] == ((first, 0.5), (second, 0.8))
    assert sum(duration for _, duration in future) == 1.7
    assert future[2][0] == Control(-0.4, 0.0)
    assert abs(future[2][1] - 0.4) <= 1e-12


def test_fixed_map_sac_trains_complete_safe_episode_and_saves_checkpoint(
    tmp_path,
):
    trainer = FixedMapSACTrainer(
        seed=31,
        hidden_dim=16,
        forward_profile=_forward_profile(),
        reverse_profile=_reverse_profile(),
        calibration_status="calibrated",
        reverse_calibration_status="calibrated",
    )

    assert trainer.dynamics.allow_reverse
    assert trainer.planning_controls[-1] == _reverse_profile().control
    starts = tuple(
        trainer._initial_state(100_000 + index)
        for index in range(20)
    )
    assert len({(state.x, state.y, state.yaw) for state in starts}) == 20
    assert all(
        abs(state.x - LIVE_RESET_SPAWN_X_M) <= 1.0
        and abs(state.y - LIVE_RESET_SPAWN_Y_M) <= 1.0
        and abs(
            (
                state.yaw
                - LIVE_RESET_SPAWN_YAW_RAD
                + 3.141592653589793
            )
            % (2.0 * 3.141592653589793)
            - 3.141592653589793
        )
        <= 3.141592653589793 / 6.0
        and trainer.compiled_map.snapshot.clearance_at(state) >= 3.0
        for state in starts
    )

    training, episodes = trainer.train(
        episodes=1,
        updates_per_episode=1,
        batch_size=1,
        burn_in=1,
        unroll=2,
    )

    assert trainer.observation_dim == 162
    assert training.completed_episodes == 1
    assert training.safety_stops == 0
    assert training.updates == 1
    assert training.total_steps == episodes[0].steps
    assert episodes[0].completed
    assert not episodes[0].timeout
    assert episodes[0].mission_index == 13
    assert len(episodes[0].waypoint_min_distances_m) == 13
    assert all(
        distance <= FIXED_ROUTE_TOLERANCE_M + 1e-9
        for distance in episodes[0].waypoint_min_distances_m
    )
    reached_steps = episodes[0].waypoint_reached_steps
    assert len(reached_steps) == 13
    assert all(step is not None for step in reached_steps)
    assert all(
        first < second
        for first, second in zip(reached_steps, reached_steps[1:])
    )
    assert (
        reached_steps[10]
        < episodes[0].narrow_escape_release_step
        < reached_steps[11]
    )
    assert (
        episodes[0].minimum_clearance_m
        > trainer.compiled_map.snapshot.required_clearance
    )

    checkpoint, manifest_path = trainer.save_checkpoint(
        tmp_path / "national_test_sac.pt",
        training,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == (
        "national-test-sac-checkpoint-v4"
    )
    assert manifest["route_guidance_version"] == (
        "national-test-reversible-composite-v37"
    )
    assert manifest["algorithm"] == "discrete-recurrent-sac"
    assert manifest["dynamics_version"] == trainer.dynamics.version
    assert manifest["reverse_control_profile"]["profile_hash"] == (
        _reverse_profile().profile_hash
    )
    assert manifest["reverse_calibration_status"] == "calibrated"
    assert manifest["route_id"] == trainer.compiled_map.manifest.route_id
    assert manifest["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert manifest["evaluation_summary"] is None
    assert not manifest["offline_ready"]
    assert not manifest["live_ready"]
    serialized = json.dumps(manifest).lower()
    assert "device_id" not in serialized
    assert '"host"' not in serialized
