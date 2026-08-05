"""Focused contracts for the National_Test direct waypoint controller."""

from __future__ import annotations

import math

import pytest

from usvlib4ros.navigation.fixed_corridor import (
    DEFAULT_CORRIDOR_PATH,
    FROZEN_CORRIDOR_SCHEMA,
    FrozenRouteCorridor,
)
from usvlib4ros.navigation.waypoint_control import (
    ACTION_SCHEMA_V3,
    CHECKPOINT_SCHEMA_V6,
    OBSERVATION_SCHEMA_V3,
    ActuatorTransitionGuard,
    CalibratedActionSet,
    NoSafeActionWindow,
)
from usvlib4ros.planning.fixed_route import (
    FIXED_ROUTE_TOLERANCE_M,
    compile_offline_national_map,
    fixed_route_goal_xy,
)
from usvlib4ros.planning.forward_control_profile import ForwardControlProfile
from usvlib4ros.planning.kinodynamic_informed_rrtstar import Control, VesselState
from usvlib4ros.policy.recurrent_sac import LocalWaypointObservationV3


def _calibrated_profile() -> ForwardControlProfile:
    return ForwardControlProfile(
        calibration_hash="1" * 64,
        minimum_steerage_throttle=0.3,
        cruise_throttle=0.4,
        action_controls=(
            Control(0.3, -0.5),
            Control(0.4, -0.2),
            Control(0.4, 0.0),
            Control(0.4, 0.2),
            Control(0.3, 0.5),
        ),
        positive_rudder_yaw_rate_gain=0.8,
        negative_rudder_yaw_rate_gain=0.9,
        action_schema=ACTION_SCHEMA_V3,
    )


def _observation() -> LocalWaypointObservationV3:
    return LocalWaypointObservationV3(
        laser_ranges=tuple(float(index) for index in range(72)),
        laser_valid_mask=tuple(index % 2 == 0 for index in range(72)),
        scan_age_s=0.1,
        pose_age_s=0.2,
        device_age_s=0.3,
        speed_mps=0.4,
        yaw_rate_rad_s=0.5,
        actual_throttle=0.6,
        actual_rudder=-0.7,
        current_waypoint_body_xy=(1.0, 2.0),
        next_waypoint_body_xy=(3.0, 4.0),
        next_waypoint_valid=True,
        mission_progress=0.25,
        corridor_cross_track_m=-0.4,
        corridor_heading_error_rad=0.15,
        corridor_progress=0.5,
        map_clearance_m=1.2,
        safe_action_mask=(True, False, True, False, True),
        session_id="v3-contract",
        stamp_sim=12.0,
        hidden_reset=False,
    )


def test_local_waypoint_observation_v3_has_fixed_166_value_contract():
    observation = _observation()
    vector = observation.to_vector()

    assert observation.schema_version == OBSERVATION_SCHEMA_V3
    assert observation.feature_dim == 166
    assert vector[:72] == tuple(float(index) for index in range(72))
    assert vector[72:144] == tuple(
        1.0 if index % 2 == 0 else 0.0 for index in range(72)
    )
    assert vector[144:147] == pytest.approx((0.1, 0.2, 0.3))
    assert vector[147:151] == pytest.approx((0.4, 0.5, 0.6, -0.7))
    assert vector[151:157] == pytest.approx((1.0, 2.0, 3.0, 4.0, 1.0, 0.25))
    assert vector[157:161] == pytest.approx((-0.4, 0.15, 0.5, 1.2))
    assert vector[161:166] == pytest.approx((1.0, 0.0, 1.0, 0.0, 1.0))


def test_five_calibrated_controls_are_percent_unique_and_two_sided():
    actions = CalibratedActionSet.from_profile(_calibrated_profile())

    assert actions.schema_version == ACTION_SCHEMA_V3
    assert len(set(actions.percent_commands)) == 5
    assert actions.percent_commands[0][1] < actions.percent_commands[1][1] < 0
    assert actions.percent_commands[2][1] == 0
    assert 0 < actions.percent_commands[3][1] < actions.percent_commands[4][1]
    assert CHECKPOINT_SCHEMA_V6 == "national-test-sac-checkpoint-v6"


def test_percent_duplicate_controls_are_rejected_even_when_floats_differ():
    profile = _calibrated_profile()
    duplicate = ForwardControlProfile(
        calibration_hash=profile.calibration_hash,
        minimum_steerage_throttle=profile.minimum_steerage_throttle,
        cruise_throttle=profile.cruise_throttle,
        action_controls=(
            Control(0.301, -0.501),
            Control(0.304, -0.504),
            *profile.action_controls[2:],
        ),
        action_schema=ACTION_SCHEMA_V3,
    )

    with pytest.raises(ValueError, match="unique.*percent"):
        CalibratedActionSet.from_profile(duplicate)


def test_extreme_rudder_reversal_requires_executed_straight_transition():
    guard = ActuatorTransitionGuard()
    assert guard.reachability_mask() == (True,) * 5

    guard.record_executed(0)
    assert guard.reachability_mask() == (True, True, True, False, False)

    guard.record_executed(1)
    assert guard.reachability_mask() == (True, True, True, False, False)

    guard.record_executed(2)
    assert guard.reachability_mask() == (True,) * 5

    guard.record_executed(4)
    assert guard.reachability_mask() == (False, False, True, True, True)


def test_ten_fresh_no_safe_cycles_truncate_but_stale_does_not_count():
    window = NoSafeActionWindow(limit=10)

    for _ in range(9):
        assert not window.observe(fresh_inputs=True, has_safe_action=False)
    assert window.count == 9
    assert not window.observe(fresh_inputs=False, has_safe_action=False)
    assert window.count == 0
    for _ in range(9):
        assert not window.observe(fresh_inputs=True, has_safe_action=False)
    assert window.observe(fresh_inputs=True, has_safe_action=False)
    assert window.count == 10
    assert not window.observe(fresh_inputs=True, has_safe_action=True)
    assert window.count == 0


def test_frozen_corridor_is_hash_bound_safe_and_keeps_thirteen_task_points():
    compiled = compile_offline_national_map(
        session_id="corridor-contract",
        required_clearance_m=0.2,
    )
    corridor = FrozenRouteCorridor.load(DEFAULT_CORRIDOR_PATH, compiled)

    assert corridor.schema_version == FROZEN_CORRIDOR_SCHEMA
    assert corridor.required_clearance_m == pytest.approx(0.2)
    assert corridor.source_artifact_hash == compiled.snapshot.source_artifact_hash
    assert corridor.map_payload_hash == compiled.snapshot.payload_content_hash
    assert len(corridor.task_points) == 13
    assert len(corridor.task_anchors) == 13
    assert len(corridor.polyline) > len(corridor.task_points)

    for index, (task_point, anchor) in enumerate(
        zip(corridor.task_points, corridor.task_anchors)
    ):
        assert task_point == pytest.approx(fixed_route_goal_xy(compiled.manifest, index))
        assert math.dist(task_point, anchor) <= FIXED_ROUTE_TOLERANCE_M + 1e-9
        anchor_state = VesselState(
            x=anchor[0],
            y=anchor[1],
            yaw=0.0,
            speed=0.0,
            yaw_rate=0.0,
        )
        assert compiled.snapshot.is_state_valid(anchor_state)
        assert compiled.snapshot.clearance_at(anchor_state) >= 0.2


def test_corridor_projection_is_monotonic_across_narrow_segment_envelopes():
    compiled = compile_offline_national_map(
        session_id="corridor-envelope",
        required_clearance_m=0.2,
    )
    corridor = FrozenRouteCorridor.load(DEFAULT_CORRIDOR_PATH, compiled)
    previous_progress = 0.0

    for index in (2, 3, 4, 5, 10, 11):
        x, y = corridor.task_anchors[index]
        state = VesselState(
            x=x + 0.2,
            y=y - 0.15,
            yaw=0.35,
            speed=0.35,
            yaw_rate=0.2,
        )
        projection = corridor.project(state, previous_progress)
        assert math.isfinite(projection.cross_track_error_m)
        assert math.isfinite(projection.heading_error_rad)
        assert projection.route_progress >= previous_progress
        previous_progress = projection.route_progress
