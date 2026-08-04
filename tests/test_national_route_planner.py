"""Fixed National_Test route regressions.

The affine profile was captured from the live, read-only Get_Route response
for route version 46.  It contains no host, device id, or GPS coordinates.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from usvlib4ros.mapping import (
    SidecarCompilerConfig,
    compile_beihu_sidecar,
    load_sidecar_artifact,
)
from usvlib4ros.planning import (
    Control,
    CostConfig,
    GoalRegion,
    KinodynamicInformedRRTStarPlanner,
    PlannerConfig,
    PlanningRequest,
    PrototypeReducedDynamics,
    VesselState,
)
from usvlib4ros.planning.fixed_route import (
    CLEARANCE_APPROACH_GATE,
    CLEARANCE_COMPOSITE_ROUTE_INDEX,
    CLEARANCE_HANDOFF_TOLERANCE_M,
    CLEARANCE_HANDOFF_XY,
    NARROW_EGRESS_HANDOFF_Y_M,
    NARROW_ESCAPE_RELEASE_X_M,
    NARROW_ESCAPE_XY,
    NARROW_ROUTE_INDEX,
    NarrowCompositeInfeasibleError,
    build_fixed_leg_request,
    compile_offline_national_map,
    fixed_route_goal_xy,
    fixed_route_geometry_candidates,
    is_clearance_composite_trajectory,
    is_clearance_exit_trajectory,
    is_narrow_composite_trajectory,
    is_terminal_route_trajectory,
    narrow_escape_released,
    fixed_route_planning_gate,
    fixed_route_tolerance,
    fixed_route_waypoint_reached,
    plan_fixed_leg,
    plan_clearance_exit,
    plan_clearance_turn,
    plan_terminal_approach,
    plan_narrow_with_geometry_evidence,
)
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    diagnostic_forward_control_profile,
    reduced_dynamics_from_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR = (
    PROJECT_ROOT
    / "usvlib4ros"
    / "mapping"
    / "data"
    / "beihu_static_world_sidecar.json"
)
LIVE_ROUTE_AFFINE = (
    0.63215819453121191,
    -0.14629506646630225,
    0.19025639089919313,
    0.82212003241410347,
    -191.1658098488378,
    -299.80159718108752,
)


def test_live_non_collision_recovery_pose_keeps_point_two_planning_margin():
    compiled = compile_offline_national_map(
        session_id="live-point-one-margin-evidence",
    )
    recovery = VesselState(
        x=38.57,
        y=73.66,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
    )
    clearance = compiled.snapshot.clearance_at(recovery)

    assert 0.1 < clearance < 0.2
    assert compiled.snapshot.required_clearance == 0.2
    assert compiled.snapshot.geometry_version == (
        "circle-0.4-margin-0.2-live-recovery-v1"
    )
    assert not compiled.snapshot.is_state_valid(recovery)


@pytest.mark.parametrize(
    ("mission_index", "start"),
    (
        (
            4,
            VesselState(
                x=40.581295305118424,
                y=80.8301923354814,
                yaw=1.6329513211908342,
                speed=0.1268131711339526,
                yaw_rate=0.00115,
                throttle_state=0.1,
                rudder_state=-0.0014417537414617328,
                stamp_sim=115.8,
            ),
        ),
        (
            7,
            VesselState(
                x=38.406701,
                y=90.545456,
                yaw=1.879720,
                speed=0.126813,
                yaw_rate=0.001183,
                stamp_sim=100.0,
            ),
        ),
        (
            9,
            VesselState(
                x=34.44859350540107,
                y=95.36486319606753,
                yaw=1.3050734272233728,
                speed=0.19072958478941768,
                yaw_rate=-0.068879,
                throttle_state=0.1,
                rudder_state=0.05,
                stamp_sim=167.4,
            ),
        ),
        (
            9,
            VesselState(
                x=34.35032185630328,
                y=95.36035244732389,
                yaw=1.7281096131523364,
                speed=0.1268131711339526,
                yaw_rate=-0.02,
                throttle_state=0.1,
                rudder_state=0.03429066809911529,
                stamp_sim=266.4,
            ),
        ),
        (
            11,
            VesselState(
                x=34.992242240145565,
                y=99.4049267448261,
                yaw=2.2196044059495414,
                speed=0.12,
                yaw_rate=0.0,
                throttle_state=0.05,
                rudder_state=0.0,
                stamp_sim=76.8,
            ),
        ),
    ),
)
def test_gate_seed_uses_enough_budget_for_low_rudder_replan_states(
    mission_index,
    start,
):
    compiled = compile_offline_national_map(
        session_id="low-rudder-replan-regression",
    )
    profile = _formal_profile_shape()
    dynamics = reduced_dynamics_from_profile(profile)
    cost = CostConfig()
    request = build_fixed_leg_request(
        compiled,
        start_state=start,
        mission_index=mission_index,
        dynamics=dynamics,
        cost_config=cost,
        time_budget_ms=5_000.0,
        seed=31,
        lookahead_count=0,
    )
    planner = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=1_200,
            edge_durations=(0.2, 0.5, 1.0, 2.0),
            goal_bias=0.25,
            global_sample_ratio=0.3,
            rewire_radius=2.5,
            connect_tolerance=1.2,
            stop_on_first_solution=True,
            grid_seed_enabled=True,
            max_request_age_s=60.0,
            max_map_age_s=1.0e9,
            max_throttle=0.1,
            max_abs_rudder=0.1,
            forward_action_controls=profile.action_controls,
        )
    )

    result = planner.plan(
        request,
        compiled.snapshot,
        dynamics,
        cost,
        now_sim=start.stamp_sim,
    )

    assert result.trajectory is not None
    assert result.elapsed_ms <= 5_000.0
    assert set(result.trajectory.controls) <= set(profile.action_controls)
    if mission_index == 9:
        assert tuple(
            (region.x, region.y, region.position_tolerance)
            for region in request.required_visit_regions
        ) == ((34.8, 98.6, 0.5),)
        assert result.trajectory.min_clearance >= 0.3


def test_narrow_escape_releases_only_after_east_exit_and_south_bypass():
    compiled = compile_offline_national_map(
        session_id="narrow-safe-release-plane",
    )
    east_exit = VesselState(
        x=NARROW_ESCAPE_RELEASE_X_M,
        y=NARROW_ESCAPE_XY[1],
        yaw=3.0,
        speed=-0.12,
        yaw_rate=0.0,
    )
    released = replace(
        east_exit,
        x=NARROW_ESCAPE_XY[0] + 0.6,
        y=NARROW_EGRESS_HANDOFF_Y_M,
        yaw=-math.pi + 0.1,
        speed=0.28,
    )

    assert not narrow_escape_released(compiled, east_exit)
    assert narrow_escape_released(compiled, released)


def test_only_post_narrow_tail_legs_use_deterministic_global_trajectory():
    assert not is_terminal_route_trajectory(
        SimpleNamespace(mission_index=10, request_id="fixed-route-live-leg-10")
    )
    assert is_terminal_route_trajectory(
        SimpleNamespace(mission_index=11, request_id="fixed-route-live-leg-11")
    )
    assert is_terminal_route_trajectory(
        SimpleNamespace(mission_index=12, request_id="fixed-route-live-leg-12")
    )


def test_point_four_request_visits_published_point_then_ends_at_safe_handoff():
    compiled = compile_offline_national_map(
        session_id="point-four-composite-request",
    )
    profile = _formal_profile_shape()
    start = VesselState(
        x=40.49,
        y=77.5,
        yaw=1.29,
        speed=0.13,
        yaw_rate=0.04,
    )
    request = build_fixed_leg_request(
        compiled,
        start_state=start,
        mission_index=CLEARANCE_COMPOSITE_ROUTE_INDEX,
        dynamics=reduced_dynamics_from_profile(profile),
        cost_config=CostConfig(),
        time_budget_ms=5_000.0,
        seed=31,
        lookahead_count=0,
    )
    published = fixed_route_goal_xy(
        compiled.manifest,
        CLEARANCE_COMPOSITE_ROUTE_INDEX,
    )

    assert (
        request.goal_region.x,
        request.goal_region.y,
        request.goal_region.position_tolerance,
    ) == (*CLEARANCE_HANDOFF_XY, CLEARANCE_HANDOFF_TOLERANCE_M)
    assert request.goal_region.speed_limit == 1.8
    assert tuple(
        (region.x, region.y, region.position_tolerance)
        for region in request.required_visit_regions
    ) == (CLEARANCE_APPROACH_GATE, (*published, 0.5))
    assert request.route_gate is None
    assert request.continuation_targets == ()

    trajectory = plan_fixed_leg(
        compiled,
        start_state=start,
        mission_index=CLEARANCE_COMPOSITE_ROUTE_INDEX,
        dynamics=reduced_dynamics_from_profile(profile),
        forward_action_controls=profile.action_controls,
        seed=31,
    )
    assert is_clearance_composite_trajectory(trajectory)


def test_point_four_replan_preserves_completed_approach_progress():
    compiled = compile_offline_national_map(
        session_id="point-four-progress-replan",
    )
    profile = _formal_profile_shape()
    dynamics = reduced_dynamics_from_profile(profile)
    start = VesselState(
        x=40.558159,
        y=80.502191,
        yaw=1.440828,
        speed=0.503731,
        yaw_rate=0.0,
        throttle_state=0.4,
        rudder_state=0.0,
        stamp_sim=100.0,
    )
    request = build_fixed_leg_request(
        compiled,
        start_state=start,
        mission_index=CLEARANCE_COMPOSITE_ROUTE_INDEX,
        dynamics=dynamics,
        cost_config=CostConfig(),
        time_budget_ms=5_000.0,
        seed=31,
        lookahead_count=0,
        clearance_approach_completed=True,
    )
    published = fixed_route_goal_xy(
        compiled.manifest,
        CLEARANCE_COMPOSITE_ROUTE_INDEX,
    )

    assert tuple(
        (region.x, region.y, region.position_tolerance)
        for region in request.required_visit_regions
    ) == ((*published, 0.5),)

    trajectory = plan_fixed_leg(
        compiled,
        start_state=start,
        mission_index=CLEARANCE_COMPOSITE_ROUTE_INDEX,
        dynamics=dynamics,
        forward_action_controls=profile.action_controls,
        seed=31,
        clearance_approach_completed=True,
    )
    assert is_clearance_composite_trajectory(trajectory)
    assert trajectory.min_clearance >= 0.2


def test_clearance_exit_recovers_varied_point_five_arrival_heading():
    compiled = compile_offline_national_map(
        session_id="clearance-exit-arrival-regression",
    )
    profile = _formal_profile_shape()
    start = VesselState(
        x=39.09507386816462,
        y=87.4101532326401,
        yaw=1.963642848598805,
        speed=0.1322264628949831,
        yaw_rate=-0.18125023188045386,
        throttle_state=0.1,
        rudder_state=0.2,
        stamp_sim=115.7,
    )

    trajectory = plan_clearance_exit(
        compiled,
        start_state=start,
        dynamics=reduced_dynamics_from_profile(profile),
    )

    assert is_clearance_exit_trajectory(trajectory)
    assert fixed_route_waypoint_reached(
        compiled,
        5,
        trajectory.states[-1],
    )
    assert trajectory.min_clearance > 0.2


def test_clearance_exit_recovers_high_speed_point_five_arrival():
    compiled = compile_offline_national_map(
        session_id="clearance-exit-high-speed-regression",
    )
    profile = _formal_profile_shape()
    start = VesselState(
        x=40.317378035516036,
        y=84.64333508398141,
        yaw=0.07672302833736547,
        speed=0.5058500616077077,
        yaw_rate=0.31057508390152194,
        throttle_state=0.4,
        rudder_state=-0.2,
        stamp_sim=90.3,
    )

    trajectory = plan_clearance_exit(
        compiled,
        start_state=start,
        dynamics=reduced_dynamics_from_profile(profile),
    )

    assert is_clearance_exit_trajectory(trajectory)
    assert fixed_route_waypoint_reached(
        compiled,
        5,
        trajectory.states[-1],
    )
    assert trajectory.min_clearance > 0.2


def test_post_narrow_egress_stays_east_then_hands_off_south_heading_west():
    compiled = compile_offline_national_map(
        session_id="post-narrow-egress-regression",
    )
    profile = _formal_profile_shape()
    base_dynamics = reduced_dynamics_from_profile(profile)
    dynamics = replace(
        base_dynamics,
        version=f"{base_dynamics.version}-reverse-v1",
        allow_reverse=True,
        max_reverse_speed=0.2,
        reverse_throttle_speed_gain=0.3064056291910699,
    )
    east_exit = VesselState(
        x=NARROW_ESCAPE_RELEASE_X_M,
        y=NARROW_ESCAPE_XY[1],
        yaw=math.pi,
        speed=-0.12256225167642798,
        yaw_rate=-0.00078,
    )

    trajectory = plan_fixed_leg(
        compiled,
        start_state=east_exit,
        mission_index=NARROW_ROUTE_INDEX,
        dynamics=dynamics,
        forward_action_controls=(
            *profile.action_controls,
            Control(-0.4, 0.0),
        ),
        narrow_visit_completed=True,
        time_budget_ms=5_000.0,
        _allow_retry=False,
    )

    assert trajectory.validation_status == "VALID"
    assert trajectory.times[-1] <= 30.0
    assert trajectory.min_clearance > compiled.snapshot.required_clearance
    assert compiled.snapshot.check_motion(
        tuple(
            state
            for rollout in trajectory.edge_rollouts
            for state in rollout
        )
    ).valid
    assert trajectory.states[-1].y <= NARROW_EGRESS_HANDOFF_Y_M
    assert abs(abs(trajectory.states[-1].yaw) - math.pi) <= 0.2
    assert trajectory.controls[-1] == Control(0.4, 0.05)
    assert narrow_escape_released(compiled, trajectory.states[-1])


def test_clearance_turn_reaches_point_five_with_continuous_safety():
    compiled = compile_offline_national_map(
        session_id="clearance-turn-regression",
    )
    profile = _formal_profile_shape()
    start = VesselState(
        x=39.05185377426736,
        y=82.85347462347535,
        yaw=2.7358295104325276,
        speed=0.1498707794378663,
        yaw_rate=0.061668,
        throttle_state=0.1,
        rudder_state=-0.05,
    )

    trajectory = plan_clearance_turn(
        compiled,
        start_state=start,
        dynamics=reduced_dynamics_from_profile(profile),
    )

    assert trajectory.validation_status == "VALID"
    assert trajectory.times[-1] <= 30.0
    assert trajectory.min_clearance > 0.3
    assert fixed_route_waypoint_reached(
        compiled,
        4,
        next(
            state
            for state in trajectory.states
            if fixed_route_waypoint_reached(compiled, 4, state)
        ),
    )
    assert fixed_route_waypoint_reached(compiled, 5, trajectory.states[-1])

    fast = plan_clearance_turn(
        compiled,
        start_state=replace(
            start,
            speed=0.49,
            yaw_rate=0.0,
            throttle_state=0.4,
            rudder_state=0.0,
        ),
        dynamics=replace(
            reduced_dynamics_from_profile(profile),
            allow_reverse=True,
            max_reverse_speed=0.2,
            reverse_throttle_speed_gain=0.3064056291910699,
        ),
    )
    assert fast.controls[0] == Control(-0.4, 0.0)
    assert fast.min_clearance > compiled.snapshot.required_clearance
    assert any(
        fixed_route_waypoint_reached(compiled, 4, state)
        for state in fast.states
    )
    assert fixed_route_waypoint_reached(compiled, 5, fast.states[-1])


def test_clearance_exit_replans_point_six_without_revisiting_point_five():
    compiled = compile_offline_national_map(
        session_id="clearance-exit-replan-regression",
    )
    profile = _formal_profile_shape()
    point_five_entry = VesselState(
        x=40.254323,
        y=84.488819,
        yaw=-0.023457,
        speed=0.505137,
        yaw_rate=-0.407061,
        throttle_state=0.4,
        rudder_state=0.2,
    )

    trajectory = plan_clearance_exit(
        compiled,
        start_state=point_five_entry,
        dynamics=replace(
            reduced_dynamics_from_profile(profile),
            allow_reverse=True,
            max_reverse_speed=0.2,
            reverse_throttle_speed_gain=0.3064056291910699,
        ),
    )

    assert trajectory.validation_status == "VALID"
    assert trajectory.times[-1] <= 15.0
    assert trajectory.min_clearance > compiled.snapshot.required_clearance
    assert fixed_route_waypoint_reached(compiled, 5, trajectory.states[-1])


def test_terminal_approach_uses_calibrated_hard_turn_then_safe_straight():
    compiled = compile_offline_national_map(
        session_id="terminal-approach-regression",
    )
    profile = _formal_profile_shape()
    start = VesselState(
        x=26.67000858607491,
        y=96.90086507063388,
        yaw=-3.9222702764231503,
        speed=0.4275277886123331,
        yaw_rate=0.06678828721461431,
        throttle_state=0.1,
        rudder_state=-0.05,
    )

    trajectory = plan_terminal_approach(
        compiled,
        start_state=start,
        dynamics=reduced_dynamics_from_profile(profile),
    )

    assert trajectory.validation_status == "VALID"
    assert trajectory.times[-1] <= 15.0
    assert trajectory.min_clearance >= 0.3
    assert fixed_route_waypoint_reached(compiled, 12, trajectory.states[-1])


def _compiled_live_route():
    artifact, artifact_hash = load_sidecar_artifact(SIDECAR)
    return compile_beihu_sidecar(
        artifact,
        source_artifact_hash=artifact_hash,
        session_id="national-route-regression",
        stamp_sim=0.0,
        config=SidecarCompilerConfig(
            transform_model="route_fitted_affine",
            coverage_status="complete_prior",
            promotion_note="operator-authorization:offline-live-route-regression",
            fitted_affine=LIVE_ROUTE_AFFINE,
        ),
    )


def _route_planner():
    return KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=600,
            edge_durations=(0.2, 0.5, 1.0, 2.0),
            goal_bias=0.25,
            global_sample_ratio=0.3,
            rewire_radius=2.5,
            connect_tolerance=1.2,
            stop_on_first_solution=True,
            grid_seed_enabled=True,
            max_request_age_s=60.0,
            max_map_age_s=1.0e9,
        )
    )


def _formal_profile_shape() -> ForwardControlProfile:
    """Deterministic unit fixture matching the approved live profile shape."""

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
        positive_rudder_yaw_rate_gain=2.0353,
        negative_rudder_yaw_rate_gain=2.0871,
    )


def test_fixed_route_preserves_published_yellow_waypoint_coordinates():
    """Safety inflation may constrain approach, but must not move the task."""

    compiled = compile_offline_national_map(
        session_id="national-route-waypoint-semantics",
    )
    manifest = compiled.manifest
    snapshot = compiled.snapshot

    for mission_index, point_enu in enumerate(manifest.route_points_enu):
        expected = (
            point_enu[0] - manifest.origin_enu[0],
            point_enu[1] - manifest.origin_enu[1],
        )
        assert fixed_route_goal_xy(manifest, mission_index) == expected

        cell = snapshot._cell_for(*expected)
        assert cell is not None
        assert snapshot.rows[cell[1]][cell[0]] == "."
        gate = fixed_route_planning_gate(compiled, mission_index)
        gate_state = VesselState(
            x=gate[0],
            y=gate[1],
            yaw=0.0,
            speed=0.0,
            yaw_rate=0.0,
            stamp_sim=snapshot.stamp_sim,
        )
        assert snapshot.is_state_valid(gate_state)
        assert snapshot.clearance_at(gate_state) >= 0.3
        assert math.hypot(
            gate[0] - expected[0],
            gate[1] - expected[1],
        ) <= 0.5
        assert fixed_route_tolerance(compiled, mission_index) == 0.5


def test_waypoint_reach_requires_ship_centre_within_point_five_metres():
    compiled = compile_offline_national_map(
        session_id="national-route-reach-boundary",
    )
    goal_x, goal_y = fixed_route_goal_xy(compiled.manifest, 0)
    on_boundary = VesselState(
        x=goal_x + 0.5,
        y=goal_y,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=compiled.snapshot.stamp_sim,
    )

    assert fixed_route_waypoint_reached(compiled, 0, on_boundary)
    assert not fixed_route_waypoint_reached(
        compiled,
        0,
        VesselState(
            x=goal_x + 0.5001,
            y=goal_y,
            yaw=0.0,
            speed=0.0,
            yaw_rate=0.0,
            stamp_sim=compiled.snapshot.stamp_sim,
        ),
    )


def test_narrow_point_is_one_visit_then_east_escape_planning_request():
    compiled = compile_offline_national_map(
        session_id="national-route-narrow-composite",
    )
    dynamics = PrototypeReducedDynamics()
    previous = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX - 1,
    )
    original = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX,
    )
    next_goal = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX + 1,
    )
    start = VesselState(
        x=previous[0],
        y=previous[1],
        yaw=math.atan2(original[1] - previous[1], original[0] - previous[0]),
        speed=0.3,
        yaw_rate=0.0,
    )
    request = build_fixed_leg_request(
        compiled,
        start_state=start,
        mission_index=NARROW_ROUTE_INDEX,
        dynamics=dynamics,
        cost_config=CostConfig(),
        time_budget_ms=5_000.0,
        seed=31,
        lookahead_count=0,
    )

    assert (request.goal_region.x, request.goal_region.y) == NARROW_ESCAPE_XY
    assert len(request.required_visit_regions) == 2
    assert (
        request.required_visit_regions[1].x,
        request.required_visit_regions[1].y,
        request.required_visit_regions[1].position_tolerance,
    ) == (*original, 0.5)
    direct = compiled.snapshot.check_motion(
        (
            VesselState(
                x=request.required_visit_regions[0].x,
                y=request.required_visit_regions[0].y,
                yaw=0.0,
                speed=0.3,
                yaw_rate=0.0,
            ),
            VesselState(
                x=next_goal[0],
                y=next_goal[1],
                yaw=0.0,
                speed=0.3,
                yaw_rate=0.0,
            ),
        )
    )
    assert not direct.valid


def test_live_third_leg_uses_calibrated_east_bend_with_reverse_profile_loaded():
    """The calibrated primitives keep the live recovery leg east of buoy 15."""

    compiled = compile_offline_national_map(
        session_id="live-third-leg-clearance-gate",
    )
    profile = _formal_profile_shape()
    dynamics = replace(
        reduced_dynamics_from_profile(profile),
        version="live-third-leg-reverse-enabled",
        allow_reverse=True,
        max_reverse_speed=0.2,
        reverse_throttle_speed_gain=0.30640562919106995,
    )
    start = VesselState(
        x=38.71,
        y=74.52,
        yaw=1.56,
        speed=0.10,
        yaw_rate=0.21,
        stamp_sim=45.89,
    )
    request = build_fixed_leg_request(
        compiled,
        start_state=start,
        mission_index=2,
        dynamics=dynamics,
        cost_config=CostConfig(),
        time_budget_ms=5_000.0,
        seed=31,
        lookahead_count=0,
    )

    assert not request.required_visit_regions

    trajectory = plan_fixed_leg(
        compiled,
        start_state=start,
        mission_index=2,
        dynamics=dynamics,
        forward_action_controls=(
            *profile.action_controls,
            Control(-0.4, 0.0),
        ),
        time_budget_ms=5_000.0,
        optimize_with_rrtstar=False,
        seed=31,
        _allow_retry=False,
    )

    assert all(control.throttle >= 0.0 for control in trajectory.controls)
    assert trajectory.min_clearance >= 0.5


def test_escape_replan_does_not_require_revisiting_completed_narrow_point():
    compiled = compile_offline_national_map(
        session_id="national-route-escape-replan",
    )
    dynamics = PrototypeReducedDynamics()
    original = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX,
    )
    request = build_fixed_leg_request(
        compiled,
        start_state=VesselState(
            x=original[0],
            y=original[1],
            yaw=0.0,
            speed=0.3,
            yaw_rate=0.0,
        ),
        mission_index=NARROW_ROUTE_INDEX,
        dynamics=dynamics,
        cost_config=CostConfig(),
        time_budget_ms=5_000.0,
        seed=31,
        lookahead_count=0,
        narrow_visit_completed=True,
    )

    assert (request.goal_region.x, request.goal_region.y) == NARROW_ESCAPE_XY
    assert request.required_visit_regions == ()


def test_geometry_evidence_gate_is_circle_then_capsule_then_point_one_margin():
    compiled = compile_offline_national_map(
        session_id="national-route-geometry-gate",
    )

    candidates = fixed_route_geometry_candidates(compiled)

    assert tuple(
        candidate.snapshot.geometry_version for candidate in candidates
    ) == (
        "circle-0.4-margin-0.2-v1",
        "official-capsule-1.3x0.64-margin-0.2-v1",
        "official-capsule-1.3x0.64-margin-0.1-v1",
    )
    assert tuple(
        candidate.snapshot.required_clearance for candidate in candidates
    ) == (0.2, 0.2, 0.1)
    assert len(
        {candidate.snapshot.payload_content_hash for candidate in candidates}
    ) == 3


def test_narrow_composite_fails_closed_after_all_approved_geometry_gates():
    compiled = compile_offline_national_map(
        session_id="national-route-narrow-trajectory",
    )
    previous = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX - 1,
    )
    gate = fixed_route_planning_gate(compiled, NARROW_ROUTE_INDEX)
    start = VesselState(
        x=previous[0],
        y=previous[1],
        yaw=math.atan2(gate[1] - previous[1], gate[0] - previous[0]),
        speed=0.3,
        yaw_rate=0.0,
    )

    with pytest.raises(NarrowCompositeInfeasibleError) as captured:
        plan_narrow_with_geometry_evidence(
            compiled,
            start_state=start,
            time_budget_ms=400.0,
            seed=71,
            forward_action_controls=(
                diagnostic_forward_control_profile().action_controls
            ),
        )

    evidence = captured.value.evidence
    assert len(evidence) == 3
    assert not any(item.feasible for item in evidence)
    assert tuple(item.geometry_version for item in evidence) == (
        "circle-0.4-margin-0.2-v1",
        "official-capsule-1.3x0.64-margin-0.2-v1",
        "official-capsule-1.3x0.64-margin-0.1-v1",
    )


def test_narrow_composite_can_reverse_through_its_single_entry():
    compiled = compile_offline_national_map(
        session_id="national-route-reverse-escape",
    )
    profile = _formal_profile_shape()
    base_dynamics = reduced_dynamics_from_profile(profile)
    dynamics = replace(
        base_dynamics,
        version=f"{base_dynamics.version}-reverse-v1",
        allow_reverse=True,
        max_reverse_speed=0.2,
        reverse_throttle_speed_gain=0.306,
    )
    previous = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX - 1,
    )
    gate = fixed_route_planning_gate(compiled, NARROW_ROUTE_INDEX)
    start = VesselState(
        x=previous[0],
        y=previous[1],
        yaw=math.atan2(gate[1] - previous[1], gate[0] - previous[0]),
        speed=0.3,
        yaw_rate=0.0,
    )
    reverse = Control(-0.4, 0.0)

    trajectory = plan_fixed_leg(
        compiled,
        start_state=start,
        mission_index=NARROW_ROUTE_INDEX,
        dynamics=dynamics,
        forward_action_controls=(*profile.action_controls, reverse),
        time_budget_ms=5_000.0,
        seed=71,
        _allow_retry=False,
    )

    original = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX,
    )
    assert reverse in trajectory.controls
    assert is_narrow_composite_trajectory(trajectory)
    assert any(
        math.hypot(state.x - original[0], state.y - original[1]) <= 0.5
        for rollout in trajectory.edge_rollouts
        for state in rollout
    )
    assert math.hypot(
        trajectory.states[-1].x - NARROW_ESCAPE_XY[0],
        trajectory.states[-1].y - NARROW_ESCAPE_XY[1],
    ) <= 0.3


@pytest.mark.parametrize(
    "entry_state",
    (
        (
            32.95171342055915,
            99.26520585035696,
            2.819942972022514,
            0.44881696654570113,
            0.6233649894781259,
            0.1,
            -0.4508191167693636,
        ),
        (
            32.92501609984386,
            99.23545963004646,
            2.896899542842471,
            0.5072523199396324,
            0.34187009513995664,
            0.4,
            -0.2,
        ),
    ),
)
def test_narrow_composite_recovers_a_safe_high_yaw_rate_entry_state(
    entry_state,
):
    compiled = compile_offline_national_map(
        session_id="national-route-reverse-entry-recovery",
    )
    profile = _formal_profile_shape()
    base_dynamics = reduced_dynamics_from_profile(profile)
    dynamics = replace(
        base_dynamics,
        version=f"{base_dynamics.version}-reverse-v1",
        allow_reverse=True,
        max_reverse_speed=0.2,
        reverse_throttle_speed_gain=0.306,
    )
    start = VesselState(
        x=entry_state[0],
        y=entry_state[1],
        yaw=entry_state[2],
        speed=entry_state[3],
        yaw_rate=entry_state[4],
        throttle_state=entry_state[5],
        rudder_state=entry_state[6],
    )
    reverse = Control(-0.4, 0.0)

    trajectory = plan_fixed_leg(
        compiled,
        start_state=start,
        mission_index=NARROW_ROUTE_INDEX,
        dynamics=dynamics,
        forward_action_controls=(*profile.action_controls, reverse),
        time_budget_ms=5_000.0,
        seed=100,
        _allow_retry=False,
    )

    original = fixed_route_goal_xy(
        compiled.manifest,
        NARROW_ROUTE_INDEX,
    )
    assert compiled.snapshot.check_motion(
        trajectory.edge_rollouts[0]
    ).valid
    assert any(
        math.hypot(state.x - original[0], state.y - original[1]) <= 0.5
        for rollout in trajectory.edge_rollouts
        for state in rollout
    )
    assert reverse in trajectory.controls
    assert math.hypot(
        trajectory.states[-1].x - NARROW_ESCAPE_XY[0],
        trajectory.states[-1].y - NARROW_ESCAPE_XY[1],
    ) <= 0.3


def test_planner_solves_first_buoy_gate_with_kinodynamic_seed():
    """The first fixed leg needs a bend around a buoy, not a straight edge."""

    compiled = _compiled_live_route()
    snapshot = compiled.snapshot
    manifest = compiled.manifest
    dynamics = PrototypeReducedDynamics()
    cost = CostConfig()
    (start_e, goal_e) = manifest.route_points_enu[:2]
    sx = start_e[0] - manifest.origin_enu[0]
    sy = start_e[1] - manifest.origin_enu[1]
    gx = goal_e[0] - manifest.origin_enu[0]
    gy = goal_e[1] - manifest.origin_enu[1]
    yaw = math.atan2(gy - sy, gx - sx)
    start = VesselState(
        x=sx,
        y=sy,
        yaw=yaw,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    request = PlanningRequest(
        request_id="national-leg-0-1",
        session_id=snapshot.session_id,
        start_state=start,
        goal_region=GoalRegion(
            x=gx,
            y=gy,
            position_tolerance=2.5,
            speed_limit=1.2,
            yaw_rate_limit=1.2,
        ),
        map_snapshot_id=snapshot.snapshot_id,
        dynamics_version=dynamics.version,
        cost_config_version=cost.version,
        time_budget_ms=2_000.0,
        seed=31,
        mission_index=1,
        stamp_sim=0.0,
        mission_version=f"route-v{manifest.route_version}",
    )
    planner = _route_planner()

    result = planner.plan(
        request,
        snapshot,
        dynamics,
        cost,
        now_sim=0.0,
    )

    assert result.trajectory is not None, (
        f"fixed route leg failed: {result.status.value} {result.reason}"
    )
    assert result.trajectory.validation_status == "VALID"
    assert request.goal_region.contains(result.trajectory.states[-1])
    assert result.trajectory.min_clearance > snapshot.required_clearance


def test_fixed_leg_uses_only_calibrated_forward_motion_primitives():
    compiled = compile_offline_national_map(
        session_id="national-route-profile-controls",
    )
    profile = _formal_profile_shape()
    dynamics = reduced_dynamics_from_profile(profile)
    start_xy = fixed_route_goal_xy(compiled.manifest, 0)
    goal_xy = fixed_route_goal_xy(compiled.manifest, 1)
    start = VesselState(
        x=start_xy[0],
        y=start_xy[1],
        yaw=math.atan2(
            goal_xy[1] - start_xy[1],
            goal_xy[0] - start_xy[0],
        ),
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )

    trajectory = plan_fixed_leg(
        compiled,
        start_state=start,
        mission_index=1,
        dynamics=dynamics,
        forward_action_controls=profile.action_controls,
        time_budget_ms=5_000.0,
        optimize_with_rrtstar=False,
    )

    assert trajectory.controls
    assert set(trajectory.controls) <= set(profile.action_controls)
    assert trajectory.times[-1] <= 300.0


def test_planner_chains_ordinary_legs_until_narrow_geometry_gate():
    """Ordinary legs must use calibrated primitives before the known blocker."""

    compiled = compile_offline_national_map(
        session_id="national-route-chain",
    )
    profile = _formal_profile_shape()
    dynamics = reduced_dynamics_from_profile(profile)
    start_xy = fixed_route_goal_xy(compiled.manifest, 0)
    goal_xy = fixed_route_goal_xy(compiled.manifest, 1)
    state = VesselState(
        x=start_xy[0],
        y=start_xy[1],
        yaw=math.atan2(
            goal_xy[1] - start_xy[1],
            goal_xy[0] - start_xy[0],
        ),
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    trajectories = []
    ordinary_trajectories = []
    for mission_index in (1, 2, 3):
        trajectory = plan_fixed_leg(
            compiled,
            start_state=state,
            mission_index=mission_index,
            dynamics=dynamics,
            forward_action_controls=profile.action_controls,
            time_budget_ms=5_000.0,
            optimize_with_rrtstar=False,
        )
        trajectories.append(trajectory)
        ordinary_trajectories.append(trajectory)
        state = trajectory.states[-1]

    clearance_turn = plan_clearance_turn(
        compiled,
        start_state=state,
        dynamics=dynamics,
    )
    trajectories.append(clearance_turn)
    state = clearance_turn.states[-1]

    for mission_index in range(6, NARROW_ROUTE_INDEX):
        trajectory = plan_fixed_leg(
            compiled,
            start_state=state,
            mission_index=mission_index,
            dynamics=dynamics,
            forward_action_controls=profile.action_controls,
            time_budget_ms=5_000.0,
            optimize_with_rrtstar=False,
        )
        trajectories.append(trajectory)
        ordinary_trajectories.append(trajectory)
        state = trajectory.states[-1]

    assert len(trajectories) == 8
    assert all(
        trajectory.validation_status == "VALID"
        for trajectory in trajectories
    )
    assert all(
        trajectory.min_clearance
        > compiled.snapshot.required_clearance
        for trajectory in trajectories
    )
    assert all(
        control in profile.action_controls
        for trajectory in ordinary_trajectories
        for control in trajectory.controls
    )
    assert sum(trajectory.times[-1] for trajectory in trajectories) < 300.0


def test_fixed_leg_recovers_after_safe_policy_exploration():
    compiled = compile_offline_national_map(
        session_id="national-route-recovery",
    )
    off_path_state = VesselState(
        x=34.992242240145565,
        y=99.4049267448261,
        yaw=2.2196044059495414,
        speed=0.12,
        yaw_rate=0.0,
        throttle_state=0.05,
        rudder_state=0.0,
        stamp_sim=76.8,
    )
    profile = _formal_profile_shape()

    trajectory = plan_fixed_leg(
        compiled,
        start_state=off_path_state,
        mission_index=11,
        dynamics=reduced_dynamics_from_profile(profile),
        forward_action_controls=profile.action_controls,
        time_budget_ms=5_000.0,
        seed=809,
    )

    assert trajectory.validation_status == "VALID"
    assert trajectory.controls
    assert (
        trajectory.min_clearance
        > compiled.snapshot.required_clearance
    )
