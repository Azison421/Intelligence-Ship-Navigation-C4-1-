import hashlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from usvlib4ros.mapping import GpsProjector
from usvlib4ros.navigation.fixed_map_runtime import (
    DEFAULT_CHECKPOINT,
    FixedMapControllerCore,
    RuntimeManeuverProfile,
    RuntimeSafetyProfile,
    RuntimeInput,
    build_live_route_context,
    load_live_ready_policy,
    load_tested_candidate_policy,
    runtime_maneuver_profile_from_manifest,
    runtime_safety_profile_from_manifest,
)
from usvlib4ros.navigation.reverse_control_calibration import (
    ReverseControlProfile,
    enable_reverse_dynamics,
)
from usvlib4ros.planning import Control, VesselState
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    reduced_dynamics_from_profile,
)
from usvlib4ros.planning.fixed_route import (
    CLEARANCE_HANDOFF_XY,
    NARROW_ESCAPE_XY,
    NARROW_ROUTE_INDEX,
    compile_offline_national_map,
    fixed_route_goal_xy,
    fixed_route_guidance_hash,
    fixed_route_planning_gate,
    is_narrow_egress_trajectory,
    plan_fixed_leg,
)
from usvlib4ros.policy import RecurrentDiscreteSAC


def _live_route_and_pose():
    compiled = compile_offline_national_map(session_id="route-fixture")
    manifest = compiled.manifest
    projector = GpsProjector(*manifest.gps_origin)
    points = []
    for x, y in manifest.route_points_enu:
        lat, lng = projector.enu_to_gps(x, y)
        points.append(SimpleNamespace(lat=lat, lng=lng))
    route = SimpleNamespace(
        id=manifest.route_id,
        name=manifest.route_name,
        version=1785568402934,
        start_index=0,
        points=points,
    )
    pose = SimpleNamespace(
        lat=points[0].lat,
        lng=points[0].lng,
        yaw=0.0,
        speed=0.0,
        rotate_speed=0.0,
    )
    return route, pose


def test_runtime_defaults_to_operator_tested_candidate_checkpoint():
    assert (
        DEFAULT_CHECKPOINT.name
        == "national_test_sac_live_v10_tested.pt"
    )


def test_v5_live_loader_enables_full_predictive_safe_mask_authority_without_changing_v4(tmp_path):
    source = (
        __import__("pathlib").Path("artifacts/checkpoints")
        / "national_test_sac_v37_zero_clearance_conservative_345_unity_test.pt"
    )
    candidate = tmp_path / "generation-001.pt"
    candidate.write_bytes(source.read_bytes())
    manifest = json.loads(
        source.with_suffix(".pt.json").read_text(encoding="utf-8")
    )
    manifest.update(
        {
            "schema_version": "national-test-sac-checkpoint-v5",
            "checkpoint_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "policy_gate_version": "sac-predictive-safe-mask-v1",
            "offline_ready": True,
            "live_ready": True,
            "unity_validation_log_hashes": [f"embedded-{index}" for index in range(5)],
            "evaluation_evidence": {
                "offline": {
                    "attempted": 20,
                    "completed": 20,
                    "collisions": 0,
                    "laser_stops": 0,
                    "safety_stops": 0,
                    "timeouts": 0,
                    "unrecovered_unsafe_events": 0,
                },
                "unity": {
                    "attempted": 5,
                    "completed": 5,
                    "collisions": 0,
                    "laser_stops": 0,
                    "safety_stops": 0,
                    "timeouts": 0,
                    "unrecovered_unsafe_events": 0,
                },
            },
            "promotion_decision": {"promote": True},
        }
    )
    manifest["safety_profile"]["unity_test_only"] = False
    manifest["clearance_maneuver_profile"]["unity_test_only"] = False
    candidate.with_suffix(".pt.json").write_text(json.dumps(manifest), encoding="utf-8")
    route, pose = _live_route_and_pose()
    safety = runtime_safety_profile_from_manifest(manifest)
    maneuver = runtime_maneuver_profile_from_manifest(manifest)
    context = build_live_route_context(
        route,
        pose,
        session_id="v5-live-loader",
        safety_profile=safety,
        maneuver_profile=maneuver,
    )

    v5 = load_live_ready_policy(candidate, context)
    original_manifest = json.loads(source.with_suffix(".pt.json").read_text(encoding="utf-8"))
    v4_context = build_live_route_context(
        route,
        pose,
        session_id="v4-loader",
        safety_profile=runtime_safety_profile_from_manifest(original_manifest),
        maneuver_profile=runtime_maneuver_profile_from_manifest(original_manifest),
    )
    v4 = load_tested_candidate_policy(source, v4_context)

    assert v5.full_safe_action_authority is True
    assert v4.full_safe_action_authority is False
    assert FixedMapControllerCore(context, v5).full_safe_action_authority is True


def _runtime_state(context):
    manifest = context.compiled_map.manifest
    first, second = manifest.route_points_enu[:2]
    x = first[0] - manifest.origin_enu[0]
    y = first[1] - manifest.origin_enu[1]
    return VesselState(
        x=x,
        y=y,
        yaw=math.atan2(second[1] - first[1], second[0] - first[0]),
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )


def _sample(context, **changes):
    values = {
        "vessel_state": _runtime_state(context),
        "laser_ranges": (20.0,) * 72,
        "laser_valid_mask": (False,) * 72,
        "pose_age_s": 0.0,
        "scan_age_s": 0.0,
        "device_age_s": 0.0,
        "work_model": 2,
        "task_status": 1,
    }
    values.update(changes)
    return RuntimeInput(**values)


def test_live_route_context_accepts_only_approved_national_route():
    route, pose = _live_route_and_pose()

    context = build_live_route_context(
        route,
        pose,
        session_id="live-route-test",
    )

    assert context.fit_residual_m < 0.05
    assert context.route_version == route.version
    assert context.compiled_map.manifest.route_id == route.id

    route.id = "wrong-route"
    with pytest.raises(ValueError, match="route id"):
        build_live_route_context(
            route,
            pose,
            session_id="wrong-route-test",
        )


def test_runtime_safety_profile_keeps_legacy_defaults_and_parses_zero():
    from usvlib4ros.navigation.fixed_map_runtime import (
        runtime_safety_profile_from_manifest,
    )

    baseline = runtime_safety_profile_from_manifest({})
    zero = runtime_safety_profile_from_manifest(
        {
            "safety_profile": {
                "id": "national-test-zero-clearance-unity-v1",
                "required_clearance_m": 0.0,
                "laser_emergency_distance_m": 0.0,
                "unity_test_only": True,
            }
        }
    )

    assert baseline.required_clearance_m == 0.2
    assert baseline.laser_emergency_distance_m == 0.6
    assert not baseline.unity_test_only
    assert zero.required_clearance_m == 0.0
    assert zero.laser_emergency_distance_m == 0.0
    assert zero.unity_test_only

    with pytest.raises(ValueError, match="invalid"):
        runtime_safety_profile_from_manifest(
            {
                "safety_profile": {
                    "id": "invalid-negative-clearance",
                    "required_clearance_m": -0.1,
                    "laser_emergency_distance_m": 0.0,
                    "unity_test_only": True,
                }
            }
        )


def test_runtime_maneuver_profile_keeps_legacy_defaults_and_parses_slow_turn():
    from usvlib4ros.navigation.fixed_map_runtime import (
        runtime_maneuver_profile_from_manifest,
    )

    baseline = runtime_maneuver_profile_from_manifest({})
    slow = runtime_maneuver_profile_from_manifest(
        {
            "clearance_maneuver_profile": {
                "id": "points-three-five-conservative-unity-v2",
                "approach_throttle_cap": 0.1,
                "approach_rudder_cap": 0.1,
                "turn_throttle": 0.1,
                "turn_rudder": 0.12,
                "turn_max_edges": 180,
                "turn_entry_speed_limit_mps": 0.15,
                "unity_test_only": True,
            }
        }
    )

    assert baseline.approach_throttle_cap == 0.4
    assert baseline.approach_rudder_cap == 1.0
    assert baseline.turn_control == Control(0.4, 0.2)
    assert baseline.turn_max_edges == 80
    assert not baseline.unity_test_only
    assert slow.approach_throttle_cap == 0.1
    assert slow.approach_rudder_cap == 0.1
    assert slow.turn_control == Control(0.1, 0.12)
    assert slow.turn_max_edges == 180
    assert slow.turn_entry_speed_limit_mps == 0.15
    assert slow.unity_test_only

    with pytest.raises(ValueError, match="invalid"):
        runtime_maneuver_profile_from_manifest(
            {
                "clearance_maneuver_profile": {
                    "id": "invalid-fast-profile",
                    "approach_throttle_cap": 1.1,
                    "approach_rudder_cap": 0.1,
                    "turn_throttle": 0.1,
                    "turn_rudder": 0.15,
                    "turn_max_edges": 180,
                    "turn_entry_speed_limit_mps": 0.15,
                    "unity_test_only": True,
                }
            }
        )


def test_zero_clearance_map_removes_buffer_but_keeps_footprint_collision():
    baseline = compile_offline_national_map(session_id="baseline-clearance")
    zero = compile_offline_national_map(
        session_id="zero-clearance",
        required_clearance_m=0.0,
    )
    near_obstacle = VesselState(
        x=38.56912943606869,
        y=73.6600965399968,
        yaw=1.102951261381084,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    obstacle = zero.snapshot.circular_obstacles[0]
    collision = VesselState(
        x=obstacle.x,
        y=obstacle.y,
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )

    assert 0.0 < baseline.snapshot.clearance_at(near_obstacle) < 0.2
    assert not baseline.snapshot.is_state_valid(near_obstacle)
    assert zero.snapshot.is_state_valid(near_obstacle)
    assert zero.snapshot.footprint_radius == baseline.snapshot.footprint_radius
    assert zero.snapshot.payload_content_hash != baseline.snapshot.payload_content_hash
    assert not zero.snapshot.is_state_valid(collision)


def test_zero_clearance_runtime_only_stops_laser_at_contact():
    from usvlib4ros.navigation.fixed_map_runtime import (
        RuntimeSafetyProfile,
    )

    route, pose = _live_route_and_pose()
    profile = RuntimeSafetyProfile(
        profile_id="national-test-zero-clearance-unity-v1",
        required_clearance_m=0.0,
        laser_emergency_distance_m=0.0,
        unity_test_only=True,
    )
    context = build_live_route_context(
        route,
        pose,
        session_id="zero-laser-threshold",
        safety_profile=profile,
    )
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=16,
        seed=31,
    )

    positive = FixedMapControllerCore(context, policy).step(
        _sample(
            context,
            laser_ranges=(0.01,) + (20.0,) * 71,
            laser_valid_mask=(True,) + (False,) * 71,
        )
    )
    contact = FixedMapControllerCore(context, policy).step(
        _sample(
            context,
            laser_ranges=(0.0,) + (20.0,) * 71,
            laser_valid_mask=(True,) + (False,) * 71,
        )
    )

    assert positive.reason != "LASER_EMERGENCY_STOP"
    assert contact.stop
    assert contact.reason == "LASER_EMERGENCY_STOP"


def test_zero_clearance_candidate_reuses_v37_weights_and_is_unity_only():
    from usvlib4ros.navigation.fixed_map_runtime import (
        DEFAULT_UNITY_TEST_CHECKPOINT,
        load_runtime_maneuver_profile,
        load_runtime_safety_profile,
        load_tested_candidate_policy,
    )
    from usvlib4ros.policy.checkpoint_promotion import PolicyMode

    baseline_checkpoint = (
        DEFAULT_UNITY_TEST_CHECKPOINT.parent
        / "national_test_sac_v37_unity_test.pt"
    )
    previous_zero_checkpoint = (
        DEFAULT_UNITY_TEST_CHECKPOINT.parent
        / "national_test_sac_v37_zero_clearance_unity_test.pt"
    )
    previous_slow_checkpoint = (
        DEFAULT_UNITY_TEST_CHECKPOINT.parent
        / "national_test_sac_v37_zero_clearance_slow_turn_unity_test.pt"
    )
    manifest = json.loads(
        DEFAULT_UNITY_TEST_CHECKPOINT.with_suffix(".pt.json").read_text(
            encoding="utf-8"
        )
    )

    assert DEFAULT_UNITY_TEST_CHECKPOINT.name == (
        "national_test_sac_v37_zero_clearance_conservative_345_unity_test.pt"
    )
    assert hashlib.sha256(DEFAULT_UNITY_TEST_CHECKPOINT.read_bytes()).digest() == (
        hashlib.sha256(baseline_checkpoint.read_bytes()).digest()
    )
    assert manifest["offline_ready"] is False
    assert manifest["live_ready"] is False
    profile = load_runtime_safety_profile(
        DEFAULT_UNITY_TEST_CHECKPOINT,
        PolicyMode.UNITY_TEST,
    )
    maneuver_profile = load_runtime_maneuver_profile(
        DEFAULT_UNITY_TEST_CHECKPOINT,
        PolicyMode.UNITY_TEST,
    )
    with pytest.raises(ValueError, match="unity_test"):
        load_runtime_safety_profile(
            DEFAULT_UNITY_TEST_CHECKPOINT,
            PolicyMode.LIVE,
        )
    with pytest.raises(ValueError, match="unity_test"):
        load_runtime_maneuver_profile(
            DEFAULT_UNITY_TEST_CHECKPOINT,
            PolicyMode.LIVE,
        )

    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="zero-candidate-load",
        safety_profile=profile,
        maneuver_profile=maneuver_profile,
    )
    policy = load_tested_candidate_policy(
        DEFAULT_UNITY_TEST_CHECKPOINT,
        context,
    )
    core = FixedMapControllerCore(context, policy)
    core.mission_index = 3
    decision = core.step(
        _sample(
            context,
            vessel_state=VesselState(
                x=39.99182511134593,
                y=77.3913731240417,
                yaw=1.2087803047104109,
                speed=0.12681330183438116,
                yaw_rate=0.007204442569118387,
                stamp_sim=65.5,
            ),
        )
    )
    baseline_context = build_live_route_context(
        route,
        pose,
        session_id="baseline-candidate-load",
    )
    baseline_policy = load_tested_candidate_policy(
        baseline_checkpoint,
        baseline_context,
    )
    previous_zero_safety = load_runtime_safety_profile(
        previous_zero_checkpoint,
        PolicyMode.UNITY_TEST,
    )
    previous_zero_maneuver = load_runtime_maneuver_profile(
        previous_zero_checkpoint,
        PolicyMode.UNITY_TEST,
    )
    previous_zero_context = build_live_route_context(
        route,
        pose,
        session_id="previous-zero-candidate-load",
        safety_profile=previous_zero_safety,
        maneuver_profile=previous_zero_maneuver,
    )
    previous_zero_policy = load_tested_candidate_policy(
        previous_zero_checkpoint,
        previous_zero_context,
    )

    assert context.compiled_map.snapshot.required_clearance == 0.0
    assert context.maneuver_profile.approach_throttle_cap == 0.1
    assert context.maneuver_profile.approach_rudder_cap == 0.1
    assert context.maneuver_profile.turn_control == Control(0.1, 0.12)
    assert policy.observation_dim == 162
    assert not decision.stop
    assert decision.control is not None
    assert decision.control.throttle <= 0.1
    assert core.trajectory is not None
    assert max(
        control.throttle for control in core.trajectory.controls
    ) <= 0.1
    assert max(
        abs(control.rudder) for control in core.trajectory.controls
    ) <= 0.1
    assert baseline_context.compiled_map.snapshot.required_clearance == 0.2
    assert baseline_policy.observation_dim == 162
    assert previous_zero_context.compiled_map.snapshot.required_clearance == 0.0
    assert previous_zero_context.maneuver_profile.turn_control == Control(
        0.4,
        0.2,
    )
    assert previous_zero_policy.observation_dim == 162
    previous_slow_profile = load_runtime_maneuver_profile(
        previous_slow_checkpoint,
        PolicyMode.UNITY_TEST,
    )
    assert previous_slow_profile.approach_rudder_cap == 1.0
    assert previous_slow_profile.turn_control == Control(0.1, 0.15)


def test_runtime_core_plans_then_emits_only_fresh_safe_control():
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-core-test",
    )
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=16,
        seed=31,
    )

    core = FixedMapControllerCore(context, policy)
    assert core.supervisor.prediction_horizon_s == 2.0
    hold = core.step(_sample(context))
    decision = core.step(_sample(context))

    assert hold.stop
    assert hold.reason == "PLANNING_HOLD"
    assert not decision.stop
    assert decision.mission_index == 1
    assert decision.action is not None
    assert any(decision.safe_mask)
    assert decision.replanned

    running_core = FixedMapControllerCore(context, policy)
    running_core.step(_sample(context, task_status=2))
    running = running_core.step(_sample(context, task_status=2))
    assert not running.stop

    inactive_core = FixedMapControllerCore(context, policy)
    inactive = inactive_core.step(_sample(context, task_status=0))
    assert inactive.stop
    assert inactive.reason == "TASK_INACTIVE"

    stale_core = FixedMapControllerCore(context, policy)
    stale = stale_core.step(_sample(context, scan_age_s=1.1))
    assert stale.stop
    assert stale.reason == "SCAN_STALE"

    laser_core = FixedMapControllerCore(context, policy)
    laser = laser_core.step(
        _sample(
            context,
            laser_ranges=(0.5,) + (20.0,) * 71,
            laser_valid_mask=(True,) + (False,) * 71,
        )
    )
    assert laser.stop
    assert laser.reason == "LASER_EMERGENCY_STOP"


def test_runtime_preserves_point_four_composite_until_safe_handoff():
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-point-four-composite",
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(observation_dim=162, hidden_dim=16, seed=31),
    )
    core.mission_index = 3
    retained_trajectory = SimpleNamespace(trajectory_id="point-four-composite")
    core.trajectory = retained_trajectory
    x, y = fixed_route_goal_xy(context.compiled_map.manifest, 3)

    assert not core._advance_reached_goals(
        VesselState(x=x, y=y, yaw=2.0, speed=0.2, yaw_rate=0.0)
    )
    assert core.mission_index == 4
    assert core.maneuver_phase == "CLEARANCE_PENDING"
    assert core.trajectory is retained_trajectory

    core._complete_composite_if_reached(
        VesselState(
            x=CLEARANCE_HANDOFF_XY[0],
            y=CLEARANCE_HANDOFF_XY[1],
            yaw=2.3,
            speed=0.15,
            yaw_rate=0.0,
        )
    )
    assert core.maneuver_phase == "CLEARANCE_TURN_PENDING"
    assert core.trajectory is None
    assert core.planning_hold_pending

    turn_trajectory = SimpleNamespace(trajectory_id="clearance-turn")
    core.trajectory = turn_trajectory
    point_five = fixed_route_goal_xy(context.compiled_map.manifest, 4)
    assert not core._advance_reached_goals(
        VesselState(
            x=point_five[0],
            y=point_five[1],
            yaw=0.0,
            speed=0.2,
            yaw_rate=0.0,
        )
    )
    assert core.mission_index == 5
    assert core.maneuver_phase == "CLEARANCE_EXIT_PENDING"
    assert core.trajectory is None
    assert core.planning_hold_pending

    point_six = fixed_route_goal_xy(context.compiled_map.manifest, 5)
    assert not core._advance_reached_goals(
        VesselState(
            x=point_six[0],
            y=point_six[1],
            yaw=2.0,
            speed=0.2,
            yaw_rate=0.0,
        )
    )
    assert core.mission_index == 6
    assert core.maneuver_phase == "NORMAL"
    assert core.trajectory is None

def test_runtime_recovers_live_laser_safe_pose_without_relaxing_global_map():
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-clearance-recovery-test",
    )
    state = VesselState(
        x=38.56912943606869,
        y=73.6600965399968,
        yaw=1.102951261381084,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    assert 0.1 < context.compiled_map.snapshot.clearance_at(state) < 0.2
    assert not context.compiled_map.snapshot.is_state_valid(state)
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )

    decision = core.step(
        _sample(
            context,
            vessel_state=state,
            laser_ranges=(1.8,) * 72,
            laser_valid_mask=(True,) * 72,
        )
    )

    assert not decision.stop
    assert decision.reason == "CLEARANCE_RECOVERY"
    assert decision.control == Control(0.1, 0.0)


def test_runtime_does_not_recover_map_invalid_pose_with_emergency_laser():
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-clearance-recovery-laser-test",
    )
    state = VesselState(
        x=38.56912943606869,
        y=73.6600965399968,
        yaw=1.102951261381084,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )

    decision = core.step(
        _sample(
            context,
            vessel_state=state,
            laser_ranges=(0.6,) * 72,
            laser_valid_mask=(True,) * 72,
        )
    )

    assert decision.stop
    assert decision.reason == "MAP_INVALID"


def test_runtime_defers_initial_planning_failure_with_zero_control(monkeypatch):
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-planning-deferred-test",
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )
    monkeypatch.setattr(
        "usvlib4ros.navigation.fixed_map_runtime.plan_fixed_leg",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("forced planner timeout")
        ),
    )

    hold = core.step(_sample(context))
    decision = core.step(_sample(context))

    assert hold.stop
    assert hold.reason == "PLANNING_HOLD"
    assert decision.stop
    assert decision.reason == "PLANNING_DEFERRED"
    assert decision.control is None


def test_runtime_keeps_last_safe_trajectory_when_replan_times_out(
    monkeypatch,
):
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-replan-recovery-test",
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )
    hold = core.step(_sample(context))
    assert hold.reason == "PLANNING_HOLD"
    first = core.step(_sample(context))
    assert not first.stop
    old_trajectory = core.trajectory
    assert old_trajectory is not None
    core.trajectory_index = 0
    reference = old_trajectory.states[0]
    candidates = (
        replace(reference, x=reference.x + 1.0),
        replace(reference, x=reference.x - 1.0),
        replace(reference, y=reference.y + 1.0),
        replace(reference, y=reference.y - 1.0),
    )
    displaced = next(
        state
        for state in candidates
        if context.compiled_map.snapshot.is_state_valid(state)
    )
    monkeypatch.setattr(
        "usvlib4ros.navigation.fixed_map_runtime.plan_fixed_leg",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("forced planner timeout")
        ),
    )

    decision = core.step(_sample(context, vessel_state=displaced))

    assert core.trajectory is None
    assert decision.stop
    assert decision.reason == "REPLANNING_HOLD"

    deferred = core.step(_sample(context, vessel_state=displaced))

    assert deferred.stop
    assert deferred.reason == "PLANNING_DEFERRED"


def test_runtime_brakes_before_planning_after_a_fast_waypoint_crossing():
    route, pose = _live_route_and_pose()
    context = replace(
        build_live_route_context(
            route,
            pose,
            session_id="runtime-transition-brake-test",
        ),
        start_index=5,
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )
    gate = fixed_route_planning_gate(context.compiled_map, 5)
    previous = fixed_route_planning_gate(context.compiled_map, 4)
    fast = VesselState(
        x=gate[0],
        y=gate[1],
        yaw=math.atan2(gate[1] - previous[1], gate[0] - previous[0]),
        speed=0.4,
        yaw_rate=0.0,
        stamp_sim=context.compiled_map.snapshot.stamp_sim,
    )
    core.trajectory = object()

    braking = core.step(_sample(context, vessel_state=fast))

    assert braking.reason == "PLANNING_BRAKE"
    assert braking.control == Control(-0.4, 0.0)
    assert core.mission_index == 6
    assert core.trajectory is None

    stopped = replace(fast, speed=0.1)
    hold = core.step(_sample(context, vessel_state=stopped))

    assert hold.stop
    assert hold.reason == "PLANNING_HOLD"


def test_runtime_uses_short_horizon_for_safe_overspeed_braking():
    route, pose = _live_route_and_pose()
    context = replace(
        build_live_route_context(
            route,
            pose,
            session_id="runtime-short-brake-fallback-test",
        ),
        start_index=1,
    )
    forward = ForwardControlProfile(
        calibration_hash="0" * 64,
        minimum_steerage_throttle=0.1,
        cruise_throttle=0.4,
        action_controls=(
            Control(0.1, -0.5),
            Control(0.4, -0.2),
            Control(0.4, 0.0),
            Control(0.4, 0.2),
            Control(0.4, 0.5),
        ),
        throttle_speed_gain=1.2681317113395243,
        positive_rudder_yaw_rate_gain=1.962635624471142,
        negative_rudder_yaw_rate_gain=2.048615259634089,
    )
    reverse = ReverseControlProfile(
        source_log_sha256="1" * 64,
        command_throttle=-0.4,
        command_signed_speed_mps=-0.12256225167642798,
        reverse_throttle_speed_gain=0.3064056291910699,
        max_reverse_speed_mps=0.2,
    )
    dynamics = enable_reverse_dynamics(
        reduced_dynamics_from_profile(forward),
        reverse,
    )
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=16,
        seed=31,
    )
    policy.forward_control_profile = forward
    policy.reverse_control_profile = reverse
    policy.reduced_dynamics = dynamics
    state = VesselState(
        x=38.422363707454906,
        y=72.75204248409835,
        yaw=1.5879662589748555,
        speed=0.35749474692653355,
        yaw_rate=0.021398576733705245,
        stamp_sim=27.73399999999674,
    )
    future_states = tuple(
        replace(
            state,
            x=state.x,
            y=state.y + 0.2 * index,
            stamp_sim=state.stamp_sim + 0.5 * index,
        )
        for index in range(4)
    )
    core = FixedMapControllerCore(context, policy)
    core.trajectory = SimpleNamespace(
        states=future_states,
        controls=(Control(0.22, 0.0),) * 3,
        durations=(0.5,) * 3,
        times=(0.0, 0.5, 1.0, 1.5),
    )

    decision = core.step(_sample(context, vessel_state=state))

    assert not decision.stop
    assert decision.reason == "OVERSPEED_REVERSE_BRAKE"
    assert decision.control == Control(-0.4, 0.0)


def test_runtime_records_waypoint_only_after_reaching_its_safe_gate():
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="runtime-original-waypoint-test",
    )
    context = replace(context, start_index=5)
    compiled = context.compiled_map
    goal = fixed_route_goal_xy(compiled.manifest, 5)
    gate = fixed_route_planning_gate(compiled, 5)
    candidates = []
    resolution = compiled.snapshot.resolution
    min_x = max(0, int((goal[0] - 0.5) // resolution))
    max_x = min(
        compiled.snapshot.width - 1,
        int((goal[0] + 0.5) // resolution),
    )
    min_y = max(0, int((goal[1] - 0.5) // resolution))
    max_y = min(
        compiled.snapshot.height - 1,
        int((goal[1] + 0.5) // resolution),
    )
    for cell_y in range(min_y, max_y + 1):
        for cell_x in range(min_x, max_x + 1):
            state = VesselState(
                x=(cell_x + 0.5) * compiled.snapshot.resolution,
                y=(cell_y + 0.5) * compiled.snapshot.resolution,
                yaw=0.0,
                speed=0.0,
                yaw_rate=0.0,
                stamp_sim=compiled.snapshot.stamp_sim,
            )
            if (
                math.hypot(state.x - goal[0], state.y - goal[1]) <= 0.5
                and compiled.snapshot.is_state_valid(state)
            ):
                candidates.append(
                    (math.hypot(state.x - gate[0], state.y - gate[1]), state)
                )
    gate_distance, reached_state = max(candidates, key=lambda item: item[0])
    assert gate_distance > 0.2

    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )
    assert not core._advance_reached_goals(reached_state)
    assert core.mission_index == 5

    gate_state = replace(
        reached_state,
        x=gate[0],
        y=gate[1],
    )
    assert math.hypot(
        gate_state.x - goal[0],
        gate_state.y - goal[1],
    ) <= 0.5
    assert not core._advance_reached_goals(gate_state)
    assert core.mission_index == 6


def test_narrow_waypoint_completion_preserves_composite_trajectory_and_hidden_state():
    route, pose = _live_route_and_pose()
    context = replace(
        build_live_route_context(
            route,
            pose,
            session_id="runtime-narrow-escape-test",
        ),
        start_index=NARROW_ROUTE_INDEX,
    )
    core = FixedMapControllerCore(
        context,
        RecurrentDiscreteSAC(
            observation_dim=162,
            hidden_dim=16,
            seed=31,
        ),
    )
    original = fixed_route_goal_xy(
        context.compiled_map.manifest,
        NARROW_ROUTE_INDEX,
    )
    gate = fixed_route_planning_gate(
        context.compiled_map,
        NARROW_ROUTE_INDEX,
    )
    reached = VesselState(
        x=gate[0],
        y=gate[1],
        yaw=0.0,
        speed=0.3,
        yaw_rate=0.0,
    )
    assert math.hypot(
        reached.x - original[0],
        reached.y - original[1],
    ) <= 0.5
    trajectory = object()
    hidden = object()
    core.trajectory = trajectory
    core.trajectory_index = 7
    core.hidden = hidden
    core.hidden_reset = False

    assert not core._advance_reached_goals(reached)
    assert core.mission_index == NARROW_ROUTE_INDEX + 1
    assert core.maneuver_phase == "ESCAPE_PENDING"
    assert core.trajectory is trajectory
    assert core.trajectory_index == 7
    assert core.hidden is hidden
    assert not core.hidden_reset


def test_runtime_executes_safe_reverse_escape_without_calling_sac():
    route, pose = _live_route_and_pose()
    context = replace(
        build_live_route_context(
            route,
            pose,
            session_id="runtime-reverse-escape-test",
        ),
        start_index=NARROW_ROUTE_INDEX + 1,
    )
    forward = ForwardControlProfile(
        calibration_hash="0" * 64,
        minimum_steerage_throttle=0.1,
        cruise_throttle=0.4,
        action_controls=(
            Control(0.1, -0.5),
            Control(0.4, -0.2),
            Control(0.4, 0.0),
            Control(0.4, 0.2),
            Control(0.4, 0.5),
        ),
        throttle_speed_gain=1.2681317113395243,
        positive_rudder_yaw_rate_gain=1.962635624471142,
        negative_rudder_yaw_rate_gain=2.048615259634089,
    )
    reverse = ReverseControlProfile(
        source_log_sha256="1" * 64,
        command_throttle=-0.4,
        command_signed_speed_mps=-0.12256225167642798,
        reverse_throttle_speed_gain=0.3064056291910699,
        max_reverse_speed_mps=0.2,
    )
    dynamics = enable_reverse_dynamics(
        reduced_dynamics_from_profile(forward),
        reverse,
    )
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=16,
        seed=31,
    )
    policy.forward_control_profile = forward
    policy.reverse_control_profile = reverse
    policy.reduced_dynamics = dynamics
    policy.act = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("SAC must not replace the reverse escape primitive")
    )
    core = FixedMapControllerCore(context, policy)
    state = VesselState(
        x=31.0,
        y=99.5,
        yaw=math.pi,
        speed=-0.1,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    rollout = dynamics.propagate(state, reverse.control, 0.8)
    core.maneuver_phase = "ESCAPE_PENDING"
    core.trajectory = SimpleNamespace(
        states=(state, rollout[-1]),
        controls=(reverse.control,),
        durations=(0.8,),
        times=(0.0, 0.8),
    )
    hidden = object()
    core.hidden = hidden
    core.hidden_reset = False

    decision = core.step(_sample(context, vessel_state=state))

    assert not decision.stop
    assert decision.reason == "REVERSE_ESCAPE_NOMINAL"
    assert decision.control == reverse.control
    assert decision.action == 2
    assert core.hidden is hidden
    assert not core.hidden_reset


def test_runtime_executes_closed_loop_narrow_ingress_without_calling_sac():
    route, pose = _live_route_and_pose()
    context = replace(
        build_live_route_context(
            route,
            pose,
            session_id="runtime-narrow-ingress-test",
        ),
        start_index=NARROW_ROUTE_INDEX,
    )
    forward = ForwardControlProfile(
        calibration_hash="0" * 64,
        minimum_steerage_throttle=0.1,
        cruise_throttle=0.4,
        action_controls=(
            Control(0.1, -0.5),
            Control(0.4, -0.2),
            Control(0.4, 0.0),
            Control(0.4, 0.2),
            Control(0.4, 0.5),
        ),
        throttle_speed_gain=1.2681317113395243,
        positive_rudder_yaw_rate_gain=1.962635624471142,
        negative_rudder_yaw_rate_gain=2.048615259634089,
    )
    reverse = ReverseControlProfile(
        source_log_sha256="1" * 64,
        command_throttle=-0.4,
        command_signed_speed_mps=-0.12256225167642798,
        reverse_throttle_speed_gain=0.3064056291910699,
        max_reverse_speed_mps=0.2,
    )
    dynamics = enable_reverse_dynamics(
        reduced_dynamics_from_profile(forward),
        reverse,
    )
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=16,
        seed=31,
    )
    policy.forward_control_profile = forward
    policy.reverse_control_profile = reverse
    policy.reduced_dynamics = dynamics
    policy.act = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("SAC must not replace the narrow ingress controller")
    )
    core = FixedMapControllerCore(context, policy)
    previous = fixed_route_goal_xy(
        context.compiled_map.manifest,
        NARROW_ROUTE_INDEX - 1,
    )
    gate = fixed_route_planning_gate(
        context.compiled_map,
        NARROW_ROUTE_INDEX,
    )
    state = VesselState(
        x=previous[0],
        y=previous[1],
        yaw=math.atan2(gate[1] - previous[1], gate[0] - previous[0]),
        speed=0.3,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    core.trajectory = plan_fixed_leg(
        context.compiled_map,
        start_state=state,
        mission_index=NARROW_ROUTE_INDEX,
        dynamics=dynamics,
        forward_action_controls=(*forward.action_controls, reverse.control),
        seed=71,
        _allow_retry=False,
    )

    decision = core.step(_sample(context, vessel_state=state))

    assert not decision.stop
    assert decision.reason == "NARROW_INGRESS_NOMINAL"
    assert decision.action is not None


def test_runtime_replans_finished_reverse_branch_into_south_bypass():
    route, pose = _live_route_and_pose()
    context = replace(
        build_live_route_context(
            route,
            pose,
            session_id="runtime-post-narrow-egress-test",
        ),
        start_index=NARROW_ROUTE_INDEX + 1,
    )
    forward = ForwardControlProfile(
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
    reverse = ReverseControlProfile(
        source_log_sha256="1" * 64,
        command_throttle=-0.4,
        command_signed_speed_mps=-0.12256225167642798,
        reverse_throttle_speed_gain=0.3064056291910699,
        max_reverse_speed_mps=0.2,
    )
    dynamics = enable_reverse_dynamics(
        reduced_dynamics_from_profile(forward),
        reverse,
    )
    policy = RecurrentDiscreteSAC(
        observation_dim=162,
        hidden_dim=16,
        seed=31,
    )
    policy.forward_control_profile = forward
    policy.reverse_control_profile = reverse
    policy.reduced_dynamics = dynamics
    policy.act = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("SAC must not replace the fixed-map egress")
    )
    core = FixedMapControllerCore(context, policy)
    state = VesselState(
        x=NARROW_ESCAPE_XY[0],
        y=NARROW_ESCAPE_XY[1],
        yaw=math.pi,
        speed=-0.12256225167642798,
        yaw_rate=0.0,
        stamp_sim=0.0,
    )
    rollout = dynamics.propagate(state, reverse.control, 0.4)
    core.maneuver_phase = "ESCAPE_PENDING"
    core.trajectory = SimpleNamespace(
        trajectory_id="completed-reverse-branch",
        states=(state, rollout[-1]),
        controls=(reverse.control,),
        durations=(0.4,),
        times=(0.0, 0.4),
    )

    decision = core.step(_sample(context, vessel_state=state))

    assert not decision.stop
    assert decision.replanned
    assert decision.reason == "REVERSE_ESCAPE_NOMINAL"
    assert is_narrow_egress_trajectory(core.trajectory)


def test_live_policy_loader_rejects_checkpoint_without_evaluation(
    tmp_path,
):
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="checkpoint-gate-test",
    )
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    checkpoint.with_suffix(".pt.json").write_text(
        '{"schema_version":"national-test-sac-checkpoint-v4",'
        '"offline_ready":false,'
        '"live_ready":false}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="offline and Unity"):
        load_live_ready_policy(checkpoint, context)


def test_live_policy_loader_rejects_checkpoint_from_other_dynamics(tmp_path):
    route, pose = _live_route_and_pose()
    context = build_live_route_context(
        route,
        pose,
        session_id="checkpoint-dynamics-gate-test",
    )
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    compiled = context.compiled_map
    checkpoint.with_suffix(".pt.json").write_text(
        json.dumps(
            {
                "schema_version": "national-test-sac-checkpoint-v4",
                "offline_ready": True,
                "live_ready": True,
                "unity_validation_log_hashes": ["a", "b", "c"],
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "route_id": compiled.manifest.route_id,
                "map_source_artifact_hash": (
                    compiled.snapshot.source_artifact_hash
                ),
                "map_payload_hash": compiled.snapshot.payload_content_hash,
                "observation_schema": "local-observation-v2-reduced",
                "observation_dim": 162,
                "action_schema": "five-discrete-forward-bias-v2",
                "action_dim": 5,
                "dynamics_version": "obsolete-turn-in-place-model",
                "route_guidance_version": (
                    "national-test-reversible-composite-v16"
                ),
                "route_guidance_hash": fixed_route_guidance_hash(
                    compiled
                ),
                "geometry_version": (
                    compiled.snapshot.geometry_version
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dynamics_version"):
        load_live_ready_policy(checkpoint, context)
