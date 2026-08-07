"""End-to-end state transition tests for the planning-free runtime."""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

from usvlib4ros.navigation.fixed_map_runtime import (
    FixedMapControllerCore,
    LiveInputAdapter,
    MOTION_STALL_CYCLES,
    RuntimeInput,
    build_fixed_route_context,
    laser_emergency_distance_m,
)
from usvlib4ros.navigation.waypoint_control import ACTION_SCHEMA_V3
from usvlib4ros.planning import Control, VesselState
from usvlib4ros.planning.forward_control_profile import (
    ForwardControlProfile,
    reduced_dynamics_from_profile,
)
from usvlib4ros.policy.recurrent_sac import LocalWaypointObservationV3


class _RecordingPolicy:
    action_schema = ACTION_SCHEMA_V3

    def __init__(self) -> None:
        self.forward_control_profile = ForwardControlProfile(
            calibration_hash="2" * 64,
            minimum_steerage_throttle=0.1,
            cruise_throttle=0.4,
            action_controls=(
                Control(0.1, -0.1),
                Control(0.1, -0.05),
                Control(0.4, 0.0),
                Control(0.1, 0.05),
                Control(0.1, 0.1),
            ),
            action_schema=ACTION_SCHEMA_V3,
        )
        self.reduced_dynamics = reduced_dynamics_from_profile(
            self.forward_control_profile
        )
        self.hidden_inputs = []
        self.observations = []

    def act(self, observation, safe_action_mask, *, hidden, deterministic):
        assert isinstance(observation, LocalWaypointObservationV3)
        assert observation.safe_action_mask == tuple(safe_action_mask)
        self.observations.append(observation)
        self.hidden_inputs.append(hidden)
        next_hidden = object()
        return SimpleNamespace(action=2), next_hidden


class _PermissiveSupervisor:
    def precheck(self, state, candidates, map_snapshot, dynamics, **kwargs):
        return (True,) * 5, ("SAFE",) * 5, (1.0,) * 5

    def finalize(
        self,
        *,
        policy_action,
        candidate_mask,
        candidates,
        **kwargs,
    ):
        return SimpleNamespace(
            policy_action=policy_action,
            final_action=policy_action,
            control=candidates[policy_action].control,
            candidate_mask=tuple(candidate_mask),
            reasons=("SAFE",) * 5,
            reason="POLICY_ACTION_SAFE",
            stop=False,
            overridden=False,
        )


class _NoSafeSupervisor(_PermissiveSupervisor):
    def precheck(self, state, candidates, map_snapshot, dynamics, **kwargs):
        return (False,) * 5, ("MOTION_COLLISION",) * 5, (0.0,) * 5


def _context():
    return build_fixed_route_context(
        session_id="direct-runtime-fixture",
    )


def _state(context, index: int, *, stamp: float = 0.0) -> VesselState:
    x, y = context.corridor.task_anchors[index]
    next_index = min(index + 1, 12)
    next_x, next_y = context.corridor.task_anchors[next_index]
    return VesselState(
        x=x,
        y=y,
        yaw=math.atan2(next_y - y, next_x - x) if index < 12 else 0.0,
        speed=0.25,
        yaw_rate=0.05,
        throttle_state=0.4,
        rudder_state=0.0,
        stamp_sim=stamp,
    )


def _sample(state: VesselState, **changes) -> RuntimeInput:
    values = {
        "vessel_state": state,
        "laser_ranges": (20.0,) * 72,
        "laser_valid_mask": (True,) * 72,
        "pose_age_s": 0.0,
        "scan_age_s": 0.0,
        "device_age_s": 0.0,
        "work_model": 2,
        "task_status": 1,
    }
    values.update(changes)
    return RuntimeInput(**values)


def _core(supervisor=None):
    context = _context()
    policy = _RecordingPolicy()
    core = FixedMapControllerCore(context, policy)
    core.supervisor = supervisor or _PermissiveSupervisor()
    return context, policy, core


def test_live_input_adapter_normalizes_only_negative_numerical_zero_speed():
    context = _context()
    pose = SimpleNamespace(
        lat=context.projector.origin_lat_deg,
        lng=context.projector.origin_lon_deg,
        yaw=0.0,
        speed=-4.264008204077991e-11,
        rotate_speed=0.0,
    )
    data = SimpleNamespace(
        scada_data=SimpleNamespace(pose=pose),
        laser_data=SimpleNamespace(ranges=(20.0,) * 72),
        device_data=SimpleNamespace(
            throttle_percent=0.0,
            rudder_percent=0.0,
            work_model=2,
            task_status=2,
        ),
    )
    adapter = LiveInputAdapter(data, context)
    dynamics = _RecordingPolicy().reduced_dynamics

    numerical_zero = adapter.build().vessel_state

    assert numerical_zero.speed == 0.0
    assert dynamics.is_state_valid(numerical_zero)

    pose.speed = -0.01
    reverse = adapter.build().vessel_state

    assert reverse.speed == -0.01
    assert not dynamics.is_state_valid(reverse)


def test_runtime_core_contains_no_online_planner_or_planning_state():
    source = inspect.getsource(FixedMapControllerCore)

    assert "plan_fixed_leg" not in source
    assert "plan_clearance" not in source
    assert "trajectory" not in source.lower()
    assert "PLANNING_" not in source


def test_thirteen_waypoints_advance_without_stop_or_gru_reset():
    context, policy, core = _core()
    decisions = []

    assert math.isclose(
        laser_emergency_distance_m(
            context.compiled_map.snapshot,
            _state(context, 9),
        ),
        0.0,
    )
    assert math.isclose(
        laser_emergency_distance_m(
            context.compiled_map.snapshot,
            _state(context, 10),
        ),
        0.0,
    )

    for index in range(13):
        decisions.append(core.step(_sample(_state(context, index, stamp=index * 0.1))))

    assert decisions[-1].completed
    assert decisions[-1].mission_index == 13
    assert decisions[-1].control is None
    assert all("PLANNING" not in decision.reason for decision in decisions)
    assert all(decision.control is not None for decision in decisions[:-1])
    assert len(policy.hidden_inputs) == 12
    assert policy.hidden_inputs[0] is None
    assert all(hidden is not None for hidden in policy.hidden_inputs[1:])
    assert policy.observations[0].hidden_reset
    assert all(not observation.hidden_reset for observation in policy.observations[1:])


def test_runtime_uses_exact_calibrated_candidates_without_point_clipping():
    context, policy, core = _core()
    core.mission_index = 3

    decision = core.step(_sample(_state(context, 2)))

    assert tuple(candidate.control for candidate in core.candidates) == (
        policy.forward_control_profile.action_controls
    )
    percent_commands = {
        (
            int(round(candidate.control.throttle * 100.0)),
            int(round(candidate.control.rudder * 100.0)),
        )
        for candidate in core.candidates
    }
    assert len(percent_commands) == 5
    assert decision.control == policy.forward_control_profile.action_controls[2]
    assert decision.policy_action == 2
    assert decision.action == 2
    assert not decision.safety_intervened


def test_named_false_collision_zones_allow_one_map_invalid_incident(monkeypatch):
    context = _context()
    snapshot_type = type(context.compiled_map.snapshot)
    original_is_state_valid = snapshot_type.is_state_valid
    forced_invalid = {"value": False}

    def is_state_valid(snapshot, state):
        if forced_invalid["value"]:
            return False
        return original_is_state_valid(snapshot, state)

    monkeypatch.setattr(snapshot_type, "is_state_valid", is_state_valid)

    for mission_index, state_index, zone in (
        (2, 2, "P2_P3"),
        (4, 4, "P4_P5"),
        (7, 7, "P7_P8"),
        (9, 9, "P9_P10"),
    ):
        context, _, core = _core()
        core.mission_index = mission_index
        state = _state(context, state_index)
        assert core.step(_sample(state)).control is not None

        forced_invalid["value"] = True
        grace = [core.step(_sample(state)) for _ in range(30)]
        assert all(
            decision.reason == f"MAP_INVALID_GRACE_{zone}"
            for decision in grace
        )
        assert all(decision.control is not None for decision in grace)
        exhausted = core.step(_sample(state))
        assert exhausted.reason == "MAP_INVALID"
        assert exhausted.control is None

        forced_invalid["value"] = False
        assert core.step(_sample(state)).control is not None
        forced_invalid["value"] = True
        repeated = core.step(_sample(state))
        assert repeated.reason == "MAP_INVALID"
        assert repeated.control is None
        forced_invalid["value"] = False

    context, _, outside = _core()
    outside.mission_index = 6
    outside_state = _state(context, 5)
    assert outside.step(_sample(outside_state)).control is not None
    forced_invalid["value"] = True
    assert outside.step(_sample(outside_state)).reason == "MAP_INVALID"

    forced_invalid["value"] = False
    context, _, laser_core = _core()
    laser_core.mission_index = 3
    laser_state = _state(context, 2)
    assert laser_core.step(_sample(laser_state)).control is not None
    forced_invalid["value"] = True
    laser = laser_core.step(
        _sample(
            laser_state,
            laser_ranges=(0.0,) + (20.0,) * 71,
        )
    )
    assert laser.reason == "LASER_EMERGENCY_STOP"
    assert laser.control is None


def test_point_eleven_map_invalid_grace_is_six_seconds_and_three_incidents(
    monkeypatch,
):
    context = _context()
    snapshot_type = type(context.compiled_map.snapshot)
    original_is_state_valid = snapshot_type.is_state_valid
    forced_invalid = {"value": False}

    def is_state_valid(snapshot, state):
        if forced_invalid["value"]:
            return False
        return original_is_state_valid(snapshot, state)

    monkeypatch.setattr(snapshot_type, "is_state_valid", is_state_valid)

    context, _, point_eleven = _core()
    point_eleven.mission_index = 11
    point_eleven_state = _state(context, 10)

    for _ in range(3):
        forced_invalid["value"] = False
        assert point_eleven.step(_sample(point_eleven_state)).control is not None
        forced_invalid["value"] = True
        grace = [
            point_eleven.step(_sample(point_eleven_state))
            for _ in range(60)
        ]
        assert all(
            decision.reason == "MAP_INVALID_GRACE_P11"
            for decision in grace
        )
        assert all(decision.control is not None for decision in grace)
        exhausted = point_eleven.step(_sample(point_eleven_state))
        assert exhausted.reason == "MAP_INVALID"
        assert exhausted.control is None

    forced_invalid["value"] = False
    assert point_eleven.step(_sample(point_eleven_state)).control is not None
    forced_invalid["value"] = True
    repeated = point_eleven.step(_sample(point_eleven_state))
    assert repeated.reason == "MAP_INVALID"
    assert repeated.control is None


def test_high_speed_turn_coasts_without_changing_the_policy_action():
    context, _, core = _core()
    core.mission_index = 3
    state = _state(context, 2)
    fast_turn = VesselState(
        x=state.x,
        y=state.y,
        yaw=state.yaw + 0.4,
        speed=0.85,
        yaw_rate=state.yaw_rate,
        throttle_state=state.throttle_state,
        rudder_state=state.rudder_state,
        stamp_sim=state.stamp_sim,
    )

    decision = core.step(_sample(fast_turn))

    assert decision.reason == "SPEED_GOVERNOR"
    assert decision.policy_action == 2
    assert decision.action == 2
    assert decision.control == Control(throttle=0.0, rudder=0.0)
    assert decision.training_trace.final_control == decision.control
    assert decision.safety_intervened


def test_unity_adapt_hands_off_to_rrt_only_after_confirmed_point_eleven(
    monkeypatch,
):
    context = _context()
    policy = _RecordingPolicy()
    guided_missions = []

    def guided_action(mission_index, *args, **kwargs):
        del args
        guided_missions.append(mission_index)
        return 1, kwargs["start_index"]

    core = FixedMapControllerCore(
        context,
        policy,
        corridor_guidance=True,
    )
    core.supervisor = _PermissiveSupervisor()
    core.route_guide = SimpleNamespace(suffix_action=guided_action)
    core.mission_index = 3
    state = _state(context, 2)
    misaligned = VesselState(
        x=state.x,
        y=state.y,
        yaw=state.yaw + 0.4,
        speed=0.25,
        yaw_rate=state.yaw_rate,
        throttle_state=state.throttle_state,
        rudder_state=state.rudder_state,
        stamp_sim=state.stamp_sim,
    )

    decision = core.step(_sample(misaligned))

    assert decision.reason == "POLICY_ACTION_SAFE"
    assert decision.policy_action == 2
    assert decision.action == decision.policy_action
    assert guided_missions == []

    handoff_core = FixedMapControllerCore(
        context,
        _RecordingPolicy(),
        corridor_guidance=True,
    )
    handoff_core.supervisor = _PermissiveSupervisor()
    handoff_core.route_guide = SimpleNamespace(suffix_action=guided_action)
    handoff_core.mission_index = 10
    start = context.corridor.task_anchors[9]
    point_eleven = context.corridor.task_anchors[10]
    leg_length = math.dist(start, point_eleven)
    ratio = (leg_length - 1.0) / leg_length
    early = VesselState(
        x=start[0] + (point_eleven[0] - start[0]) * ratio,
        y=start[1] + (point_eleven[1] - start[1]) * ratio,
        yaw=math.atan2(
            point_eleven[1] - start[1],
            point_eleven[0] - start[0],
        ),
        speed=0.25,
        yaw_rate=0.0,
        throttle_state=0.4,
        rudder_state=0.0,
    )

    assert math.dist(
        (early.x, early.y),
        context.corridor.task_points[10],
    ) < 2.0
    before_handoff = handoff_core.step(_sample(early))
    assert before_handoff.mission_index == 10
    assert before_handoff.action == before_handoff.policy_action == 2
    assert guided_missions == []

    after_handoff = handoff_core.step(_sample(_state(context, 10)))
    assert after_handoff.mission_index == 11
    assert after_handoff.policy_action == 2
    assert after_handoff.action == 1
    assert after_handoff.reason == "ROUTE_TRAINING_GUIDANCE_OVERRIDE"
    assert guided_missions == [11]

    exit_core = FixedMapControllerCore(
        context,
        _RecordingPolicy(),
        corridor_guidance=True,
    )
    exit_core.supervisor = _PermissiveSupervisor()
    exit_core.mission_index = 11
    suffix = exit_core.route_guide.suffix_plans[0]
    route = suffix["route_xy"]

    before_exit_x, before_exit_y = route[-20]
    before_exit_state = VesselState(
        x=before_exit_x,
        y=before_exit_y,
        yaw=float(suffix["terminal_state"]["yaw"]),
        speed=0.2,
        yaw_rate=0.0,
        throttle_state=0.1,
        rudder_state=0.0,
    )
    assert exit_core.step(_sample(before_exit_state)).control is not None

    exit_x, exit_y = route[-1]
    exit_state = VesselState(
        x=exit_x,
        y=exit_y,
        yaw=float(suffix["terminal_state"]["yaw"]),
        speed=0.2,
        yaw_rate=0.0,
        throttle_state=0.1,
        rudder_state=0.0,
    )
    assert math.dist(
        (exit_x, exit_y),
        context.corridor.task_points[10],
    ) > 1.0
    assert exit_core._map_invalid_grace_zone(exit_state) == "P11"

    finish_core = FixedMapControllerCore(
        context,
        _RecordingPolicy(),
        corridor_guidance=True,
    )
    finish_core.supervisor = _PermissiveSupervisor()
    finish_core.mission_index = 12
    point_twelve = context.corridor.task_points[11]
    finish = context.corridor.task_points[12]
    final_leg_length = math.dist(point_twelve, finish)
    unit_x = (finish[0] - point_twelve[0]) / final_leg_length
    unit_y = (finish[1] - point_twelve[1]) / final_leg_length
    before_finish = VesselState(
        x=finish[0] - unit_x * 0.55,
        y=finish[1] - unit_y * 0.55,
        yaw=math.atan2(unit_y, unit_x),
        speed=0.2,
        yaw_rate=0.0,
        throttle_state=0.1,
        rudder_state=0.0,
        stamp_sim=0.0,
    )
    not_finished = finish_core.step(_sample(before_finish))
    assert not_finished.mission_index == 12
    assert not not_finished.completed

    after_finish = VesselState(
        x=finish[0] + unit_x * 0.55,
        y=finish[1] + unit_y * 0.55,
        yaw=before_finish.yaw,
        speed=0.2,
        yaw_rate=0.0,
        throttle_state=0.1,
        rudder_state=0.0,
        stamp_sim=0.1,
    )
    finished = finish_core.step(_sample(after_finish))
    assert finished.mission_index == 13
    assert finished.completed
    assert finished.reason == "MISSION_COMPLETE"

    snapshot_type = type(context.compiled_map.snapshot)
    monkeypatch.setattr(
        snapshot_type,
        "is_state_valid",
        lambda snapshot, state: False,
    )
    continued = exit_core.step(_sample(exit_state))

    assert continued.mission_index == 12
    assert continued.reason == "MAP_INVALID_GRACE_P11"
    assert continued.control is not None


def test_tenth_fresh_no_safe_decision_truncates_immediately():
    context, policy, core = _core(_NoSafeSupervisor())
    x, y = context.corridor.polyline[0]
    state = VesselState(x=x, y=y, yaw=math.pi / 2, speed=0.1, yaw_rate=0.0)

    for _ in range(9):
        decision = core.step(_sample(state))
        assert decision.reason == "NO_SAFE_ACTION"
        assert not decision.safety_truncated
        assert decision.control is None
    decision = core.step(_sample(state))

    assert decision.reason == "NO_SAFE_ACTION_TRUNCATED"
    assert decision.safety_truncated
    assert decision.control is None
    assert policy.observations == []


def test_stale_input_stops_but_resets_no_safe_failure_window():
    context, _, core = _core(_NoSafeSupervisor())
    x, y = context.corridor.polyline[0]
    state = VesselState(x=x, y=y, yaw=math.pi / 2, speed=0.1, yaw_rate=0.0)

    for _ in range(9):
        assert not core.step(_sample(state)).safety_truncated
    stale = core.step(_sample(state, scan_age_s=2.0))
    assert stale.reason == "SCAN_STALE"
    for _ in range(9):
        assert not core.step(_sample(state)).safety_truncated
    assert core.step(_sample(state)).safety_truncated


def test_commanded_throttle_with_no_motion_truncates_as_stalled():
    context, _, core = _core()
    x, y = context.corridor.polyline[0]
    state = VesselState(x=x, y=y, yaw=math.pi / 2, speed=0.0, yaw_rate=0.0)

    assert core.step(_sample(state)).control is not None
    for _ in range(MOTION_STALL_CYCLES - 1):
        assert core.step(_sample(state)).reason == "POLICY_ACTION_SAFE"
    stalled = core.step(_sample(state))

    assert stalled.reason == "MOTION_STALLED"
    assert stalled.safety_truncated
    assert stalled.control is None
