"""End-to-end state transition tests for the planning-free runtime."""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

from usvlib4ros.navigation.fixed_map_runtime import (
    FixedMapControllerCore,
    RuntimeInput,
    build_fixed_route_context,
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


def test_runtime_core_contains_no_online_planner_or_planning_state():
    source = inspect.getsource(FixedMapControllerCore)

    assert "plan_fixed_leg" not in source
    assert "plan_clearance" not in source
    assert "trajectory" not in source.lower()
    assert "PLANNING_" not in source


def test_thirteen_waypoints_advance_without_stop_or_gru_reset():
    context, policy, core = _core()
    decisions = []

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
