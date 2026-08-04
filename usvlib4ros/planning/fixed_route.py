"""Fixed National_Test route compilation and kinodynamic planning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Optional

from usvlib4ros.mapping import (
    CompiledSidecarMap,
    SidecarCompilerConfig,
    compile_beihu_sidecar,
    load_sidecar_artifact,
)

from .kinodynamic_informed_rrtstar import (
    Control,
    CostConfig,
    GoalRegion,
    KinodynamicInformedRRTStarPlanner,
    PlannerConfig,
    PlanningRequest,
    PrototypeReducedDynamics,
    Trajectory,
    TrajectoryValidator,
    VesselState,
)
from .forward_control_profile import (
    diagnostic_forward_control_profile,
    reduced_dynamics_from_profile,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "mapping" / "data"
SIDECAR_PATH = DATA_DIR / "beihu_static_world_sidecar.json"
LIVE_PROFILE_PATH = DATA_DIR / "national_test_live_profile.json"
FIXED_ROUTE_TOLERANCE_M = 0.5
MAX_FIXED_ROUTE_TOLERANCE_M = FIXED_ROUTE_TOLERANCE_M
DISPLACED_GATE_TOLERANCE_M = 0.3
SAFE_GATE_CLEARANCE_M = 0.3
ORDINARY_PLAN_CLEARANCE_BY_INDEX = {2: 0.3, 3: 0.2, 11: 0.3}
ROUTE_GUIDANCE_VERSION = "national-test-reversible-composite-v33"
CLEARANCE_COMPOSITE_ROUTE_INDEX = 3
CLEARANCE_APPROACH_GATE = (40.2, 78.8, 0.5)
CLEARANCE_HANDOFF_XY = (39.0, 82.9)
CLEARANCE_HANDOFF_TOLERANCE_M = 0.6
CLEARANCE_TURN_CONTROL = Control(0.4, 0.2)
CLEARANCE_TURN_BRAKE_CONTROL = Control(-0.4, 0.0)
CLEARANCE_TURN_EDGE_DURATION_S = 0.2
CLEARANCE_EXIT_TURN_CONTROL = Control(0.4, -0.2)
CLEARANCE_EXIT_TAIL_CONTROLS = (
    Control(0.1, 0.2),
    Control(0.1, 0.1),
    Control(0.4, 0.2),
    Control(0.4, 0.0),
)
CLEARANCE_EXIT_EDGE_DURATION_S = 0.2
NARROW_ROUTE_INDEX = 10
NARROW_ESCAPE_XY = (31.6, 99.5)
NARROW_ESCAPE_TOLERANCE_M = 0.3
NARROW_ESCAPE_RELEASE_X_M = 33.1
NARROW_EGRESS_HANDOFF_Y_M = 94.0
NARROW_EGRESS_HEADING_TOLERANCE_RAD = 0.3
NARROW_EGRESS_SOUTH_TURN_Y_M = 95.2
NARROW_EGRESS_CONTROL = Control(0.4, 0.2)
NARROW_EGRESS_STRAIGHT_CONTROL = Control(0.4, 0.0)
NARROW_EGRESS_TERMINAL_CONTROL = Control(0.4, 0.05)
NARROW_EGRESS_RECOVERY_CONTROL = Control(0.4, 0.0)
NARROW_EGRESS_REVERSE_CONTROL = Control(-0.4, 0.0)
NARROW_EGRESS_EDGE_DURATION_S = 0.4
TERMINAL_ROUTE_INDEX = 12
TERMINAL_TURN_CONTROLS = (
    Control(0.4, -0.3),
    Control(0.4, -0.4),
    Control(0.4, -0.5),
)
TERMINAL_TAIL_CONTROLS = (Control(0.4, 0.0), Control(0.1, 0.0))
TERMINAL_EDGE_DURATION_S = 0.2
ORDINARY_CLEARANCE_VISIT_REGIONS: dict[
    int,
    tuple[tuple[float, float, float], ...],
] = {
    4: ((39.0, 82.9, 0.6),),
    9: ((34.8, 98.6, 0.5),),
    11: ((29.5, 94.0, 0.6),),
}
ORDINARY_VISIT_MAX_START_Y = {11: 94.5}
_ROUTE_GATE_CACHE: dict[tuple[str, int], tuple[float, float]] = {}


def _validate_route_index(point_count: int, mission_index: int) -> None:
    """Validate a published fixed-route index."""

    if point_count <= 0 or not 0 <= mission_index < point_count:
        raise ValueError("fixed route tolerance index is invalid")


def fixed_route_goal_xy(manifest, mission_index: int) -> tuple[float, float]:
    """Return one unchanged published National_Test waypoint."""

    point_count = len(manifest.route_points_enu)
    _validate_route_index(point_count, mission_index)
    point = manifest.route_points_enu[mission_index]
    x = point[0] - manifest.origin_enu[0]
    y = point[1] - manifest.origin_enu[1]
    return x, y


def narrow_escape_released(
    compiled_map: CompiledSidecarMap,
    state: VesselState,
) -> bool:
    """Release ordinary navigation south of the obstacle, heading west."""

    snapshot = compiled_map.snapshot
    west_heading_error = abs(
        (state.yaw - math.pi + math.pi) % (2.0 * math.pi)
        - math.pi
    )
    return (
        state.y <= NARROW_EGRESS_HANDOFF_Y_M
        and west_heading_error <= NARROW_EGRESS_HEADING_TOLERANCE_RAD
        and state.speed >= 0.05
        and snapshot.is_state_valid(state)
        and snapshot.clearance_at(state)
        >= snapshot.required_clearance
    )


def clearance_handoff_reached(
    compiled_map: CompiledSidecarMap,
    state: VesselState,
) -> bool:
    """Whether the point-four composite reached its west safety gate."""

    return (
        math.hypot(
            state.x - CLEARANCE_HANDOFF_XY[0],
            state.y - CLEARANCE_HANDOFF_XY[1],
        )
        <= CLEARANCE_HANDOFF_TOLERANCE_M + 1e-9
        and compiled_map.snapshot.is_state_valid(state)
    )


def clearance_approach_reached(state: VesselState) -> bool:
    """Whether point four's non-scoring approach gate was visited."""

    return (
        math.hypot(
            state.x - CLEARANCE_APPROACH_GATE[0],
            state.y - CLEARANCE_APPROACH_GATE[1],
        )
        <= CLEARANCE_APPROACH_GATE[2] + 1e-9
    )


def is_narrow_egress_trajectory(trajectory: object) -> bool:
    """Whether a trajectory is the deterministic post-target bypass."""

    return str(getattr(trajectory, "trajectory_id", "")).endswith(
        "-post-narrow-egress"
    )


def is_narrow_composite_trajectory(trajectory: object) -> bool:
    """Whether a trajectory is the deterministic point-eleven composite."""

    return (
        getattr(trajectory, "mission_index", None) == NARROW_ROUTE_INDEX
        and "fixed-route-live-leg-10" in str(
            getattr(trajectory, "request_id", "")
        )
        and not is_narrow_egress_trajectory(trajectory)
    )


def is_terminal_route_trajectory(trajectory: object) -> bool:
    """Whether an open-water post-narrow leg uses nominal time control."""

    mission_index = getattr(trajectory, "mission_index", None)
    return (
        isinstance(mission_index, int)
        and mission_index > NARROW_ROUTE_INDEX
        and "fixed-route-live-leg-" in str(
            getattr(trajectory, "request_id", "")
        )
    )


def is_clearance_composite_trajectory(trajectory: object) -> bool:
    """Whether a trajectory preserves point four through its handoff."""

    return (
        getattr(trajectory, "mission_index", None)
        == CLEARANCE_COMPOSITE_ROUTE_INDEX
        and "fixed-route-live-leg-3" in str(
            getattr(trajectory, "request_id", "")
        )
    )


def is_clearance_turn_trajectory(trajectory: object) -> bool:
    """Whether a trajectory is the fixed west-gate turn to point five."""

    return str(getattr(trajectory, "trajectory_id", "")).endswith(
        "-clearance-turn"
    )


def is_clearance_exit_trajectory(trajectory: object) -> bool:
    """Whether a trajectory is a local replan from point five to six."""

    return str(getattr(trajectory, "trajectory_id", "")).endswith(
        "-clearance-exit"
    )


def plan_clearance_exit(
    compiled_map: CompiledSidecarMap,
    *,
    start_state: VesselState,
    dynamics: PrototypeReducedDynamics,
) -> Trajectory:
    """Search the calibrated local primitives around the point-six buoy."""

    snapshot = compiled_map.snapshot
    if not snapshot.is_state_valid(start_state):
        raise RuntimeError("clearance exit must start from a valid state")

    prefix_controls: list[Control] = []
    prefix_durations: list[float] = []
    prefix_states = [start_state]
    prefix_rollouts: list[tuple[VesselState, ...]] = []
    prefix_minimum = snapshot.clearance_at(start_state)
    candidates = []

    for turn_count in range(1, 56):
        rollout = dynamics.propagate(
            prefix_states[-1],
            CLEARANCE_EXIT_TURN_CONTROL,
            CLEARANCE_EXIT_EDGE_DURATION_S,
        )
        motion = snapshot.check_motion(rollout)
        if not motion.valid:
            break
        prefix_controls.append(CLEARANCE_EXIT_TURN_CONTROL)
        prefix_durations.append(CLEARANCE_EXIT_EDGE_DURATION_S)
        prefix_rollouts.append(rollout)
        prefix_states.append(rollout[-1])
        prefix_minimum = min(prefix_minimum, motion.min_clearance)
        if turn_count < 24:
            continue

        for tail_control in CLEARANCE_EXIT_TAIL_CONTROLS:
            controls = list(prefix_controls)
            durations = list(prefix_durations)
            states = list(prefix_states)
            edge_rollouts = list(prefix_rollouts)
            minimum_clearance = prefix_minimum
            for _ in range(30):
                rollout = dynamics.propagate(
                    states[-1],
                    tail_control,
                    CLEARANCE_EXIT_EDGE_DURATION_S,
                )
                motion = snapshot.check_motion(rollout)
                if not motion.valid:
                    break
                controls.append(tail_control)
                durations.append(CLEARANCE_EXIT_EDGE_DURATION_S)
                edge_rollouts.append(rollout)
                states.append(rollout[-1])
                minimum_clearance = min(
                    minimum_clearance,
                    motion.min_clearance,
                )
                if not fixed_route_waypoint_reached(
                    compiled_map,
                    5,
                    states[-1],
                ):
                    continue
                next_x, next_y = fixed_route_goal_xy(
                    compiled_map.manifest,
                    6,
                )
                desired_heading = math.atan2(
                    next_y - states[-1].y,
                    next_x - states[-1].x,
                )
                heading_error = abs(
                    (states[-1].yaw - desired_heading + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                )
                goal_x, goal_y = fixed_route_goal_xy(
                    compiled_map.manifest,
                    5,
                )
                candidates.append(
                    (
                        minimum_clearance,
                        states[-1].speed,
                        len(controls),
                        heading_error,
                        math.hypot(
                            states[-1].x - goal_x,
                            states[-1].y - goal_y,
                        ),
                        controls,
                        durations,
                        states,
                        edge_rollouts,
                    )
                )
                break

    if not candidates:
        goal_x, goal_y = fixed_route_goal_xy(
            compiled_map.manifest,
            5,
        )
        next_x, next_y = fixed_route_goal_xy(
            compiled_map.manifest,
            6,
        )
        controls: list[Control] = []
        durations: list[float] = []
        states = [start_state]
        edge_rollouts: list[tuple[VesselState, ...]] = []
        minimum_clearance = snapshot.clearance_at(start_state)
        for _ in range(240):
            current = states[-1]
            desired_yaw = math.atan2(
                goal_y - current.y,
                goal_x - current.x,
            )
            yaw_error = (
                desired_yaw - current.yaw + math.pi
            ) % (2.0 * math.pi) - math.pi
            rudder = dynamics.rudder_yaw_sign * (
                1.2 * yaw_error - 0.4 * current.yaw_rate
            )
            control = Control(
                0.1,
                max(-0.5, min(0.5, rudder)),
            )
            rollout = dynamics.propagate(current, control, 0.1)
            motion = snapshot.check_motion(rollout)
            if not motion.valid:
                break
            controls.append(control)
            durations.append(0.1)
            edge_rollouts.append(rollout)
            states.append(rollout[-1])
            minimum_clearance = min(
                minimum_clearance,
                motion.min_clearance,
            )
            if not fixed_route_waypoint_reached(
                compiled_map,
                5,
                states[-1],
            ):
                continue
            desired_heading = math.atan2(
                next_y - states[-1].y,
                next_x - states[-1].x,
            )
            heading_error = abs(
                (states[-1].yaw - desired_heading + math.pi)
                % (2.0 * math.pi)
                - math.pi
            )
            candidates.append(
                (
                    minimum_clearance,
                    states[-1].speed,
                    len(controls),
                    heading_error,
                    math.hypot(
                        states[-1].x - goal_x,
                        states[-1].y - goal_y,
                    ),
                    controls,
                    durations,
                    states,
                    edge_rollouts,
                )
            )
            break

    if not candidates:
        raise RuntimeError("clearance exit has no collision-free primitive chain")
    preferred = [
        candidate
        for candidate in candidates
        if candidate[0] >= 0.25 and candidate[1] <= 0.25
    ]
    pool = preferred or candidates
    selected = min(
        pool,
        key=lambda candidate: (
            candidate[2],
            candidate[3],
            candidate[4],
            -candidate[0],
        ),
    )
    (
        minimum_clearance,
        _,
        _,
        _,
        terminal_error,
        controls,
        durations,
        states,
        edge_rollouts,
    ) = selected
    times = [0.0]
    for duration in durations:
        times.append(times[-1] + duration)
    request_id = "fixed-route-live-leg-5-clearance-exit"
    terminal = states[-1]
    return Trajectory(
        trajectory_id=f"{request_id}-clearance-exit",
        request_id=request_id,
        session_id=snapshot.session_id,
        map_snapshot_id=snapshot.snapshot_id,
        map_source_version=snapshot.source_version,
        map_payload_content_hash=snapshot.payload_content_hash,
        dynamics_version=dynamics.version,
        validator_version=TrajectoryValidator.version,
        frame_id=snapshot.map_frame,
        mission_index=5,
        mission_version=f"route-v{compiled_map.manifest.route_version}",
        map_source_artifact_hash=snapshot.source_artifact_hash,
        map_compiler_config_hash=snapshot.compiler_config_hash,
        state_version=start_state.state_version,
        states=tuple(states),
        controls=tuple(controls),
        durations=tuple(durations),
        times=tuple(times),
        edge_rollouts=tuple(edge_rollouts),
        cost=times[-1],
        min_clearance=minimum_clearance,
        validation_status="VALID",
        terminal_position_error=terminal_error,
        terminal_heading_error=0.0,
        terminal_speed=terminal.speed,
        terminal_yaw_rate=terminal.yaw_rate,
    )


def plan_clearance_turn(
    compiled_map: CompiledSidecarMap,
    *,
    start_state: VesselState,
    dynamics: PrototypeReducedDynamics,
) -> Trajectory:
    """Propagate the calibrated clockwise arc from the west gate."""

    if not clearance_handoff_reached(compiled_map, start_state):
        raise RuntimeError("clearance turn must start in the handoff")
    snapshot = compiled_map.snapshot
    controls: list[Control] = []
    durations: list[float] = []
    states = [start_state]
    edge_rollouts: list[tuple[VesselState, ...]] = []
    minimum_clearance = snapshot.clearance_at(start_state)
    def append(control: Control) -> VesselState:
        nonlocal minimum_clearance
        rollout = dynamics.propagate(
            states[-1],
            control,
            CLEARANCE_TURN_EDGE_DURATION_S,
        )
        motion = snapshot.check_motion(rollout)
        if not motion.valid:
            raise RuntimeError("clearance turn primitive is not collision-free")
        controls.append(control)
        durations.append(CLEARANCE_TURN_EDGE_DURATION_S)
        edge_rollouts.append(rollout)
        states.append(rollout[-1])
        minimum_clearance = min(minimum_clearance, motion.min_clearance)
        return states[-1]

    for _ in range(20):
        if states[-1].speed <= 0.15:
            break
        append(CLEARANCE_TURN_BRAKE_CONTROL)
    else:
        raise RuntimeError("clearance turn braking did not settle")

    for _ in range(80):
        append(CLEARANCE_TURN_CONTROL)
        if fixed_route_waypoint_reached(compiled_map, 4, states[-1]):
            break
    else:
        raise RuntimeError("clearance turn did not reach point five")
    exit_trajectory = plan_clearance_exit(
        compiled_map,
        start_state=states[-1],
        dynamics=dynamics,
    )
    controls.extend(exit_trajectory.controls)
    durations.extend(exit_trajectory.durations)
    states.extend(exit_trajectory.states[1:])
    edge_rollouts.extend(exit_trajectory.edge_rollouts)
    minimum_clearance = min(
        minimum_clearance,
        exit_trajectory.min_clearance,
    )
    times = [0.0]
    for duration in durations:
        times.append(times[-1] + duration)
    request_id = "fixed-route-live-leg-4-clearance-turn"
    terminal = states[-1]
    return Trajectory(
        trajectory_id=f"{request_id}-clearance-turn",
        request_id=request_id,
        session_id=snapshot.session_id,
        map_snapshot_id=snapshot.snapshot_id,
        map_source_version=snapshot.source_version,
        map_payload_content_hash=snapshot.payload_content_hash,
        dynamics_version=dynamics.version,
        validator_version=TrajectoryValidator.version,
        frame_id=snapshot.map_frame,
        mission_index=4,
        mission_version=f"route-v{compiled_map.manifest.route_version}",
        map_source_artifact_hash=snapshot.source_artifact_hash,
        map_compiler_config_hash=snapshot.compiler_config_hash,
        state_version=start_state.state_version,
        states=tuple(states),
        controls=tuple(controls),
        durations=tuple(durations),
        times=tuple(times),
        edge_rollouts=tuple(edge_rollouts),
        cost=times[-1],
        min_clearance=minimum_clearance,
        validation_status="VALID",
        terminal_position_error=0.0,
        terminal_heading_error=0.0,
        terminal_speed=terminal.speed,
        terminal_yaw_rate=terminal.yaw_rate,
    )


def plan_terminal_approach(
    compiled_map: CompiledSidecarMap,
    *,
    start_state: VesselState,
    dynamics: PrototypeReducedDynamics,
) -> Trajectory:
    """Search the calibrated hard-turn primitives for the final buoy gap."""

    snapshot = compiled_map.snapshot
    if not snapshot.is_state_valid(start_state):
        raise RuntimeError("terminal approach must start from a valid state")
    candidates = []
    for turn_control in TERMINAL_TURN_CONTROLS:
        prefix_controls: list[Control] = []
        prefix_durations: list[float] = []
        prefix_states = [start_state]
        prefix_rollouts: list[tuple[VesselState, ...]] = []
        prefix_minimum = snapshot.clearance_at(start_state)
        for turn_count in range(1, 51):
            rollout = dynamics.propagate(
                prefix_states[-1],
                turn_control,
                TERMINAL_EDGE_DURATION_S,
            )
            motion = snapshot.check_motion(rollout)
            if not motion.valid:
                break
            prefix_controls.append(turn_control)
            prefix_durations.append(TERMINAL_EDGE_DURATION_S)
            prefix_rollouts.append(rollout)
            prefix_states.append(rollout[-1])
            prefix_minimum = min(prefix_minimum, motion.min_clearance)
            if turn_count < 5:
                continue
            for tail_control in TERMINAL_TAIL_CONTROLS:
                controls = list(prefix_controls)
                durations = list(prefix_durations)
                states = list(prefix_states)
                edge_rollouts = list(prefix_rollouts)
                minimum_clearance = prefix_minimum
                for _ in range(80):
                    rollout = dynamics.propagate(
                        states[-1],
                        tail_control,
                        TERMINAL_EDGE_DURATION_S,
                    )
                    motion = snapshot.check_motion(rollout)
                    if not motion.valid:
                        break
                    controls.append(tail_control)
                    durations.append(TERMINAL_EDGE_DURATION_S)
                    edge_rollouts.append(rollout)
                    states.append(rollout[-1])
                    minimum_clearance = min(
                        minimum_clearance,
                        motion.min_clearance,
                    )
                    if fixed_route_waypoint_reached(
                        compiled_map,
                        TERMINAL_ROUTE_INDEX,
                        states[-1],
                    ):
                        goal_x, goal_y = fixed_route_goal_xy(
                            compiled_map.manifest,
                            TERMINAL_ROUTE_INDEX,
                        )
                        candidates.append(
                            (
                                minimum_clearance,
                                len(controls),
                                math.hypot(
                                    states[-1].x - goal_x,
                                    states[-1].y - goal_y,
                                ),
                                controls,
                                durations,
                                states,
                                edge_rollouts,
                            )
                        )
                        break
    if not candidates:
        raise RuntimeError("terminal approach has no collision-free chain")
    preferred = [candidate for candidate in candidates if candidate[0] >= 0.3]
    selected = min(
        preferred or candidates,
        key=lambda candidate: (candidate[1], -candidate[0], candidate[2]),
    )
    (
        minimum_clearance,
        _,
        terminal_error,
        controls,
        durations,
        states,
        edge_rollouts,
    ) = selected
    times = [0.0]
    for duration in durations:
        times.append(times[-1] + duration)
    request_id = "fixed-route-live-leg-12-terminal-approach"
    terminal = states[-1]
    return Trajectory(
        trajectory_id=f"{request_id}-terminal-approach",
        request_id=request_id,
        session_id=snapshot.session_id,
        map_snapshot_id=snapshot.snapshot_id,
        map_source_version=snapshot.source_version,
        map_payload_content_hash=snapshot.payload_content_hash,
        dynamics_version=dynamics.version,
        validator_version=TrajectoryValidator.version,
        frame_id=snapshot.map_frame,
        mission_index=TERMINAL_ROUTE_INDEX,
        mission_version=f"route-v{compiled_map.manifest.route_version}",
        map_source_artifact_hash=snapshot.source_artifact_hash,
        map_compiler_config_hash=snapshot.compiler_config_hash,
        state_version=start_state.state_version,
        states=tuple(states),
        controls=tuple(controls),
        durations=tuple(durations),
        times=tuple(times),
        edge_rollouts=tuple(edge_rollouts),
        cost=times[-1],
        min_clearance=minimum_clearance,
        validation_status="VALID",
        terminal_position_error=terminal_error,
        terminal_heading_error=0.0,
        terminal_speed=terminal.speed,
        terminal_yaw_rate=terminal.yaw_rate,
    )


def _post_narrow_egress_trajectory(
    compiled_map: CompiledSidecarMap,
    *,
    start_state: VesselState,
    dynamics: PrototypeReducedDynamics,
) -> Trajectory:
    """Build the fixed-map east, clockwise, south, then west bypass."""

    snapshot = compiled_map.snapshot
    controls: list[Control] = []
    durations: list[float] = []
    states = [start_state]
    edge_rollouts: list[tuple[VesselState, ...]] = []
    minimum_clearance = snapshot.clearance_at(start_state)

    def append(control: Control) -> VesselState:
        nonlocal minimum_clearance
        rollout = dynamics.propagate(
            states[-1],
            control,
            NARROW_EGRESS_EDGE_DURATION_S,
        )
        motion = snapshot.check_motion(rollout)
        if not motion.valid:
            raise RuntimeError(
                "post-narrow egress primitive is not collision-free"
            )
        controls.append(control)
        durations.append(NARROW_EGRESS_EDGE_DURATION_S)
        edge_rollouts.append(rollout)
        states.append(rollout[-1])
        minimum_clearance = min(
            minimum_clearance,
            motion.min_clearance,
        )
        return rollout[-1]

    # Continue through the single narrow entrance until the clockwise loop
    # has enough room to clear the north-east buoy.  The original composite
    # ends at x=31.6; the measured straight reverse primitive extends it to
    # the independently scanned x=33.0 release plane.
    for _ in range(80):
        if states[-1].x >= NARROW_ESCAPE_RELEASE_X_M:
            break
        append(NARROW_EGRESS_REVERSE_CONTROL)
    else:
        raise RuntimeError("post-narrow egress did not reach east exit")

    # Remove the measured reverse velocity before beginning the clockwise
    # open-water loop.  Two 0.4 s edges are the bounded forward recovery
    # already used by the calibrated reduced model.
    append(NARROW_EGRESS_RECOVERY_CONTROL)
    append(NARROW_EGRESS_RECOVERY_CONTROL)

    for _ in range(180):
        state = append(NARROW_EGRESS_CONTROL)
        if -1.75 <= state.yaw <= -1.40:
            break
    else:
        raise RuntimeError("post-narrow egress did not acquire south heading")

    for _ in range(100):
        state = append(NARROW_EGRESS_STRAIGHT_CONTROL)
        if state.y <= NARROW_EGRESS_SOUTH_TURN_Y_M:
            break
    else:
        raise RuntimeError("post-narrow egress did not pass obstacle south")

    for _ in range(100):
        state = append(NARROW_EGRESS_CONTROL)
        west_error = abs(
            (state.yaw - math.pi + math.pi) % (2.0 * math.pi)
            - math.pi
        )
        if west_error <= 0.15:
            break
    else:
        raise RuntimeError("post-narrow egress did not acquire west heading")

    # End on a stable westbound primitive.  Without this tail, a tracker
    # that reaches the final sample keeps repeating the last turn forever.
    append(NARROW_EGRESS_STRAIGHT_CONTROL)
    append(NARROW_EGRESS_STRAIGHT_CONTROL)
    append(NARROW_EGRESS_TERMINAL_CONTROL)

    terminal = states[-1]
    if not narrow_escape_released(compiled_map, terminal):
        raise RuntimeError(
            "post-narrow egress did not reach its handoff set: "
            f"start=({start_state.x:.6f},{start_state.y:.6f},"
            f"{start_state.yaw:.6f},{start_state.speed:.6f}); "
            f"terminal=({terminal.x:.6f},{terminal.y:.6f},"
            f"{terminal.yaw:.6f},{terminal.speed:.6f})"
        )
    times = [0.0]
    for duration in durations:
        times.append(times[-1] + duration)
    request_id = "fixed-route-live-leg-10-post-narrow-egress"
    return Trajectory(
        trajectory_id=f"{request_id}-post-narrow-egress",
        request_id=request_id,
        session_id=snapshot.session_id,
        map_snapshot_id=snapshot.snapshot_id,
        map_source_version=snapshot.source_version,
        map_payload_content_hash=snapshot.payload_content_hash,
        dynamics_version=dynamics.version,
        validator_version=TrajectoryValidator.version,
        frame_id=snapshot.map_frame,
        mission_index=NARROW_ROUTE_INDEX,
        mission_version=f"route-v{compiled_map.manifest.route_version}",
        map_source_artifact_hash=snapshot.source_artifact_hash,
        map_compiler_config_hash=snapshot.compiler_config_hash,
        state_version=start_state.state_version,
        states=tuple(states),
        controls=tuple(controls),
        durations=tuple(durations),
        times=tuple(times),
        edge_rollouts=tuple(edge_rollouts),
        cost=times[-1],
        min_clearance=minimum_clearance,
        validation_status="VALID",
        terminal_position_error=0.0,
        terminal_heading_error=0.0,
        terminal_speed=terminal.speed,
        terminal_yaw_rate=terminal.yaw_rate,
    )


def fixed_route_tolerance(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
) -> float:
    """Required ship-centre radius around the unchanged published target."""

    manifest = compiled_map.manifest
    point_count = len(manifest.route_points_enu)
    _validate_route_index(point_count, mission_index)
    return FIXED_ROUTE_TOLERANCE_M


def fixed_route_waypoint_reached(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
    state: VesselState,
) -> bool:
    """Whether the ship centre entered the unchanged waypoint region."""

    if not state.is_finite():
        return False
    goal_x, goal_y = fixed_route_goal_xy(
        compiled_map.manifest,
        mission_index,
    )
    return (
        math.hypot(state.x - goal_x, state.y - goal_y)
        <= FIXED_ROUTE_TOLERANCE_M + 1e-9
    )


def fixed_route_planning_gate(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
) -> tuple[float, float]:
    """Map-derived safe pass gate; published waypoint coordinates stay fixed."""

    manifest = compiled_map.manifest
    snapshot = compiled_map.snapshot
    point_count = len(manifest.route_points_enu)
    _validate_route_index(point_count, mission_index)
    cache_key = (snapshot.payload_content_hash, mission_index)
    cached = _ROUTE_GATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    goal = fixed_route_goal_xy(manifest, mission_index)
    goal_state = VesselState(
        x=goal[0],
        y=goal[1],
        yaw=0.0,
        speed=0.0,
        yaw_rate=0.0,
        stamp_sim=snapshot.stamp_sim,
    )
    if (
        snapshot.is_state_valid(goal_state)
        and snapshot.clearance_at(goal_state)
        >= SAFE_GATE_CLEARANCE_M
    ):
        _ROUTE_GATE_CACHE[cache_key] = goal
        return goal

    resolution = snapshot.resolution
    safe_cells: list[tuple[float, float, tuple[int, int]]] = []
    min_cell_x = max(
        0,
        int((goal[0] - FIXED_ROUTE_TOLERANCE_M) // resolution),
    )
    max_cell_x = min(
        snapshot.width - 1,
        int((goal[0] + FIXED_ROUTE_TOLERANCE_M) // resolution),
    )
    min_cell_y = max(
        0,
        int((goal[1] - FIXED_ROUTE_TOLERANCE_M) // resolution),
    )
    max_cell_y = min(
        snapshot.height - 1,
        int((goal[1] + FIXED_ROUTE_TOLERANCE_M) // resolution),
    )
    for cell_y in range(min_cell_y, max_cell_y + 1):
        for cell_x in range(min_cell_x, max_cell_x + 1):
            x = (cell_x + 0.5) * resolution
            y = (cell_y + 0.5) * resolution
            distance = math.hypot(x - goal[0], y - goal[1])
            if distance > MAX_FIXED_ROUTE_TOLERANCE_M:
                continue
            state = VesselState(
                x=x,
                y=y,
                yaw=0.0,
                speed=0.0,
                yaw_rate=0.0,
                stamp_sim=snapshot.stamp_sim,
            )
            clearance = snapshot.clearance_at(state)
            if (
                snapshot.is_state_valid(state)
                and clearance >= SAFE_GATE_CLEARANCE_M
            ):
                safe_cells.append(
                    (distance, -clearance, (cell_x, cell_y))
                )
    if not safe_cells:
        raise ValueError(
            "fixed route point has no safe pass gate within 0.5 m"
        )
    gate_cell = min(safe_cells)[2]
    gate = (
        (gate_cell[0] + 0.5) * resolution,
        (gate_cell[1] + 0.5) * resolution,
    )
    _ROUTE_GATE_CACHE[cache_key] = gate
    return gate


def fixed_route_gate_region(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
) -> tuple[float, float, float]:
    manifest = compiled_map.manifest
    published = fixed_route_goal_xy(manifest, mission_index)
    gate = fixed_route_planning_gate(compiled_map, mission_index)
    displaced = math.hypot(
        gate[0] - published[0],
        gate[1] - published[1],
    ) > 1e-9
    tolerance = (
        DISPLACED_GATE_TOLERANCE_M
        if displaced
        else FIXED_ROUTE_TOLERANCE_M
    )
    return gate[0], gate[1], tolerance


def fixed_route_continuations(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
) -> tuple[tuple[float, float, float], ...]:
    """Return every remaining fixed waypoint region for global lookahead."""

    manifest = compiled_map.manifest
    point_count = len(manifest.route_points_enu)
    _validate_route_index(point_count, mission_index)
    return tuple(
        (
            *fixed_route_gate_region(compiled_map, next_index),
        )
        for next_index in range(
            mission_index + 1,
            min(point_count, mission_index + 3),
        )
    )


def fixed_route_guidance_hash(compiled_map: CompiledSidecarMap) -> str:
    payload = {
        "version": ROUTE_GUIDANCE_VERSION,
        "published_goals": [
            fixed_route_goal_xy(compiled_map.manifest, index)
            for index in range(
                len(compiled_map.manifest.route_points_enu)
            )
        ],
        "planning_gates": [
            fixed_route_gate_region(compiled_map, index)
            for index in range(
                len(compiled_map.manifest.route_points_enu)
            )
        ],
        "narrow_route_index": NARROW_ROUTE_INDEX,
        "clearance_composite_route_index": (
            CLEARANCE_COMPOSITE_ROUTE_INDEX
        ),
        "clearance_approach_gate": CLEARANCE_APPROACH_GATE,
        "clearance_handoff": (
            *CLEARANCE_HANDOFF_XY,
            CLEARANCE_HANDOFF_TOLERANCE_M,
        ),
        "clearance_turn_control": (
            CLEARANCE_TURN_CONTROL.throttle,
            CLEARANCE_TURN_CONTROL.rudder,
            CLEARANCE_TURN_EDGE_DURATION_S,
        ),
        "clearance_turn_brake_control": (
            CLEARANCE_TURN_BRAKE_CONTROL.throttle,
            CLEARANCE_TURN_BRAKE_CONTROL.rudder,
        ),
        "clearance_exit_turn_control": (
            CLEARANCE_EXIT_TURN_CONTROL.throttle,
            CLEARANCE_EXIT_TURN_CONTROL.rudder,
            CLEARANCE_EXIT_EDGE_DURATION_S,
        ),
        "clearance_exit_tail_controls": [
            (control.throttle, control.rudder)
            for control in CLEARANCE_EXIT_TAIL_CONTROLS
        ],
        "narrow_escape_xy": NARROW_ESCAPE_XY,
        "narrow_escape_tolerance_m": NARROW_ESCAPE_TOLERANCE_M,
        "narrow_escape_release_x_m": NARROW_ESCAPE_RELEASE_X_M,
        "narrow_egress_handoff_y_m": NARROW_EGRESS_HANDOFF_Y_M,
        "narrow_egress_heading_tolerance_rad": (
            NARROW_EGRESS_HEADING_TOLERANCE_RAD
        ),
        "narrow_egress_south_turn_y_m": (
            NARROW_EGRESS_SOUTH_TURN_Y_M
        ),
        "narrow_egress_controls": [
            (control.throttle, control.rudder)
            for control in (
                NARROW_EGRESS_REVERSE_CONTROL,
                NARROW_EGRESS_RECOVERY_CONTROL,
                NARROW_EGRESS_CONTROL,
                NARROW_EGRESS_STRAIGHT_CONTROL,
                NARROW_EGRESS_TERMINAL_CONTROL,
            )
        ],
        "terminal_route_index": TERMINAL_ROUTE_INDEX,
        "terminal_turn_controls": [
            (control.throttle, control.rudder)
            for control in TERMINAL_TURN_CONTROLS
        ],
        "terminal_tail_controls": [
            (control.throttle, control.rudder)
            for control in TERMINAL_TAIL_CONTROLS
        ],
        "terminal_edge_duration_s": TERMINAL_EDGE_DURATION_S,
        "ordinary_clearance_visit_regions": (
            ORDINARY_CLEARANCE_VISIT_REGIONS
        ),
        "ordinary_plan_clearance_by_index": (
            ORDINARY_PLAN_CLEARANCE_BY_INDEX
        ),
        "ordinary_visit_max_start_y": ORDINARY_VISIT_MAX_START_Y,
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def fixed_route_geometry_candidates(
    compiled_map: CompiledSidecarMap,
) -> tuple[CompiledSidecarMap, ...]:
    """Ordered collision-model evidence gate; never weakens below 0.1 m."""

    snapshot = compiled_map.snapshot
    configurations = (
        {
            "footprint_radius": 0.4,
            "required_clearance": 0.2,
            "vessel_capsule_length": 0.0,
            "vessel_capsule_width": 0.0,
            "geometry_version": "circle-0.4-margin-0.2-v1",
        },
        {
            "footprint_radius": 0.0,
            "required_clearance": 0.2,
            "vessel_capsule_length": 1.3,
            "vessel_capsule_width": 0.64,
            "geometry_version": (
                "official-capsule-1.3x0.64-margin-0.2-v1"
            ),
        },
        {
            "footprint_radius": 0.0,
            "required_clearance": 0.1,
            "vessel_capsule_length": 1.3,
            "vessel_capsule_width": 0.64,
            "geometry_version": (
                "official-capsule-1.3x0.64-margin-0.1-v1"
            ),
        },
    )
    return tuple(
        replace(
            compiled_map,
            snapshot=replace(
                snapshot,
                payload_content_hash="",
                **configuration,
            ),
        )
        for configuration in configurations
    )


def build_fixed_leg_request(
    compiled_map: CompiledSidecarMap,
    *,
    start_state: VesselState,
    mission_index: int,
    dynamics: PrototypeReducedDynamics,
    cost_config: CostConfig,
    time_budget_ms: float,
    seed: int,
    lookahead_count: int,
    narrow_visit_completed: bool = False,
    clearance_approach_completed: bool = False,
) -> PlanningRequest:
    """Compile one ordinary or narrow composite leg without changing task points."""

    manifest = compiled_map.manifest
    snapshot = compiled_map.snapshot
    _validate_route_index(len(manifest.route_points_enu), mission_index)
    continuations = fixed_route_continuations(compiled_map, mission_index)
    if not 0 <= lookahead_count <= len(continuations):
        raise ValueError("lookahead_count is outside the available route")
    goal_x, goal_y = fixed_route_goal_xy(manifest, mission_index)
    published_goal = GoalRegion(
        x=goal_x,
        y=goal_y,
        position_tolerance=FIXED_ROUTE_TOLERANCE_M,
        heading_tolerance=math.pi,
        speed_limit=1.8,
        yaw_rate_limit=1.2,
    )
    route_gate = fixed_route_gate_region(compiled_map, mission_index)
    visit_regions = ORDINARY_CLEARANCE_VISIT_REGIONS.get(
        mission_index,
        (),
    )
    if start_state.y > ORDINARY_VISIT_MAX_START_Y.get(
        mission_index,
        math.inf,
    ):
        visit_regions = ()
    required_visit_regions = tuple(
        GoalRegion(
            x=x,
            y=y,
            position_tolerance=tolerance,
            heading_tolerance=math.pi,
            speed_limit=1.8,
            yaw_rate_limit=1.2,
        )
        for x, y, tolerance in visit_regions
    )
    goal = published_goal
    continuation_targets = continuations[:lookahead_count]
    if mission_index == NARROW_ROUTE_INDEX:
        goal = GoalRegion(
            x=NARROW_ESCAPE_XY[0],
            y=NARROW_ESCAPE_XY[1],
            position_tolerance=NARROW_ESCAPE_TOLERANCE_M,
            heading_tolerance=math.pi,
            speed_limit=1.8,
            yaw_rate_limit=1.2,
        )
        if not narrow_visit_completed:
            required_visit_regions = (
                GoalRegion(
                    x=route_gate[0],
                    y=route_gate[1],
                    position_tolerance=route_gate[2],
                    heading_tolerance=math.pi,
                    speed_limit=1.8,
                    yaw_rate_limit=1.2,
                ),
                published_goal,
            )
        route_gate = None
        continuation_targets = ()
    elif mission_index == CLEARANCE_COMPOSITE_ROUTE_INDEX:
        goal = GoalRegion(
            x=CLEARANCE_HANDOFF_XY[0],
            y=CLEARANCE_HANDOFF_XY[1],
            position_tolerance=CLEARANCE_HANDOFF_TOLERANCE_M,
            heading_tolerance=math.pi,
            speed_limit=1.8,
            yaw_rate_limit=1.2,
        )
        required_visit_regions = (published_goal,)
        if not clearance_approach_completed:
            required_visit_regions = (
                GoalRegion(
                    x=CLEARANCE_APPROACH_GATE[0],
                    y=CLEARANCE_APPROACH_GATE[1],
                    position_tolerance=CLEARANCE_APPROACH_GATE[2],
                    heading_tolerance=math.pi,
                    speed_limit=1.8,
                    yaw_rate_limit=1.2,
                ),
                *required_visit_regions,
            )
        route_gate = None
        continuation_targets = ()
    while (
        required_visit_regions
        and required_visit_regions[0].contains(start_state)
    ):
        required_visit_regions = required_visit_regions[1:]
    return PlanningRequest(
        request_id=(
            f"fixed-route-live-leg-{mission_index}"
            f"-lookahead-{lookahead_count}"
        ),
        session_id=snapshot.session_id,
        start_state=start_state,
        goal_region=goal,
        map_snapshot_id=snapshot.snapshot_id,
        dynamics_version=dynamics.version,
        cost_config_version=cost_config.version,
        time_budget_ms=time_budget_ms,
        seed=seed + mission_index,
        mission_index=mission_index,
        stamp_sim=start_state.stamp_sim,
        mission_version=f"route-v{manifest.route_version}",
        route_gate=route_gate,
        continuation_targets=continuation_targets,
        required_visit_regions=required_visit_regions,
    )


@dataclass(frozen=True)
class FixedRoutePlan:
    """A sequentially certified plan over every fixed task waypoint."""

    compiled_map: CompiledSidecarMap
    trajectories: tuple[Trajectory, ...]
    start_mission_index: int
    final_state: VesselState


@dataclass(frozen=True)
class GeometryGateEvidence:
    geometry_version: str
    map_payload_hash: str
    required_clearance_m: float
    feasible: bool
    reason: str


class NarrowCompositeInfeasibleError(RuntimeError):
    def __init__(self, evidence: tuple[GeometryGateEvidence, ...]) -> None:
        self.evidence = evidence
        summary = "; ".join(
            f"{item.geometry_version}:{item.reason}" for item in evidence
        )
        super().__init__(
            "narrow composite is infeasible under all approved geometry "
            f"gates: {summary}"
        )


def _route_planner(
    *,
    optimize_with_rrtstar: bool,
    forward_action_controls: tuple[Control, ...] = (),
) -> KinodynamicInformedRRTStarPlanner:
    return KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=1_200,
            edge_durations=(0.2, 0.5, 1.0, 2.0),
            goal_bias=0.25,
            global_sample_ratio=0.3,
            rewire_radius=2.5,
            connect_tolerance=1.2,
            stop_on_first_solution=not optimize_with_rrtstar,
            grid_seed_enabled=True,
            max_request_age_s=60.0,
            max_map_age_s=1.0e9,
            max_throttle=0.1,
            max_abs_rudder=0.1,
            forward_action_controls=forward_action_controls,
        )
    )


def compile_offline_national_map(
    *,
    session_id: str,
    stamp_sim: float = 0.0,
) -> CompiledSidecarMap:
    """Compile the current verified live affine profile without ROS access."""

    artifact, artifact_hash = load_sidecar_artifact(SIDECAR_PATH)
    profile = json.loads(LIVE_PROFILE_PATH.read_text(encoding="utf-8"))
    if profile.get("schema_version") != "national-test-live-affine-v1":
        raise ValueError("National_Test live profile schema is incompatible")
    if profile.get("source_artifact_sha256") != artifact_hash:
        raise ValueError("National_Test profile and sidecar hash do not match")
    if profile.get("route_id") != artifact["route"]["route_id"]:
        raise ValueError("National_Test profile route id does not match")
    coefficients = tuple(float(value) for value in profile["fitted_affine"])
    if len(coefficients) != 6 or not all(
        math.isfinite(value) for value in coefficients
    ):
        raise ValueError("National_Test affine coefficients are invalid")
    return compile_beihu_sidecar(
        artifact,
        source_artifact_hash=artifact_hash,
        session_id=session_id,
        stamp_sim=stamp_sim,
        config=SidecarCompilerConfig(
            required_clearance_m=0.2,
            geometry_version=(
                "circle-0.4-margin-0.2-live-recovery-v1"
            ),
            transform_model="route_fitted_affine",
            coverage_status="complete_prior",
            promotion_note=(
                "operator-authorization:verified-live-route-offline-profile"
            ),
            fitted_affine=coefficients,
        ),
    )


def plan_fixed_leg(
    compiled_map: CompiledSidecarMap,
    *,
    start_state: VesselState,
    mission_index: int,
    dynamics: Optional[PrototypeReducedDynamics] = None,
    cost_config: Optional[CostConfig] = None,
    time_budget_ms: float = 5_000.0,
    optimize_with_rrtstar: bool = False,
    seed: int = 31,
    forward_action_controls: tuple[Control, ...] = (),
    narrow_visit_completed: bool = False,
    clearance_approach_completed: bool = False,
    _allow_retry: bool = True,
) -> Trajectory:
    """Plan one task leg from the latest measured or simulated state."""

    manifest = compiled_map.manifest
    snapshot = compiled_map.snapshot
    if dynamics is None and not forward_action_controls:
        profile = diagnostic_forward_control_profile()
        dynamics = reduced_dynamics_from_profile(profile)
        forward_action_controls = profile.action_controls
    else:
        dynamics = dynamics or PrototypeReducedDynamics()
    cost_config = cost_config or CostConfig()
    if not 0 <= mission_index < len(manifest.route_points_enu):
        raise ValueError("mission_index is outside the fixed route")
    if not snapshot.is_state_valid(start_state):
        raise ValueError("fixed leg start state is not valid")
    if not math.isfinite(time_budget_ms) or time_budget_ms <= 0.0:
        raise ValueError("time_budget_ms must be positive and finite")
    if (
        mission_index == NARROW_ROUTE_INDEX
        and narrow_visit_completed
    ):
        return _post_narrow_egress_trajectory(
            compiled_map,
            start_state=start_state,
            dynamics=dynamics,
        )
    if mission_index == TERMINAL_ROUTE_INDEX:
        return plan_terminal_approach(
            compiled_map,
            start_state=start_state,
            dynamics=dynamics,
        )
    continuations = fixed_route_continuations(
        compiled_map,
        mission_index,
    )
    planner_controls = (
        forward_action_controls
        if mission_index == NARROW_ROUTE_INDEX
        else tuple(
            control
            for control in forward_action_controls
            if control.throttle >= 0.0
        )
    )
    result = None
    ordinary_clearance_floor = min(
        ORDINARY_PLAN_CLEARANCE_BY_INDEX.get(
            mission_index,
            snapshot.required_clearance,
        ),
        snapshot.clearance_at(start_state),
    )
    if start_state.y > ORDINARY_VISIT_MAX_START_Y.get(
        mission_index,
        math.inf,
    ):
        ordinary_clearance_floor = min(
            ordinary_clearance_floor,
            snapshot.required_clearance,
        )

    def acceptable(trajectory: Trajectory) -> bool:
        return (
            mission_index == NARROW_ROUTE_INDEX
            or trajectory.min_clearance + 1e-9
            >= ordinary_clearance_floor
        )

    lookahead_counts = (
        (0,)
        if mission_index == NARROW_ROUTE_INDEX
        else range(len(continuations), -1, -1)
    )
    for lookahead_count in lookahead_counts:
        request = build_fixed_leg_request(
            compiled_map,
            start_state=start_state,
            mission_index=mission_index,
            dynamics=dynamics,
            cost_config=cost_config,
            time_budget_ms=time_budget_ms,
            seed=seed,
            lookahead_count=lookahead_count,
            narrow_visit_completed=narrow_visit_completed,
            clearance_approach_completed=clearance_approach_completed,
        )
        result = _route_planner(
            optimize_with_rrtstar=optimize_with_rrtstar,
            forward_action_controls=planner_controls,
        ).plan(
            request,
            snapshot,
            dynamics,
            cost_config,
            now_sim=start_state.stamp_sim,
        )
        if (
            result.trajectory is not None
            and result.trajectory.controls
            and acceptable(result.trajectory)
        ):
            return result.trajectory
    if mission_index == NARROW_ROUTE_INDEX:
        assert result is not None
        raise RuntimeError(
            f"fixed route leg {mission_index} failed: "
            f"{result.status.value} {result.reason}"
        )
    if mission_index == CLEARANCE_COMPOSITE_ROUTE_INDEX:
        assert result is not None
        raise RuntimeError(
            "point-four composite failed its 0.3 m clearance floor: "
            f"{result.status.value} {result.reason}; "
            f"start=({start_state.x:.6f},{start_state.y:.6f},"
            f"{start_state.yaw:.6f},{start_state.speed:.6f},"
            f"{start_state.yaw_rate:.6f})"
        )
    if request.required_visit_regions:
        assert result is not None
        raise RuntimeError(
            f"fixed route leg {mission_index} failed its required "
            f"clearance visit: {result.status.value} {result.reason}; "
            f"start=({start_state.x:.6f},{start_state.y:.6f},"
            f"{start_state.yaw:.6f},{start_state.speed:.6f},"
            f"{start_state.yaw_rate:.6f})"
        )
    for recovery_attempt in range(2):
        goal_x, goal_y = fixed_route_goal_xy(
            manifest,
            mission_index,
        )
        fallback_request = PlanningRequest(
            request_id=(
                f"fixed-route-live-leg-{mission_index}"
                f"-recovery-{recovery_attempt}"
            ),
            session_id=snapshot.session_id,
            start_state=start_state,
            goal_region=GoalRegion(
                x=goal_x,
                y=goal_y,
                position_tolerance=fixed_route_tolerance(
                    compiled_map,
                    mission_index,
                ),
                heading_tolerance=math.pi,
                speed_limit=1.2,
                yaw_rate_limit=1.2,
            ),
            map_snapshot_id=snapshot.snapshot_id,
            dynamics_version=dynamics.version,
            cost_config_version=cost_config.version,
            time_budget_ms=time_budget_ms,
            seed=seed + mission_index + 1_009 * recovery_attempt,
            mission_index=mission_index,
            stamp_sim=start_state.stamp_sim,
            mission_version=f"route-v{manifest.route_version}",
        )
        result = _route_planner(
            optimize_with_rrtstar=optimize_with_rrtstar,
            forward_action_controls=planner_controls,
        ).plan(
            fallback_request,
            snapshot,
            dynamics,
            cost_config,
            now_sim=start_state.stamp_sim,
        )
        if (
            result.trajectory is not None
            and result.trajectory.controls
            and acceptable(result.trajectory)
        ):
            return result.trajectory
    if _allow_retry:
        return plan_fixed_leg(
            compiled_map,
            start_state=start_state,
            mission_index=mission_index,
            dynamics=dynamics,
            cost_config=cost_config,
            time_budget_ms=time_budget_ms,
            optimize_with_rrtstar=optimize_with_rrtstar,
            seed=seed + 10_009,
            forward_action_controls=forward_action_controls,
            narrow_visit_completed=narrow_visit_completed,
            clearance_approach_completed=clearance_approach_completed,
            _allow_retry=False,
        )
    assert result is not None
    raise RuntimeError(
        f"fixed route leg {mission_index} failed: "
        f"{result.status.value} {result.reason}; "
        f"start=({start_state.x:.6f},{start_state.y:.6f},"
        f"{start_state.yaw:.6f},{start_state.speed:.6f},"
        f"{start_state.yaw_rate:.6f})"
    )


def plan_narrow_with_geometry_evidence(
    compiled_map: CompiledSidecarMap,
    *,
    start_state: VesselState,
    dynamics: Optional[PrototypeReducedDynamics] = None,
    cost_config: Optional[CostConfig] = None,
    time_budget_ms: float = 5_000.0,
    seed: int = 31,
    forward_action_controls: tuple[Control, ...] = (),
) -> tuple[
    CompiledSidecarMap,
    Trajectory,
    tuple[GeometryGateEvidence, ...],
]:
    """Try the three approved geometry gates in order and fail explicitly."""

    if dynamics is None and not forward_action_controls:
        profile = diagnostic_forward_control_profile()
        dynamics = reduced_dynamics_from_profile(profile)
        forward_action_controls = profile.action_controls
    else:
        dynamics = dynamics or PrototypeReducedDynamics()
    cost_config = cost_config or CostConfig()
    evidence = []
    for candidate in fixed_route_geometry_candidates(compiled_map):
        try:
            trajectory = plan_fixed_leg(
                candidate,
                start_state=start_state,
                mission_index=NARROW_ROUTE_INDEX,
                dynamics=dynamics,
                cost_config=cost_config,
                time_budget_ms=time_budget_ms,
                seed=seed,
                forward_action_controls=forward_action_controls,
                _allow_retry=False,
            )
        except RuntimeError as exc:
            evidence.append(
                GeometryGateEvidence(
                    candidate.snapshot.geometry_version,
                    candidate.snapshot.payload_content_hash,
                    candidate.snapshot.required_clearance,
                    False,
                    str(exc),
                )
            )
            continue
        evidence.append(
            GeometryGateEvidence(
                candidate.snapshot.geometry_version,
                candidate.snapshot.payload_content_hash,
                candidate.snapshot.required_clearance,
                True,
                "FEASIBLE",
            )
        )
        return candidate, trajectory, tuple(evidence)
    raise NarrowCompositeInfeasibleError(tuple(evidence))


def plan_fixed_route(
    compiled_map: CompiledSidecarMap,
    *,
    dynamics: Optional[PrototypeReducedDynamics] = None,
    cost_config: Optional[CostConfig] = None,
    start_state: Optional[VesselState] = None,
    start_mission_index: int = 1,
    time_budget_ms: float = 5_000.0,
    optimize_with_rrtstar: bool = False,
    seed: int = 31,
    forward_action_controls: tuple[Control, ...] = (),
) -> FixedRoutePlan:
    """Plan and independently validate every remaining fixed route leg.

    The deterministic grid/lattice rollout supplies a feasible kinodynamic
    warm start.  With ``optimize_with_rrtstar`` enabled, the planner spends
    the remaining per-leg budget on informed sampling and rewiring.
    """

    manifest = compiled_map.manifest
    snapshot = compiled_map.snapshot
    if dynamics is None and not forward_action_controls:
        profile = diagnostic_forward_control_profile()
        dynamics = reduced_dynamics_from_profile(profile)
        forward_action_controls = profile.action_controls
    else:
        dynamics = dynamics or PrototypeReducedDynamics()
    cost_config = cost_config or CostConfig()
    if not 0 <= start_mission_index < len(manifest.route_points_enu):
        raise ValueError("start_mission_index is outside the fixed route")
    if not math.isfinite(time_budget_ms) or time_budget_ms <= 0.0:
        raise ValueError("time_budget_ms must be positive and finite")

    if start_state is None:
        current_enu = manifest.route_points_enu[start_mission_index - 1]
        goal_enu = manifest.route_points_enu[start_mission_index]
        x = current_enu[0] - manifest.origin_enu[0]
        y = current_enu[1] - manifest.origin_enu[1]
        goal_x = goal_enu[0] - manifest.origin_enu[0]
        goal_y = goal_enu[1] - manifest.origin_enu[1]
        start_state = VesselState(
            x=x,
            y=y,
            yaw=math.atan2(goal_y - y, goal_x - x),
            speed=0.0,
            yaw_rate=0.0,
            stamp_sim=snapshot.stamp_sim,
        )
    if not snapshot.is_state_valid(start_state):
        raise ValueError("fixed route start state is not valid")

    trajectories = []
    state = start_state
    for mission_index in range(
        start_mission_index,
        len(manifest.route_points_enu),
    ):
        trajectory = plan_fixed_leg(
            compiled_map,
            start_state=state,
            mission_index=mission_index,
            dynamics=dynamics,
            cost_config=cost_config,
            time_budget_ms=time_budget_ms,
            optimize_with_rrtstar=optimize_with_rrtstar,
            seed=seed,
            forward_action_controls=forward_action_controls,
        )
        trajectories.append(trajectory)
        state = trajectory.states[-1]

    return FixedRoutePlan(
        compiled_map=compiled_map,
        trajectories=tuple(trajectories),
        start_mission_index=start_mission_index,
        final_state=state,
    )


__all__ = [
    "CLEARANCE_APPROACH_GATE",
    "CLEARANCE_COMPOSITE_ROUTE_INDEX",
    "CLEARANCE_HANDOFF_TOLERANCE_M",
    "CLEARANCE_HANDOFF_XY",
    "CLEARANCE_TURN_CONTROL",
    "CLEARANCE_TURN_BRAKE_CONTROL",
    "CLEARANCE_TURN_EDGE_DURATION_S",
    "CLEARANCE_EXIT_EDGE_DURATION_S",
    "CLEARANCE_EXIT_TAIL_CONTROLS",
    "CLEARANCE_EXIT_TURN_CONTROL",
    "FIXED_ROUTE_TOLERANCE_M",
    "MAX_FIXED_ROUTE_TOLERANCE_M",
    "ROUTE_GUIDANCE_VERSION",
    "SAFE_GATE_CLEARANCE_M",
    "NARROW_EGRESS_CONTROL",
    "NARROW_EGRESS_HANDOFF_Y_M",
    "NARROW_EGRESS_HEADING_TOLERANCE_RAD",
    "NARROW_EGRESS_SOUTH_TURN_Y_M",
    "NARROW_EGRESS_TERMINAL_CONTROL",
    "NARROW_ESCAPE_TOLERANCE_M",
    "NARROW_ESCAPE_RELEASE_X_M",
    "NARROW_ESCAPE_XY",
    "NARROW_ROUTE_INDEX",
    "ORDINARY_CLEARANCE_VISIT_REGIONS",
    "ORDINARY_PLAN_CLEARANCE_BY_INDEX",
    "ORDINARY_VISIT_MAX_START_Y",
    "TERMINAL_EDGE_DURATION_S",
    "TERMINAL_ROUTE_INDEX",
    "TERMINAL_TAIL_CONTROLS",
    "TERMINAL_TURN_CONTROLS",
    "FixedRoutePlan",
    "GeometryGateEvidence",
    "NarrowCompositeInfeasibleError",
    "compile_offline_national_map",
    "clearance_approach_reached",
    "clearance_handoff_reached",
    "fixed_route_continuations",
    "fixed_route_geometry_candidates",
    "fixed_route_guidance_hash",
    "fixed_route_gate_region",
    "fixed_route_goal_xy",
    "fixed_route_planning_gate",
    "fixed_route_tolerance",
    "fixed_route_waypoint_reached",
    "is_clearance_composite_trajectory",
    "is_clearance_exit_trajectory",
    "is_clearance_turn_trajectory",
    "is_narrow_composite_trajectory",
    "is_narrow_egress_trajectory",
    "is_terminal_route_trajectory",
    "narrow_escape_released",
    "plan_fixed_leg",
    "plan_clearance_turn",
    "plan_clearance_exit",
    "plan_narrow_with_geometry_evidence",
    "plan_terminal_approach",
    "plan_fixed_route",
    "build_fixed_leg_request",
]
