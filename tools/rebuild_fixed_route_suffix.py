"""Rebuild the frozen National_Test waypoint 11->12->13 RRT* suffix."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usvlib4ros.navigation.route_training_guide import (
    DEFAULT_ROUTE_GUIDE_PATH,
)
from usvlib4ros.planning.fixed_route import (
    compile_offline_national_map,
    fixed_route_goal_xy,
)
from usvlib4ros.planning.forward_control_profile import (
    forward_control_profile_from_dict,
    reduced_dynamics_from_profile,
)
from usvlib4ros.planning.kinodynamic_informed_rrtstar import (
    CostConfig,
    GoalRegion,
    KinodynamicInformedRRTStarPlanner,
    PlannerConfig,
    PlanningRequest,
    VesselState,
)

ACTIVE_CHECKPOINT_PATH = (
    ROOT / "artifacts" / "checkpoints" / "national_test_sac_active.json"
)
GOAL_TOLERANCE_M = 0.5
HANDOFF_DISTANCE_M = 0.7
STAGE_TOLERANCES_M = (2.0, 0.75, GOAL_TOLERANCE_M)
STAGE_SEEDS = {
    11: (31, 312, 315),
    12: (32, 322, 325),
}


def _state_payload(state: VesselState) -> dict[str, float]:
    return {
        "x": round(state.x, 12),
        "y": round(state.y, 12),
        "yaw": round(state.yaw, 12),
        "speed": round(state.speed, 12),
        "yaw_rate": round(state.yaw_rate, 12),
        "throttle_state": round(state.throttle_state, 12),
        "rudder_state": round(state.rudder_state, 12),
        "stamp_sim": round(state.stamp_sim, 12),
    }


def _nearest_action(control, action_controls) -> int:
    return min(
        range(len(action_controls)),
        key=lambda index: (
            ((control.throttle - action_controls[index].throttle) / 0.3) ** 2
            + ((control.rudder - action_controls[index].rudder) / 0.05) ** 2
        ),
    )


def _append_trajectory(
    route_xy: list[list[float]],
    route_actions: list[int],
    edges: list[dict[str, float | int]],
    trajectory,
    action_controls,
) -> None:
    for control, duration, rollout in zip(
        trajectory.controls,
        trajectory.durations,
        trajectory.edge_rollouts,
    ):
        action = _nearest_action(control, action_controls)
        if not route_xy:
            route_xy.append(
                [round(rollout[0].x, 12), round(rollout[0].y, 12)]
            )
            route_actions.append(action)
        route_xy.extend(
            [round(state.x, 12), round(state.y, 12)]
            for state in rollout[1:]
        )
        route_actions.extend(action for _ in rollout[1:])
        edges.append(
            {
                "action": action,
                "duration_s": round(float(duration), 12),
            }
        )


def _plan_leg(
    planner,
    compiled_map,
    dynamics,
    cost_config,
    action_controls,
    mission_index: int,
    start_state: VesselState,
    goal_xy: tuple[float, float],
) -> tuple[dict[str, object], VesselState]:
    current = replace(start_state, stamp_sim=0.0)
    route_xy: list[list[float]] = []
    route_actions: list[int] = []
    edges: list[dict[str, float | int]] = []
    reasons: list[str] = []
    minimum_clearance = math.inf

    for tolerance, seed in zip(
        STAGE_TOLERANCES_M,
        STAGE_SEEDS[mission_index],
    ):
        if math.dist((current.x, current.y), goal_xy) <= tolerance:
            continue
        request = PlanningRequest(
            request_id=(
                f"national-test-suffix-{mission_index}-{tolerance:g}"
            ),
            session_id=compiled_map.snapshot.session_id,
            start_state=current,
            goal_region=GoalRegion(
                x=goal_xy[0],
                y=goal_xy[1],
                position_tolerance=tolerance,
                desired_yaw=None,
                heading_tolerance=math.pi,
                speed_limit=0.7,
                yaw_rate_limit=0.5,
            ),
            map_snapshot_id=compiled_map.snapshot.snapshot_id,
            dynamics_version=dynamics.version,
            cost_config_version=cost_config.version,
            time_budget_ms=30_000.0,
            seed=seed,
            mission_index=mission_index,
            stamp_sim=0.0,
        )
        result = planner.plan(
            request,
            compiled_map.snapshot,
            dynamics,
            cost_config,
            now_sim=0.0,
        )
        if result.trajectory is None:
            raise RuntimeError(
                f"RRT* suffix {mission_index} failed at {tolerance:g} m: "
                f"{result.status.value}:{result.reason}"
            )
        trajectory = result.trajectory
        _append_trajectory(
            route_xy,
            route_actions,
            edges,
            trajectory,
            action_controls,
        )
        reasons.append(result.reason)
        minimum_clearance = min(
            minimum_clearance,
            trajectory.min_clearance,
        )
        current = replace(trajectory.states[-1], stamp_sim=0.0)

    terminal_error = math.dist((current.x, current.y), goal_xy)
    if terminal_error > GOAL_TOLERANCE_M + 1e-9:
        raise RuntimeError(
            f"RRT* suffix {mission_index} ended {terminal_error:.3f} m "
            "from its goal"
        )
    return (
        {
            "mission_index": mission_index,
            "status": "SUCCESS",
            "reason": "+".join(reasons),
            "planner_variant": planner.variant,
            "stage_seeds": list(STAGE_SEEDS[mission_index]),
            "goal_xy": [round(goal_xy[0], 12), round(goal_xy[1], 12)],
            "goal_tolerance_m": GOAL_TOLERANCE_M,
            "start_state": _state_payload(start_state),
            "terminal_state": _state_payload(current),
            "terminal_position_error_m": round(terminal_error, 12),
            "duration_s": round(
                sum(float(edge["duration_s"]) for edge in edges),
                12,
            ),
            "min_clearance_m": round(minimum_clearance, 12),
            "edges": edges,
            "route_xy": route_xy,
            "route_actions": route_actions,
        },
        current,
    )


def main() -> int:
    registry = json.loads(
        ACTIVE_CHECKPOINT_PATH.read_text(encoding="utf-8")
    )
    checkpoint_meta_path = (
        ACTIVE_CHECKPOINT_PATH.parent
        / f"{registry['checkpoint']}.json"
    )
    checkpoint_meta = json.loads(
        checkpoint_meta_path.read_text(encoding="utf-8")
    )
    profile = forward_control_profile_from_dict(
        checkpoint_meta["forward_control_profile"]
    )
    dynamics = reduced_dynamics_from_profile(profile)
    compiled_map = compile_offline_national_map(
        session_id="national-test-frozen-rrt-suffix-v2",
        required_clearance_m=0.0,
    )
    task_points = tuple(
        fixed_route_goal_xy(compiled_map.manifest, index)
        for index in range(13)
    )
    previous = task_points[9]
    point_eleven = task_points[10]
    incoming_length = math.dist(previous, point_eleven)
    ratio = (incoming_length - HANDOFF_DISTANCE_M) / incoming_length
    current = VesselState(
        x=previous[0] + (point_eleven[0] - previous[0]) * ratio,
        y=previous[1] + (point_eleven[1] - previous[1]) * ratio,
        yaw=math.atan2(
            point_eleven[1] - previous[1],
            point_eleven[0] - previous[0],
        ),
        speed=0.3,
        yaw_rate=0.0,
        throttle_state=0.1,
        rudder_state=0.0,
        stamp_sim=0.0,
    )
    planner = KinodynamicInformedRRTStarPlanner(
        PlannerConfig(
            max_nodes=5_000,
            edge_durations=(0.1, 0.2, 0.4, 0.8, 1.2),
            goal_bias=0.4,
            global_sample_ratio=0.05,
            rewire_radius=2.0,
            max_neighbors=48,
            connect_tolerance=0.85,
            stop_on_first_solution=True,
            grid_seed_enabled=True,
            max_request_age_s=60.0,
            max_map_age_s=60.0,
            max_throttle=profile.cruise_throttle,
            max_abs_rudder=max(
                abs(control.rudder)
                for control in profile.action_controls
            ),
            forward_action_controls=profile.action_controls,
        )
    )
    cost_config = CostConfig()
    suffix_plans = []
    for mission_index in (11, 12):
        plan, current = _plan_leg(
            planner,
            compiled_map,
            dynamics,
            cost_config,
            profile.action_controls,
            mission_index,
            current,
            task_points[mission_index],
        )
        suffix_plans.append(plan)

    payload = json.loads(
        DEFAULT_ROUTE_GUIDE_PATH.read_text(encoding="utf-8")
    )
    payload["suffix_plans"] = suffix_plans
    payload.pop("guide_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["guide_sha256"] = hashlib.sha256(encoded).hexdigest()
    DEFAULT_ROUTE_GUIDE_PATH.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "guide_sha256": payload["guide_sha256"],
                "suffixes": [
                    {
                        "mission_index": plan["mission_index"],
                        "terminal_position_error_m": (
                            plan["terminal_position_error_m"]
                        ),
                        "min_clearance_m": plan["min_clearance_m"],
                        "route_points": len(plan["route_xy"]),
                    }
                    for plan in suffix_plans
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
