"""Frozen Unity and RRT* demonstrations for the fixed National_Test route."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np


ROUTE_GUIDE_SCHEMA = "fixed-route-training-guide-v1"
GUIDED_ROUTE_WAYPOINTS = 13
SUFFIX_GOAL_TOLERANCE_M = 0.5
SUFFIX_LOOKAHEAD_M = 0.75
DEFAULT_ROUTE_GUIDE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mapping"
    / "data"
    / "national_test_route_training_guide_v1.json"
)
_FEATURE_ORDER = (
    "corridor_progress",
    "cross_track_m",
    "heading_error_rad",
    "speed_mps",
    "yaw_rate_rad_s",
)


class FrozenRouteTrainingGuide:
    def __init__(
        self,
        payload: object,
        expected_corridor_hash: str,
        expected_action_protocol_hash: str,
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError("route guide artifact must be an object")
        stored_hash = payload.get("guide_sha256")
        canonical = dict(payload)
        canonical.pop("guide_sha256", None)
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if stored_hash != hashlib.sha256(encoded).hexdigest():
            raise ValueError("route guide artifact hash does not match content")
        if payload.get("schema_version") != ROUTE_GUIDE_SCHEMA:
            raise ValueError("route guide schema is incompatible")
        if payload.get("corridor_sha256") != expected_corridor_hash:
            raise ValueError("route guide corridor identity is incompatible")
        if (
            payload.get("action_protocol_sha256")
            != expected_action_protocol_hash
        ):
            raise ValueError("route guide action protocol is incompatible")
        if payload.get("feature_order") != list(_FEATURE_ORDER):
            raise ValueError("route guide feature contract is incompatible")
        if payload.get("guided_waypoints") != GUIDED_ROUTE_WAYPOINTS:
            raise ValueError("route guide waypoint count is incompatible")
        sessions = payload.get("source_sessions")
        if not isinstance(sessions, list) or len(sessions) != 3 or any(
            not isinstance(value, str) or not value for value in sessions
        ):
            raise ValueError("route guide must identify three Unity sources")
        suffix_plans = payload.get("suffix_plans")
        if not isinstance(suffix_plans, list) or any(
            not isinstance(item, dict) for item in suffix_plans
        ):
            raise ValueError("route guide suffix plans are invalid")
        if (
            [item.get("mission_index") for item in suffix_plans] != [11, 12]
            or any(
                item.get("status") != "SUCCESS"
                or float(item.get("min_clearance_m", -1.0)) < 0.0
                for item in suffix_plans
            )
        ):
            raise ValueError("route guide suffix plans are invalid")
        suffix_routes = {}
        suffix_actions = {}
        for item in suffix_plans:
            route = np.asarray(item.get("route_xy"), dtype=np.float64)
            actions = item.get("route_actions")
            goal = np.asarray(item.get("goal_xy"), dtype=np.float64)
            terminal_error = float(
                item.get("terminal_position_error_m", math.inf)
            )
            if (
                route.ndim != 2
                or route.shape[0] < 2
                or route.shape[1] != 2
                or not np.isfinite(route).all()
                or goal.shape != (2,)
                or not np.isfinite(goal).all()
                or not math.isclose(
                    float(item.get("goal_tolerance_m", math.nan)),
                    SUFFIX_GOAL_TOLERANCE_M,
                )
                or not math.isfinite(terminal_error)
                or terminal_error > SUFFIX_GOAL_TOLERANCE_M
                or np.linalg.norm(route[-1] - goal)
                > SUFFIX_GOAL_TOLERANCE_M
                or not isinstance(actions, list)
                or len(actions) != route.shape[0]
                or any(
                    isinstance(action, bool)
                    or not isinstance(action, int)
                    or not 0 <= action < 5
                    for action in actions
                )
            ):
                raise ValueError("route guide suffix route is invalid")
            mission_index = int(item["mission_index"])
            suffix_routes[mission_index] = route
            suffix_actions[mission_index] = tuple(actions)
        weights = np.asarray(payload.get("feature_weights"), dtype=np.float64)
        if weights.shape != (5,) or not np.isfinite(weights).all() or np.any(
            weights <= 0.0
        ):
            raise ValueError("route guide feature weights are invalid")

        groups: list[list[list[tuple[float, ...]]]] = [
            [[] for _ in range(5)] for _ in range(GUIDED_ROUTE_WAYPOINTS)
        ]
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("route guide samples are missing")
        for sample in samples:
            if not isinstance(sample, list) or len(sample) != 7:
                raise ValueError("route guide sample is invalid")
            mission_index, *features, action = sample
            if (
                isinstance(mission_index, bool)
                or not isinstance(mission_index, int)
                or not 0 <= mission_index < GUIDED_ROUTE_WAYPOINTS
                or isinstance(action, bool)
                or not isinstance(action, int)
                or not 0 <= action < 5
            ):
                raise ValueError("route guide sample index is invalid")
            feature_values = tuple(float(value) for value in features)
            if not all(math.isfinite(value) for value in feature_values):
                raise ValueError("route guide sample must be finite")
            groups[mission_index][action].append(feature_values)
        if any(not any(action_groups) for action_groups in groups):
            raise ValueError("route guide must cover every mission leg")
        if any(not action_groups[2] for action_groups in groups):
            raise ValueError("route guide requires a straight transition action")

        self.guide_hash = str(stored_hash)
        self.source_sessions = tuple(sessions)
        self.suffix_plans = tuple(suffix_plans)
        self.sample_count = len(samples)
        self._weights = weights
        self._suffix_routes = suffix_routes
        self._suffix_actions = suffix_actions
        self._groups = tuple(
            tuple(
                (
                    np.asarray(action_group, dtype=np.float64)
                    if action_group
                    else None
                )
                for action_group in mission
            )
            for mission in groups
        )

    def action(
        self,
        mission_index: int,
        corridor_progress: float,
        cross_track_m: float,
        heading_error_rad: float,
        speed_mps: float,
        yaw_rate_rad_s: float,
        safe_action_mask: tuple[bool, ...],
    ) -> int:
        values = (
            corridor_progress,
            cross_track_m,
            heading_error_rad,
            speed_mps,
            yaw_rate_rad_s,
        )
        if (
            isinstance(mission_index, bool)
            or not isinstance(mission_index, int)
            or not 0 <= mission_index < GUIDED_ROUTE_WAYPOINTS
            or not all(math.isfinite(value) for value in values)
            or len(safe_action_mask) != 5
            or any(type(value) is not bool for value in safe_action_mask)
            or not any(safe_action_mask)
        ):
            raise ValueError("route guide input is invalid")
        query = np.asarray(values, dtype=np.float64)
        best_action = None
        best_distance = math.inf
        for action, allowed in enumerate(safe_action_mask):
            samples = self._groups[mission_index][action]
            if not allowed or samples is None:
                continue
            delta = (samples - query) * self._weights
            distance = float(np.min(np.einsum("ij,ij->i", delta, delta)))
            if distance < best_distance:
                best_action = action
                best_distance = distance
        if best_action is None:
            if safe_action_mask[2]:
                return 2
            raise ValueError("route guide has no reachable action")
        return best_action

    @staticmethod
    def _distance_to_route(
        route: np.ndarray,
        x: float,
        y: float,
        start_index: int = 0,
    ) -> float:
        points = route[max(0, min(start_index, len(route) - 2)) :]
        starts = points[:-1]
        vectors = points[1:] - starts
        lengths = np.einsum("ij,ij->i", vectors, vectors)
        offset = np.asarray((x, y), dtype=np.float64) - starts
        ratios = np.divide(
            np.einsum("ij,ij->i", offset, vectors),
            lengths,
            out=np.zeros_like(lengths),
            where=lengths > 0.0,
        )
        ratios = np.clip(ratios, 0.0, 1.0)
        nearest = starts + ratios[:, None] * vectors
        return float(
            np.min(
                np.linalg.norm(
                    nearest - np.asarray((x, y), dtype=np.float64),
                    axis=1,
                )
            )
        )

    def distance_to_suffix(
        self,
        mission_index: int,
        x: float,
        y: float,
    ) -> float:
        route = self._suffix_routes.get(mission_index)
        if (
            route is None
            or not math.isfinite(x)
            or not math.isfinite(y)
        ):
            return math.inf
        return self._distance_to_route(route, x, y)

    def suffix_action(
        self,
        mission_index: int,
        x: float,
        y: float,
        yaw: float,
        safe_action_mask: tuple[bool, ...],
        *,
        start_index: int = 0,
    ) -> tuple[int, int]:
        route = self._suffix_routes.get(mission_index)
        actions = self._suffix_actions.get(mission_index)
        if (
            route is None
            or actions is None
            or not all(math.isfinite(value) for value in (x, y, yaw))
            or len(safe_action_mask) != 5
            or any(type(value) is not bool for value in safe_action_mask)
            or not any(safe_action_mask)
            or isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or start_index < 0
        ):
            raise ValueError("route guide suffix input is invalid")

        first = min(start_index, len(route) - 1)
        offset = route[first:] - np.asarray((x, y), dtype=np.float64)
        nearest_index = first + int(
            np.argmin(np.einsum("ij,ij->i", offset, offset))
        )

        target_index = nearest_index
        lookahead = 0.0
        while (
            target_index + 1 < len(route)
            and lookahead < SUFFIX_LOOKAHEAD_M
        ):
            lookahead += float(
                np.linalg.norm(route[target_index + 1] - route[target_index])
            )
            target_index += 1
        target = route[target_index]
        desired_yaw = math.atan2(target[1] - y, target[0] - x)
        heading_error = math.atan2(
            math.sin(desired_yaw - yaw),
            math.cos(desired_yaw - yaw),
        )

        reference_action = actions[nearest_index]
        if heading_error > 0.35:
            correction = -2
        elif heading_error > 0.10:
            correction = -1
        elif heading_error < -0.35:
            correction = 2
        elif heading_error < -0.10:
            correction = 1
        else:
            correction = 0
        requested_action = max(
            0,
            min(4, reference_action + correction),
        )
        allowed_actions = [
            action
            for action, allowed in enumerate(safe_action_mask)
            if allowed
        ]
        selected_action = min(
            allowed_actions,
            key=lambda action: (
                abs(action - requested_action),
                abs(action - 2),
            ),
        )
        return selected_action, nearest_index


@lru_cache(maxsize=2)
def load_route_training_guide(
    expected_corridor_hash: str,
    expected_action_protocol_hash: str,
) -> FrozenRouteTrainingGuide:
    payload = json.loads(DEFAULT_ROUTE_GUIDE_PATH.read_text(encoding="utf-8"))
    return FrozenRouteTrainingGuide(
        payload,
        expected_corridor_hash,
        expected_action_protocol_hash,
    )
