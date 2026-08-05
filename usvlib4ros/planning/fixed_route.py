"""Static National_Test map and published waypoint contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

from usvlib4ros.mapping import (
    CompiledSidecarMap,
    SidecarCompilerConfig,
    compile_beihu_sidecar,
    load_sidecar_artifact,
)

from .kinodynamic_informed_rrtstar import VesselState


DATA_DIR = Path(__file__).resolve().parents[1] / "mapping" / "data"
SIDECAR_PATH = DATA_DIR / "beihu_static_world_sidecar.json"
LIVE_PROFILE_PATH = DATA_DIR / "national_test_live_profile.json"
FIXED_ROUTE_TOLERANCE_M = 0.5


def _validate_route_index(point_count: int, mission_index: int) -> None:
    if (
        isinstance(mission_index, bool)
        or not isinstance(mission_index, int)
        or point_count <= 0
        or not 0 <= mission_index < point_count
    ):
        raise ValueError("fixed route index is invalid")


def fixed_route_goal_xy(manifest, mission_index: int) -> tuple[float, float]:
    """Return one unchanged published National_Test waypoint."""

    point_count = len(manifest.route_points_enu)
    _validate_route_index(point_count, mission_index)
    point = manifest.route_points_enu[mission_index]
    return (
        point[0] - manifest.origin_enu[0],
        point[1] - manifest.origin_enu[1],
    )


def fixed_route_tolerance(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
) -> float:
    _validate_route_index(
        len(compiled_map.manifest.route_points_enu),
        mission_index,
    )
    return FIXED_ROUTE_TOLERANCE_M


def fixed_route_waypoint_reached(
    compiled_map: CompiledSidecarMap,
    mission_index: int,
    state: VesselState,
) -> bool:
    """Whether the ship centre entered the published 0.5 m waypoint circle."""

    if not isinstance(state, VesselState) or not state.is_finite():
        return False
    goal_x, goal_y = fixed_route_goal_xy(
        compiled_map.manifest,
        mission_index,
    )
    return (
        math.hypot(state.x - goal_x, state.y - goal_y)
        <= FIXED_ROUTE_TOLERANCE_M + 1e-9
    )


def compile_offline_national_map(
    *,
    session_id: str,
    stamp_sim: float = 0.0,
    required_clearance_m: float = 0.2,
) -> CompiledSidecarMap:
    """Compile the approved affine map with an explicit collision buffer."""

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required")
    if (
        isinstance(required_clearance_m, bool)
        or not isinstance(required_clearance_m, (int, float))
        or not math.isfinite(float(required_clearance_m))
        or float(required_clearance_m) < 0.0
    ):
        raise ValueError("required_clearance_m must be finite and non-negative")
    if not math.isfinite(float(stamp_sim)):
        raise ValueError("stamp_sim must be finite")
    required_clearance_m = float(required_clearance_m)
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
        stamp_sim=float(stamp_sim),
        config=SidecarCompilerConfig(
            required_clearance_m=required_clearance_m,
            geometry_version=(
                "circle-0.4-margin-"
                f"{required_clearance_m}-live-recovery-v1"
            ),
            transform_model="route_fitted_affine",
            coverage_status="complete_prior",
            promotion_note=(
                "operator-authorization:verified-live-route-offline-profile"
            ),
            fitted_affine=coefficients,
        ),
    )


__all__ = [
    "FIXED_ROUTE_TOLERANCE_M",
    "SIDECAR_PATH",
    "compile_offline_national_map",
    "fixed_route_goal_xy",
    "fixed_route_tolerance",
    "fixed_route_waypoint_reached",
]
