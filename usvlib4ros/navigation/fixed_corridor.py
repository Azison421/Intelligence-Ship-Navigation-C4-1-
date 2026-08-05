"""Hash-bound frozen reference corridor for the fixed National_Test map."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from usvlib4ros.mapping import CompiledSidecarMap
from usvlib4ros.planning.fixed_route import (
    FIXED_ROUTE_TOLERANCE_M,
    fixed_route_goal_xy,
)
from usvlib4ros.planning.kinodynamic_informed_rrtstar import VesselState


FROZEN_CORRIDOR_SCHEMA = "fixed-route-corridor-v1"
DEFAULT_CORRIDOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "mapping"
    / "data"
    / "national_test_fixed_route_corridor_v1.json"
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _point(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"corridor {name} must contain x and y")
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"corridor {name} must be numeric") from exc
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"corridor {name} must be finite")
    return point


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class CorridorProjection:
    cross_track_error_m: float
    heading_error_rad: float
    route_progress: float
    lookahead_xy: tuple[float, float]


@dataclass(frozen=True)
class FrozenRouteCorridor:
    corridor_id: str
    route_id: str
    source_artifact_hash: str
    map_payload_hash: str
    compiler_config_hash: str
    route_points_hash: str
    corridor_hash: str
    required_clearance_m: float
    task_points: tuple[tuple[float, float], ...]
    task_anchors: tuple[tuple[float, float], ...]
    polyline: tuple[tuple[float, float], ...]
    anchor_polyline_indices: tuple[int, ...]
    generator: str
    schema_version: str = FROZEN_CORRIDOR_SCHEMA

    @classmethod
    def load(
        cls,
        path: Path,
        compiled_map: CompiledSidecarMap,
    ) -> "FrozenRouteCorridor":
        if not isinstance(compiled_map, CompiledSidecarMap):
            raise ValueError("corridor requires a compiled National_Test map")
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("corridor artifact must be an object")
        if payload.get("schema_version") != FROZEN_CORRIDOR_SCHEMA:
            raise ValueError("corridor schema is incompatible")
        stored_hash = payload.get("corridor_sha256")
        canonical = dict(payload)
        canonical.pop("corridor_sha256", None)
        if stored_hash != _canonical_hash(canonical):
            raise ValueError("corridor artifact hash does not match content")

        task_points = tuple(
            _point(value, "task point") for value in payload.get("task_points", ())
        )
        task_anchors = tuple(
            _point(value, "task anchor") for value in payload.get("task_anchors", ())
        )
        polyline = tuple(
            _point(value, "polyline point") for value in payload.get("polyline", ())
        )
        raw_indices = payload.get("anchor_polyline_indices")
        if not isinstance(raw_indices, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_indices
        ):
            raise ValueError("corridor anchor indices are invalid")
        indices = tuple(raw_indices)
        corridor = cls(
            corridor_id=str(payload.get("corridor_id", "")),
            route_id=str(payload.get("route_id", "")),
            source_artifact_hash=str(payload.get("source_artifact_sha256", "")),
            map_payload_hash=str(payload.get("map_payload_sha256", "")),
            compiler_config_hash=str(payload.get("compiler_config_sha256", "")),
            route_points_hash=str(payload.get("route_points_sha256", "")),
            corridor_hash=str(stored_hash or ""),
            required_clearance_m=float(payload.get("required_clearance_m", -1.0)),
            task_points=task_points,
            task_anchors=task_anchors,
            polyline=polyline,
            anchor_polyline_indices=indices,
            generator=str(payload.get("generator", "")),
        )
        corridor._validate(compiled_map)
        return corridor

    def _validate(self, compiled_map: CompiledSidecarMap) -> None:
        snapshot = compiled_map.snapshot
        manifest = compiled_map.manifest
        if not self.corridor_id or self.route_id != manifest.route_id:
            raise ValueError("corridor route identity is incompatible")
        if self.source_artifact_hash != snapshot.source_artifact_hash:
            raise ValueError("corridor source artifact hash is incompatible")
        if self.map_payload_hash != snapshot.payload_content_hash:
            raise ValueError("corridor map payload hash is incompatible")
        if self.compiler_config_hash != snapshot.compiler_config_hash:
            raise ValueError("corridor compiler configuration is incompatible")
        if self.required_clearance_m != 0.2 or snapshot.required_clearance != 0.2:
            raise ValueError("corridor requires the 0.2 m map clearance baseline")
        if len(self.task_points) != 13 or len(self.task_anchors) != 13:
            raise ValueError("corridor must bind exactly thirteen task points")
        if len(self.polyline) <= len(self.task_points):
            raise ValueError("corridor must contain non-scoring guidance points")
        if len(self.anchor_polyline_indices) != 13:
            raise ValueError("corridor must index all thirteen task anchors")
        if tuple(sorted(self.anchor_polyline_indices)) != self.anchor_polyline_indices:
            raise ValueError("corridor anchor indices must be monotonic")
        if self.anchor_polyline_indices[-1] >= len(self.polyline):
            raise ValueError("corridor anchor index is out of range")
        if self.route_points_hash != _canonical_hash(list(map(list, self.task_points))):
            raise ValueError("corridor route point hash does not match content")

        for index, (task_point, anchor, polyline_index) in enumerate(
            zip(
                self.task_points,
                self.task_anchors,
                self.anchor_polyline_indices,
            )
        ):
            expected = fixed_route_goal_xy(manifest, index)
            if math.dist(task_point, expected) > 1e-6:
                raise ValueError("corridor task points differ from National_Test")
            if math.dist(task_point, anchor) > FIXED_ROUTE_TOLERANCE_M + 1e-9:
                raise ValueError("corridor anchor lies outside task tolerance")
            if math.dist(anchor, self.polyline[polyline_index]) > 1e-9:
                raise ValueError("corridor task anchor index does not match polyline")
            self._validate_point(snapshot, anchor)

        for point in self.polyline:
            self._validate_point(snapshot, point)
        for start, end in zip(self.polyline, self.polyline[1:]):
            distance = math.dist(start, end)
            sample_count = max(1, math.ceil(distance / 0.05))
            for sample_index in range(sample_count + 1):
                ratio = sample_index / sample_count
                point = (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
                self._validate_point(snapshot, point)

    def _validate_point(self, snapshot, point: tuple[float, float]) -> None:
        state = VesselState(
            x=point[0],
            y=point[1],
            yaw=0.0,
            speed=0.0,
            yaw_rate=0.0,
            stamp_sim=snapshot.stamp_sim,
        )
        if not snapshot.is_state_valid(state):
            raise ValueError("corridor enters invalid map space")
        if snapshot.clearance_at(state) + 1e-9 < self.required_clearance_m:
            raise ValueError("corridor violates the map clearance baseline")

    @property
    def total_length_m(self) -> float:
        return sum(
            math.dist(start, end)
            for start, end in zip(self.polyline, self.polyline[1:])
        )

    def project(
        self,
        state: VesselState,
        previous_progress: float,
        mission_index: int,
    ) -> CorridorProjection:
        if not state.is_finite():
            raise ValueError("corridor projection requires a finite state")
        if not math.isfinite(previous_progress) or not 0.0 <= previous_progress <= 1.0:
            raise ValueError("previous corridor progress must be in [0, 1]")
        if (
            isinstance(mission_index, bool)
            or not isinstance(mission_index, int)
            or not 0 <= mission_index < 13
        ):
            raise ValueError("corridor projection mission index is invalid")
        total = self.total_length_m
        if total <= 0.0:
            raise ValueError("corridor must have positive length")

        last_polyline_index = self.anchor_polyline_indices[mission_index]
        maximum_distance = sum(
            math.dist(start, end)
            for start, end in zip(
                self.polyline[:last_polyline_index],
                self.polyline[1 : last_polyline_index + 1],
            )
        )
        maximum_progress = maximum_distance / total
        if previous_progress > maximum_progress + 1e-12:
            raise ValueError("corridor progress is beyond the current mission leg")

        best = None
        traversed = 0.0
        for start, end in zip(
            self.polyline[:last_polyline_index],
            self.polyline[1 : last_polyline_index + 1],
        ):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1e-12:
                continue
            ratio = max(
                0.0,
                min(
                    1.0,
                    ((state.x - start[0]) * dx + (state.y - start[1]) * dy)
                    / (length * length),
                ),
            )
            progress = (traversed + ratio * length) / total
            if progress + 1e-12 < previous_progress:
                traversed += length
                continue
            projected = (start[0] + ratio * dx, start[1] + ratio * dy)
            distance = math.dist((state.x, state.y), projected)
            candidate = (distance, progress, start, end, projected, length)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            traversed += length

        if best is None:
            progress = previous_progress
            projected, start, end = self._point_at_progress(progress)
            length = math.dist(start, end)
        else:
            _, progress, start, end, projected, length = best
            progress = max(previous_progress, progress)
            if progress > best[1] + 1e-12:
                projected, start, end = self._point_at_progress(progress)
                length = math.dist(start, end)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if length <= 1e-12:
            heading = state.yaw
            cross_track = 0.0
        else:
            heading = math.atan2(dy, dx)
            cross_track = (
                dx * (state.y - projected[1])
                - dy * (state.x - projected[0])
            ) / length
        lookahead_progress = min(
            maximum_progress,
            progress + 1.0 / total,
        )
        lookahead, _, _ = self._point_at_progress(lookahead_progress)
        return CorridorProjection(
            cross_track_error_m=cross_track,
            heading_error_rad=_wrap_angle(heading - state.yaw),
            route_progress=progress,
            lookahead_xy=lookahead,
        )

    def _point_at_progress(
        self,
        progress: float,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        target = min(1.0, max(0.0, progress)) * self.total_length_m
        traversed = 0.0
        for start, end in zip(self.polyline, self.polyline[1:]):
            length = math.dist(start, end)
            if traversed + length + 1e-12 >= target:
                ratio = 0.0 if length <= 1e-12 else (target - traversed) / length
                return (
                    (
                        start[0] + ratio * (end[0] - start[0]),
                        start[1] + ratio * (end[1] - start[1]),
                    ),
                    start,
                    end,
                )
            traversed += length
        return self.polyline[-1], self.polyline[-2], self.polyline[-1]
