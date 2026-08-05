"""Sensor feature extraction shared by live control and offline training."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .recurrent_sac import LASER_COUNT


def front_arc_laser_features(
    ranges: Sequence[object] | Iterable[object],
    *,
    max_range_m: float = 20.0,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Return the sample-compatible first/last 36 front-arc beams."""

    if not math.isfinite(max_range_m) or max_range_m <= 0.0:
        raise ValueError("maximum laser range must be positive and finite")
    values = tuple(ranges)
    if not values:
        return (max_range_m,) * LASER_COUNT, (False,) * LASER_COUNT
    if len(values) >= LASER_COUNT:
        indices = (
            *range(LASER_COUNT // 2),
            *range(len(values) - LASER_COUNT // 2, len(values)),
        )
    else:
        indices = tuple(
            round(index * (len(values) - 1) / (LASER_COUNT - 1))
            for index in range(LASER_COUNT)
        )
    normalized: list[float] = []
    valid_mask: list[bool] = []
    for index in indices:
        try:
            value = float(values[index])
        except (TypeError, ValueError, OverflowError):
            normalized.append(max_range_m)
            valid_mask.append(False)
            continue
        if math.isinf(value) and value > 0.0:
            normalized.append(max_range_m)
            valid_mask.append(True)
        elif math.isfinite(value) and value > 0.0:
            normalized.append(min(value, max_range_m))
            valid_mask.append(True)
        else:
            normalized.append(max_range_m)
            valid_mask.append(False)
    return tuple(normalized), tuple(valid_mask)


__all__ = ["front_arc_laser_features"]
