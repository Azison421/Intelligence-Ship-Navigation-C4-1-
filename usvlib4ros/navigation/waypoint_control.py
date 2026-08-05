"""Pure control contracts for the frozen National_Test waypoint route."""

from __future__ import annotations

from dataclasses import dataclass

from usvlib4ros.planning.forward_control_profile import ForwardControlProfile
from usvlib4ros.planning.kinodynamic_informed_rrtstar import Control
from usvlib4ros.policy.recurrent_sac import (
    LOCAL_WAYPOINT_OBSERVATION_SCHEMA_V3,
)


OBSERVATION_SCHEMA_V3 = LOCAL_WAYPOINT_OBSERVATION_SCHEMA_V3
ACTION_SCHEMA_V3 = "five-calibrated-controls-v3"
CHECKPOINT_SCHEMA_V6 = "national-test-sac-checkpoint-v6"
REPLAY_SCHEMA_V3 = "national-test-replay-v3"


def _percent_command(control: Control) -> tuple[int, int]:
    return (
        int(round(control.throttle * 100.0)),
        int(round(control.rudder * 100.0)),
    )


@dataclass(frozen=True)
class CalibratedActionSet:
    """Five semantically ordered controls that remain unique on the wire."""

    controls: tuple[Control, ...]
    percent_commands: tuple[tuple[int, int], ...]
    calibration_hash: str
    schema_version: str = ACTION_SCHEMA_V3

    @classmethod
    def from_profile(
        cls,
        profile: ForwardControlProfile,
    ) -> "CalibratedActionSet":
        if not isinstance(profile, ForwardControlProfile):
            raise ValueError("calibrated action set requires a forward profile")
        if profile.action_schema != ACTION_SCHEMA_V3:
            raise ValueError("forward profile action schema is incompatible")
        controls = tuple(profile.action_controls)
        percentages = tuple(_percent_command(control) for control in controls)
        if len(set(percentages)) != 5:
            raise ValueError("controls must remain unique after percent rounding")
        rudders = tuple(command[1] for command in percentages)
        if not (
            rudders[0] < rudders[1] < 0
            and rudders[2] == 0
            and 0 < rudders[3] < rudders[4]
        ):
            raise ValueError(
                "controls must be ordered hard-left, soft-left, straight, "
                "soft-right, hard-right"
            )
        if any(command[0] <= 0 for command in percentages):
            raise ValueError("all calibrated actions must provide forward steerage")
        return cls(
            controls=controls,
            percent_commands=percentages,
            calibration_hash=profile.calibration_hash,
        )

    def control(self, action: int) -> Control:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("action must be an integer")
        try:
            return self.controls[action]
        except IndexError as exc:
            raise ValueError("action must be one of five calibrated controls") from exc


class ActuatorTransitionGuard:
    """Require a straight command before crossing between extreme rudders."""

    def __init__(self) -> None:
        self._latched_side = 0

    def reachability_mask(self) -> tuple[bool, ...]:
        if self._latched_side < 0:
            return (True, True, True, False, False)
        if self._latched_side > 0:
            return (False, False, True, True, True)
        return (True,) * 5

    def record_executed(self, action: int) -> None:
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 5:
            raise ValueError("executed action must be one of five controls")
        if not self.reachability_mask()[action]:
            raise ValueError("extreme rudder reversal requires a straight transition")
        if action == 2:
            self._latched_side = 0
        elif action == 0:
            self._latched_side = -1
        elif action == 4:
            self._latched_side = 1

    def reset(self) -> None:
        self._latched_side = 0


class NoSafeActionWindow:
    """Count only consecutive fresh cycles with no reachable safe control."""

    def __init__(self, limit: int = 10) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("no-safe-action limit must be a positive integer")
        self.limit = limit
        self.count = 0

    def observe(self, *, fresh_inputs: bool, has_safe_action: bool) -> bool:
        if type(fresh_inputs) is not bool or type(has_safe_action) is not bool:
            raise ValueError("freshness and safety flags must be boolean")
        if not fresh_inputs or has_safe_action:
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.limit

    def reset(self) -> None:
        self.count = 0


def combine_action_masks(
    reachability: tuple[bool, ...],
    safety: tuple[bool, ...],
) -> tuple[bool, ...]:
    if len(reachability) != 5 or len(safety) != 5:
        raise ValueError("action masks must contain five values")
    if any(type(value) is not bool for value in (*reachability, *safety)):
        raise ValueError("action masks must contain booleans")
    return tuple(reachable and safe for reachable, safe in zip(reachability, safety))
