"""Focused two-sided Unity control calibration contracts."""

from __future__ import annotations

from dataclasses import asdict
import json

from usvlib4ros.planning.forward_control_profile import (
    ForwardProbe,
    ForwardProbeSafetySample,
    action_protocol_hash,
    build_forward_control_profile,
    forward_control_profile_from_dict,
    forward_probe_abort_reason,
)


def _probes() -> tuple[ForwardProbe, ...]:
    return (
        ForwardProbe(0.1, 0.0, 0.11, 0.0, 0.02),
        ForwardProbe(0.4, 0.0, 0.44, 0.0, 0.08),
        ForwardProbe(0.1, -0.05, 0.11, 0.10, 0.02),
        ForwardProbe(0.1, -0.1, 0.11, 0.22, 0.02),
        ForwardProbe(0.1, 0.05, 0.11, -0.11, 0.02),
        ForwardProbe(0.1, 0.1, 0.11, -0.23, 0.02),
    )


def test_two_sided_probes_build_five_unique_v3_controls():
    profile = build_forward_control_profile(_probes())
    commands = tuple(
        (
            round(control.throttle * 100),
            round(control.rudder * 100),
        )
        for control in profile.action_controls
    )

    assert profile.action_schema == "five-calibrated-controls-v3"
    assert profile.minimum_steerage_throttle == 0.1
    assert profile.cruise_throttle == 0.4
    assert commands == ((10, -10), (10, -5), (40, 0), (10, 5), (10, 10))
    assert len(set(commands)) == 5


def test_calibration_profile_round_trip_preserves_protocol_identity():
    profile = build_forward_control_profile(_probes())
    payload = json.loads(json.dumps(asdict(profile)))

    restored = forward_control_profile_from_dict(payload)

    assert restored == profile
    assert action_protocol_hash(restored) == action_protocol_hash(profile)


def test_probe_safety_fails_closed_before_motion():
    sample = ForwardProbeSafetySample(
        feedback_ok=True,
        pose_age_s=0.0,
        scan_age_s=0.0,
        clearance_m=2.9,
        minimum_laser_m=4.0,
        displacement_m=0.0,
    )

    assert forward_probe_abort_reason(sample, starting=True) == "START_CLEARANCE"
