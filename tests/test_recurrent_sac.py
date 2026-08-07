"""Focused V3 recurrent SAC contracts."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from usvlib4ros.policy.recurrent_sac import (
    LocalWaypointObservationV3,
    RecurrentDiscreteSAC,
    SequenceReplay,
    SequenceTransition,
)


def _observation(
    stamp: float,
    *,
    safe_mask: tuple[bool, ...] = (True,) * 5,
) -> LocalWaypointObservationV3:
    return LocalWaypointObservationV3(
        laser_ranges=(20.0,) * 72,
        laser_valid_mask=(True,) * 72,
        scan_age_s=0.0,
        pose_age_s=0.0,
        device_age_s=0.0,
        speed_mps=0.1,
        yaw_rate_rad_s=0.0,
        actual_throttle=0.1,
        actual_rudder=0.0,
        current_waypoint_body_xy=(2.0, 0.0),
        next_waypoint_body_xy=(4.0, 0.0),
        next_waypoint_valid=True,
        mission_progress=0.25,
        corridor_cross_track_m=0.0,
        corridor_heading_error_rad=0.0,
        corridor_progress=0.25,
        map_clearance_m=1.0,
        safe_action_mask=safe_mask,
        session_id="sac-v3-test",
        stamp_sim=stamp,
        hidden_reset=stamp == 0.0,
    )


def test_masked_policy_uses_only_safe_action_and_keeps_gru_state():
    policy = RecurrentDiscreteSAC(hidden_dim=16, seed=7)
    observation = _observation(0.0, safe_mask=(False, False, False, True, False))

    proposal, hidden = policy.act(
        observation,
        observation.safe_action_mask,
        deterministic=True,
    )

    assert proposal.action == 3
    assert proposal.masked_probabilities[3] == pytest.approx(1.0)
    assert hidden.actor is not None
    assert hidden.critic1 is not None
    assert hidden.critic2 is not None


def test_v3_episode_replay_updates_sac_without_crossing_boundaries():
    transitions = []
    for index in range(12):
        transitions.append(
            SequenceTransition(
                observation=_observation(index * 0.1),
                next_observation=_observation((index + 1) * 0.1),
                executed_action=2,
                reward=1.0,
                terminated=False,
                truncated=index == 11,
                reason="TIME_LIMIT" if index == 11 else "STEP",
            )
        )
    replay = SequenceReplay(capacity=4, seed=11)
    replay.add_episode(transitions)
    batch = replay.sample(batch_size=4, burn_in=2, unroll=4)
    policy = RecurrentDiscreteSAC(hidden_dim=16, seed=11)
    initial_alpha = float(policy.alpha.detach().item())

    result = policy.update(batch)

    assert result["updated"] is True
    assert result["critic_samples"] > 0
    assert result["entropy"] > 0.5 * math.log(5.0)
    assert result["alpha"] < initial_alpha
    assert SequenceReplay.from_state_dict(replay.state_dict()).state_dict()[
        "schema_version"
    ] == "national-test-replay-v3"


def test_replay_learns_from_known_zero_hidden_episode_boundary():
    transition = SequenceTransition(
        observation=_observation(0.0),
        next_observation=_observation(0.1),
        executed_action=4,
        reward=1.0,
        terminated=False,
        truncated=True,
        reason="TIME_LIMIT",
    )
    replay = SequenceReplay(capacity=1, seed=5)
    replay.add_episode((transition,))

    batch = replay.sample(batch_size=1, burn_in=2, unroll=1)

    assert batch.hidden_reset[0, 0].item() is True
    assert batch.learning_mask[0, 0].item() == 1.0


def test_demonstration_warmup_increases_executed_action_probability():
    transitions = tuple(
        SequenceTransition(
            observation=_observation(index * 0.1),
            next_observation=_observation((index + 1) * 0.1),
            executed_action=4,
            reward=1.0,
            terminated=False,
            truncated=index == 11,
            reason="TIME_LIMIT" if index == 11 else "STEP",
        )
        for index in range(12)
    )
    replay = SequenceReplay(capacity=4, seed=13)
    replay.add_episode(transitions)
    batch = replay.sample(batch_size=4, burn_in=2, unroll=4)
    policy = RecurrentDiscreteSAC(hidden_dim=16, seed=13)
    initial_alpha = float(policy.alpha.detach().item())
    before, _ = policy.act(
        _observation(0.0),
        (True,) * 5,
        deterministic=True,
    )

    for _ in range(20):
        result = policy.update(batch, demonstration=True)

    after, _ = policy.act(
        _observation(0.0),
        (True,) * 5,
        deterministic=True,
    )
    assert result["actor_objective"] == "BEHAVIOR_CLONING"
    assert result["behavior_clone_loss"] > 0.0
    assert result["alpha"] == pytest.approx(initial_alpha)
    assert after.masked_probabilities[4] > before.masked_probabilities[4]


def test_dagger_expert_action_trains_actor_without_temperature_drift():
    transitions = tuple(
        SequenceTransition(
            observation=_observation(index * 0.1),
            next_observation=_observation((index + 1) * 0.1),
            executed_action=4,
            reward=1.0,
            terminated=False,
            truncated=index == 11,
            reason="TIME_LIMIT" if index == 11 else "STEP",
        )
        for index in range(12)
    )
    replay = SequenceReplay(capacity=4, seed=23)
    replay.add_episode(transitions)
    batch = replay.sample(batch_size=4, burn_in=2, unroll=4)
    policy = RecurrentDiscreteSAC(hidden_dim=16, seed=23)
    expert_actions = torch.zeros_like(batch.actions)
    initial_alpha = float(policy.alpha.detach().item())
    before, _ = policy.act(
        _observation(0.0),
        (True,) * 5,
        deterministic=True,
    )

    for _ in range(20):
        result = policy.update(batch, expert_actions=expert_actions)

    after, _ = policy.act(
        _observation(0.0),
        (True,) * 5,
        deterministic=True,
    )
    assert result["actor_objective"] == "SAC_WITH_DAGGER"
    assert result["alpha"] == pytest.approx(initial_alpha)
    assert policy.actor_optimizer.param_groups[0]["lr"] == pytest.approx(
        policy.actor_optimizer.defaults["lr"]
    )
    assert after.masked_probabilities[0] > before.masked_probabilities[0]


def test_v6_policy_checkpoint_round_trip_is_non_overwriting(tmp_path: Path):
    checkpoint = tmp_path / "policy-v6.pt"
    policy = RecurrentDiscreteSAC(hidden_dim=16, seed=17)
    policy.save_checkpoint(checkpoint)

    restored = RecurrentDiscreteSAC(hidden_dim=16, seed=18)
    restored.load_checkpoint(checkpoint)

    assert restored.training_step == policy.training_step
    with pytest.raises(FileExistsError):
        policy.save_checkpoint(checkpoint)
