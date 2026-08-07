"""V3-only masked recurrent discrete SAC for National_Test."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ACTION_COUNT = 5
LASER_COUNT = 72
OBSERVATION_DIM = 166
ACTION_SCHEMA = "five-calibrated-controls-v3"
CHECKPOINT_FORMAT = "recurrent-sac-v4"
LOCAL_WAYPOINT_OBSERVATION_SCHEMA_V3 = "local-waypoint-observation-v3"
REPLAY_SCHEMA_V3 = "national-test-replay-v3"


def _finite_tensor(tensor: Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite")


def _finite_modules(modules: Iterable[nn.Module], name: str = "parameters") -> None:
    for module in modules:
        for parameter in module.parameters():
            _finite_tensor(parameter.detach(), name)


def _finite_value_tree(value: object, name: str) -> None:
    if isinstance(value, Tensor):
        _finite_tensor(value.detach(), name)
    elif isinstance(value, Mapping):
        for nested in value.values():
            _finite_value_tree(nested, name)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _finite_value_tree(nested, name)
    elif isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _float_tuple(values: Sequence[float] | Iterable[float]) -> tuple[float, ...]:
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation features must be numeric") from exc


@dataclass(frozen=True)
class LocalWaypointObservationV3:
    """Fixed 166-value observation shared by live control and training."""

    laser_ranges: tuple[float, ...]
    laser_valid_mask: tuple[bool, ...]
    scan_age_s: float
    pose_age_s: float
    device_age_s: float
    speed_mps: float
    yaw_rate_rad_s: float
    actual_throttle: float
    actual_rudder: float
    current_waypoint_body_xy: tuple[float, float]
    next_waypoint_body_xy: tuple[float, float]
    next_waypoint_valid: bool
    mission_progress: float
    corridor_cross_track_m: float
    corridor_heading_error_rad: float
    corridor_progress: float
    map_clearance_m: float
    safe_action_mask: tuple[bool, ...]
    session_id: str
    stamp_sim: float
    hidden_reset: bool = False
    schema_version: str = LOCAL_WAYPOINT_OBSERVATION_SCHEMA_V3

    def __post_init__(self) -> None:
        laser = _float_tuple(self.laser_ranges)
        laser_mask = tuple(self.laser_valid_mask)
        safe_mask = tuple(self.safe_action_mask)
        current = _float_tuple(self.current_waypoint_body_xy)
        next_waypoint = _float_tuple(self.next_waypoint_body_xy)
        object.__setattr__(self, "laser_ranges", laser)
        object.__setattr__(self, "laser_valid_mask", laser_mask)
        object.__setattr__(self, "safe_action_mask", safe_mask)
        object.__setattr__(self, "current_waypoint_body_xy", current)
        object.__setattr__(self, "next_waypoint_body_xy", next_waypoint)
        if len(laser) != LASER_COUNT or len(laser_mask) != LASER_COUNT:
            raise ValueError("observation requires 72 laser values and masks")
        if any(type(value) is not bool for value in laser_mask):
            raise ValueError("laser_valid_mask must contain booleans")
        if len(safe_mask) != ACTION_COUNT or any(
            type(value) is not bool for value in safe_mask
        ):
            raise ValueError("safe_action_mask must contain five booleans")
        if len(current) != 2 or len(next_waypoint) != 2:
            raise ValueError("waypoint body coordinates must contain x and y")
        if type(self.next_waypoint_valid) is not bool:
            raise ValueError("next_waypoint_valid must be boolean")
        if type(self.hidden_reset) is not bool:
            raise ValueError("hidden_reset must be boolean")
        if self.schema_version != LOCAL_WAYPOINT_OBSERVATION_SCHEMA_V3:
            raise ValueError("observation schema is incompatible")
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("observation session is required")
        numeric = (
            *laser,
            self.scan_age_s,
            self.pose_age_s,
            self.device_age_s,
            self.speed_mps,
            self.yaw_rate_rad_s,
            self.actual_throttle,
            self.actual_rudder,
            *current,
            *next_waypoint,
            self.mission_progress,
            self.corridor_cross_track_m,
            self.corridor_heading_error_rad,
            self.corridor_progress,
            self.map_clearance_m,
            self.stamp_sim,
        )
        if not all(isfinite(float(value)) for value in numeric):
            raise ValueError("observation values must be finite")
        if min(self.scan_age_s, self.pose_age_s, self.device_age_s) < 0.0:
            raise ValueError("observation ages must be non-negative")
        if not 0.0 <= self.mission_progress <= 1.0:
            raise ValueError("mission progress must be in [0, 1]")
        if not 0.0 <= self.corridor_progress <= 1.0:
            raise ValueError("corridor progress must be in [0, 1]")
        if self.map_clearance_m < 0.0:
            raise ValueError("map clearance must be non-negative")

    @property
    def feature_dim(self) -> int:
        return OBSERVATION_DIM

    def to_vector(self) -> tuple[float, ...]:
        vector = (
            *self.laser_ranges,
            *(1.0 if valid else 0.0 for valid in self.laser_valid_mask),
            float(self.scan_age_s),
            float(self.pose_age_s),
            float(self.device_age_s),
            float(self.speed_mps),
            float(self.yaw_rate_rad_s),
            float(self.actual_throttle),
            float(self.actual_rudder),
            *self.current_waypoint_body_xy,
            *self.next_waypoint_body_xy,
            1.0 if self.next_waypoint_valid else 0.0,
            float(self.mission_progress),
            float(self.corridor_cross_track_m),
            float(self.corridor_heading_error_rad),
            float(self.corridor_progress),
            float(self.map_clearance_m),
            *(1.0 if safe else 0.0 for safe in self.safe_action_mask),
        )
        if len(vector) != OBSERVATION_DIM:
            raise RuntimeError("observation dimension is inconsistent")
        return vector

    def tensor(self) -> Tensor:
        return torch.tensor(self.to_vector(), dtype=torch.float32)

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LocalWaypointObservationV3":
        if not isinstance(payload, Mapping):
            raise ValueError("observation payload is invalid")
        try:
            return cls(**dict(payload))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("observation payload is invalid") from exc


@dataclass(frozen=True)
class PolicyProposal:
    probabilities: tuple[float, ...]
    masked_probabilities: tuple[float, ...]
    action: int
    q1_values: tuple[float, ...]
    q2_values: tuple[float, ...]
    entropy: float
    hidden_state_input_id: str
    hidden_state_output_id: str
    observation_schema: str
    action_schema: str
    inference_ms: float

    @property
    def q_values(self) -> tuple[float, ...]:
        return tuple(
            min(first, second)
            for first, second in zip(self.q1_values, self.q2_values)
        )


@dataclass(frozen=True)
class RecurrentHiddenState:
    actor: Optional[Tensor]
    critic1: Optional[Tensor]
    critic2: Optional[Tensor]


@dataclass(frozen=True)
class SequenceTransition:
    observation: LocalWaypointObservationV3
    next_observation: LocalWaypointObservationV3
    executed_action: int
    reward: float
    terminated: bool
    truncated: bool
    reason: str
    operator_truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.observation, LocalWaypointObservationV3) or not isinstance(
            self.next_observation, LocalWaypointObservationV3
        ):
            raise ValueError("transition observations must use the V3 schema")
        if self.observation.session_id != self.next_observation.session_id:
            raise ValueError("a transition cannot cross sessions")
        for name in ("terminated", "truncated", "operator_truncated"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated cannot both be true")
        if self.operator_truncated and not self.truncated:
            raise ValueError("operator_truncated requires truncated")
        if isinstance(self.executed_action, bool) or not isinstance(
            self.executed_action, int
        ) or not 0 <= self.executed_action < ACTION_COUNT:
            raise ValueError("executed_action must be one of five actions")
        if not self.observation.safe_action_mask[self.executed_action]:
            raise ValueError("executed_action must be safe in the observation")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("transition reason is required")
        if not isfinite(float(self.reward)):
            raise ValueError("reward must be finite")

    def to_payload(self) -> dict[str, object]:
        return {
            "observation": self.observation.to_payload(),
            "next_observation": self.next_observation.to_payload(),
            "executed_action": self.executed_action,
            "reward": float(self.reward),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "reason": self.reason,
            "operator_truncated": self.operator_truncated,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SequenceTransition":
        if not isinstance(payload, Mapping):
            raise ValueError("transition payload is invalid")
        required = {
            "observation",
            "next_observation",
            "executed_action",
            "reward",
            "terminated",
            "truncated",
            "reason",
            "operator_truncated",
        }
        if set(payload) != required:
            raise ValueError("transition payload schema is incompatible")
        try:
            return cls(
                observation=LocalWaypointObservationV3.from_payload(payload["observation"]),
                next_observation=LocalWaypointObservationV3.from_payload(
                    payload["next_observation"]
                ),
                executed_action=payload["executed_action"],
                reward=float(payload["reward"]),
                terminated=payload["terminated"],
                truncated=payload["truncated"],
                reason=payload["reason"],
                operator_truncated=payload["operator_truncated"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("transition payload is invalid") from exc


@dataclass
class ReplaySequenceBatch:
    observations: Tensor
    next_observations: Tensor
    actions: Tensor
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    safe_action_mask: Tensor
    next_safe_action_mask: Tensor
    learning_mask: Tensor
    padding_mask: Tensor
    hidden_reset: Tensor
    next_hidden_reset: Tensor
    session_ids: tuple[str, ...]
    episode_ids: tuple[int, ...]

    @property
    def observation_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.safe_action_mask.shape[-1])


class SequenceReplay:
    """Episode-aware V3 replay that never crosses a session boundary."""

    def __init__(self, capacity: int = 1024, seed: int = 0) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.capacity = capacity
        self._rng = Random(seed)
        self._episodes: list[tuple[SequenceTransition, ...]] = []
        self._next_episode_id = 0

    def __len__(self) -> int:
        return len(self._episodes)

    def add_episode(self, transitions: Sequence[SequenceTransition]) -> None:
        episode = tuple(transitions)
        if not episode:
            raise ValueError("episode must contain at least one transition")
        if any(not isinstance(item, SequenceTransition) for item in episode):
            raise ValueError("episode contains an invalid transition")
        session = episode[0].observation.session_id
        if any(item.observation.session_id != session for item in episode):
            raise ValueError("episode transitions must share a session")
        for index, transition in enumerate(episode):
            boundary = transition.terminated or transition.truncated
            if boundary != (index == len(episode) - 1):
                raise ValueError("the episode boundary must be the final transition")
        self._episodes.append(episode)
        self._next_episode_id += 1
        if len(self._episodes) > self.capacity:
            self._episodes.pop(0)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPLAY_SCHEMA_V3,
            "capacity": self.capacity,
            "rng_state": deepcopy(self._rng.getstate()),
            "episodes": [
                [transition.to_payload() for transition in episode]
                for episode in self._episodes
            ],
            "next_episode_id": self._next_episode_id,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "SequenceReplay":
        required = {
            "schema_version",
            "capacity",
            "rng_state",
            "episodes",
            "next_episode_id",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("replay state schema is incompatible")
        if state.get("schema_version") != REPLAY_SCHEMA_V3:
            raise ValueError("only national-test-replay-v3 is supported")
        capacity = state.get("capacity")
        episodes = state.get("episodes")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("replay capacity is invalid")
        if not isinstance(episodes, (list, tuple)) or len(episodes) > capacity:
            raise ValueError("replay episodes are invalid")
        replay = cls(capacity=capacity)
        for raw_episode in episodes:
            if not isinstance(raw_episode, (list, tuple)):
                raise ValueError("replay episode is invalid")
            replay.add_episode(
                tuple(SequenceTransition.from_payload(item) for item in raw_episode)
            )
        counter = state.get("next_episode_id")
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < len(episodes):
            raise ValueError("replay episode counter is invalid")
        replay._next_episode_id = counter
        try:
            replay._rng.setstate(state["rng_state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("replay random state is invalid") from exc
        return replay

    def sample(self, batch_size: int, burn_in: int, unroll: int) -> ReplaySequenceBatch:
        if not self._episodes:
            raise ValueError("replay is empty")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            isinstance(burn_in, bool)
            or not isinstance(burn_in, int)
            or burn_in < 0
            or isinstance(unroll, bool)
            or not isinstance(unroll, int)
            or unroll <= 0
        ):
            raise ValueError("burn_in must be non-negative and unroll positive")
        total = burn_in + unroll
        rows: list[dict[str, object]] = []
        for _ in range(batch_size):
            episode_id = self._rng.randrange(len(self._episodes))
            episode = self._episodes[episode_id]
            start = (
                0
                if self._rng.random() < 0.5
                else self._rng.randrange(len(episode))
            )
            row: dict[str, object] = {
                "obs": [],
                "next_obs": [],
                "actions": [],
                "rewards": [],
                "terminated": [],
                "truncated": [],
                "safe": [],
                "next_safe": [],
                "learning": [],
                "padding": [],
                "reset": [],
                "next_reset": [],
                "session": episode[0].observation.session_id,
                "episode_id": episode_id,
            }
            for offset in range(total):
                index = start + offset
                if index >= len(episode):
                    row["obs"].append((0.0,) * OBSERVATION_DIM)
                    row["next_obs"].append((0.0,) * OBSERVATION_DIM)
                    row["actions"].append(0)
                    row["rewards"].append(0.0)
                    row["terminated"].append(True)
                    row["truncated"].append(False)
                    row["safe"].append((False,) * ACTION_COUNT)
                    row["next_safe"].append((False,) * ACTION_COUNT)
                    row["learning"].append(0.0)
                    row["padding"].append(True)
                    row["reset"].append(True)
                    row["next_reset"].append(True)
                    continue
                transition = episode[index]
                row["obs"].append(transition.observation.to_vector())
                row["next_obs"].append(transition.next_observation.to_vector())
                row["actions"].append(transition.executed_action)
                row["rewards"].append(float(transition.reward))
                row["terminated"].append(transition.terminated)
                row["truncated"].append(transition.truncated)
                row["safe"].append(transition.observation.safe_action_mask)
                row["next_safe"].append(transition.next_observation.safe_action_mask)
                row["learning"].append(
                    1.0 if start == 0 or offset >= burn_in else 0.0
                )
                row["padding"].append(False)
                row["reset"].append(transition.observation.hidden_reset or index == 0)
                row["next_reset"].append(transition.next_observation.hidden_reset)
            rows.append(row)

        def stack(name: str, dtype: torch.dtype) -> Tensor:
            return torch.tensor([row[name] for row in rows], dtype=dtype)

        return ReplaySequenceBatch(
            observations=stack("obs", torch.float32),
            next_observations=stack("next_obs", torch.float32),
            actions=stack("actions", torch.long),
            rewards=stack("rewards", torch.float32),
            terminated=stack("terminated", torch.bool),
            truncated=stack("truncated", torch.bool),
            safe_action_mask=stack("safe", torch.bool),
            next_safe_action_mask=stack("next_safe", torch.bool),
            learning_mask=stack("learning", torch.float32),
            padding_mask=stack("padding", torch.bool),
            hidden_reset=stack("reset", torch.bool),
            next_hidden_reset=stack("next_reset", torch.bool),
            session_ids=tuple(str(row["session"]) for row in rows),
            episode_ids=tuple(int(row["episode_id"]) for row in rows),
        )


def _scale_observation_features(observations: Tensor) -> Tensor:
    """Put heterogeneous V3 features on comparable fixed scales."""

    scaled = observations.clone()
    scaled[..., 0:72] = observations[..., 0:72].clamp(0.0, 20.0) / 20.0
    scaled[..., 144:147] = observations[..., 144:147].clamp(0.0, 1.0)
    scaled[..., 147] = observations[..., 147].clamp(-0.5, 0.5) / 0.5
    scaled[..., 148] = observations[..., 148].clamp(-0.3, 0.3) / 0.3
    scaled[..., 149] = observations[..., 149].clamp(-1.0, 1.0)
    scaled[..., 150] = observations[..., 150].clamp(-0.1, 0.1) / 0.1
    scaled[..., 151:155] = torch.tanh(observations[..., 151:155] / 5.0)
    scaled[..., 157] = observations[..., 157].clamp(-1.0, 1.0)
    scaled[..., 158] = observations[..., 158].clamp(-1.0, 1.0)
    scaled[..., 160] = observations[..., 160].clamp(0.0, 5.0) / 5.0
    return scaled


class _RecurrentHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(OBSERVATION_DIM)
        self.gru = nn.GRU(OBSERVATION_DIM, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim + OBSERVATION_DIM, output_dim)

    def _output(self, encoded: Tensor, scaled: Tensor) -> Tensor:
        return self.head(torch.cat((encoded, scaled), dim=-1))

    @staticmethod
    def _reset_hidden(
        hidden: Optional[Tensor],
        reset_mask: Tensor,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        if hidden is None:
            return None
        reset = reset_mask.to(dtype=dtype).view(1, -1, 1)
        return hidden * (1.0 - reset)

    def forward(
        self,
        observations: Tensor,
        reset_mask: Optional[Tensor] = None,
        hidden: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if observations.ndim != 3 or observations.shape[-1] != OBSERVATION_DIM:
            raise ValueError("observations must have shape [batch, time, 166]")
        scaled = _scale_observation_features(observations)
        normalized = self.input_norm(scaled)
        if reset_mask is None:
            encoded, output_hidden = self.gru(normalized, hidden)
            return self._output(encoded, scaled), output_hidden
        if reset_mask.shape != observations.shape[:2]:
            raise ValueError("reset_mask must match batch and time")
        outputs: list[Tensor] = []
        current_hidden = hidden
        for index in range(observations.shape[1]):
            current_hidden = self._reset_hidden(
                current_hidden,
                reset_mask[:, index],
                normalized.dtype,
            )
            encoded, current_hidden = self.gru(
                normalized[:, index : index + 1], current_hidden
            )
            outputs.append(
                self._output(encoded, scaled[:, index : index + 1])
            )
        if not outputs:
            raise ValueError("observations must contain a time step")
        return torch.cat(outputs, dim=1), current_hidden

    def forward_with_history(
        self,
        observations: Tensor,
        reset_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if observations.ndim != 3 or reset_mask.shape != observations.shape[:2]:
            raise ValueError("sequence shapes are incompatible")
        scaled = _scale_observation_features(observations)
        normalized = self.input_norm(scaled)
        outputs: list[Tensor] = []
        history: list[Tensor] = []
        hidden: Optional[Tensor] = None
        for index in range(observations.shape[1]):
            hidden = self._reset_hidden(hidden, reset_mask[:, index], normalized.dtype)
            encoded, hidden = self.gru(normalized[:, index : index + 1], hidden)
            outputs.append(
                self._output(encoded, scaled[:, index : index + 1])
            )
            history.append(hidden[0].unsqueeze(1))
        if not outputs:
            raise ValueError("observations must contain a time step")
        return torch.cat(outputs, dim=1), torch.cat(history, dim=1)

    def advance_from_history(
        self,
        observations: Tensor,
        history: Tensor,
        reset_mask: Tensor,
    ) -> Tensor:
        if (
            observations.ndim != 3
            or history.ndim != 3
            or observations.shape[:2] != history.shape[:2]
            or reset_mask.shape != observations.shape[:2]
        ):
            raise ValueError("next-observation history shapes are incompatible")
        scaled = _scale_observation_features(observations)
        normalized = self.input_norm(scaled)
        outputs: list[Tensor] = []
        for index in range(observations.shape[1]):
            hidden = history[:, index : index + 1].transpose(0, 1)
            hidden = self._reset_hidden(hidden, reset_mask[:, index], normalized.dtype)
            encoded, _ = self.gru(normalized[:, index : index + 1], hidden)
            outputs.append(
                self._output(encoded, scaled[:, index : index + 1])
            )
        if not outputs:
            raise ValueError("observations must contain a time step")
        return torch.cat(outputs, dim=1)


class RecurrentDiscreteSAC:
    """Twin-Q discrete SAC with GRU history and exact safety masking."""

    observation_dim = OBSERVATION_DIM
    action_dim = ACTION_COUNT
    observation_schema = LOCAL_WAYPOINT_OBSERVATION_SCHEMA_V3
    action_schema = ACTION_SCHEMA

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        learning_rate: float = 3e-4,
        seed: int = 0,
    ) -> None:
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        values = (gamma, tau, alpha_init, learning_rate)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("SAC configuration must be finite")
        if (
            not 0.0 < gamma <= 1.0
            or not 0.0 < tau <= 1.0
            or alpha_init <= 0.0
            or learning_rate <= 0.0
        ):
            raise ValueError("SAC configuration is outside bounds")
        torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.actor = _RecurrentHead(hidden_dim, ACTION_COUNT)
        self.critic1 = _RecurrentHead(hidden_dim, ACTION_COUNT)
        self.critic2 = _RecurrentHead(hidden_dim, ACTION_COUNT)
        self.target_critic1 = _RecurrentHead(hidden_dim, ACTION_COUNT)
        self.target_critic2 = _RecurrentHead(hidden_dim, ACTION_COUNT)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for parameter in (
            *self.target_critic1.parameters(),
            *self.target_critic2.parameters(),
        ):
            parameter.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=learning_rate
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=learning_rate
        )
        self.log_alpha = nn.Parameter(
            torch.tensor(alpha_init, dtype=torch.float32).log()
        )
        self.alpha_optimizer = torch.optim.Adam(
            (self.log_alpha,), lr=learning_rate
        )
        self.training_step = 0

    def parameters(self) -> Iterable[Tensor]:
        yield from self.actor.parameters()
        yield from self.critic1.parameters()
        yield from self.critic2.parameters()
        yield from self.target_critic1.parameters()
        yield from self.target_critic2.parameters()
        yield self.log_alpha

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp().clamp(1e-6, 100.0)

    @staticmethod
    def _masked_distribution(
        logits: Tensor,
        action_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if logits.shape != action_mask.shape:
            raise ValueError("logits and action mask must have the same shape")
        mask = action_mask.to(dtype=torch.bool)
        has_safe = mask.any(dim=-1)
        safe_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(safe_logits, dim=-1)
        probabilities = probabilities * mask.to(dtype=probabilities.dtype)
        probabilities = probabilities / probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        probabilities = torch.where(
            has_safe.unsqueeze(-1),
            probabilities,
            torch.zeros_like(probabilities),
        )
        log_probabilities = torch.where(
            probabilities > 0.0,
            torch.log(probabilities.clamp_min(1e-12)),
            torch.zeros_like(probabilities),
        )
        return probabilities, log_probabilities, has_safe

    def act(
        self,
        observation: LocalWaypointObservationV3,
        safe_action_mask: Sequence[bool],
        *,
        hidden: Optional[RecurrentHiddenState] = None,
        deterministic: bool = False,
    ) -> tuple[PolicyProposal, RecurrentHiddenState]:
        started = perf_counter()
        if not isinstance(observation, LocalWaypointObservationV3):
            raise ValueError("observation schema is incompatible with SAC")
        if type(deterministic) is not bool:
            raise ValueError("deterministic must be boolean")
        mask_values = tuple(safe_action_mask)
        if len(mask_values) != ACTION_COUNT or any(
            type(value) is not bool for value in mask_values
        ):
            raise ValueError("safe_action_mask must contain five booleans")
        if not any(mask_values):
            raise ValueError("policy requires at least one safe action")
        if hidden is not None and not isinstance(hidden, RecurrentHiddenState):
            raise ValueError("hidden state type is incompatible")
        recurrent = hidden or RecurrentHiddenState(None, None, None)
        vector = observation.tensor().view(1, 1, -1)
        mask = torch.tensor(mask_values, dtype=torch.bool).view(1, 1, -1)
        reset = torch.tensor([[observation.hidden_reset]], dtype=torch.bool)
        with torch.inference_mode():
            logits, actor_hidden = self.actor(vector, reset, recurrent.actor)
            q1, critic1_hidden = self.critic1(vector, reset, recurrent.critic1)
            q2, critic2_hidden = self.critic2(vector, reset, recurrent.critic2)
            probabilities = torch.softmax(logits[:, -1], dim=-1)
            masked, log_probs, _ = self._masked_distribution(logits, mask)
            masked = masked[:, -1]
            log_probs = log_probs[:, -1]
            if deterministic:
                action = int(torch.argmax(masked[0]).item())
            else:
                action = int(torch.multinomial(masked[0], 1).item())
            entropy = float(-(masked[0] * log_probs[0]).sum().item())
            _finite_tensor(probabilities, "policy probabilities")
            _finite_tensor(q1, "q1")
            _finite_tensor(q2, "q2")
        input_id = f"{observation.session_id}:{observation.stamp_sim:.6f}"
        return (
            PolicyProposal(
                probabilities=tuple(float(value) for value in probabilities[0]),
                masked_probabilities=tuple(float(value) for value in masked[0]),
                action=action,
                q1_values=tuple(float(value) for value in q1[0, -1]),
                q2_values=tuple(float(value) for value in q2[0, -1]),
                entropy=entropy,
                hidden_state_input_id=input_id,
                hidden_state_output_id=f"{input_id}:next",
                observation_schema=self.observation_schema,
                action_schema=self.action_schema,
                inference_ms=(perf_counter() - started) * 1000.0,
            ),
            RecurrentHiddenState(
                actor=actor_hidden.detach(),
                critic1=critic1_hidden.detach(),
                critic2=critic2_hidden.detach(),
            ),
        )

    def _validate_batch(self, batch: ReplaySequenceBatch) -> None:
        if not isinstance(batch, ReplaySequenceBatch):
            raise ValueError("batch type is invalid")
        if batch.observation_dim != OBSERVATION_DIM or batch.action_dim != ACTION_COUNT:
            raise ValueError("batch dimensions do not match SAC")
        shape = batch.observations.shape[:2]
        if batch.next_observations.shape != batch.observations.shape:
            raise ValueError("next observation shape is invalid")
        for name in (
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "learning_mask",
            "padding_mask",
            "hidden_reset",
            "next_hidden_reset",
        ):
            if getattr(batch, name).shape != shape:
                raise ValueError(f"{name} shape is invalid")
        for name in ("safe_action_mask", "next_safe_action_mask"):
            if getattr(batch, name).shape != (*shape, ACTION_COUNT):
                raise ValueError(f"{name} shape is invalid")
        for name in ("observations", "next_observations", "rewards", "learning_mask"):
            _finite_tensor(getattr(batch, name), name)
        if batch.actions.dtype != torch.long or torch.any(
            (batch.actions < 0) | (batch.actions >= ACTION_COUNT)
        ).item():
            raise ValueError("actions are invalid")
        for name in (
            "terminated",
            "truncated",
            "safe_action_mask",
            "next_safe_action_mask",
            "padding_mask",
            "hidden_reset",
            "next_hidden_reset",
        ):
            if getattr(batch, name).dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")
        if torch.any(batch.terminated & batch.truncated).item():
            raise ValueError("a sample cannot be terminated and truncated")

    def update(
        self,
        batch: ReplaySequenceBatch,
        *,
        demonstration: bool = False,
        expert_actions: Optional[Tensor] = None,
    ) -> dict[str, float | bool | int | str]:
        if type(demonstration) is not bool:
            raise ValueError("demonstration must be boolean")
        self._validate_batch(batch)
        if expert_actions is not None and (
            not isinstance(expert_actions, Tensor)
            or expert_actions.shape != batch.actions.shape
            or expert_actions.dtype != torch.long
            or torch.any(
                (expert_actions < 0) | (expert_actions >= ACTION_COUNT)
            ).item()
        ):
            raise ValueError("expert actions are invalid")
        modules = (
            self.actor,
            self.critic1,
            self.critic2,
            self.target_critic1,
            self.target_critic2,
        )
        _finite_modules(modules)
        learning = batch.learning_mask > 0.0
        current_has_safe = batch.safe_action_mask.any(dim=-1)
        action_is_safe = batch.safe_action_mask.gather(
            -1, batch.actions.unsqueeze(-1)
        ).squeeze(-1)
        critic_mask = learning & ~batch.padding_mask & current_has_safe & action_is_safe
        if not critic_mask.any().item():
            return {
                "updated": False,
                "reason": "NO_EXECUTED_ACTIONS",
                "critic_samples": 0,
                "bootstrap_count": 0,
            }
        logits, actor_history = self.actor.forward_with_history(
            batch.observations, batch.hidden_reset
        )
        q1, _ = self.critic1(batch.observations, batch.hidden_reset)
        q2, _ = self.critic2(batch.observations, batch.hidden_reset)
        with torch.no_grad():
            next_logits = self.actor.advance_from_history(
                batch.next_observations,
                actor_history,
                batch.next_hidden_reset,
            )
            _, target1_history = self.target_critic1.forward_with_history(
                batch.observations, batch.hidden_reset
            )
            _, target2_history = self.target_critic2.forward_with_history(
                batch.observations, batch.hidden_reset
            )
            next_q1 = self.target_critic1.advance_from_history(
                batch.next_observations,
                target1_history,
                batch.next_hidden_reset,
            )
            next_q2 = self.target_critic2.advance_from_history(
                batch.next_observations,
                target2_history,
                batch.next_hidden_reset,
            )
        current_prob, current_log_prob, _ = self._masked_distribution(
            logits, batch.safe_action_mask
        )
        next_prob, next_log_prob, next_has_safe = self._masked_distribution(
            next_logits, batch.next_safe_action_mask
        )
        q1_taken = q1.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
        q2_taken = q2.gather(-1, batch.actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_min_q = torch.minimum(next_q1, next_q2)
            next_value = (
                next_prob * (next_min_q - self.alpha.detach() * next_log_prob)
            ).sum(dim=-1)
            bootstrap = ~batch.terminated & ~batch.padding_mask & next_has_safe
            target = batch.rewards + self.gamma * bootstrap.to(torch.float32) * next_value
        critic_loss = F.mse_loss(
            q1_taken[critic_mask], target[critic_mask]
        ) + F.mse_loss(q2_taken[critic_mask], target[critic_mask])
        actor_mask = learning & ~batch.padding_mask & current_has_safe
        current_min_q = torch.minimum(q1, q2).detach()
        actor_terms = (
            current_prob * (self.alpha.detach() * current_log_prob - current_min_q)
        ).sum(dim=-1)
        behavior_actions = (
            batch.actions if expert_actions is None else expert_actions
        )
        if expert_actions is not None:
            expert_is_safe = batch.safe_action_mask.gather(
                -1,
                expert_actions.unsqueeze(-1),
            ).squeeze(-1)
            if torch.any(actor_mask & ~expert_is_safe).item():
                raise ValueError("expert action must be safe")
        behavior_mask = (
            actor_mask
            if demonstration or expert_actions is not None
            else torch.zeros_like(actor_mask)
        )
        has_behavior_samples = behavior_mask.any().item()
        behavior_clone_loss = logits.sum() * 0.0
        if has_behavior_samples:
            behavior_clone_sample_loss = -current_log_prob.gather(
                -1, behavior_actions.unsqueeze(-1)
            ).squeeze(-1)[behavior_mask]
            demonstration_actions = behavior_actions[behavior_mask]
            action_counts = torch.bincount(
                demonstration_actions,
                minlength=ACTION_COUNT,
            ).to(dtype=behavior_clone_sample_loss.dtype)
            present_actions = (action_counts > 0.0).sum().clamp_min(1)
            class_weights = demonstration_actions.numel() / (
                present_actions * action_counts.clamp_min(1.0)
            )
            action_sample_weights = class_weights.gather(
                0,
                demonstration_actions,
            )

            def clone_loss(sample_loss: Tensor) -> Tensor:
                if expert_actions is not None:
                    return sample_loss.mean()
                balanced_loss = (sample_loss * action_sample_weights).mean()
                return 0.5 * (sample_loss.mean() + balanced_loss)

            sequence_behavior_clone_loss = clone_loss(
                behavior_clone_sample_loss
            )
            reset_samples = batch.hidden_reset[behavior_mask]
            behavior_clone_loss = sequence_behavior_clone_loss
            if reset_samples.any().item():
                behavior_clone_loss = (
                    0.75 * sequence_behavior_clone_loss
                    + 0.25 * behavior_clone_sample_loss[reset_samples].mean()
                )
            stateless_observations = batch.observations.reshape(
                -1,
                1,
                OBSERVATION_DIM,
            )
            stateless_reset = torch.ones(
                stateless_observations.shape[:2],
                dtype=torch.bool,
                device=stateless_observations.device,
            )
            stateless_logits, _ = self.actor(
                stateless_observations,
                stateless_reset,
            )
            _, stateless_log_prob, _ = self._masked_distribution(
                stateless_logits,
                batch.safe_action_mask.reshape(-1, 1, ACTION_COUNT),
            )
            stateless_log_prob = stateless_log_prob.reshape(
                *batch.actions.shape,
                ACTION_COUNT,
            )
            stateless_sample_loss = -stateless_log_prob.gather(
                -1,
                behavior_actions.unsqueeze(-1),
            ).squeeze(-1)[behavior_mask]
            behavior_clone_loss = 0.5 * (
                behavior_clone_loss
                + clone_loss(stateless_sample_loss)
            )
        if demonstration:
            actor_loss = behavior_clone_loss
            actor_objective = "BEHAVIOR_CLONING"
        elif expert_actions is not None:
            q_scale = current_min_q[actor_mask].abs().mean().clamp_min(1.0)
            sac_weight = (
                (0.25 - behavior_clone_loss.detach()) / 0.25
            ).clamp(0.0, 1.0)
            actor_loss = (
                sac_weight * actor_terms[actor_mask].mean() / q_scale
                + behavior_clone_loss
            )
            actor_objective = "SAC_WITH_DAGGER"
        else:
            actor_loss = actor_terms[actor_mask].mean()
            actor_objective = "SOFT_ACTOR_CRITIC"
        entropy = -(current_prob * current_log_prob).sum(dim=-1)[actor_mask].mean()
        safe_count = batch.safe_action_mask.to(torch.float32).sum(dim=-1).clamp_min(1.0)
        desired_entropy = (0.5 * torch.log(safe_count))[actor_mask].mean().detach()
        alpha_loss = self.log_alpha * (entropy.detach() - desired_entropy)
        for loss, name in (
            (critic_loss, "critic_loss"),
            (actor_loss, "actor_loss"),
            (alpha_loss, "alpha_loss"),
        ):
            _finite_tensor(loss.detach(), name)

        before = self._snapshot_training_state()
        try:
            self.critic1_optimizer.zero_grad(set_to_none=True)
            self.critic2_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self._finite_gradients((self.critic1, self.critic2), "critic gradients")
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._finite_gradients((self.actor,), "actor gradients")
            self.actor_optimizer.step()
            if not demonstration and expert_actions is None:
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self._finite_gradients((self,), "temperature gradients")
                self.alpha_optimizer.step()
            self._soft_update(self.target_critic1, self.critic1, self.tau)
            self._soft_update(self.target_critic2, self.critic2, self.tau)
            _finite_modules(modules)
            _finite_tensor(self.log_alpha.detach(), "log_alpha")
            self._finite_optimizer_states()
        except Exception:
            self._restore_training_state(before)
            raise
        self.training_step += 1
        return {
            "updated": True,
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "actor_objective": actor_objective,
            "behavior_clone_loss": float(behavior_clone_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "entropy": float(entropy.detach().item()),
            "critic_samples": int(critic_mask.sum().item()),
            "bootstrap_count": int(bootstrap[critic_mask].sum().item()),
        }

    @staticmethod
    def _finite_gradients(
        modules: Iterable[nn.Module] | tuple["RecurrentDiscreteSAC", ...],
        name: str,
    ) -> None:
        if isinstance(modules, tuple) and modules and isinstance(
            modules[0], RecurrentDiscreteSAC
        ):
            parameters = (modules[0].log_alpha,)
        else:
            parameters = tuple(
                parameter for module in modules for parameter in module.parameters()
            )
        for parameter in parameters:
            if parameter.grad is not None:
                _finite_tensor(parameter.grad.detach(), name)

    def _snapshot_training_state(self) -> dict[str, object]:
        return {
            "actor": deepcopy(self.actor.state_dict()),
            "critic1": deepcopy(self.critic1.state_dict()),
            "critic2": deepcopy(self.critic2.state_dict()),
            "target_critic1": deepcopy(self.target_critic1.state_dict()),
            "target_critic2": deepcopy(self.target_critic2.state_dict()),
            "actor_optimizer": deepcopy(self.actor_optimizer.state_dict()),
            "critic1_optimizer": deepcopy(self.critic1_optimizer.state_dict()),
            "critic2_optimizer": deepcopy(self.critic2_optimizer.state_dict()),
            "alpha_optimizer": deepcopy(self.alpha_optimizer.state_dict()),
            "log_alpha": self.log_alpha.detach().clone(),
            "training_step": self.training_step,
            "torch_rng_state": torch.get_rng_state().clone(),
        }

    def training_state_dict(self) -> dict[str, object]:
        state = self._snapshot_training_state()
        _finite_value_tree(state, "training state")
        return state

    def load_training_state_dict(self, state: Mapping[str, object]) -> None:
        required = {
            "actor",
            "critic1",
            "critic2",
            "target_critic1",
            "target_critic2",
            "actor_optimizer",
            "critic1_optimizer",
            "critic2_optimizer",
            "alpha_optimizer",
            "log_alpha",
            "training_step",
            "torch_rng_state",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("training state is incomplete")
        step = state.get("training_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("training step is invalid")
        rng_state = state.get("torch_rng_state")
        if (
            not isinstance(rng_state, Tensor)
            or rng_state.dtype != torch.uint8
            or rng_state.ndim != 1
        ):
            raise ValueError("training random state is invalid")
        _finite_value_tree(state, "training state")
        before = self._snapshot_training_state()
        try:
            self._restore_training_state(deepcopy(dict(state)))
            self._finite_optimizer_states()
        except Exception:
            self._restore_training_state(before)
            raise

    def _restore_training_state(self, state: Mapping[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target_critic1.load_state_dict(state["target_critic1"])
        self.target_critic2.load_state_dict(state["target_critic2"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.actor_optimizer.defaults["lr"]
        self.critic1_optimizer.load_state_dict(state["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(state["critic2_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.training_step = int(state["training_step"])
        torch.set_rng_state(state["torch_rng_state"].cpu())

    def _finite_optimizer_states(self) -> None:
        for name, optimizer in (
            ("actor optimizer", self.actor_optimizer),
            ("critic1 optimizer", self.critic1_optimizer),
            ("critic2 optimizer", self.critic2_optimizer),
            ("alpha optimizer", self.alpha_optimizer),
        ):
            _finite_value_tree(optimizer.state_dict(), name)

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                target.parameters(), source.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(tau * source_parameter)

    def save_checkpoint(self, path: str | Path) -> Path:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "observation_schema": self.observation_schema,
            "action_schema": self.action_schema,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_COUNT,
            "hidden_dim": self.hidden_dim,
            "training_state": self.training_state_dict(),
        }
        with target.open("xb") as handle:
            torch.save(payload, handle)
        return target

    def load_checkpoint(self, path: str | Path) -> Path:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("checkpoint cannot be safely loaded") from exc
        required = {
            "format",
            "observation_schema",
            "action_schema",
            "observation_dim",
            "action_dim",
            "hidden_dim",
            "training_state",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("checkpoint schema is incompatible")
        expected = {
            "format": CHECKPOINT_FORMAT,
            "observation_schema": self.observation_schema,
            "action_schema": self.action_schema,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_COUNT,
            "hidden_dim": self.hidden_dim,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint schema is incompatible")
        self.load_training_state_dict(payload["training_state"])
        return source


__all__ = [
    "ACTION_SCHEMA",
    "CHECKPOINT_FORMAT",
    "LOCAL_WAYPOINT_OBSERVATION_SCHEMA_V3",
    "LocalWaypointObservationV3",
    "PolicyProposal",
    "REPLAY_SCHEMA_V3",
    "RecurrentDiscreteSAC",
    "RecurrentHiddenState",
    "ReplaySequenceBatch",
    "SequenceReplay",
    "SequenceTransition",
]
