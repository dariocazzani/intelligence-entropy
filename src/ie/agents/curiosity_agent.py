"""Curiosity-driven agent using prediction error as intrinsic reward."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ie.agents.ppo import PPOAgent
from ie.agents.training import run_training_episode
from ie.environments.base import BaseEnvironment


class ForwardDynamicsModel(nn.Module):
    """Predicts next observation given current obs and action."""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        # One-hot encode action, concatenate with obs
        self.net = nn.Sequential(
            nn.Linear(obs_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, obs_size),
        )
        self.action_size = action_size

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # One-hot encode action
        action_onehot = torch.zeros(obs.shape[0], self.action_size)
        action_onehot.scatter_(1, action.unsqueeze(1), 1.0)
        x = torch.cat([obs, action_onehot], dim=-1)
        result: torch.Tensor = self.net(x)
        return result


class CuriosityAgent:
    """Agent with intrinsic curiosity reward from prediction error."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        curiosity_coeff: float = 1.0,
        dynamics_lr: float = 1e-3,
        **ppo_kwargs: Any,
    ) -> None:
        self._ppo = PPOAgent(obs_size=obs_size, action_size=action_size, **ppo_kwargs)
        hidden_size = ppo_kwargs.get("hidden_size", 64)
        self._dynamics = ForwardDynamicsModel(obs_size, action_size, hidden_size)
        self._dynamics_optimizer = optim.Adam(self._dynamics.parameters(), lr=dynamics_lr)
        self._curiosity_coeff = curiosity_coeff
        self._obs_size = obs_size
        self._action_size = action_size

    def act(self, observation: np.ndarray) -> int:
        return self._ppo.act(observation)

    def act_deterministic(self, observation: np.ndarray) -> int:
        return self._ppo.act_deterministic(observation)

    def train_episode(self, env: BaseEnvironment, config: dict | None = None) -> dict[str, float]:
        """Run one episode with curiosity-based intrinsic reward."""

        def reward_fn(
            obs: np.ndarray, action: int, obs_next: np.ndarray, delta_entropy: float
        ) -> float:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                action_t = torch.tensor([action], dtype=torch.long)
                predicted = self._dynamics(obs_t, action_t)
                actual = torch.from_numpy(obs_next.astype(np.float32)).unsqueeze(0)
                return self._curiosity_coeff * nn.functional.mse_loss(predicted, actual).item()

        result = run_training_episode(self._ppo, env, reward_fn, config=config)
        self._update_dynamics(result.observations, result.actions, result.next_observations)
        return result.stats

    def _update_dynamics(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        next_observations: list[np.ndarray],
    ) -> None:
        """Train the forward dynamics model on episode data."""
        obs_t = torch.from_numpy(np.array(observations, dtype=np.float32))
        action_t = torch.tensor(actions, dtype=torch.long)
        next_obs_t = torch.from_numpy(np.array(next_observations, dtype=np.float32))

        predicted = self._dynamics(obs_t, action_t)
        loss = nn.functional.mse_loss(predicted, next_obs_t)

        self._dynamics_optimizer.zero_grad()
        loss.backward()
        self._dynamics_optimizer.step()

    def save(self, path: str) -> None:
        self._ppo.save(path)
        # Also save dynamics model alongside
        dynamics_path = Path(path).with_name(Path(path).stem + "_dynamics.pt")
        torch.save(self._dynamics.state_dict(), str(dynamics_path))

    def load(self, path: str) -> None:
        self._ppo.load(path)
        dynamics_path = Path(path).with_name(Path(path).stem + "_dynamics.pt")
        if dynamics_path.exists():
            self._dynamics.load_state_dict(torch.load(str(dynamics_path), weights_only=True))
