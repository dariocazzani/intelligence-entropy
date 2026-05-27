"""Reward-maximizing agent using hand-crafted entropy-reduction reward."""

from __future__ import annotations

from typing import Any

import numpy as np

from ie.agents.ppo import PPOAgent
from ie.agents.training import run_training_episode
from ie.environments.base import BaseEnvironment


class RewardMaximizer:
    """Agent trained with reward r = -delta_entropy per step.

    Trained with a standard task reward, NOT the I_cd measure.
    """

    def __init__(self, obs_size: int, action_size: int, **ppo_kwargs: Any) -> None:
        self._ppo = PPOAgent(obs_size=obs_size, action_size=action_size, **ppo_kwargs)

    def act(self, observation: np.ndarray) -> int:
        return self._ppo.act(observation)

    def act_deterministic(self, observation: np.ndarray) -> int:
        return self._ppo.act_deterministic(observation)

    def train_episode(self, env: BaseEnvironment, config: dict | None = None) -> dict[str, float]:
        """Run one episode, using -delta_entropy as reward."""

        def reward_fn(
            obs: np.ndarray, action: int, obs_next: np.ndarray, delta_entropy: float
        ) -> float:
            return -delta_entropy

        return run_training_episode(self._ppo, env, reward_fn, config=config).stats

    def save(self, path: str) -> None:
        self._ppo.save(path)

    def load(self, path: str) -> None:
        self._ppo.load(path)
