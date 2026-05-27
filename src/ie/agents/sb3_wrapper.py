"""Wrappers for Stable Baselines3 integration.

SB3Agent wraps a trained SB3 model to satisfy the Agent protocol (act(obs) -> int).
BaseEnvGymnasiumWrapper wraps any BaseEnvironment as a gymnasium.Env for SB3 training.
"""

from __future__ import annotations

from collections.abc import Callable

import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from ie.environments.base import BaseEnvironment

RewardFn = Callable[[np.ndarray, int, np.ndarray, float], float]


class SB3Agent:
    """Wrap a trained SB3 model to comply with the Agent protocol."""

    def __init__(self, model: PPO, *, deterministic: bool = True) -> None:
        self._model = model
        self._deterministic = deterministic

    def act(self, observation: np.ndarray) -> int:
        action, _ = self._model.predict(observation, deterministic=self._deterministic)
        return int(action)


class BaseEnvGymnasiumWrapper(gymnasium.Env):
    """Wrap a BaseEnvironment as a gymnasium.Env for SB3 training.

    Handles the API translation between our 4-tuple step and gymnasium's 5-tuple,
    tracks entropy for custom reward computation, and cycles through configs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: BaseEnvironment,
        configs: list[dict],
        reward_fn: RewardFn,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._env = env
        self._configs = configs
        self._reward_fn = reward_fn
        self._config_idx = 0
        self._rng = np.random.default_rng(seed)
        self._prev_entropy = 0.0
        self._prev_obs: np.ndarray = np.zeros(env.observation_shape, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(env.action_space_size)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        config = self._configs[self._config_idx % len(self._configs)]
        self._config_idx += 1

        obs = self._env.reset(config)
        self._prev_entropy = self._env.compute_entropy()
        self._prev_obs = obs.astype(np.float32)
        return self._prev_obs.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs_prev_entropy = self._prev_entropy
        obs_next, _env_reward, done, info = self._env.step(int(action))

        curr_entropy = self._env.compute_entropy()
        delta_entropy = curr_entropy - obs_prev_entropy

        reward = self._reward_fn(
            self._prev_obs,
            int(action),
            obs_next.astype(np.float32),
            delta_entropy,
        )

        self._prev_entropy = curr_entropy
        obs_out = obs_next.astype(np.float32)
        self._prev_obs = obs_out
        return obs_out, reward, False, done, info
