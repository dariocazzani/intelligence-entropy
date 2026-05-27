"""Shared training episode loop for PPO-based agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ie.agents.ppo import PPOAgent
from ie.environments.base import BaseEnvironment

RewardFn = Callable[[np.ndarray, int, np.ndarray, float], float]


@dataclass
class EpisodeResult:
    """Result of a training episode, including stats and raw buffers."""

    stats: dict[str, float]
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    next_observations: list[np.ndarray] = field(default_factory=list)


def run_training_episode(
    ppo: PPOAgent,
    env: BaseEnvironment,
    reward_fn: RewardFn,
    *,
    config: dict | None = None,
) -> EpisodeResult:
    """Run one training episode, collecting experience and updating the PPO agent.

    Args:
        ppo: The PPO agent to train.
        env: Environment instance.
        reward_fn: Callable(obs, action, obs_next, delta_entropy) -> float.
        config: Optional environment config dict.
    """
    obs = env.reset(config)
    prev_entropy = env.compute_entropy()

    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    next_observations: list[np.ndarray] = []
    total_reward = 0.0
    total_work = 0.0
    total_entropy_change = 0.0
    done = False

    while not done:
        action = ppo.act(obs)
        obs_next, _, done, info = env.step(action)

        curr_entropy = env.compute_entropy()
        delta_entropy = curr_entropy - prev_entropy
        reward = reward_fn(obs, action, obs_next, delta_entropy)

        observations.append(obs.astype(np.float32))
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        next_observations.append(obs_next.astype(np.float32))

        total_reward += reward
        total_work += info["work"]
        total_entropy_change += delta_entropy

        obs = obs_next
        prev_entropy = curr_entropy

    ppo.update(observations, actions, rewards, dones)

    return EpisodeResult(
        stats={
            "total_reward": total_reward,
            "episode_length": len(observations),
            "total_entropy_change": total_entropy_change,
            "total_work": total_work,
        },
        observations=observations,
        actions=actions,
        next_observations=next_observations,
    )
