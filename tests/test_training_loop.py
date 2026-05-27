"""Tests for the shared training loop."""

from __future__ import annotations

import numpy as np

from ie.agents.ppo import PPOAgent
from ie.agents.training import EpisodeResult, run_training_episode
from ie.environments.resource_grid import ResourceGrid


def test_returns_episode_result():
    env = ResourceGrid(seed=42)
    ppo = PPOAgent(obs_size=env.observation_shape[0], action_size=env.action_space_size, seed=42)

    def reward_fn(obs, action, obs_next, delta_entropy):
        return -0.1

    result = run_training_episode(ppo, env, reward_fn, config={"name": "basic"})
    assert isinstance(result, EpisodeResult)
    assert "total_reward" in result.stats
    assert "episode_length" in result.stats
    assert "total_entropy_change" in result.stats
    assert "total_work" in result.stats
    assert result.stats["episode_length"] == 300


def test_reward_fn_receives_delta_entropy():
    env = ResourceGrid(seed=42)
    ppo = PPOAgent(obs_size=env.observation_shape[0], action_size=env.action_space_size, seed=42)
    received_deltas = []

    def reward_fn(obs, action, obs_next, delta_entropy):
        received_deltas.append(delta_entropy)
        return 0.0

    run_training_episode(ppo, env, reward_fn, config={"name": "basic"})
    assert len(received_deltas) == 300
    assert all(isinstance(d, float) for d in received_deltas)


def test_uses_reward_fn():
    env = ResourceGrid(seed=42)
    ppo = PPOAgent(obs_size=env.observation_shape[0], action_size=env.action_space_size, seed=42)

    def constant_reward(obs, action, obs_next, delta_entropy):
        return 5.0

    result = run_training_episode(ppo, env, constant_reward, config={"name": "basic"})
    assert np.isclose(result.stats["total_reward"], 1500.0)


def test_episode_buffers_populated():
    env = ResourceGrid(seed=42)
    ppo = PPOAgent(obs_size=env.observation_shape[0], action_size=env.action_space_size, seed=42)

    def reward_fn(obs, action, obs_next, delta_entropy):
        return 0.0

    result = run_training_episode(ppo, env, reward_fn, config={"name": "basic"})
    assert len(result.observations) == 300
    assert len(result.actions) == 300
    assert len(result.next_observations) == 300
    assert result.observations[0].shape == (env.observation_shape[0],)
