"""Tests for reward maximizer and curiosity agents."""

import numpy as np

from ie.agents.curiosity_agent import CuriosityAgent
from ie.agents.reward_maximizer import RewardMaximizer
from ie.environments.resource_grid import ResourceGrid
from ie.environments.sequence_sorting import SequenceSorting


class TestRewardMaximizer:
    def test_trains_without_error(self):
        env = ResourceGrid(seed=42)
        agent = RewardMaximizer(
            obs_size=env.observation_shape[0],
            action_size=env.action_space_size,
            seed=42,
        )
        for _ in range(3):
            stats = agent.train_episode(env, config={"name": "basic"})
            assert "total_reward" in stats
            assert "episode_length" in stats
            assert "total_entropy_change" in stats
            assert "total_work" in stats
            assert stats["episode_length"] > 0

    def test_acts_in_valid_range(self):
        agent = RewardMaximizer(obs_size=28, action_size=6, seed=42)
        obs = np.zeros(28, dtype=np.float32)
        for _ in range(50):
            assert 0 <= agent.act(obs) < 6

    def test_works_with_sequence_sorting(self):
        env = SequenceSorting(seed=42)
        agent = RewardMaximizer(
            obs_size=env.observation_shape[0],
            action_size=env.action_space_size,
            seed=42,
        )
        stats = agent.train_episode(env, config={"name": "short"})
        assert stats["episode_length"] > 0


class TestCuriosityAgent:
    def test_trains_without_error(self):
        env = ResourceGrid(seed=42)
        agent = CuriosityAgent(
            obs_size=env.observation_shape[0],
            action_size=env.action_space_size,
            seed=42,
        )
        for _ in range(3):
            stats = agent.train_episode(env, config={"name": "basic"})
            assert "total_reward" in stats
            assert stats["episode_length"] > 0

    def test_acts_in_valid_range(self):
        agent = CuriosityAgent(obs_size=28, action_size=6, seed=42)
        obs = np.zeros(28, dtype=np.float32)
        for _ in range(50):
            assert 0 <= agent.act(obs) < 6

    def test_works_with_sequence_sorting(self):
        env = SequenceSorting(seed=42)
        agent = CuriosityAgent(
            obs_size=env.observation_shape[0],
            action_size=env.action_space_size,
            seed=42,
        )
        stats = agent.train_episode(env, config={"name": "short"})
        assert stats["episode_length"] > 0
