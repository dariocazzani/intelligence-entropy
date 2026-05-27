"""Tests for agent implementations."""

import numpy as np

from ie.agents.random_agent import RandomAgent


class TestRandomAgent:
    def test_returns_valid_actions(self):
        agent = RandomAgent(action_space_size=6, seed=42)
        obs = np.zeros(28)
        for _ in range(100):
            action = agent.act(obs)
            assert 0 <= action < 6

    def test_covers_action_space(self):
        agent = RandomAgent(action_space_size=6, seed=42)
        obs = np.zeros(28)
        actions = {agent.act(obs) for _ in range(1000)}
        assert actions == {0, 1, 2, 3, 4, 5}

    def test_seeded_reproducibility(self):
        obs = np.zeros(28)
        agent1 = RandomAgent(action_space_size=6, seed=42)
        agent2 = RandomAgent(action_space_size=6, seed=42)
        actions1 = [agent1.act(obs) for _ in range(50)]
        actions2 = [agent2.act(obs) for _ in range(50)]
        assert actions1 == actions2
