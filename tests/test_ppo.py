"""Tests for PPO infrastructure."""

import os
import tempfile

import numpy as np
import torch

from ie.agents.ppo import PolicyNetwork, PPOAgent, ValueNetwork


class TestNetworks:
    def test_policy_network_output_shape(self):
        net = PolicyNetwork(obs_size=28, action_size=6)
        x = torch.randn(1, 28)
        out = net(x)
        assert out.shape == (1, 6)

    def test_value_network_output_shape(self):
        net = ValueNetwork(obs_size=28)
        x = torch.randn(1, 28)
        out = net(x)
        assert out.shape == (1, 1)

    def test_policy_batch(self):
        net = PolicyNetwork(obs_size=10, action_size=4)
        x = torch.randn(32, 10)
        out = net(x)
        assert out.shape == (32, 4)


class TestPPOAgent:
    def test_act_returns_valid_action(self):
        agent = PPOAgent(obs_size=28, action_size=6, seed=42)
        obs = np.zeros(28, dtype=np.float32)
        for _ in range(50):
            action = agent.act(obs)
            assert 0 <= action < 6

    def test_act_deterministic(self):
        agent = PPOAgent(obs_size=28, action_size=6, seed=42)
        obs = np.zeros(28, dtype=np.float32)
        a1 = agent.act_deterministic(obs)
        a2 = agent.act_deterministic(obs)
        assert a1 == a2  # deterministic should give same result

    def test_update_runs_without_error(self):
        agent = PPOAgent(obs_size=10, action_size=4, seed=42)
        # Simulate a short episode
        observations = [np.random.randn(10).astype(np.float32) for _ in range(20)]
        actions = [np.random.randint(0, 4) for _ in range(20)]
        rewards = [np.random.randn() for _ in range(20)]
        dones = [False] * 19 + [True]
        agent.update(observations, actions, rewards, dones)

    def test_save_load_preserves_behavior(self):
        agent = PPOAgent(obs_size=10, action_size=4, seed=42)
        obs = np.random.randn(10).astype(np.float32)
        action_before = agent.act_deterministic(obs)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            agent.save(path)

            agent2 = PPOAgent(obs_size=10, action_size=4, seed=99)
            agent2.load(path)
            action_after = agent2.act_deterministic(obs)

        assert action_before == action_after
