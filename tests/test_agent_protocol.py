"""Tests that all agents satisfy the Agent protocol."""

import numpy as np

from ie.agents.base import Agent
from ie.agents.curiosity_agent import CuriosityAgent
from ie.agents.random_agent import RandomAgent
from ie.agents.reward_maximizer import RewardMaximizer


def _check_is_agent(agent: Agent) -> None:
    """Type-check helper — if this compiles, the agent satisfies the Protocol."""
    obs = np.zeros(10, dtype=np.float32)
    action = agent.act(obs)
    assert isinstance(action, int)


def test_random_agent_satisfies_protocol():
    agent = RandomAgent(action_space_size=6, seed=42)
    _check_is_agent(agent)


def test_reward_maximizer_satisfies_protocol():
    agent = RewardMaximizer(obs_size=10, action_size=6, seed=42)
    _check_is_agent(agent)


def test_curiosity_agent_satisfies_protocol():
    agent = CuriosityAgent(obs_size=10, action_size=6, seed=42)
    _check_is_agent(agent)
