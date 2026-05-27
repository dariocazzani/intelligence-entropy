"""Minimal PPO implementation for training agents in I_cd environments."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Simple MLP policy: obs -> action logits."""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


class ValueNetwork(nn.Module):
    """Simple MLP value function: obs -> scalar value."""

    def __init__(self, obs_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.net(x)
        return result


class PPOAgent:
    """Minimal PPO agent for discrete action spaces."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        hidden_size: int = 64,
        seed: int | None = None,
    ):
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs

        if seed is not None:
            torch.manual_seed(seed)

        self.policy = PolicyNetwork(obs_size, action_size, hidden_size)
        self.value = ValueNetwork(obs_size, hidden_size)

        # Single optimizer for both networks
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )

    def act(self, observation: np.ndarray) -> int:
        """Sample action from current policy."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
            logits = self.policy(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
        return int(action.item())

    def act_deterministic(self, observation: np.ndarray) -> int:
        """Select highest probability action."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)
            logits = self.policy(obs_tensor)
            action = torch.argmax(logits, dim=-1)
        return int(action.item())

    def update(
        self,
        observations: list[np.ndarray],
        actions: list[int],
        rewards: list[float],
        dones: list[bool],
    ) -> None:
        """Run PPO update on a batch of experience.

        Args:
            observations: list of numpy arrays (T observations)
            actions: list of ints (T actions)
            rewards: list of floats (T rewards)
            dones: list of bools (T done flags)

        Computes GAE advantages, then does multiple epochs of clipped PPO updates.
        """
        # Convert to tensors
        obs_tensor = torch.from_numpy(np.array(observations, dtype=np.float32))
        actions_tensor = torch.tensor(actions, dtype=torch.long)

        # Compute discounted returns
        returns = []
        g = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                g = 0.0
            g = reward + self.gamma * g
            returns.append(g)
        returns.reverse()
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # Get values for advantages
        with torch.no_grad():
            values = self.value(obs_tensor).squeeze(-1)
            advantages = returns_tensor - values

            # Get old log probs
            logits = self.policy(obs_tensor)
            dist = Categorical(logits=logits)
            old_log_probs = dist.log_prob(actions_tensor)

        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Get new log probs
            logits = self.policy(obs_tensor)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_tensor)

            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # Value loss
            values = self.value(obs_tensor).squeeze(-1)
            value_loss = nn.functional.mse_loss(values, returns_tensor)

            # Combined loss
            loss = policy_loss + 0.5 * value_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, path: str) -> None:
        """Save policy and value network weights."""
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load policy and value network weights."""
        checkpoint = torch.load(path, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value.load_state_dict(checkpoint["value"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
