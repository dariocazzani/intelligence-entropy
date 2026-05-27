"""Random baseline agent."""

from __future__ import annotations

import numpy as np


class RandomAgent:
    """Agent that selects actions uniformly at random."""

    def __init__(self, action_space_size: int, seed: int | None = None):
        self.action_space_size = action_space_size
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        """Select a random action."""
        return int(self._rng.integers(0, self.action_space_size))

    def reset(self) -> None:
        """No state to reset."""
        pass
