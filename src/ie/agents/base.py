"""Agent protocol definition."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Agent(Protocol):
    """Structural type for any agent usable in evaluation."""

    def act(self, observation: np.ndarray) -> int: ...
