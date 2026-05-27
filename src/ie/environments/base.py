"""Abstract base class for environments that support I_cd evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEnvironment(ABC):
    """Abstract interface for environments that support I_cd evaluation.

    Any environment used in this project must implement this interface.
    The interface is designed around the quantities needed to compute
    I_cd = (-delta_S / W) x H(C):

    - compute_entropy() provides S(E) for computing delta_S
    - step() returns work W via info['work']
    - get_context_label() provides context labels for H(C)
    """

    @abstractmethod
    def reset(self, config: dict | None = None) -> np.ndarray:
        """Reset environment to initial state.

        Args:
            config: Optional configuration dict for the environment.
                    Used to set up different contexts (e.g., number of
                    objects, grid size, perturbation settings).

        Returns:
            Initial observation as numpy array.
        """

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one action in the environment.

        Args:
            action: Integer action index.

        Returns:
            Tuple of (observation, reward, done, info) where:
            - observation: numpy array of the new state
            - reward: scalar reward (may be unused; agents have their own objectives)
            - done: whether the episode has ended
            - info: dict that MUST contain 'work' key (float >= 0) representing
                    the energy cost of this action
        """

    @abstractmethod
    def compute_entropy(self) -> float:
        """Compute Shannon entropy of the current environmental state.

        This is the domain-specific entropy definition. For a grid world,
        this might be entropy over the (position, type) joint distribution.
        For a sequence, it might be entropy over displacement distances.

        The definition must be:
        1. Precise enough that two implementations compute the same number
        2. Capturing the intuitive notion of "order" for this domain

        Returns:
            Shannon entropy S(E) in nats (natural log).
        """

    @abstractmethod
    def get_context_label(self) -> int:
        """Return the current context label for H(C) computation.

        A context label is an integer identifying the environment
        configuration or setting. Different contexts represent meaningfully
        different situations the agent must handle.

        For example, in a grid world:
        - Context 0: basic (single object type)
        - Context 1: multi-type (sorting required)
        - Context 2: perturbation (mid-episode disruption)

        Returns:
            Integer context label.
        """

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Number of possible discrete actions."""

    @property
    @abstractmethod
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of the observation array returned by reset() and step()."""
