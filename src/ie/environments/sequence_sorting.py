"""Sequence Sorting environment for I_cd evaluation.

A 1D sequence sorting environment where an agent sorts elements by swapping
adjacent pairs. Environmental entropy is measured as Shannon entropy over the
distribution of displacement distances.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from ie.environments.base import BaseEnvironment
from ie.measure import environmental_entropy


class SequenceSorting(BaseEnvironment):
    """1D sequence sorting environment where agent sorts elements by swapping.

    The agent can swap adjacent elements or perform a no-op. The goal is to
    reduce environmental entropy by sorting the sequence into ascending order.

    Actions:
        0..N-2: swap element at position i with element at position i+1
        N-1: no-op (do nothing)

    Observation:
        1D numpy array containing the current sequence as float32 values.
        Always fixed size (MAX_LENGTH) with zero-padding for shorter sequences.

    Environmental Entropy:
        Entropy over the distribution of displacement distances. For each element
        at position i, compute displacement = |i - sorted_position_i| where
        sorted_position_i is where element i should be in the sorted sequence.
        A histogram of displacement values is normalized to probabilities, then
        Shannon entropy is computed.

    Configurations:
        - "short": length=15, values=[0..14] (one of each, shuffled)
        - "medium": length=20, values=[0..19] (one of each, shuffled)
        - "duplicates": length=25, values drawn randomly from [0,1,2,3,4] (repeats)
    """

    # Maximum sequence length across all configs (for fixed observation/action space)
    MAX_LENGTH = 25

    # Configuration specifications
    CONFIGS = {
        "short": {
            "length": 15,
            "values": list(range(15)),
            "context_label": 0,
        },
        "medium": {
            "length": 20,
            "values": list(range(20)),
            "context_label": 1,
        },
        "duplicates": {
            "length": 25,
            "value_range": [0, 1, 2, 3, 4],
            "context_label": 2,
        },
    }

    def __init__(self, seed: int | None = None, heterogeneous_costs: bool = False):
        """Initialize the Sequence Sorting environment.

        Args:
            seed: Optional random seed for reproducibility.
            heterogeneous_costs: If True, swap cost depends on displacement from
                sorted position: c(swap_i) = 1 + |displacement_i| / N. Swapping
                far-from-target elements costs more. If False, all swaps cost 1.0.
        """
        self.rng = np.random.default_rng(seed)
        self.heterogeneous_costs = heterogeneous_costs

        # Environment state
        self.sequence: np.ndarray | None = None
        self.length: int | None = None

        # Episode tracking
        self.step_count = 0
        self.max_steps = 500

        # Configuration tracking
        self.current_config: str | None = None
        self.context_label = 0

    def reset(self, config: dict | None = None) -> np.ndarray:
        """Reset environment to initial state.

        Args:
            config: Configuration dict. Format: {"name": "short"} or
                   {"name": "medium"} or {"name": "duplicates"}.
                   Default is "short" if None.

        Returns:
            Initial observation as numpy array.
        """
        if config is None:
            config = {"name": "short"}

        config_name = config.get("name", "short")
        if config_name not in self.CONFIGS:
            raise ValueError(
                f"Unknown config '{config_name}'. Available: {list(self.CONFIGS.keys())}"
            )

        cfg = self.CONFIGS[config_name]
        self.current_config = config_name
        self.context_label = cast(int, cfg["context_label"])

        if config_name == "duplicates":
            # Generate random values from value_range
            self.length = cast(int, cfg["length"])
            value_range = cast(list, cfg["value_range"])
            self.sequence = self.rng.choice(value_range, size=self.length)
        else:
            # Use predefined values and shuffle
            values = cast(list, cfg["values"])
            self.length = cast(int, cfg["length"])
            self.sequence = np.array(values, dtype=np.int32)
            self.rng.shuffle(self.sequence)

        self.step_count = 0

        return self._get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one action in the environment.

        Args:
            action: Integer in range [0, MAX_LENGTH-1]:
                0..MAX_LENGTH-2: swap element at position i with element at i+1
                MAX_LENGTH-1: no-op (do nothing)
                Actions beyond the actual sequence length are treated as no-ops.

        Returns:
            Tuple of (observation, reward, done, info).
            - observation: current state as numpy array (zero-padded to MAX_LENGTH)
            - reward: always 0.0 (agents have their own objectives)
            - done: True after 500 steps
            - info: dict with 'work' key = 1.0 for swaps, 0.0 for no-op
        """
        assert self.sequence is not None and self.length is not None, "Must call reset() first"

        if action < 0 or action >= self.MAX_LENGTH:
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.MAX_LENGTH - 1}].")

        work = 0.0
        if action == self.MAX_LENGTH - 1:
            work = 0.0
        elif action < self.length - 1:
            i = action
            if self.heterogeneous_costs:
                displacement = self._displacement_at(i)
                work = 1.0 + displacement / self.length
            else:
                work = 1.0
            self.sequence[i], self.sequence[i + 1] = (
                self.sequence[i + 1],
                self.sequence[i],
            )
        else:
            work = 0.0

        self.step_count += 1

        done = self.step_count >= self.max_steps

        info = {"work": work}

        return self._get_observation(), 0.0, done, info

    def _displacement_at(self, i: int) -> int:
        """Compute displacement of element at position i from its sorted position.

        Args:
            i: Current position of the element.

        Returns:
            Absolute distance from current position to sorted position.
        """
        assert self.sequence is not None and self.length is not None
        sorted_sequence = np.sort(self.sequence, kind="stable")
        value = self.sequence[i]
        used = np.zeros(self.length, dtype=bool)
        for pos in range(self.length):
            if not used[pos] and sorted_sequence[pos] == value:
                used[pos] = True
                return abs(i - pos)
        return 0  # Fallback (should not happen)

    def _get_observation(self) -> np.ndarray:
        """Build observation array.

        Returns:
            1D numpy array containing the current sequence as float32,
            zero-padded to MAX_LENGTH.
        """
        assert self.sequence is not None and self.length is not None, "Must call reset() first"

        obs = np.zeros(self.MAX_LENGTH, dtype=np.float32)
        obs[: self.length] = self.sequence.astype(np.float32)
        return obs

    def compute_entropy(self) -> float:
        """Compute Shannon entropy over displacement distances distribution.

        For each element at position i, compute displacement = |i - sorted_position_i|
        where sorted_position_i is where element i should be in the sorted sequence.
        For sequences with duplicates, use stable sort to define canonical sorted
        positions (preserve relative order of equal elements from the sorted array).

        A histogram of displacement values is normalized to probabilities, then
        Shannon entropy is computed with natural log.

        Returns:
            Shannon entropy H in nats.
        """
        assert self.sequence is not None and self.length is not None, "Must call reset() first"

        sorted_sequence = np.sort(self.sequence, kind="stable")

        displacements = np.zeros(self.length, dtype=np.int32)
        used = np.zeros(self.length, dtype=bool)

        for current_pos in range(self.length):
            current_value = self.sequence[current_pos]
            for sorted_pos in range(self.length):
                if not used[sorted_pos] and sorted_sequence[sorted_pos] == current_value:
                    displacements[current_pos] = abs(current_pos - sorted_pos)
                    used[sorted_pos] = True
                    break

        if np.all(displacements == 0):
            return 0.0

        counts = np.bincount(displacements)
        return environmental_entropy(counts)

    def get_context_label(self) -> int:
        """Return the current context label.

        Returns:
            Integer context label based on current configuration:
            - 0 for "short"
            - 1 for "medium"
            - 2 for "duplicates"
        """
        return self.context_label

    @property
    def action_space_size(self) -> int:
        """Number of possible discrete actions.

        Returns:
            MAX_LENGTH actions: MAX_LENGTH-1 swap actions plus 1 no-op action.
            Fixed size regardless of current config.
        """
        return self.MAX_LENGTH

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of the observation array.

        Returns:
            Tuple (MAX_LENGTH,) - fixed size regardless of current config.
        """
        return (self.MAX_LENGTH,)
