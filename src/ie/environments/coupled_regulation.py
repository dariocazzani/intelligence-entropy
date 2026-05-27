"""Coupled Regulation environment for I_cd evaluation.

A continuous control environment where an agent regulates N coupled variables
toward target values. Adjusting one variable affects others through coupling,
requiring coordinated control. Environmental entropy is measured as Shannon
entropy over the discretized deviation distribution.
"""

from __future__ import annotations

import numpy as np

from ie.environments.base import BaseEnvironment
from ie.measure import environmental_entropy


class CoupledRegulation(BaseEnvironment):
    """Environment for controlling N coupled continuous variables toward targets.

    The agent adjusts N continuous variables (x_0, ..., x_{N-1}) toward
    target values. Variables are coupled: adjusting x_k affects other variables
    x_j through a coupling matrix A[j,k]. The goal is to minimize deviations
    from target values despite coupling effects.

    Actions:
        0-7: increase variable k by DELTA (k = action)
        8-15: decrease variable k by DELTA (k = action - N_MAX)
        16: no-op

    Observation:
        16-element float32 array containing:
        - Current values (padded to 8)
        - Target values (padded to 8)

    Configurations:
        - "weak": 4 variables, coupling strength 0.1
        - "moderate": 6 variables, coupling strength 0.3
        - "strong": 8 variables, coupling strength 0.5
    """

    # Constants
    N_MAX = 8  # Maximum number of variables (for fixed action/obs space)
    DELTA = 0.2  # Adjustment step size

    # Configuration specifications
    CONFIGS = {
        "weak": {
            "n_vars": 4,
            "coupling_strength": 0.1,
            "context_label": 0,
        },
        "moderate": {
            "n_vars": 6,
            "coupling_strength": 0.3,
            "context_label": 1,
        },
        "strong": {
            "n_vars": 8,
            "coupling_strength": 0.5,
            "context_label": 2,
        },
    }

    def __init__(self, seed: int | None = None, heterogeneous_costs: bool = False):
        """Initialize the Coupled Regulation environment.

        Args:
            seed: Optional random seed for reproducibility.
            heterogeneous_costs: If True, action cost depends on coupling strength
                (stronger coupling = higher cost). If False, all actions cost 1.0.
        """
        self.rng = np.random.default_rng(seed)

        # Environment state
        self.values: np.ndarray | None = None
        self.targets: np.ndarray | None = None
        self.coupling: np.ndarray | None = None
        self.n_vars: int | None = None

        # Episode tracking
        self.step_count = 0
        self.max_steps = 300

        # Configuration tracking
        self.current_config: str | None = None
        self.context_label = 0

        # Cost model
        self.heterogeneous_costs = heterogeneous_costs

    def reset(self, config: dict | None = None) -> np.ndarray:
        """Reset environment to initial state.

        Args:
            config: Configuration dict. Format: {"name": "weak"} or
                   {"name": "moderate"} or {"name": "strong"}.
                   Default is "weak" if None.

        Returns:
            Initial observation as numpy array.
        """
        if config is None:
            config = {"name": "weak"}

        config_name = config.get("name", "weak")
        if config_name not in self.CONFIGS:
            raise ValueError(
                f"Unknown config '{config_name}'. Available: {list(self.CONFIGS.keys())}"
            )

        cfg = self.CONFIGS[config_name]
        self.current_config = config_name
        self.n_vars = int(cfg["n_vars"])
        self.context_label = int(cfg["context_label"])

        n_vars = self.n_vars
        alpha = float(cfg["coupling_strength"])

        # Generate targets in range [-2.0, 2.0]
        self.targets = self.rng.uniform(-2.0, 2.0, size=n_vars)

        # Generate coupling matrix A[j,k] = effect of adjusting x_k on x_j
        # A[k,k] = 0 (adjusting a variable doesn't affect itself through coupling)
        # A[j,k] ~ Uniform(-alpha, alpha) for j != k
        self.coupling = np.zeros((n_vars, n_vars), dtype=np.float32)
        for j in range(n_vars):
            for k in range(n_vars):
                if j != k:
                    self.coupling[j, k] = self.rng.uniform(-alpha, alpha)

        # Initialize values near targets with some noise
        self.values = self.targets + self.rng.uniform(-1.5, 1.5, size=n_vars)

        self.step_count = 0

        return self._get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one action in the environment.

        Args:
            action: Integer in range [0, 16]:
                0-7: increase variable k (k = action)
                8-15: decrease variable k (k = action - N_MAX)
                16: no-op

        Returns:
            Tuple of (observation, reward, done, info).
            - observation: current state as numpy array
            - reward: always 0.0 (agents have their own objectives)
            - done: True after 300 steps
            - info: dict with 'work' key = 1.0 for valid adjustment, 0.0 for no-op
        """
        assert self.values is not None and self.coupling is not None and self.n_vars is not None, (
            "Must call reset() first"
        )

        work = 0.0

        if action == 16:
            # No-op
            work = 0.0
        elif 0 <= action < self.N_MAX:
            # Increase variable k
            k = action
            if k < self.n_vars:
                direction = 1.0
                self.values[k] += direction * self.DELTA
                # Apply coupling effects to other variables
                for j in range(self.n_vars):
                    if j != k:
                        self.values[j] += self.coupling[j, k] * direction * self.DELTA
                # Compute work cost based on coupling strength
                if self.heterogeneous_costs:
                    work = 1.0 + float(np.sum(np.abs(self.coupling[:, k])))
                else:
                    work = 1.0
            else:
                # Out of range for this context
                work = 0.0
        elif self.N_MAX <= action < 2 * self.N_MAX:
            # Decrease variable k
            k = action - self.N_MAX
            if k < self.n_vars:
                direction = -1.0
                self.values[k] += direction * self.DELTA
                # Apply coupling effects to other variables
                for j in range(self.n_vars):
                    if j != k:
                        self.values[j] += self.coupling[j, k] * direction * self.DELTA
                # Compute work cost based on coupling strength
                if self.heterogeneous_costs:
                    work = 1.0 + float(np.sum(np.abs(self.coupling[:, k])))
                else:
                    work = 1.0
            else:
                # Out of range for this context
                work = 0.0
        else:
            raise ValueError(f"Invalid action {action}. Must be in [0, 16].")

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {"work": work}

        return self._get_observation(), 0.0, done, info

    def _get_observation(self) -> np.ndarray:
        """Build observation array.

        Returns:
            16-element float32 array: [values padded to 8, targets padded to 8]
        """
        assert self.values is not None and self.targets is not None and self.n_vars is not None

        # Pad values and targets to N_MAX length
        values_padded = np.zeros(self.N_MAX, dtype=np.float32)
        targets_padded = np.zeros(self.N_MAX, dtype=np.float32)

        values_padded[: self.n_vars] = self.values
        targets_padded[: self.n_vars] = self.targets

        obs = np.concatenate([values_padded, targets_padded])

        return obs.astype(np.float32)

    def compute_entropy(self) -> float:
        """Compute Shannon entropy over discretized deviation distribution.

        For each active variable, compute deviation |value - target| and
        discretize into bins:
            0: dev < 0.1 (very close)
            1: 0.1 <= dev < 0.3 (close)
            2: 0.3 <= dev < 0.6 (moderate)
            3: 0.6 <= dev < 1.0 (far)
            4: dev >= 1.0 (very far)

        Returns:
            Shannon entropy H over bin counts in nats.
        """
        assert self.values is not None and self.targets is not None and self.n_vars is not None, (
            "Must call reset() first"
        )

        # Compute deviations for active variables
        deviations = np.abs(self.values - self.targets)

        # Discretize into bins
        bins = np.zeros(self.n_vars, dtype=np.int32)
        for i in range(self.n_vars):
            dev = deviations[i]
            if dev < 0.1:
                bins[i] = 0
            elif dev < 0.3:
                bins[i] = 1
            elif dev < 0.6:
                bins[i] = 2
            elif dev < 1.0:
                bins[i] = 3
            else:
                bins[i] = 4

        # Count frequency of each bin
        counts = np.bincount(bins, minlength=5)

        return environmental_entropy(counts)

    def get_context_label(self) -> int:
        """Return the current context label.

        Returns:
            Integer context label based on current configuration:
            - 0 for "weak"
            - 1 for "moderate"
            - 2 for "strong"
        """
        return self.context_label

    @property
    def action_space_size(self) -> int:
        """Number of possible discrete actions."""
        return 17  # 8 increase + 8 decrease + 1 no-op

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of the observation array.

        Returns:
            Tuple (16,) representing [values_padded(8), targets_padded(8)]
        """
        return (16,)
