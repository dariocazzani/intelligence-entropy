"""Resource Grid environment for I_cd evaluation.

A grid world where an agent organizes colored objects. The agent can move,
pick up, and place objects. Environmental entropy is measured as Shannon
entropy over the spatial distribution of cell contents.
"""

from __future__ import annotations

import numpy as np

from ie.environments.base import BaseEnvironment
from ie.measure import environmental_entropy


class ResourceGrid(BaseEnvironment):
    """Grid world environment where an agent organizes colored objects.

    The agent can move around a grid, pick up objects into an inventory,
    and place them down. The goal is to reduce environmental entropy by
    organizing objects into clusters.

    Actions:
        0: move up
        1: move down
        2: move left
        3: move right
        4: pick up object at current position
        5: place object from inventory at current position

    Observation:
        1D numpy array containing:
        - Flattened grid (N*N values, -1=empty, else type id)
        - Agent row position
        - Agent column position
        - Inventory content (-1=empty, else type id)

    Configurations:
        - "basic": 8×8 grid, 1 object type, 15 objects, randomly placed
        - "multi_type": 8×8 grid, 3 object types, 25 objects, sorting required
        - "perturbation": 8×8 grid, 3 object types, 30 objects, perturbation at step 150
    """

    # Configuration specifications
    CONFIGS = {
        "basic": {
            "grid_size": 8,
            "num_types": 1,
            "num_objects": 15,
            "context_label": 0,
        },
        "multi_type": {
            "grid_size": 8,
            "num_types": 3,
            "num_objects": 25,
            "context_label": 1,
        },
        "perturbation": {
            "grid_size": 8,
            "num_types": 3,
            "num_objects": 30,
            "context_label": 2,
            "perturbation_step": 150,
            "perturbation_count": 5,
        },
    }

    def __init__(self, seed: int | None = None):
        """Initialize the Resource Grid environment.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

        # Environment state
        self.grid: np.ndarray | None = None
        self.agent_pos: tuple[int, int] | None = None  # (row, col)
        self.inventory: int | None = None  # -1 or object type id
        self.grid_size: int | None = None
        self.num_types: int | None = None

        # Episode tracking
        self.step_count = 0
        self.max_steps = 300

        # Configuration tracking
        self.current_config: str | None = None
        self.context_label = 0

        # Perturbation tracking (for perturbation config)
        self.perturbation_step: int | None = None
        self.perturbation_count: int | None = None
        self.perturbation_applied = False

    def reset(self, config: dict | None = None) -> np.ndarray:
        """Reset environment to initial state.

        Args:
            config: Configuration dict. Format: {"name": "basic"} or
                   {"name": "multi_type"} or {"name": "perturbation"}.
                   Default is "basic" if None.

        Returns:
            Initial observation as numpy array.
        """
        if config is None:
            config = {"name": "basic"}

        config_name = config.get("name", "basic")
        if config_name not in self.CONFIGS:
            raise ValueError(
                f"Unknown config '{config_name}'. Available: {list(self.CONFIGS.keys())}"
            )

        cfg = self.CONFIGS[config_name]
        self.current_config = config_name
        self.grid_size = cfg["grid_size"]
        self.num_types = cfg["num_types"]
        self.context_label = cfg["context_label"]

        # Initialize empty grid (-1 = empty cell)
        self.grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int32)

        num_objects = cfg["num_objects"]
        self._place_objects_randomly(num_objects)

        # Initialize agent at (0, 0) with empty inventory
        self.agent_pos = (0, 0)
        self.inventory = -1  # -1 means empty

        self.step_count = 0

        self.perturbation_step = cfg.get("perturbation_step")
        self.perturbation_count = cfg.get("perturbation_count")
        self.perturbation_applied = False

        return self._get_observation()

    def _place_objects_randomly(self, num_objects: int) -> None:
        """Place objects randomly on the grid.

        Args:
            num_objects: Total number of objects to place.
        """
        assert self.grid is not None and self.grid_size is not None and self.num_types is not None
        total_cells = self.grid_size * self.grid_size
        if num_objects > total_cells:
            raise ValueError(f"Cannot place {num_objects} objects in {total_cells} cells")

        all_positions = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        positions = self.rng.choice(len(all_positions), size=num_objects, replace=False)

        objects_per_type = num_objects // self.num_types
        extra = num_objects % self.num_types

        object_idx = 0
        for type_id in range(self.num_types):
            count = objects_per_type + (1 if type_id < extra else 0)
            for _ in range(count):
                pos_idx = positions[object_idx]
                r, c = all_positions[pos_idx]
                self.grid[r, c] = type_id
                object_idx += 1

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one action in the environment.

        Args:
            action: Integer in range [0, 5]:
                0=up, 1=down, 2=left, 3=right, 4=pick_up, 5=place

        Returns:
            Tuple of (observation, reward, done, info).
            - observation: current state as numpy array
            - reward: always 0.0 (agents have their own objectives)
            - done: True after 500 steps
            - info: dict with 'work' key = 1.0 (every action costs 1.0)
        """
        assert self.agent_pos is not None and self.grid is not None, "Must call reset() first"

        # Note: step_count starts at 0, so step 50 means step_count == 49
        if (
            self.perturbation_step is not None
            and self.step_count == self.perturbation_step - 1
            and not self.perturbation_applied
        ):
            self._apply_perturbation()
            self.perturbation_applied = True

        # Execute action (all actions cost 1.0 work, even no-ops)
        if action == 0:  # up
            self._move(-1, 0)
        elif action == 1:  # down
            self._move(1, 0)
        elif action == 2:  # left
            self._move(0, -1)
        elif action == 3:  # right
            self._move(0, 1)
        elif action == 4:  # pick up
            self._pick_up()
        elif action == 5:  # place
            self._place()
        else:
            raise ValueError(f"Invalid action {action}. Must be in [0, 5].")

        self.step_count += 1

        done = self.step_count >= self.max_steps

        info = {"work": 1.0}

        return self._get_observation(), 0.0, done, info

    def _move(self, dr: int, dc: int) -> None:
        """Move agent by (dr, dc) if not hitting a wall.

        Args:
            dr: Row delta (-1, 0, or 1)
            dc: Column delta (-1, 0, or 1)
        """
        assert self.agent_pos is not None and self.grid_size is not None
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc

        if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
            self.agent_pos = (new_row, new_col)
        # Otherwise, no-op (wall collision)

    def _pick_up(self) -> None:
        """Pick up object at agent's position into inventory."""
        assert self.agent_pos is not None and self.grid is not None and self.inventory is not None
        r, c = self.agent_pos

        # Only pick up if inventory is empty and cell has an object
        if self.inventory == -1 and self.grid[r, c] != -1:
            self.inventory = self.grid[r, c]
            self.grid[r, c] = -1
        # Otherwise, no-op

    def _place(self) -> None:
        """Place object from inventory at agent's position."""
        assert self.agent_pos is not None and self.grid is not None and self.inventory is not None
        r, c = self.agent_pos

        # Only place if inventory has an object and cell is empty
        if self.inventory != -1 and self.grid[r, c] == -1:
            self.grid[r, c] = self.inventory
            self.inventory = -1
        # Otherwise, no-op

    def _apply_perturbation(self) -> None:
        """Randomly relocate some objects (for perturbation config)."""
        assert self.grid is not None
        if self.perturbation_count is None:
            return

        occupied = np.argwhere(self.grid != -1)
        if len(occupied) == 0:
            return

        num_to_relocate = min(self.perturbation_count, len(occupied))
        indices = self.rng.choice(len(occupied), size=num_to_relocate, replace=False)

        for idx in indices:
            r, c = occupied[idx]
            obj_type = self.grid[r, c]
            self.grid[r, c] = -1
            # Recalculate empty after each move (same semantics as original)
            empty = np.argwhere(self.grid == -1)
            if len(empty) > 0:
                new_pos = empty[self.rng.integers(0, len(empty))]
                self.grid[new_pos[0], new_pos[1]] = obj_type

    def _get_observation(self) -> np.ndarray:
        """Build observation array.

        Returns:
            1D numpy array with: [flattened_grid, agent_row, agent_col, inventory]
        """
        assert self.grid is not None and self.agent_pos is not None and self.inventory is not None
        flat_grid = self.grid.flatten()
        agent_row = self.agent_pos[0]
        agent_col = self.agent_pos[1]

        obs = np.concatenate([flat_grid, [agent_row, agent_col, self.inventory]])

        return obs.astype(np.float32)

    def compute_entropy(self) -> float:
        """Compute Shannon entropy over row-occupancy distribution.

        Counts the number of objects in each row and computes Shannon entropy
        over that distribution. Scattered objects (many rows occupied) give
        high entropy; concentrated objects (few rows) give low entropy.

        Returns:
            Shannon entropy H in nats.
        """
        assert self.grid is not None, "Must call reset() first"
        assert self.grid_size is not None
        row_counts = np.array([int(np.sum(self.grid[r, :] != -1)) for r in range(self.grid_size)])
        nonzero = row_counts[row_counts > 0]
        if len(nonzero) == 0:
            return 0.0
        return float(environmental_entropy(nonzero))

    def get_context_label(self) -> int:
        """Return the current context label.

        Returns:
            Integer context label based on current configuration:
            - 0 for "basic"
            - 1 for "multi_type"
            - 2 for "perturbation"
        """
        return self.context_label

    @property
    def action_space_size(self) -> int:
        """Number of possible discrete actions."""
        return 6  # up, down, left, right, pick_up, place

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of the observation array.

        Returns:
            Tuple (N*N + 3,) where N is grid_size.
        """
        if self.grid_size is None:
            # Before first reset, return default for basic config
            return (8 * 8 + 3,)
        return (self.grid_size * self.grid_size + 3,)
