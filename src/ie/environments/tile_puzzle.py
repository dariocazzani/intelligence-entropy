"""Sliding Tile Puzzle environment for I_cd evaluation.

A 4×4 sliding tile puzzle (15-puzzle) where the agent solves the puzzle by
sliding tiles into the empty space. Environmental entropy is measured as Shannon
entropy over the distribution of Manhattan displacement distances for each tile.
"""

from __future__ import annotations

import numpy as np

from ie.environments.base import BaseEnvironment
from ie.measure import environmental_entropy


class TilePuzzle(BaseEnvironment):
    """4×4 sliding tile puzzle (15-puzzle) environment.

    The agent can slide tiles into the empty space to solve the puzzle.
    The goal is to reduce environmental entropy by arranging tiles in order.

    Actions:
        0: slide tile from above empty space (if valid)
        1: slide tile from below empty space (if valid)
        2: slide tile from left of empty space (if valid)
        3: slide tile from right of empty space (if valid)
        Invalid moves (at edges) are no-ops with work=0.0

    Observation:
        1D numpy array (16 elements) containing the flattened 4×4 grid as float32.
        Values 1-15 are tiles, 0 is the empty space.

    Environmental Entropy:
        Entropy over the distribution of Manhattan displacement distances.
        For each tile i (1-15), compute displacement = |current_row - target_row| +
        |current_col - target_col| where target is the tile's position in the solved
        state. Displacements range from 0 (correct position) to 6 (maximum displacement).

    Configurations:
        - "easy": 20 random scramble moves
        - "medium": 50 random scramble moves
        - "hard": 100 random scramble moves

    Solved state:
        [[ 1,  2,  3,  4],
         [ 5,  6,  7,  8],
         [ 9, 10, 11, 12],
         [13, 14, 15,  0]]
    """

    GRID_SIZE = 4
    NUM_TILES = 15

    CONFIGS = {
        "easy": {
            "scramble_moves": 20,
            "context_label": 0,
        },
        "medium": {
            "scramble_moves": 50,
            "context_label": 1,
        },
        "hard": {
            "scramble_moves": 100,
            "context_label": 2,
        },
    }

    def __init__(self, seed: int | None = None):
        """Initialize the Tile Puzzle environment.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

        # Environment state
        self.grid: np.ndarray | None = None
        self.empty_pos: tuple[int, int] | None = None

        # Episode tracking
        self.step_count = 0
        self.max_steps = 500

        # Configuration tracking
        self.current_config: str | None = None
        self.context_label = 0

    def reset(self, config: dict | None = None) -> np.ndarray:
        """Reset environment to initial state.

        Args:
            config: Configuration dict. Format: {"name": "easy"} or
                   {"name": "medium"} or {"name": "hard"}.
                   Default is "easy" if None.

        Returns:
            Initial observation as numpy array (16 elements, flattened grid).
        """
        if config is None:
            config = {"name": "easy"}

        config_name = config.get("name", "easy")
        if config_name not in self.CONFIGS:
            raise ValueError(
                f"Unknown config '{config_name}'. Available: {list(self.CONFIGS.keys())}"
            )

        cfg = self.CONFIGS[config_name]
        self.current_config = config_name
        self.context_label = cfg["context_label"]

        # Create solved state
        self.grid = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]],
            dtype=np.int32,
        )
        self.empty_pos = (3, 3)

        # Scramble the puzzle
        scramble_moves = cfg["scramble_moves"]
        for _ in range(scramble_moves):
            # Pick a random valid action and apply it
            valid_actions = []
            if self.empty_pos[0] > 0:
                valid_actions.append(0)  # can slide from above
            if self.empty_pos[0] < self.GRID_SIZE - 1:
                valid_actions.append(1)  # can slide from below
            if self.empty_pos[1] > 0:
                valid_actions.append(2)  # can slide from left
            if self.empty_pos[1] < self.GRID_SIZE - 1:
                valid_actions.append(3)  # can slide from right

            action = self.rng.choice(valid_actions)
            self._apply_action(action)

        self.step_count = 0

        return self._get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one action in the environment.

        Args:
            action: Integer in range [0, 3]:
                0: slide tile from above empty space (move tile down)
                1: slide tile from below empty space (move tile up)
                2: slide tile from left of empty space (move tile right)
                3: slide tile from right of empty space (move tile left)

        Returns:
            Tuple of (observation, reward, done, info).
            - observation: current state as numpy array (16 elements)
            - reward: always 0.0 (agents have their own objectives)
            - done: True after 500 steps
            - info: dict with 'work' key = 1.0 for valid moves, 0.0 for invalid
        """
        assert self.grid is not None and self.empty_pos is not None, "Must call reset() first"

        if action < 0 or action > 3:
            raise ValueError(f"Invalid action {action}. Must be in [0, 3].")

        work = self._apply_action(action)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {"work": work}

        return self._get_observation(), 0.0, done, info

    def _apply_action(self, action: int) -> float:
        """Apply an action and return the work done.

        Args:
            action: Action to apply (0-3).

        Returns:
            Work done (1.0 for valid moves, 0.0 for invalid moves).
        """
        assert self.grid is not None and self.empty_pos is not None

        empty_row, empty_col = self.empty_pos

        # Determine the tile position to swap with empty space
        tile_row, tile_col = empty_row, empty_col

        if action == 0:  # slide from above
            if empty_row > 0:
                tile_row = empty_row - 1
            else:
                return 0.0  # invalid move
        elif action == 1:  # slide from below
            if empty_row < self.GRID_SIZE - 1:
                tile_row = empty_row + 1
            else:
                return 0.0  # invalid move
        elif action == 2:  # slide from left
            if empty_col > 0:
                tile_col = empty_col - 1
            else:
                return 0.0  # invalid move
        elif action == 3:  # slide from right
            if empty_col < self.GRID_SIZE - 1:
                tile_col = empty_col + 1
            else:
                return 0.0  # invalid move

        # Swap tile with empty space
        self.grid[empty_row, empty_col], self.grid[tile_row, tile_col] = (
            self.grid[tile_row, tile_col],
            self.grid[empty_row, empty_col],
        )
        self.empty_pos = (tile_row, tile_col)

        return 1.0

    def _get_observation(self) -> np.ndarray:
        """Build observation array.

        Returns:
            1D numpy array (16 elements) containing the flattened grid as float32.
        """
        assert self.grid is not None, "Must call reset() first"
        return self.grid.flatten().astype(np.float32)

    def compute_entropy(self) -> float:
        """Compute Shannon entropy over displacement distances distribution.

        For each tile i (1-15), compute Manhattan displacement from its current
        position to its target position in the solved state. Displacements range
        from 0 (correct position) to 6 (maximum displacement). Histogram these
        displacements and compute Shannon entropy.

        Returns:
            Shannon entropy H in nats. Returns 0.0 if solved (all displacements zero).
        """
        assert self.grid is not None, "Must call reset() first"

        displacements = np.zeros(self.NUM_TILES, dtype=np.int32)

        # For each tile 1-15, compute its displacement
        for tile in range(1, self.NUM_TILES + 1):
            # Find current position
            current_pos = np.where(self.grid == tile)
            current_row, current_col = current_pos[0][0], current_pos[1][0]

            # Compute target position (1-indexed tiles, 0-indexed positions)
            target_row = (tile - 1) // self.GRID_SIZE
            target_col = (tile - 1) % self.GRID_SIZE

            # Manhattan distance
            displacement = abs(current_row - target_row) + abs(current_col - target_col)
            displacements[tile - 1] = displacement

        # Special case: if solved, return 0.0
        if np.all(displacements == 0):
            return 0.0

        # Count displacements (0-6, so minlength=7)
        counts = np.bincount(displacements, minlength=7)

        return environmental_entropy(counts)

    def get_context_label(self) -> int:
        """Return the current context label.

        Returns:
            Integer context label based on current configuration:
            - 0 for "easy"
            - 1 for "medium"
            - 2 for "hard"
        """
        return self.context_label

    @property
    def action_space_size(self) -> int:
        """Number of possible discrete actions.

        Returns:
            4 actions: slide from above, below, left, or right.
        """
        return 4

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of the observation array.

        Returns:
            Tuple (16,) - flattened 4×4 grid.
        """
        return (16,)
