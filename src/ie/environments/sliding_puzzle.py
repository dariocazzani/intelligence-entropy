"""Variable-size Sliding Puzzle environment for I_cd evaluation.

Sliding tile puzzles from 2x2 (3-puzzle) through 5x5 (24-puzzle).
The agent slides tiles into the blank space to restore the canonical order.

Environmental entropy is Shannon entropy over the distribution of Manhattan
displacement distances, identical to the existing TilePuzzle but generalized
to variable grid widths. PPO is proven to converge on w=3 (ICML 2025) and
degrades on larger grids, creating natural eta variation across contexts.

Key design properties for I_cd:
- 10 contexts spanning 4 grid sizes x 3 shuffle depths
- PPO solves w=2,3 reliably (high eta) but degrades on w=4,5 (lower eta)
- Specialist agents trained on easy contexts have high eta but low H(C)
- Generalist agents have moderate eta across many contexts, so higher H(C)
- The product form I_cd = eta x H(C) discriminates specialist vs generalist
"""

from __future__ import annotations

import numpy as np

from ie.environments.base import BaseEnvironment
from ie.measure import environmental_entropy

W_MAX = 5
OBS_SIZE = W_MAX * W_MAX + 1  # 25 grid cells + 1 grid-size indicator = 26
MAX_TILE_VALUE = W_MAX * W_MAX - 1  # 24, for normalization
MAX_STEPS = 500

# 10 contexts: 4 grid sizes x varying shuffle depths
CONFIGS = [
    {"context_id": 0, "w": 2, "scramble": 10, "name": "2x2_trivial"},
    {"context_id": 1, "w": 3, "scramble": 20, "name": "3x3_easy"},
    {"context_id": 2, "w": 3, "scramble": 50, "name": "3x3_medium"},
    {"context_id": 3, "w": 3, "scramble": 100, "name": "3x3_hard"},
    {"context_id": 4, "w": 4, "scramble": 20, "name": "4x4_easy"},
    {"context_id": 5, "w": 4, "scramble": 50, "name": "4x4_medium"},
    {"context_id": 6, "w": 4, "scramble": 100, "name": "4x4_hard"},
    {"context_id": 7, "w": 5, "scramble": 20, "name": "5x5_easy"},
    {"context_id": 8, "w": 5, "scramble": 50, "name": "5x5_medium"},
    {"context_id": 9, "w": 5, "scramble": 100, "name": "5x5_hard"},
]


class SlidingPuzzle(BaseEnvironment):
    """Variable-size sliding tile puzzle.

    Supports w=2 (3-puzzle) through w=5 (24-puzzle). Observation is a
    fixed-size 26-element array: 25 normalized tile values (padded with -1
    for unused cells) plus a grid-size indicator.

    Actions:
        0: slide tile from above blank into blank
        1: slide tile from below blank into blank
        2: slide tile from left of blank into blank
        3: slide tile from right of blank into blank
        Invalid moves (at edges) are no-ops with work=0.0

    Observation:
        26-element float32 array:
        - First w*w elements: tile values / 24 (0.0 for blank)
        - Remaining positions up to 25: -1.0 (padding)
        - Element 25: w / 5 (grid size indicator)

    Solved state (example w=3):
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 0]]
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.w: int = 3
        self.grid: np.ndarray | None = None
        self.empty_pos: tuple[int, int] = (0, 0)
        self.step_count: int = 0
        self.context_id: int = 0

    def reset(self, config: dict | None = None) -> np.ndarray:
        if config is None:
            config = CONFIGS[0]

        context_id: int = config["context_id"]  # type: ignore[assignment]
        w: int = config["w"]  # type: ignore[assignment]
        scramble: int = config.get("scramble", 20)  # type: ignore[assignment]
        self.context_id = context_id
        self.w = w

        # Build solved state: tiles 1..n-1, blank=0 in bottom-right
        n = self.w * self.w
        self.grid = np.zeros((self.w, self.w), dtype=np.int32)
        for i in range(n - 1):
            self.grid[i // self.w, i % self.w] = i + 1
        self.empty_pos = (self.w - 1, self.w - 1)

        # Scramble by applying random valid moves (guarantees solvability)
        for _ in range(scramble):
            valid = self._valid_actions()
            action = int(self.rng.choice(valid))
            self._apply_action(action)

        self.step_count = 0
        return self._get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert self.grid is not None, "Must call reset() first"

        if action < 0 or action > 3:
            raise ValueError(f"Invalid action {action}. Must be 0-3.")

        self._apply_action(action)
        self.step_count += 1
        done = self.step_count >= MAX_STEPS

        return self._get_observation(), 0.0, done, {"work": 1.0}

    def _valid_actions(self) -> list[int]:
        er, ec = self.empty_pos
        actions = []
        if er > 0:
            actions.append(0)
        if er < self.w - 1:
            actions.append(1)
        if ec > 0:
            actions.append(2)
        if ec < self.w - 1:
            actions.append(3)
        return actions

    def _apply_action(self, action: int) -> float:
        assert self.grid is not None
        er, ec = self.empty_pos

        # Map action to tile position that slides into blank
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = deltas[action]
        tr, tc = er + dr, ec + dc

        if not (0 <= tr < self.w and 0 <= tc < self.w):
            return 0.0  # invalid move, no work

        # Swap tile with blank
        self.grid[er, ec] = self.grid[tr, tc]
        self.grid[tr, tc] = 0
        self.empty_pos = (tr, tc)
        return 1.0

    def _get_observation(self) -> np.ndarray:
        assert self.grid is not None
        obs = np.full(OBS_SIZE, -1.0, dtype=np.float32)
        n = self.w * self.w
        obs[:n] = self.grid.flatten().astype(np.float32) / MAX_TILE_VALUE
        obs[W_MAX * W_MAX] = self.w / W_MAX  # grid size indicator
        return obs

    def compute_entropy(self) -> float:
        assert self.grid is not None
        num_tiles = self.w * self.w - 1
        if num_tiles == 0:
            return 0.0

        displacements = np.zeros(num_tiles, dtype=np.int32)
        max_displacement = 2 * (self.w - 1)  # max Manhattan in w x w grid

        for tile in range(1, num_tiles + 1):
            pos = np.where(self.grid == tile)
            curr_r, curr_c = int(pos[0][0]), int(pos[1][0])
            goal_r = (tile - 1) // self.w
            goal_c = (tile - 1) % self.w
            displacements[tile - 1] = abs(curr_r - goal_r) + abs(curr_c - goal_c)

        if np.all(displacements == 0):
            return 0.0

        counts = np.bincount(displacements, minlength=max_displacement + 1)
        return environmental_entropy(counts)

    def get_context_label(self) -> int:
        return self.context_id

    @property
    def action_space_size(self) -> int:
        return 4

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (OBS_SIZE,)
