"""Multi-Room Grid Cleaning environment for I_cd evaluation.

An 8x8 grid environment where an agent navigates and cleans dirty cells.
Entropy reduction IS the task goal: cleaning reduces disorder measured as
Shannon entropy over the distribution of cell states {clean, light, medium, heavy}.

Key design properties:
- Non-uniform action costs by default (move=0.5, clean=1.0)
- 10 room layouts providing natural H(C) variation across agents
- Specialist vs. generalist agents achieve different effective context sets
- Both factors of I_cd (eta and H(C)) do joint discriminatory work
"""

from __future__ import annotations

import numpy as np

from ie.environments.base import BaseEnvironment
from ie.measure import environmental_entropy

# Grid cell values
CLEAN = 0
LIGHT = 1
MEDIUM = 2
HEAVY = 3
OBSTACLE = -1
WALL = -2

# Normalized observation values
_OBS_CLEAN = 0.0
_OBS_LIGHT = 0.33
_OBS_MEDIUM = 0.67
_OBS_HEAVY = 1.0
_OBS_BLOCKED = -1.0

# Actions
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
ACTION_CLEAN = 4

# Costs
MOVE_COST = 0.5
CLEAN_COST = 1.0

GRID_SIZE = 8
MAX_STEPS = 200


def _generate_layout(layout_id: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a deterministic room layout from its ID.

    Args:
        layout_id: Integer 0-9 identifying the room.
        rng: Seeded RNG for reproducibility.

    Returns:
        8x8 int array with cell values (CLEAN, LIGHT, MEDIUM, HEAVY, OBSTACLE, WALL).
    """
    grid = np.full((GRID_SIZE, GRID_SIZE), CLEAN, dtype=np.int32)

    if layout_id == 0:
        # empty-light: no obstacles, 20% light dirt
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if rng.random() < 0.20:
                    grid[r, c] = LIGHT

    elif layout_id == 1:
        # empty-mixed: no obstacles, 40% mixed dirt (L/M/H uniform)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if rng.random() < 0.40:
                    grid[r, c] = rng.choice([LIGHT, MEDIUM, HEAVY])

    elif layout_id == 2:
        # sparse-light: 5 scattered obstacles, 25% light dirt
        obs_positions: set[tuple[int, int]] = set()
        while len(obs_positions) < 5:
            pos = (int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE)))
            obs_positions.add(pos)
        for r, c in obs_positions:
            grid[r, c] = OBSTACLE
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == CLEAN and rng.random() < 0.25:
                    grid[r, c] = LIGHT

    elif layout_id == 3:
        # sparse-mixed: 5 scattered obstacles, 40% mixed dirt
        obs_positions = set()
        while len(obs_positions) < 5:
            pos = (int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE)))
            obs_positions.add(pos)
        for r, c in obs_positions:
            grid[r, c] = OBSTACLE
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == CLEAN and rng.random() < 0.40:
                    grid[r, c] = rng.choice([LIGHT, MEDIUM, HEAVY])

    elif layout_id == 4:
        # moderate-mixed: 12 scattered obstacles, 35% mixed dirt
        obs_positions = set()
        while len(obs_positions) < 12:
            pos = (int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE)))
            obs_positions.add(pos)
        for r, c in obs_positions:
            grid[r, c] = OBSTACLE
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == CLEAN and rng.random() < 0.35:
                    grid[r, c] = rng.choice([LIGHT, MEDIUM, HEAVY])

    elif layout_id == 5:
        # corridor: central wall with 2 gaps, 40% mixed dirt
        for r in range(GRID_SIZE):
            grid[r, GRID_SIZE // 2] = WALL
        gap1 = rng.integers(0, GRID_SIZE)
        gap2 = rng.integers(0, GRID_SIZE)
        while gap2 == gap1:
            gap2 = rng.integers(0, GRID_SIZE)
        grid[gap1, GRID_SIZE // 2] = CLEAN
        grid[gap2, GRID_SIZE // 2] = CLEAN
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == CLEAN and rng.random() < 0.40:
                    grid[r, c] = rng.choice([LIGHT, MEDIUM, HEAVY])

    elif layout_id == 6:
        # four-rooms: walls dividing into quadrants with 1 gap each
        mid = GRID_SIZE // 2
        # Horizontal wall
        for c in range(GRID_SIZE):
            grid[mid, c] = WALL
        # Vertical wall
        for r in range(GRID_SIZE):
            grid[r, mid] = WALL
        # Gaps (one per wall segment)
        grid[rng.integers(0, mid), mid] = CLEAN  # top vertical
        grid[rng.integers(mid + 1, GRID_SIZE), mid] = CLEAN  # bottom vertical
        grid[mid, rng.integers(0, mid)] = CLEAN  # left horizontal
        grid[mid, rng.integers(mid + 1, GRID_SIZE)] = CLEAN  # right horizontal
        # 30% mixed dirt
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == CLEAN and rng.random() < 0.30:
                    grid[r, c] = rng.choice([LIGHT, MEDIUM, HEAVY])

    elif layout_id == 7:
        # dense-heavy: 20 scattered obstacles, 30% heavy dirt
        obs_positions = set()
        while len(obs_positions) < 20:
            pos = (int(rng.integers(0, GRID_SIZE)), int(rng.integers(0, GRID_SIZE)))
            obs_positions.add(pos)
        for r, c in obs_positions:
            grid[r, c] = OBSTACLE
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == CLEAN and rng.random() < 0.30:
                    grid[r, c] = HEAVY

    elif layout_id == 8:
        # maze: DFS maze with narrow corridors, 35% heavy dirt
        grid = _generate_dfs_maze(rng)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == CLEAN and rng.random() < 0.35:
                    grid[r, c] = HEAVY

    elif layout_id == 9:
        # L-room: quarter blocked by L-shaped wall, 50% heavy dirt
        mid = GRID_SIZE // 2
        # L-shape: vertical wall from top to mid, horizontal wall from mid to right
        for r in range(mid + 1):
            grid[r, mid] = WALL
        for c in range(mid, GRID_SIZE):
            grid[mid, c] = WALL
        # One gap in each segment
        grid[rng.integers(0, mid), mid] = CLEAN
        grid[mid, rng.integers(mid + 1, GRID_SIZE)] = CLEAN
        # 50% heavy dirt
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if grid[r, c] == CLEAN and rng.random() < 0.50:
                    grid[r, c] = HEAVY

    return grid


def _generate_dfs_maze(rng: np.random.Generator) -> np.ndarray:
    """Generate a simple maze using randomized DFS on an 8x8 grid.

    Creates walls everywhere, then carves passages. Cells at even coordinates
    are potential passage cells; odd coordinates are walls or passages.

    Returns:
        8x8 int array with WALL for walls and CLEAN for passages.
    """
    grid = np.full((GRID_SIZE, GRID_SIZE), WALL, dtype=np.int32)

    # Use even-coordinate cells as nodes in the maze graph
    start = (0, 0)
    grid[start] = CLEAN
    stack = [start]
    visited = {start}

    while stack:
        r, c = stack[-1]
        # Find unvisited neighbors (2 steps away)
        neighbors = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in visited:
                neighbors.append((nr, nc))

        if neighbors:
            nr, nc = neighbors[rng.integers(0, len(neighbors))]
            # Carve passage between current and neighbor
            grid[(r + nr) // 2, (c + nc) // 2] = CLEAN
            grid[nr, nc] = CLEAN
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    return grid


class GridCleaning(BaseEnvironment):
    """Multi-Room Grid Cleaning environment.

    An 8x8 grid where the agent navigates and cleans dirty cells. Each cell
    can be CLEAN, LIGHT (1 clean action), MEDIUM (2 actions), or HEAVY
    (3 actions), plus OBSTACLE and WALL cells that block movement.

    Actions:
        0: Move up
        1: Move down
        2: Move left
        3: Move right
        4: Clean (reduce dirt level of current cell by 1)

    Observation:
        66-element float32 array:
        - Flattened 8x8 grid (64 values, normalized)
        - Agent row (normalized to [0,1])
        - Agent col (normalized to [0,1])

    Costs:
        - Move: 0.5
        - Clean: 1.0

    Entropy:
        Shannon entropy over the distribution of cleanable cell states
        {clean, light, medium, heavy}. When all cells are clean, S = 0.
    """

    LAYOUT_NAMES = [
        "empty-light",
        "empty-mixed",
        "sparse-light",
        "sparse-mixed",
        "moderate-mixed",
        "corridor",
        "four-rooms",
        "dense-heavy",
        "maze",
        "L-room",
    ]

    def __init__(self, seed: int | None = None):
        """Initialize the Grid Cleaning environment.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.grid: np.ndarray | None = None
        self.agent_row: int = 0
        self.agent_col: int = 0
        self.step_count: int = 0
        self.layout_id: int = 0
        self._n_cleanable: int = 0

    def reset(self, config: dict | None = None) -> np.ndarray:
        """Reset environment with the specified layout.

        Args:
            config: Configuration dict. Format: {"layout_id": 0-9}.
                   Default is layout 0 if None.

        Returns:
            Initial observation as numpy array.
        """
        if config is None:
            config = {"layout_id": 0}

        self.layout_id = config.get("layout_id", 0)
        if not 0 <= self.layout_id < len(self.LAYOUT_NAMES):
            raise ValueError(
                f"layout_id must be 0-{len(self.LAYOUT_NAMES) - 1}, got {self.layout_id}"
            )

        # Generate layout deterministically from layout_id and a per-reset seed
        layout_seed = self.rng.integers(0, 2**31)
        layout_rng = np.random.default_rng(self.layout_id * 100_000 + layout_seed)
        self.grid = _generate_layout(self.layout_id, layout_rng)

        # Count cleanable cells (not OBSTACLE, not WALL)
        self._n_cleanable = int(np.sum((self.grid != OBSTACLE) & (self.grid != WALL)))

        # Place agent at a random empty cell (CLEAN or dirty, not obstacle/wall)
        empty_cells = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r, c] >= CLEAN:
                    empty_cells.append((r, c))

        if not empty_cells:
            raise RuntimeError(f"Layout {self.layout_id} has no empty cells")

        idx = self.rng.integers(0, len(empty_cells))
        self.agent_row, self.agent_col = empty_cells[idx]

        self.step_count = 0
        return self._get_observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one action.

        Args:
            action: 0=up, 1=down, 2=left, 3=right, 4=clean.

        Returns:
            (observation, reward, done, info) where info['work'] is action cost.
        """
        assert self.grid is not None, "Must call reset() first"

        work = 0.0
        reward = 0.0

        if action == ACTION_CLEAN:
            work = CLEAN_COST
            if self.grid[self.agent_row, self.agent_col] > CLEAN:
                self.grid[self.agent_row, self.agent_col] -= 1
                reward = 1.0  # reward for successful cleaning
        elif 0 <= action <= 3:
            work = MOVE_COST
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            nr, nc = self.agent_row + dr, self.agent_col + dc
            if (
                0 <= nr < GRID_SIZE
                and 0 <= nc < GRID_SIZE
                and self.grid[nr, nc] != OBSTACLE
                and self.grid[nr, nc] != WALL
            ):
                self.agent_row = nr
                self.agent_col = nc
        else:
            raise ValueError(f"Invalid action {action}. Must be 0-4.")

        self.step_count += 1
        done = self.step_count >= MAX_STEPS

        return self._get_observation(), reward, done, {"work": work}

    def compute_entropy(self) -> float:
        """Compute Shannon entropy over cleanable cell state distribution.

        Counts cells in each state {clean, light, medium, heavy} among all
        cleanable cells (excludes obstacles and walls).

        Returns:
            Shannon entropy in nats. 0 when all cleanable cells are clean.
        """
        assert self.grid is not None, "Must call reset() first"

        counts = np.zeros(4, dtype=np.int64)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                val = self.grid[r, c]
                if val >= CLEAN:
                    counts[val] += 1

        return environmental_entropy(counts)

    def get_context_label(self) -> int:
        """Return layout ID as context label (0-9)."""
        return self.layout_id

    @property
    def action_space_size(self) -> int:
        return 5  # up, down, left, right, clean

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (66,)  # 64 grid + 2 agent position

    def _get_observation(self) -> np.ndarray:
        """Build the observation array.

        Returns:
            66-element float32: [flattened grid (64), agent_row_norm, agent_col_norm]
        """
        assert self.grid is not None

        obs_grid = np.zeros(GRID_SIZE * GRID_SIZE, dtype=np.float32)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                idx = r * GRID_SIZE + c
                val = self.grid[r, c]
                if val == CLEAN:
                    obs_grid[idx] = _OBS_CLEAN
                elif val == LIGHT:
                    obs_grid[idx] = _OBS_LIGHT
                elif val == MEDIUM:
                    obs_grid[idx] = _OBS_MEDIUM
                elif val == HEAVY:
                    obs_grid[idx] = _OBS_HEAVY
                else:  # OBSTACLE or WALL
                    obs_grid[idx] = _OBS_BLOCKED

        agent_pos = np.array(
            [
                self.agent_row / (GRID_SIZE - 1),
                self.agent_col / (GRID_SIZE - 1),
            ],
            dtype=np.float32,
        )

        return np.concatenate([obs_grid, agent_pos])

    def get_grid_copy(self) -> np.ndarray:
        """Return a copy of the current grid (for heuristic agent pathfinding)."""
        assert self.grid is not None, "Must call reset() first"
        return self.grid.copy()

    def get_agent_position(self) -> tuple[int, int]:
        """Return current agent position."""
        return (self.agent_row, self.agent_col)
