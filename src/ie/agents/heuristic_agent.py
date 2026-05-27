"""Hand-coded heuristic agents for baseline comparison."""

from __future__ import annotations

from collections import deque

import numpy as np


class GridHeuristicAgent:
    """Heuristic agent for ResourceGrid environment.

    Strategy:
    - If not holding an object: find nearest object and navigate toward it,
      picking it up when positioned on it
    - If holding an object: navigate to target area (row 0, then row 1, etc.)
      and place it when positioned on an empty target cell
    """

    def __init__(self) -> None:
        """Initialize the grid heuristic agent."""
        pass

    def act(self, observation: np.ndarray) -> int:
        """Select action based on simple heuristic strategy.

        Args:
            observation: 1D array with [flattened_grid (N*N), agent_row, agent_col, inventory]

        Returns:
            Action integer in [0, 5]:
                0=up, 1=down, 2=left, 3=right, 4=pickup, 5=place
        """
        # Parse observation - dynamically infer grid size
        grid_cells = len(observation) - 3  # last 3 are agent_row, agent_col, inventory
        grid_size = int(np.sqrt(grid_cells))
        grid = observation[:grid_cells].reshape(grid_size, grid_size)
        agent_row = int(observation[grid_cells])
        agent_col = int(observation[grid_cells + 1])
        inventory = int(observation[grid_cells + 2])

        if inventory == -1:  # Not holding anything
            # Find all cells with objects
            object_positions = []
            for r in range(grid_size):
                for c in range(grid_size):
                    if grid[r, c] >= 0:
                        object_positions.append((r, c))

            # If no objects on grid, do nothing (move up as default)
            if not object_positions:
                return 0

            # Check if agent is on an object
            if grid[agent_row, agent_col] >= 0:
                return 4  # pickup

            # Find nearest object (Manhattan distance)
            min_distance = float("inf")
            target_row, target_col = object_positions[0]
            for r, c in object_positions:
                distance = abs(r - agent_row) + abs(c - agent_col)
                if distance < min_distance:
                    min_distance = distance
                    target_row, target_col = r, c

        else:  # Holding an object
            # Find first empty cell in target area (row 0, then row 1, etc.)
            target_row, target_col = 0, 0
            found_target = False
            for r in range(grid_size):
                for c in range(grid_size):
                    if grid[r, c] == -1:
                        target_row, target_col = r, c
                        found_target = True
                        break
                if found_target:
                    break

            # If agent is at target and cell is empty, place object
            at_target = agent_row == target_row and agent_col == target_col
            if at_target and grid[agent_row, agent_col] == -1:
                return 5  # place

        # Navigate toward target
        if target_row < agent_row:
            return 0  # up
        elif target_row > agent_row:
            return 1  # down
        elif target_col < agent_col:
            return 2  # left
        elif target_col > agent_col:
            return 3  # right
        else:
            # Already at target but can't perform target action
            return 0  # default: move up


class SeqHeuristicAgent:
    """Heuristic agent for SequenceSorting environment.

    Strategy:
    - Single-pass bubble sort step: scan sequence left to right
    - When first inversion found (obs[i] > obs[i+1]), swap at position i
    - If no inversions, perform no-op
    """

    def __init__(self) -> None:
        """Initialize the sequence heuristic agent."""
        pass

    def act(self, observation: np.ndarray) -> int:
        """Select action based on bubble sort heuristic.

        Args:
            observation: 1D array (zero-padded sequence)

        Returns:
            Action integer in [0, len(observation)-1]:
                0 to len-2: swap at position i
                len-1: no-op
        """
        # Scan for first inversion, but only in non-zero padded region
        # We need to avoid false inversions from zero-padding
        # Strategy: scan left to right, but skip inversions where next value is 0
        # unless current value is also 0
        for i in range(len(observation) - 1):
            if observation[i] > observation[i + 1]:
                # Check if this is a real inversion or just zero-padding boundary
                # If next value is 0 and current is not, might be at padding boundary
                # Only swap if both are non-zero, or if we have evidence this is real
                if observation[i + 1] == 0 and observation[i] != 0:
                    # Could be padding boundary, skip it
                    # But check if there are more non-zeros ahead
                    has_more_data = any(observation[j] != 0 for j in range(i + 2, len(observation)))
                    if not has_more_data:
                        # This is the padding boundary, skip
                        continue
                return i  # swap at position i

        # No inversions found
        return len(observation) - 1  # no-op


class RegulationHeuristicAgent:
    """Heuristic agent for CoupledRegulation environment.

    Strategy: Greedy — adjust variable with largest absolute deviation
    toward target. Ignores coupling effects.
    """

    def __init__(self) -> None:
        pass

    def act(self, observation: np.ndarray) -> int:
        """Select action based on greedy deviation reduction.

        Observation: [x_1..x_8, t_1..t_8] (16 floats, zero-padded)
        Actions: 0-7 increase var k, 8-15 decrease var k, 16 no-op
        """
        n_max = 8
        values = observation[:n_max]
        targets = observation[n_max : 2 * n_max]

        # Find active variables (non-zero targets or non-zero values)
        # Actually, just find variable with largest absolute deviation
        deviations = values - targets
        abs_devs = np.abs(deviations)

        # Find variable with largest deviation
        k = int(np.argmax(abs_devs))

        # If deviation is negligible, no-op
        if abs_devs[k] < 0.05:
            return 16  # no-op

        # If value > target, decrease; if value < target, increase
        if deviations[k] > 0:
            return k + n_max  # decrease action
        else:
            return k  # increase action


class GridCleaningHeuristicAgent:
    """Heuristic agent for GridCleaning environment.

    Strategy: BFS to find nearest dirty cell, navigate to it, clean it. Repeat.
    Optimal pathfinding but greedy dirt selection (always nearest, not globally optimal).
    """

    def __init__(self) -> None:
        self._path: list[int] = []
        self._cleaning_target: tuple[int, int] | None = None

    def act(self, observation: np.ndarray) -> int:
        """Select action using BFS nearest-dirty heuristic.

        Args:
            observation: 66-element array [flattened 8x8 grid, agent_row_norm, agent_col_norm]

        Returns:
            Action integer 0-4: up/down/left/right/clean
        """
        grid_size = 8
        grid_flat = observation[:64]
        agent_row = int(round(observation[64] * (grid_size - 1)))
        agent_col = int(round(observation[65] * (grid_size - 1)))

        # Reconstruct grid from normalized observation
        grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        for r in range(grid_size):
            for c in range(grid_size):
                val = grid_flat[r * grid_size + c]
                if val < -0.5:
                    grid[r, c] = -1  # obstacle or wall
                elif val < 0.15:
                    grid[r, c] = 0  # clean
                elif val < 0.50:
                    grid[r, c] = 1  # light
                elif val < 0.85:
                    grid[r, c] = 2  # medium
                else:
                    grid[r, c] = 3  # heavy

        # If standing on a dirty cell, clean it
        if grid[agent_row, agent_col] > 0:
            return 4  # clean

        # BFS to find nearest dirty cell
        target = self._bfs_nearest_dirty(grid, agent_row, agent_col, grid_size)
        if target is None:
            return 0  # nothing to clean, default move

        target_row, target_col = target

        # Navigate toward target
        if target_row < agent_row:
            return 0  # up
        elif target_row > agent_row:
            return 1  # down
        elif target_col < agent_col:
            return 2  # left
        elif target_col > agent_col:
            return 3  # right

        return 0  # fallback

    @staticmethod
    def _bfs_nearest_dirty(
        grid: np.ndarray, start_row: int, start_col: int, grid_size: int
    ) -> tuple[int, int] | None:
        """BFS to find the nearest dirty cell, respecting obstacles."""
        visited = set()
        queue: deque[tuple[int, int]] = deque()
        queue.append((start_row, start_col))
        visited.add((start_row, start_col))

        while queue:
            r, c = queue.popleft()
            if grid[r, c] > 0:
                return (r, c)

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < grid_size
                    and 0 <= nc < grid_size
                    and (nr, nc) not in visited
                    and grid[nr, nc] >= 0  # not obstacle/wall
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return None


class TilePuzzleHeuristicAgent:
    """Heuristic agent for TilePuzzle environment (fixed 4x4).

    Strategy: Greedy Manhattan — try all 4 possible slides, pick the one
    that reduces total Manhattan distance the most. Break ties randomly.
    """

    def __init__(self) -> None:
        self._rng = np.random.default_rng(42)

    def act(self, observation: np.ndarray) -> int:
        """Select action that maximally reduces total Manhattan distance.

        Observation: 16 floats (flattened 4x4 grid, 0=empty, 1-15=tiles)
        Actions: 0=slide from above, 1=below, 2=left, 3=right into empty
        """
        grid = observation.reshape(4, 4).astype(int)
        return _greedy_manhattan_action(grid, 4, self._rng)

    @staticmethod
    def _total_manhattan(grid: np.ndarray) -> int:
        """Compute total Manhattan distance for all tiles."""
        return _total_manhattan(grid, 4)


class SlidingPuzzleHeuristicAgent:
    """Heuristic agent for variable-size SlidingPuzzle environment.

    Same greedy Manhattan strategy as TilePuzzleHeuristicAgent but reads
    grid size from the observation (element 25 = w/5).
    """

    def __init__(self) -> None:
        self._rng = np.random.default_rng(42)

    def act(self, observation: np.ndarray) -> int:
        """Select action that maximally reduces total Manhattan distance.

        Observation: 26 floats [25 grid cells (normalized), grid_size/5]
        Actions: 0=slide from above, 1=below, 2=left, 3=right into empty
        """
        w = round(observation[25] * 5)
        n = w * w
        raw = observation[:n] * 24  # denormalize
        grid = np.round(raw).astype(int).reshape(w, w)
        return _greedy_manhattan_action(grid, w, self._rng)


class RestrictedSlidingPuzzleHeuristicAgent:
    """Heuristic agent restricted to specific grid sizes.

    Uses greedy Manhattan on allowed grid widths, random actions on others.
    Creates agents with different η/H(C) profiles for joint I_cd testing.
    """

    def __init__(self, allowed_widths: set[int], seed: int = 42) -> None:
        self._allowed_widths = allowed_widths
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        w = round(observation[25] * 5)
        if w in self._allowed_widths:
            n = w * w
            raw = observation[:n] * 24
            grid = np.round(raw).astype(int).reshape(w, w)
            return _greedy_manhattan_action(grid, w, self._rng)
        return int(self._rng.integers(0, 4))


def _total_manhattan(grid: np.ndarray, w: int) -> int:
    """Compute total Manhattan distance for all tiles in a w x w grid."""
    total = 0
    for r in range(w):
        for c in range(w):
            tile = grid[r, c]
            if tile == 0:
                continue
            target_r = (tile - 1) // w
            target_c = (tile - 1) % w
            total += abs(r - target_r) + abs(c - target_c)
    return total


def _greedy_manhattan_action(grid: np.ndarray, w: int, rng: np.random.Generator) -> int:
    """Pick the slide action that best reduces total Manhattan distance."""
    empty_pos = None
    for r in range(w):
        for c in range(w):
            if grid[r, c] == 0:
                empty_pos = (r, c)
                break
        if empty_pos is not None:
            break

    if empty_pos is None:
        return 0

    er, ec = empty_pos
    current_dist = _total_manhattan(grid, w)

    best_actions: list[int] = []
    best_improvement = -float("inf")
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for action, (dr, dc) in enumerate(deltas):
        tr, tc = er + dr, ec + dc
        if 0 <= tr < w and 0 <= tc < w:
            new_grid = grid.copy()
            new_grid[er, ec] = new_grid[tr, tc]
            new_grid[tr, tc] = 0
            new_dist = _total_manhattan(new_grid, w)
            improvement = current_dist - new_dist

            if improvement > best_improvement:
                best_improvement = improvement
                best_actions = [action]
            elif improvement == best_improvement:
                best_actions.append(action)

    if not best_actions:
        return 0

    return int(rng.choice(best_actions))
