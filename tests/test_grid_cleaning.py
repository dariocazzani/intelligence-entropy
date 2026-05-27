"""Tests for the Multi-Room Grid Cleaning environment."""

import numpy as np
import pytest

from ie.agents.heuristic_agent import GridCleaningHeuristicAgent
from ie.environments.grid_cleaning import (
    CLEAN,
    CLEAN_COST,
    GRID_SIZE,
    HEAVY,
    LIGHT,
    MAX_STEPS,
    MEDIUM,
    MOVE_COST,
    OBSTACLE,
    WALL,
    GridCleaning,
)


class TestGridCleaningBasic:
    """Basic environment interface tests."""

    def test_reset_returns_correct_obs_shape(self):
        env = GridCleaning(seed=42)
        obs = env.reset({"layout_id": 0})
        assert obs.shape == (66,)
        assert env.observation_shape == (66,)

    def test_action_space_size(self):
        env = GridCleaning(seed=42)
        assert env.action_space_size == 5

    def test_default_config(self):
        env = GridCleaning(seed=42)
        obs = env.reset(None)
        assert obs.shape == (66,)
        assert env.get_context_label() == 0

    def test_invalid_layout_raises_error(self):
        env = GridCleaning(seed=42)
        with pytest.raises(ValueError, match="layout_id"):
            env.reset({"layout_id": 10})
        with pytest.raises(ValueError, match="layout_id"):
            env.reset({"layout_id": -1})

    def test_invalid_action_raises_error(self):
        env = GridCleaning(seed=42)
        env.reset()
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(5)
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)


class TestGridCleaningLayouts:
    """Test all 10 layouts generate valid grids."""

    @pytest.mark.parametrize("layout_id", range(10))
    def test_layout_generates_valid_grid(self, layout_id: int):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": layout_id})
        assert env.grid is not None
        assert env.grid.shape == (GRID_SIZE, GRID_SIZE)
        # All values should be in valid range
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                assert env.grid[r, c] in (CLEAN, LIGHT, MEDIUM, HEAVY, OBSTACLE, WALL)

    @pytest.mark.parametrize("layout_id", range(10))
    def test_layout_has_walkable_cells(self, layout_id: int):
        """Every layout must have at least some walkable cells."""
        env = GridCleaning(seed=42)
        env.reset({"layout_id": layout_id})
        walkable = np.sum(env.grid >= CLEAN)
        assert walkable > 0

    @pytest.mark.parametrize("layout_id", range(10))
    def test_context_label_matches_layout(self, layout_id: int):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": layout_id})
        assert env.get_context_label() == layout_id

    def test_layout_names_count(self):
        assert len(GridCleaning.LAYOUT_NAMES) == 10

    def test_obstacle_layouts_have_obstacles(self):
        """Layouts 2-4 and 7 should have obstacles."""
        for layout_id in [2, 3, 4, 7]:
            env = GridCleaning(seed=42)
            env.reset({"layout_id": layout_id})
            n_obstacles = np.sum(env.grid == OBSTACLE)
            assert n_obstacles > 0, f"Layout {layout_id} should have obstacles"

    def test_wall_layouts_have_walls(self):
        """Layouts 5, 6, 8, 9 should have walls."""
        for layout_id in [5, 6, 8, 9]:
            env = GridCleaning(seed=42)
            env.reset({"layout_id": layout_id})
            n_walls = np.sum(env.grid == WALL)
            assert n_walls > 0, f"Layout {layout_id} should have walls"


class TestGridCleaningActions:
    """Test movement and cleaning actions."""

    def test_move_blocked_by_obstacle(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})  # empty layout, no obstacles
        # Place an obstacle manually for testing
        env.grid[0, 0] = OBSTACLE
        env.agent_row, env.agent_col = 1, 0
        _, _, _, info = env.step(0)  # try to move up into obstacle
        assert env.agent_row == 1  # didn't move
        assert info["work"] == MOVE_COST  # still costs work

    def test_move_blocked_by_wall(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        env.grid[0, 0] = WALL
        env.agent_row, env.agent_col = 1, 0
        _, _, _, info = env.step(0)  # try to move up into wall
        assert env.agent_row == 1
        assert info["work"] == MOVE_COST

    def test_move_blocked_by_boundary(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        env.agent_row, env.agent_col = 0, 0
        env.step(0)  # try to move up from top row
        assert env.agent_row == 0  # didn't move

    def test_valid_move(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        env.agent_row, env.agent_col = 4, 4
        env.grid[3, 4] = CLEAN  # ensure target is walkable
        _, _, _, info = env.step(0)  # move up
        assert env.agent_row == 3
        assert env.agent_col == 4
        assert info["work"] == MOVE_COST

    def test_all_four_directions(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        # Place agent in center with clear surroundings
        env.agent_row, env.agent_col = 4, 4
        for r in range(3, 6):
            for c in range(3, 6):
                env.grid[r, c] = CLEAN

        # Up
        env.agent_row, env.agent_col = 4, 4
        env.step(0)
        assert (env.agent_row, env.agent_col) == (3, 4)

        # Down
        env.agent_row, env.agent_col = 4, 4
        env.step(1)
        assert (env.agent_row, env.agent_col) == (5, 4)

        # Left
        env.agent_row, env.agent_col = 4, 4
        env.step(2)
        assert (env.agent_row, env.agent_col) == (4, 3)

        # Right
        env.agent_row, env.agent_col = 4, 4
        env.step(3)
        assert (env.agent_row, env.agent_col) == (4, 5)

    def test_clean_reduces_dirt(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        env.agent_row, env.agent_col = 3, 3
        env.grid[3, 3] = HEAVY

        _, _, _, info = env.step(4)  # clean
        assert env.grid[3, 3] == MEDIUM
        assert info["work"] == CLEAN_COST

        env.step(4)  # clean again
        assert env.grid[3, 3] == LIGHT

        env.step(4)  # clean again
        assert env.grid[3, 3] == CLEAN

    def test_clean_on_clean_cell_no_effect(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        env.agent_row, env.agent_col = 3, 3
        env.grid[3, 3] = CLEAN

        _, _, _, info = env.step(4)
        assert env.grid[3, 3] == CLEAN
        assert info["work"] == CLEAN_COST  # still costs work

    def test_move_cost_is_half_clean_cost(self):
        assert MOVE_COST == 0.5
        assert CLEAN_COST == 1.0


class TestGridCleaningEntropy:
    """Test entropy computation."""

    def test_all_clean_entropy_is_zero(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        # Set all cleanable cells to CLEAN
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if env.grid[r, c] >= CLEAN:
                    env.grid[r, c] = CLEAN
        assert env.compute_entropy() == 0.0

    def test_uniform_distribution_has_max_entropy(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        # Set cells to be uniformly distributed across 4 states
        idx = 0
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if env.grid[r, c] >= CLEAN:
                    env.grid[r, c] = idx % 4
                    idx += 1
        entropy = env.compute_entropy()
        # Should be close to ln(4) = 1.386
        assert abs(entropy - np.log(4)) < 0.1

    def test_cleaning_reduces_entropy(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 1})  # mixed dirt
        initial_entropy = env.compute_entropy()
        assert initial_entropy > 0

        # Clean many cells
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if env.grid[r, c] > CLEAN:
                    env.grid[r, c] = CLEAN

        final_entropy = env.compute_entropy()
        assert final_entropy < initial_entropy
        assert final_entropy == 0.0

    def test_entropy_nonnegative(self):
        env = GridCleaning(seed=42)
        for layout_id in range(10):
            env.reset({"layout_id": layout_id})
            assert env.compute_entropy() >= 0.0

    def test_entropy_ignores_obstacles_and_walls(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        # Set everything to clean except one obstacle
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                env.grid[r, c] = CLEAN
        env.grid[0, 0] = OBSTACLE
        env.grid[0, 1] = WALL
        # Entropy should be 0 because all cleanable cells are clean
        assert env.compute_entropy() == 0.0


class TestGridCleaningEpisode:
    """Test episode mechanics."""

    def test_episode_terminates_after_max_steps(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        for i in range(MAX_STEPS - 1):
            _, _, done, _ = env.step(4)  # clean action
            assert not done
        _, _, done, _ = env.step(4)
        assert done

    def test_reward_for_successful_clean(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        env.agent_row, env.agent_col = 3, 3
        env.grid[3, 3] = LIGHT
        _, reward, _, _ = env.step(4)
        assert reward == 1.0

    def test_no_reward_for_cleaning_clean_cell(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        env.agent_row, env.agent_col = 3, 3
        env.grid[3, 3] = CLEAN
        _, reward, _, _ = env.step(4)
        assert reward == 0.0


class TestGridCleaningDeterminism:
    """Test reproducibility."""

    def test_same_seed_same_layout(self):
        env1 = GridCleaning(seed=123)
        env2 = GridCleaning(seed=123)
        obs1 = env1.reset({"layout_id": 3})
        obs2 = env2.reset({"layout_id": 3})
        assert np.array_equal(obs1, obs2)
        assert np.array_equal(env1.grid, env2.grid)

    def test_different_seeds_different_layouts(self):
        env1 = GridCleaning(seed=123)
        env2 = GridCleaning(seed=456)
        env1.reset({"layout_id": 3})
        env2.reset({"layout_id": 3})
        assert not np.array_equal(env1.grid, env2.grid)


class TestGridCleaningObservation:
    """Test observation encoding."""

    def test_observation_dtype(self):
        env = GridCleaning(seed=42)
        obs = env.reset({"layout_id": 0})
        assert obs.dtype == np.float32

    def test_agent_position_in_observation(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        env.agent_row, env.agent_col = 0, 0
        obs = env._get_observation()
        assert obs[64] == pytest.approx(0.0)
        assert obs[65] == pytest.approx(0.0)

        env.agent_row, env.agent_col = GRID_SIZE - 1, GRID_SIZE - 1
        obs = env._get_observation()
        assert obs[64] == pytest.approx(1.0)
        assert obs[65] == pytest.approx(1.0)

    def test_grid_values_in_observation(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        # Set a known cell
        env.grid[0, 0] = HEAVY
        obs = env._get_observation()
        assert obs[0] == pytest.approx(1.0)  # HEAVY = 1.0

        env.grid[0, 0] = CLEAN
        obs = env._get_observation()
        assert obs[0] == pytest.approx(0.0)

        env.grid[0, 0] = OBSTACLE
        obs = env._get_observation()
        assert obs[0] == pytest.approx(-1.0)


class TestGridCleaningHeuristicAgent:
    """Test the BFS heuristic agent."""

    def test_agent_cleans_when_on_dirt(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        env.agent_row, env.agent_col = 3, 3
        env.grid[3, 3] = LIGHT
        obs = env._get_observation()
        agent = GridCleaningHeuristicAgent()
        action = agent.act(obs)
        assert action == 4  # should clean

    def test_agent_moves_toward_dirt(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        # Clear entire grid
        env.grid[:] = CLEAN
        # Place one dirty cell
        env.grid[0, 3] = LIGHT
        env.agent_row, env.agent_col = 4, 3
        obs = env._get_observation()
        agent = GridCleaningHeuristicAgent()
        action = agent.act(obs)
        assert action == 0  # should move up toward dirt

    def test_agent_reduces_entropy_over_episode(self):
        """Heuristic agent should reduce entropy in a simple layout."""
        env = GridCleaning(seed=42)
        obs = env.reset({"layout_id": 0})
        initial_entropy = env.compute_entropy()

        agent = GridCleaningHeuristicAgent()
        for _ in range(MAX_STEPS):
            action = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break

        final_entropy = env.compute_entropy()
        assert final_entropy < initial_entropy

    def test_agent_handles_all_layouts(self):
        """Agent should not crash on any layout."""
        agent = GridCleaningHeuristicAgent()
        for layout_id in range(10):
            env = GridCleaning(seed=42)
            obs = env.reset({"layout_id": layout_id})
            for _ in range(10):
                action = agent.act(obs)
                assert 0 <= action <= 4
                obs, _, done, _ = env.step(action)
                if done:
                    break


class TestGridCleaningHelpers:
    """Test helper methods."""

    def test_get_grid_copy(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        copy = env.get_grid_copy()
        assert np.array_equal(copy, env.grid)
        # Modifying copy shouldn't affect env
        copy[0, 0] = 99
        assert env.grid[0, 0] != 99

    def test_get_agent_position(self):
        env = GridCleaning(seed=42)
        env.reset({"layout_id": 0})
        pos = env.get_agent_position()
        assert pos == (env.agent_row, env.agent_col)
