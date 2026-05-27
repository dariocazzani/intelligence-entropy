"""Tests for the Resource Grid environment."""

import numpy as np
import pytest

from ie.environments.resource_grid import ResourceGrid
from ie.measure import environmental_entropy


class TestResourceGrid:
    """Test suite for ResourceGrid environment."""

    def test_reset_produces_valid_state(self):
        """After reset, grid has correct number of objects, agent at (0,0), inventory empty."""
        env = ResourceGrid(seed=42)
        obs = env.reset({"name": "basic"})

        # Check observation shape (8×8 + 3 = 67)
        assert obs.shape == (67,)

        # Extract components
        grid = obs[:64].reshape(8, 8)
        agent_row, agent_col, inventory = obs[64], obs[65], obs[66]

        # Check agent position
        assert agent_row == 0
        assert agent_col == 0

        # Check inventory is empty
        assert inventory == -1

        # Check grid has exactly 15 objects (basic config)
        num_objects = np.sum(grid != -1)
        assert num_objects == 15

        # Check objects are type 0 (single type in basic)
        objects = grid[grid != -1]
        assert np.all(objects == 0)

    def test_move_actions(self):
        """Test up/down/left/right modify agent position correctly, wall collision is no-op."""
        env = ResourceGrid(seed=42)
        env.reset({"name": "basic"})

        # Move right from (0, 0) -> (0, 1)
        obs, _, _, _ = env.step(3)  # right
        assert obs[64] == 0  # row unchanged (updated index for 8×8)
        assert obs[65] == 1  # col = 1

        # Move down from (0, 1) -> (1, 1)
        obs, _, _, _ = env.step(1)  # down
        assert obs[64] == 1  # row = 1
        assert obs[65] == 1  # col = 1

        # Move left from (1, 1) -> (1, 0)
        obs, _, _, _ = env.step(2)  # left
        assert obs[64] == 1  # row = 1
        assert obs[65] == 0  # col = 0

        # Move up from (1, 0) -> (0, 0)
        obs, _, _, _ = env.step(0)  # up
        assert obs[64] == 0  # row = 0
        assert obs[65] == 0  # col = 0

        # Try to move up from (0, 0) - should be no-op (wall)
        obs, _, _, _ = env.step(0)  # up
        assert obs[64] == 0  # row = 0 (unchanged)
        assert obs[65] == 0  # col = 0 (unchanged)

        # Try to move left from (0, 0) - should be no-op (wall)
        obs, _, _, _ = env.step(2)  # left
        assert obs[64] == 0  # row = 0 (unchanged)
        assert obs[65] == 0  # col = 0 (unchanged)

    def test_pick_up_and_place(self):
        """Pick up object, verify inventory, place it elsewhere, verify grid changed."""
        env = ResourceGrid(seed=42)
        obs = env.reset({"name": "basic"})

        # Find a cell with an object (updated for 8×8)
        grid = obs[:64].reshape(8, 8)
        obj_positions = np.argwhere(grid != -1)
        assert len(obj_positions) > 0

        # Move agent to that position
        target_row, target_col = obj_positions[0]
        env.agent_pos = (int(target_row), int(target_col))

        # Pick up object
        obs, _, _, _ = env.step(4)  # pick_up
        inventory = obs[66]  # Updated index
        assert inventory == 0  # Should have picked up object type 0

        # Verify cell is now empty
        grid = obs[:64].reshape(8, 8)
        assert grid[target_row, target_col] == -1

        # Move to a different empty cell
        empty_positions = np.argwhere(grid == -1)
        new_row, new_col = empty_positions[0]
        if (new_row, new_col) == (target_row, target_col):
            new_row, new_col = empty_positions[1]
        env.agent_pos = (int(new_row), int(new_col))

        # Place object
        obs, _, _, _ = env.step(5)  # place
        inventory = obs[66]  # Updated index
        assert inventory == -1  # Inventory should be empty

        # Verify cell now has object
        grid = obs[:64].reshape(8, 8)
        assert grid[new_row, new_col] == 0

    def test_entropy_known_state(self):
        """Construct a known grid state, verify row-occupancy entropy matches manual calc."""
        env = ResourceGrid(seed=42)
        env.reset({"name": "basic"})

        # Place 3 objects in row 0, 2 objects in row 1
        env.grid = np.full((8, 8), -1, dtype=np.int32)
        env.grid[0, 0] = 0
        env.grid[0, 1] = 0
        env.grid[0, 2] = 0
        env.grid[1, 0] = 0
        env.grid[1, 1] = 0

        # Row occupancy: row 0 has 3 objects, row 1 has 2 objects
        # Non-zero counts: [3, 2]
        expected_counts = np.array([3, 2])
        expected_entropy = environmental_entropy(expected_counts)

        computed_entropy = env.compute_entropy()
        assert np.isclose(computed_entropy, expected_entropy, rtol=1e-6)

    def test_organizing_reduces_entropy(self):
        """Concentrated objects (few rows) have lower entropy than scattered (many rows)."""
        env = ResourceGrid(seed=42)
        env.reset({"name": "basic"})

        # Scattered: 2 objects in each of 5 rows
        env.grid = np.full((8, 8), -1, dtype=np.int32)
        for r in range(5):
            env.grid[r, 0] = 0
            env.grid[r, 1] = 0
        scattered_entropy = env.compute_entropy()

        # Concentrated: all 10 objects packed into row 0
        env.grid = np.full((8, 8), -1, dtype=np.int32)
        for c in range(8):
            env.grid[0, c] = 0
        # Remaining 2 in row 1
        env.grid[1, 0] = 0
        env.grid[1, 1] = 0
        concentrated_entropy = env.compute_entropy()

        assert concentrated_entropy < scattered_entropy

    def test_entropy_zero_when_empty(self):
        """Empty grid has zero entropy."""
        env = ResourceGrid(seed=42)
        env.reset({"name": "basic"})
        env.grid = np.full((8, 8), -1, dtype=np.int32)
        assert env.compute_entropy() == 0.0

    def test_perturbation_disrupts(self):
        """In perturbation config, run 50+ steps and verify grid changed at step 150."""
        env = ResourceGrid(seed=42)
        env.reset({"name": "perturbation"})

        # Get initial grid state
        obs = env.reset({"name": "perturbation"})
        _ = obs[:64].copy()  # Updated for 8×8

        # Run exactly 149 steps (perturbation happens at step 150)
        for _ in range(149):
            obs, _, _, _ = env.step(0)  # just move up (no-op at wall)

        grid_at_149 = obs[:64].copy()

        # Run one more step (step 150)
        obs, _, _, _ = env.step(0)
        grid_at_150 = obs[:64].copy()

        # Grid should have changed at step 150 (perturbation applied)
        # At least some cells should be different
        assert not np.array_equal(grid_at_149, grid_at_150)

    def test_context_labels(self):
        """Each config returns correct context label."""
        env = ResourceGrid(seed=42)

        env.reset({"name": "basic"})
        assert env.get_context_label() == 0

        env.reset({"name": "multi_type"})
        assert env.get_context_label() == 1

        env.reset({"name": "perturbation"})
        assert env.get_context_label() == 2

    def test_observation_shape(self):
        """Observation has expected shape (N*N + 3)."""
        env = ResourceGrid(seed=42)

        # Basic config (8×8)
        obs = env.reset({"name": "basic"})
        assert obs.shape == (67,)  # 64 + 3
        assert env.observation_shape == (67,)

        # Multi_type config (also 8×8)
        obs = env.reset({"name": "multi_type"})
        assert obs.shape == (67,)
        assert env.observation_shape == (67,)

    def test_action_space_size(self):
        """Returns 6."""
        env = ResourceGrid(seed=42)
        assert env.action_space_size == 6

    def test_episode_length(self):
        """done=True after 300 steps."""
        env = ResourceGrid(seed=42)
        env.reset({"name": "basic"})

        # Run 299 steps
        for _ in range(299):
            _, _, done, _ = env.step(0)
            assert done is False

        # Step 300 should set done=True
        _, _, done, _ = env.step(0)
        assert done is True

    def test_work_in_info(self):
        """Every step returns info dict with 'work' key = 1.0."""
        env = ResourceGrid(seed=42)
        env.reset({"name": "basic"})

        # Test each action type
        for action in range(6):
            env.reset({"name": "basic"})
            _, _, _, info = env.step(action)
            assert "work" in info
            assert info["work"] == 1.0

    def test_pick_up_no_op_when_inventory_full(self):
        """Pick up action is no-op when inventory is already full."""
        env = ResourceGrid(seed=42)
        env.reset({"name": "basic"})

        # Find an object and pick it up (updated for 8×8)
        obs = env.reset({"name": "basic"})
        grid = obs[:64].reshape(8, 8)
        obj_positions = np.argwhere(grid != -1)
        target_row, target_col = obj_positions[0]
        env.agent_pos = (int(target_row), int(target_col))

        # Pick up first object
        obs, _, _, _ = env.step(4)
        assert obs[66] == 0  # inventory has object (updated index)

        # Move to another object
        target_row2, target_col2 = obj_positions[1]
        env.agent_pos = (int(target_row2), int(target_col2))

        # Try to pick up again - should be no-op
        obs_before = obs.copy()
        obs, _, _, _ = env.step(4)

        # Inventory should be unchanged
        assert obs[66] == obs_before[66]

        # Object should still be on grid
        grid = obs[:64].reshape(8, 8)
        assert grid[target_row2, target_col2] != -1

    def test_place_no_op_when_cell_occupied(self):
        """Place action is no-op when target cell is occupied."""
        env = ResourceGrid(seed=42)
        env.reset({"name": "basic"})

        # Find an object, pick it up (updated for 8×8)
        obs = env.reset({"name": "basic"})
        grid = obs[:64].reshape(8, 8)
        obj_positions = np.argwhere(grid != -1)
        target_row, target_col = obj_positions[0]
        env.agent_pos = (int(target_row), int(target_col))

        obs, _, _, _ = env.step(4)  # pick up
        assert obs[66] == 0  # Updated index

        # Move to a cell that has an object
        target_row2, target_col2 = obj_positions[1]
        env.agent_pos = (int(target_row2), int(target_col2))

        # Try to place - should be no-op (cell occupied)
        obs_before = obs.copy()
        obs, _, _, _ = env.step(5)

        # Inventory should be unchanged (still has object)
        assert obs[66] == obs_before[66]
        assert obs[66] == 0

    def test_seeded_reset_is_reproducible(self):
        """Two environments with same seed produce identical initial states."""
        env1 = ResourceGrid(seed=123)
        env2 = ResourceGrid(seed=123)

        obs1 = env1.reset({"name": "multi_type"})
        obs2 = env2.reset({"name": "multi_type"})

        assert np.array_equal(obs1, obs2)

    def test_different_seeds_produce_different_states(self):
        """Two environments with different seeds produce different initial states."""
        env1 = ResourceGrid(seed=123)
        env2 = ResourceGrid(seed=456)

        obs1 = env1.reset({"name": "multi_type"})
        obs2 = env2.reset({"name": "multi_type"})

        # Should be different (with very high probability)
        assert not np.array_equal(obs1, obs2)

    def test_multi_type_has_three_types(self):
        """Multi-type config places objects of three different types."""
        env = ResourceGrid(seed=42)
        obs = env.reset({"name": "multi_type"})

        grid = obs[:64].reshape(8, 8)  # Updated for 8×8
        objects = grid[grid != -1]

        # Should have 25 objects total (updated from 8)
        assert len(objects) == 25

        # Should have types 0, 1, and 2 (3 types now)
        assert 0 in objects
        assert 1 in objects
        assert 2 in objects

    def test_invalid_action_raises_error(self):
        """Invalid action index raises ValueError."""
        env = ResourceGrid(seed=42)
        env.reset({"name": "basic"})

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(6)

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)

    def test_invalid_config_raises_error(self):
        """Invalid config name raises ValueError."""
        env = ResourceGrid(seed=42)

        with pytest.raises(ValueError, match="Unknown config"):
            env.reset({"name": "nonexistent_config"})
