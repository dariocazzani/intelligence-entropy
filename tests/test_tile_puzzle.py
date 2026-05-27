"""Tests for the Tile Puzzle environment."""

import numpy as np
import pytest

from ie.environments.tile_puzzle import TilePuzzle


class TestTilePuzzle:
    """Test suite for TilePuzzle environment."""

    def test_reset_returns_correct_obs_shape(self):
        """After reset, observation has shape (16,)."""
        env = TilePuzzle(seed=42)
        obs = env.reset({"name": "easy"})
        assert obs.shape == (16,)
        assert env.observation_shape == (16,)

    def test_action_space_size(self):
        """Action space has 4 actions."""
        env = TilePuzzle(seed=42)
        assert env.action_space_size == 4

    def test_all_configs_work(self):
        """All configs (easy/medium/hard) reset successfully."""
        env = TilePuzzle(seed=42)

        # Easy config
        obs = env.reset({"name": "easy"})
        assert obs.shape == (16,)
        assert env.context_label == 0

        # Medium config
        obs = env.reset({"name": "medium"})
        assert obs.shape == (16,)
        assert env.context_label == 1

        # Hard config
        obs = env.reset({"name": "hard"})
        assert obs.shape == (16,)
        assert env.context_label == 2

    def test_solved_state_has_zero_entropy(self):
        """Manually set to solved state, entropy = 0."""
        env = TilePuzzle(seed=42)
        env.reset({"name": "easy"})

        # Set to solved state
        env.grid = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]],
            dtype=np.int32,
        )

        entropy = env.compute_entropy()
        assert entropy == 0.0

    def test_scrambled_has_positive_entropy(self):
        """After reset with any config, entropy > 0."""
        env = TilePuzzle(seed=42)
        env.reset({"name": "easy"})

        entropy = env.compute_entropy()
        assert entropy > 0.0

    def test_valid_slide_moves_tile(self):
        """Valid slide action moves a tile into empty space."""
        env = TilePuzzle(seed=42)
        env.reset({"name": "easy"})

        # Set up a known state: empty at (3, 3)
        env.grid = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]],
            dtype=np.int32,
        )
        env.empty_pos = (3, 3)

        # Action 0: slide from above (tile 15 at (3,2) moves down)
        # Wait, action 0 is slide from above, meaning tile above empty slides down
        # Empty at (3,3), tile above is at (2,3) = 12
        obs, _, _, info = env.step(0)
        assert info["work"] == 1.0
        grid = obs.reshape(4, 4).astype(int)
        assert grid[3, 3] == 12  # Tile 12 moved into empty
        assert grid[2, 3] == 0  # Empty is now at (2,3)

    def test_invalid_slide_is_noop(self):
        """Invalid slide (at edge) is no-op with work=0.0."""
        env = TilePuzzle(seed=42)
        env.reset({"name": "easy"})

        # Set up state: empty at corner (0, 0)
        env.grid = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            dtype=np.int32,
        )
        env.empty_pos = (0, 0)

        # Store initial state
        initial_grid = env.grid.copy()

        # Action 0: slide from above (invalid, empty is at top edge)
        obs, _, _, info = env.step(0)
        assert info["work"] == 0.0
        assert np.array_equal(env.grid, initial_grid)

        # Action 2: slide from left (invalid, empty is at left edge)
        obs, _, _, info = env.step(2)
        assert info["work"] == 0.0
        assert np.array_equal(env.grid, initial_grid)

    def test_episode_terminates(self):
        """Episode terminates after 500 steps."""
        env = TilePuzzle(seed=42)
        env.reset({"name": "easy"})

        # Run 499 steps
        for _ in range(499):
            # Try a valid action (action 1: slide from below if possible)
            _, _, done, _ = env.step(1)
            if done:
                # Episode ended early, that's fine for this test
                break
        else:
            # If we made it through 499 steps, check that step 500 sets done=True
            _, _, done, _ = env.step(1)
            assert done

    def test_harder_scrambles_higher_entropy(self):
        """Harder scrambles (more moves) tend to have higher entropy."""
        # Run multiple trials and check average entropy
        easy_entropies = []
        hard_entropies = []

        for seed in range(10):
            env_easy = TilePuzzle(seed=seed)
            env_easy.reset({"name": "easy"})
            easy_entropies.append(env_easy.compute_entropy())

            env_hard = TilePuzzle(seed=seed)
            env_hard.reset({"name": "hard"})
            hard_entropies.append(env_hard.compute_entropy())

        avg_easy = np.mean(easy_entropies)
        avg_hard = np.mean(hard_entropies)

        # On average, hard should have higher entropy
        assert avg_hard >= avg_easy

    def test_context_labels(self):
        """Each config returns correct context label."""
        env = TilePuzzle(seed=42)

        env.reset({"name": "easy"})
        assert env.get_context_label() == 0

        env.reset({"name": "medium"})
        assert env.get_context_label() == 1

        env.reset({"name": "hard"})
        assert env.get_context_label() == 2

    def test_invalid_action_raises_error(self):
        """Invalid action index raises ValueError."""
        env = TilePuzzle(seed=42)
        env.reset({"name": "easy"})

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(4)

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)

    def test_invalid_config_raises_error(self):
        """Invalid config name raises ValueError."""
        env = TilePuzzle(seed=42)

        with pytest.raises(ValueError, match="Unknown config"):
            env.reset({"name": "nonexistent_config"})

    def test_default_config(self):
        """None config defaults to easy."""
        env = TilePuzzle(seed=42)
        obs = env.reset(None)
        assert obs.shape == (16,)
        assert env.get_context_label() == 0

    def test_deterministic_with_seed(self):
        """Same seed produces same initial state."""
        env1 = TilePuzzle(seed=123)
        env2 = TilePuzzle(seed=123)

        obs1 = env1.reset({"name": "medium"})
        obs2 = env2.reset({"name": "medium"})

        assert np.array_equal(obs1, obs2)

    def test_different_seeds_produce_different_states(self):
        """Different seeds produce different initial states."""
        env1 = TilePuzzle(seed=123)
        env2 = TilePuzzle(seed=456)

        obs1 = env1.reset({"name": "medium"})
        obs2 = env2.reset({"name": "medium"})

        # Should be different (with very high probability)
        assert not np.array_equal(obs1, obs2)

    def test_entropy_computation(self):
        """Entropy computation returns valid values."""
        env = TilePuzzle(seed=42)
        env.reset({"name": "easy"})

        entropy = env.compute_entropy()
        assert entropy >= 0.0
        assert np.isfinite(entropy)

    def test_work_in_info(self):
        """Valid actions return work=1.0, invalid return work=0.0."""
        env = TilePuzzle(seed=42)
        env.reset({"name": "easy"})

        # Set up state where we know which actions are valid
        env.grid = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 0], [13, 14, 15, 12]],
            dtype=np.int32,
        )
        env.empty_pos = (2, 3)

        # Action 0: slide from above (valid, tile 8 at (1,3))
        _, _, _, info = env.step(0)
        assert info["work"] == 1.0

        # Reset to same state
        env.grid = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 0], [13, 14, 15, 12]],
            dtype=np.int32,
        )
        env.empty_pos = (2, 3)

        # Action 3: slide from right (invalid, at right edge)
        _, _, _, info = env.step(3)
        assert info["work"] == 0.0

    def test_slide_directions(self):
        """Test all four slide directions work correctly."""
        env = TilePuzzle(seed=42)
        env.reset({"name": "easy"})

        # Set up state: empty at center (2, 2)
        env.grid = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 0, 11], [13, 14, 15, 12]],
            dtype=np.int32,
        )
        env.empty_pos = (2, 2)

        # Action 0: slide from above (tile 7 at (1,2))
        obs, _, _, _ = env.step(0)
        grid = obs.reshape(4, 4).astype(int)
        assert grid[2, 2] == 7
        assert grid[1, 2] == 0

        # Reset and test action 1: slide from below
        env.grid = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 0, 11], [13, 14, 15, 12]],
            dtype=np.int32,
        )
        env.empty_pos = (2, 2)

        obs, _, _, _ = env.step(1)
        grid = obs.reshape(4, 4).astype(int)
        assert grid[2, 2] == 15
        assert grid[3, 2] == 0

        # Reset and test action 2: slide from left
        env.grid = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 0, 11], [13, 14, 15, 12]],
            dtype=np.int32,
        )
        env.empty_pos = (2, 2)

        obs, _, _, _ = env.step(2)
        grid = obs.reshape(4, 4).astype(int)
        assert grid[2, 2] == 10
        assert grid[2, 1] == 0

        # Reset and test action 3: slide from right
        env.grid = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 0, 11], [13, 14, 15, 12]],
            dtype=np.int32,
        )
        env.empty_pos = (2, 2)

        obs, _, _, _ = env.step(3)
        grid = obs.reshape(4, 4).astype(int)
        assert grid[2, 2] == 11
        assert grid[2, 3] == 0
