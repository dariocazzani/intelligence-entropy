"""Tests for the variable-size SlidingPuzzle environment and SB3 wrapper."""

import numpy as np
import pytest

from ie.agents.heuristic_agent import (
    RestrictedSlidingPuzzleHeuristicAgent,
    SlidingPuzzleHeuristicAgent,
)
from ie.environments.sliding_puzzle import (
    CONFIGS,
    MAX_STEPS,
    MAX_TILE_VALUE,
    OBS_SIZE,
    W_MAX,
    SlidingPuzzle,
)


class TestSlidingPuzzle:
    """Test suite for SlidingPuzzle environment."""

    def test_obs_shape_all_configs(self):
        """All configs produce observations of shape (26,)."""
        env = SlidingPuzzle(seed=42)
        for cfg in CONFIGS:
            obs = env.reset(cfg)
            assert obs.shape == (OBS_SIZE,), f"Config {cfg['name']}: {obs.shape}"

    def test_observation_shape_property(self):
        env = SlidingPuzzle(seed=42)
        assert env.observation_shape == (OBS_SIZE,)

    def test_action_space_size(self):
        env = SlidingPuzzle(seed=42)
        assert env.action_space_size == 4

    def test_context_labels(self):
        """Each config returns its context_id."""
        env = SlidingPuzzle(seed=42)
        for cfg in CONFIGS:
            env.reset(cfg)
            assert env.get_context_label() == cfg["context_id"]

    def test_default_config(self):
        env = SlidingPuzzle(seed=42)
        obs = env.reset(None)
        assert obs.shape == (OBS_SIZE,)
        assert env.get_context_label() == 0

    def test_padding_for_small_grids(self):
        """Unused positions in obs are padded with -1."""
        env = SlidingPuzzle(seed=42)
        obs = env.reset(CONFIGS[0])  # 2x2
        # First 4 elements: grid values (>=0)
        assert all(obs[i] >= 0 for i in range(4))
        # Elements 4-24: padding = -1
        assert all(obs[i] == -1.0 for i in range(4, W_MAX * W_MAX))
        # Element 25: grid size indicator
        assert obs[25] == pytest.approx(2 / W_MAX)

    def test_grid_size_indicator(self):
        """Observation element 25 encodes grid size."""
        env = SlidingPuzzle(seed=42)
        for cfg in CONFIGS:
            obs = env.reset(cfg)
            expected = cfg["w"] / W_MAX
            assert obs[25] == pytest.approx(expected), f"Config {cfg['name']}"

    def test_solved_state_zero_entropy(self):
        """Manually set to solved state, entropy = 0."""
        env = SlidingPuzzle(seed=42)
        env.reset(CONFIGS[1])  # 3x3

        env.grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]], dtype=np.int32)
        assert env.compute_entropy() == 0.0

    def test_scrambled_positive_entropy(self):
        """After reset, entropy > 0 for non-trivial scrambles."""
        env = SlidingPuzzle(seed=42)
        for cfg in CONFIGS:
            env.reset(cfg)
            assert env.compute_entropy() > 0.0, f"Config {cfg['name']}"

    def test_valid_move_does_work(self):
        """Valid slide returns work=1.0."""
        env = SlidingPuzzle(seed=42)
        env.reset(CONFIGS[1])  # 3x3
        # Set solved state with blank at (2,2)
        env.grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]], dtype=np.int32)
        env.empty_pos = (2, 2)

        # Action 0: slide from above (tile 6 at (1,2))
        _, _, _, info = env.step(0)
        assert info["work"] == 1.0

    def test_invalid_move_still_costs_work(self):
        """Invalid slide at edge is no-op but still costs work=1.0."""
        env = SlidingPuzzle(seed=42)
        env.reset(CONFIGS[1])
        # Blank at (0,0)
        env.grid = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
        env.empty_pos = (0, 0)

        initial_grid = env.grid.copy()
        _, _, _, info = env.step(0)  # slide from above, invalid
        assert info["work"] == 1.0  # every step costs work
        assert np.array_equal(env.grid, initial_grid)

    def test_slide_moves_correct_tile(self):
        """Action 0 slides tile above blank into blank."""
        env = SlidingPuzzle(seed=42)
        env.reset(CONFIGS[1])
        env.grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]], dtype=np.int32)
        env.empty_pos = (2, 2)

        env.step(0)  # slide tile from (1,2)=6 into blank
        assert env.grid[2, 2] == 6
        assert env.grid[1, 2] == 0
        assert env.empty_pos == (1, 2)

    def test_episode_terminates_at_max_steps(self):
        env = SlidingPuzzle(seed=42)
        env.reset(CONFIGS[1])

        for i in range(MAX_STEPS - 1):
            _, _, done, _ = env.step(0)
            assert not done, f"Terminated early at step {i + 1}"

        _, _, done, _ = env.step(0)
        assert done

    def test_invalid_action_raises(self):
        env = SlidingPuzzle(seed=42)
        env.reset(CONFIGS[1])
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(4)
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)

    def test_deterministic_with_seed(self):
        env1 = SlidingPuzzle(seed=123)
        env2 = SlidingPuzzle(seed=123)
        obs1 = env1.reset(CONFIGS[2])
        obs2 = env2.reset(CONFIGS[2])
        assert np.array_equal(obs1, obs2)

    def test_different_seeds_differ(self):
        env1 = SlidingPuzzle(seed=123)
        env2 = SlidingPuzzle(seed=456)
        obs1 = env1.reset(CONFIGS[2])
        obs2 = env2.reset(CONFIGS[2])
        assert not np.array_equal(obs1, obs2)

    def test_tile_normalization(self):
        """Tile values are normalized by MAX_TILE_VALUE (24)."""
        env = SlidingPuzzle(seed=42)
        env.reset(CONFIGS[1])  # 3x3
        env.grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]], dtype=np.int32)
        obs = env._get_observation()
        assert obs[0] == pytest.approx(1.0 / MAX_TILE_VALUE)
        assert obs[7] == pytest.approx(8.0 / MAX_TILE_VALUE)
        assert obs[8] == pytest.approx(0.0)  # blank

    def test_all_grid_sizes_work(self):
        """w=2,3,4,5 all produce valid states."""
        env = SlidingPuzzle(seed=42)
        for w in [2, 3, 4, 5]:
            cfg = next(c for c in CONFIGS if c["w"] == w)
            obs = env.reset(cfg)
            assert obs.shape == (OBS_SIZE,)
            h = env.compute_entropy()
            assert h >= 0.0 and np.isfinite(h)

    def test_solvability(self):
        """Scrambling by valid moves guarantees solvability."""
        env = SlidingPuzzle(seed=42)
        heuristic = SlidingPuzzleHeuristicAgent()

        # Heuristic should solve easy 3x3 within 500 steps
        env.reset(CONFIGS[1])  # 3x3_easy
        solved = False
        for _ in range(500):
            action = heuristic.act(env._get_observation())
            env.step(action)
            if env.compute_entropy() == 0.0:
                solved = True
                break
        assert solved, "Heuristic failed to solve 3x3_easy within 500 steps"


class TestSlidingPuzzleHeuristic:
    """Test the SlidingPuzzleHeuristicAgent."""

    def test_reduces_entropy(self):
        """Heuristic reduces entropy on 3x3_easy."""
        env = SlidingPuzzle(seed=42)
        agent = SlidingPuzzleHeuristicAgent()

        obs = env.reset(CONFIGS[1])
        h_start = env.compute_entropy()
        for _ in range(200):
            action = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break
        h_end = env.compute_entropy()
        assert h_end <= h_start
        assert h_end < h_start or h_end == 0.0  # solved or reduced

    def test_solves_2x2(self):
        """Heuristic reliably solves 2x2 puzzle."""
        agent = SlidingPuzzleHeuristicAgent()

        solves = 0
        for seed in range(20):
            env_t = SlidingPuzzle(seed=seed)
            obs = env_t.reset(CONFIGS[0])  # 2x2_trivial
            for _ in range(100):
                action = agent.act(obs)
                obs, _, done, _ = env_t.step(action)
                if env_t.compute_entropy() == 0.0:
                    solves += 1
                    break
        assert solves >= 15, f"Only solved {solves}/20 2x2 puzzles"

    def test_reads_grid_size_from_obs(self):
        """Heuristic correctly reads variable grid sizes."""
        env = SlidingPuzzle(seed=42)
        agent = SlidingPuzzleHeuristicAgent()

        for w in [2, 3, 4, 5]:
            cfg = next(c for c in CONFIGS if c["w"] == w)
            obs = env.reset(cfg)
            action = agent.act(obs)
            assert 0 <= action <= 3


class TestSB3Wrapper:
    """Test the SB3 gymnasium wrapper with SlidingPuzzle."""

    def test_wrapper_obs_action_spaces(self):
        """Wrapper exposes correct obs/action spaces."""
        from gymnasium import spaces

        from ie.agents.sb3_wrapper import BaseEnvGymnasiumWrapper

        env = SlidingPuzzle(seed=42)

        def dummy_reward(_o, _a, _on, de):
            return -de

        gym_env = BaseEnvGymnasiumWrapper(env, [CONFIGS[1]], dummy_reward, seed=42)
        assert isinstance(gym_env.observation_space, spaces.Box)
        assert gym_env.observation_space.shape == (OBS_SIZE,)
        assert isinstance(gym_env.action_space, spaces.Discrete)
        assert gym_env.action_space.n == 4

    def test_wrapper_reset_step(self):
        """Wrapper reset/step produce correct tuple formats."""
        from ie.agents.sb3_wrapper import BaseEnvGymnasiumWrapper

        env = SlidingPuzzle(seed=42)

        def dummy_reward(_o, _a, _on, de):
            return -de

        gym_env = BaseEnvGymnasiumWrapper(env, [CONFIGS[1]], dummy_reward, seed=42)

        obs, info = gym_env.reset()
        assert obs.shape == (OBS_SIZE,)
        assert isinstance(info, dict)

        obs2, rew, term, trunc, info2 = gym_env.step(0)
        assert obs2.shape == (OBS_SIZE,)
        assert isinstance(rew, float)
        assert term is False  # sliding puzzles only truncate
        assert isinstance(trunc, bool)
        assert "work" in info2

    def test_wrapper_passes_real_obs_to_reward(self):
        """Reward function receives actual previous observation."""
        from ie.agents.sb3_wrapper import BaseEnvGymnasiumWrapper

        env = SlidingPuzzle(seed=42)
        received_shapes = []

        def tracking_reward(obs, _a, obs_next, _de):
            received_shapes.append((obs.shape, obs_next.shape))
            return 0.0

        gym_env = BaseEnvGymnasiumWrapper(env, [CONFIGS[1]], tracking_reward, seed=42)
        gym_env.reset()
        gym_env.step(0)
        gym_env.step(1)

        assert len(received_shapes) == 2
        for obs_shape, obs_next_shape in received_shapes:
            assert obs_shape == (OBS_SIZE,)
            assert obs_next_shape == (OBS_SIZE,)

    def test_sb3_agent_protocol(self):
        """SB3Agent wraps a model and provides act(obs) -> int."""
        from stable_baselines3 import PPO

        from ie.agents.sb3_wrapper import BaseEnvGymnasiumWrapper, SB3Agent

        env = SlidingPuzzle(seed=42)

        def dummy_reward(_o, _a, _on, de):
            return -de

        gym_env = BaseEnvGymnasiumWrapper(env, [CONFIGS[0]], dummy_reward, seed=42)
        model = PPO("MlpPolicy", gym_env, n_steps=64, seed=42, verbose=0)
        model.learn(total_timesteps=128)

        agent = SB3Agent(model)
        obs = env.reset(CONFIGS[0])
        action = agent.act(obs)
        assert isinstance(action, int)
        assert 0 <= action <= 3

    def test_wrapper_cycles_configs(self):
        """Wrapper cycles through configs on successive resets."""
        from ie.agents.sb3_wrapper import BaseEnvGymnasiumWrapper

        env = SlidingPuzzle(seed=42)
        configs = [CONFIGS[0], CONFIGS[1]]  # 2x2, 3x3

        def dummy_reward(_o, _a, _on, de):
            return -de

        gym_env = BaseEnvGymnasiumWrapper(env, configs, dummy_reward, seed=42)

        gym_env.reset()
        assert env.w == 2  # first config

        gym_env.reset()
        assert env.w == 3  # second config

        gym_env.reset()
        assert env.w == 2  # cycles back


class TestRestrictedHeuristic:
    """Test RestrictedSlidingPuzzleHeuristicAgent."""

    def test_uses_heuristic_on_allowed_width(self):
        """Restricted agent matches full heuristic on allowed grid sizes."""
        env = SlidingPuzzle(seed=42)
        full = SlidingPuzzleHeuristicAgent()
        restricted = RestrictedSlidingPuzzleHeuristicAgent(allowed_widths={3}, seed=42)

        obs = env.reset(CONFIGS[1])  # 3x3
        assert restricted.act(obs) == full.act(obs)

    def test_random_on_disallowed_width(self):
        """Restricted agent does NOT use heuristic on disallowed sizes."""
        env = SlidingPuzzle(seed=42)
        restricted = RestrictedSlidingPuzzleHeuristicAgent(allowed_widths={2}, seed=99)
        obs = env.reset(CONFIGS[1])  # 3x3, not in allowed_widths

        # Should return a valid action (0-3) but from RNG, not heuristic
        action = restricted.act(obs)
        assert 0 <= action <= 3

    def test_all_widths_matches_full(self):
        """Restricted with all widths behaves identically to unrestricted."""
        env = SlidingPuzzle(seed=42)
        full = SlidingPuzzleHeuristicAgent()
        restricted = RestrictedSlidingPuzzleHeuristicAgent(allowed_widths={2, 3, 4, 5}, seed=42)

        for cfg in CONFIGS:
            obs = env.reset(cfg)
            assert restricted.act(obs) == full.act(obs), f"Config {cfg['name']}"

    def test_reads_grid_size_correctly(self):
        """Restricted agent reads w from observation for all sizes."""
        env = SlidingPuzzle(seed=42)
        restricted = RestrictedSlidingPuzzleHeuristicAgent(allowed_widths={3}, seed=42)

        for w in [2, 3, 4, 5]:
            cfg = next(c for c in CONFIGS if c["w"] == w)
            obs = env.reset(cfg)
            action = restricted.act(obs)
            assert 0 <= action <= 3
