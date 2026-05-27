"""Tests for the Coupled Regulation environment."""

import numpy as np
import pytest

from ie.environments.coupled_regulation import CoupledRegulation


class TestCoupledRegulation:
    """Test suite for CoupledRegulation environment."""

    def test_reset_returns_correct_obs_shape(self):
        """After reset, observation has shape (16,)."""
        env = CoupledRegulation(seed=42)
        obs = env.reset({"name": "weak"})
        assert obs.shape == (16,)
        assert env.observation_shape == (16,)

    def test_action_space_size(self):
        """Action space has 17 actions."""
        env = CoupledRegulation(seed=42)
        assert env.action_space_size == 17

    def test_all_configs_work(self):
        """All configs (weak/moderate/strong) reset successfully."""
        env = CoupledRegulation(seed=42)

        # Weak config
        obs = env.reset({"name": "weak"})
        assert obs.shape == (16,)
        assert env.n_vars == 4
        assert env.context_label == 0

        # Moderate config
        obs = env.reset({"name": "moderate"})
        assert obs.shape == (16,)
        assert env.n_vars == 6
        assert env.context_label == 1

        # Strong config
        obs = env.reset({"name": "strong"})
        assert obs.shape == (16,)
        assert env.n_vars == 8
        assert env.context_label == 2

    def test_valid_action_adjusts_variable(self):
        """Valid action adjusts the corresponding variable."""
        env = CoupledRegulation(seed=42)
        env.reset({"name": "weak"})

        # Record initial value of variable 0
        initial_value = env.values[0]

        # Action 0: increase variable 0
        obs, _, _, info = env.step(0)
        assert info["work"] == 1.0
        assert env.values[0] > initial_value

        # Reset and test decrease action
        env.reset({"name": "weak"})
        initial_value = env.values[0]

        # Action 8: decrease variable 0
        obs, _, _, info = env.step(8)
        assert info["work"] == 1.0
        assert env.values[0] < initial_value

    def test_noop_action(self):
        """No-op action (16) does not change state and has work=0.0."""
        env = CoupledRegulation(seed=42)
        env.reset({"name": "weak"})

        # Store initial state
        initial_values = env.values.copy()

        # Execute no-op
        obs, _, _, info = env.step(16)
        assert info["work"] == 0.0
        assert np.array_equal(env.values, initial_values)

    def test_out_of_range_action(self):
        """Action for non-existent variable is no-op with work=0.0."""
        env = CoupledRegulation(seed=42)
        env.reset({"name": "weak"})  # n_vars=4

        # Store initial state
        initial_values = env.values.copy()

        # Action 5: increase variable 5 (doesn't exist, only 0-3 exist)
        obs, _, _, info = env.step(5)
        assert info["work"] == 0.0
        assert np.array_equal(env.values, initial_values)

        # Action 13: decrease variable 5 (doesn't exist)
        obs, _, _, info = env.step(13)
        assert info["work"] == 0.0
        assert np.array_equal(env.values, initial_values)

    def test_coupling_affects_other_variables(self):
        """Adjusting one variable affects others in moderate/strong configs."""
        env = CoupledRegulation(seed=42)
        env.reset({"name": "moderate"})  # coupling_strength=0.3

        # Store initial values
        initial_values = env.values.copy()

        # Adjust variable 0
        env.step(0)  # increase variable 0

        # Check that other variables changed (due to coupling)
        # At least one other variable should have changed
        other_vars_changed = False
        for i in range(1, env.n_vars):
            if env.values[i] != initial_values[i]:
                other_vars_changed = True
                break

        assert other_vars_changed, "Coupling should affect other variables"

    def test_entropy_decreases_toward_target(self):
        """Entropy is low when values are close to targets."""
        env = CoupledRegulation(seed=42)
        env.reset({"name": "weak"})

        # Measure initial entropy (values have some deviation)
        initial_entropy = env.compute_entropy()

        # Manually set values very close to targets
        env.values = env.targets.copy()

        # Entropy should be very low (ideally 0)
        final_entropy = env.compute_entropy()
        assert final_entropy < initial_entropy
        assert final_entropy < 0.1  # Should be very close to 0

    def test_episode_terminates(self):
        """Episode terminates after 300 steps."""
        env = CoupledRegulation(seed=42)
        env.reset({"name": "weak"})

        # Run 299 steps
        for _ in range(299):
            _, _, done, _ = env.step(16)  # no-op
            assert not done

        # Step 300 should set done=True
        _, _, done, _ = env.step(16)
        assert done

    def test_context_labels(self):
        """Each config returns correct context label."""
        env = CoupledRegulation(seed=42)

        env.reset({"name": "weak"})
        assert env.get_context_label() == 0

        env.reset({"name": "moderate"})
        assert env.get_context_label() == 1

        env.reset({"name": "strong"})
        assert env.get_context_label() == 2

    def test_invalid_action_raises_error(self):
        """Invalid action index raises ValueError."""
        env = CoupledRegulation(seed=42)
        env.reset({"name": "weak"})

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(17)

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)

    def test_invalid_config_raises_error(self):
        """Invalid config name raises ValueError."""
        env = CoupledRegulation(seed=42)

        with pytest.raises(ValueError, match="Unknown config"):
            env.reset({"name": "nonexistent_config"})

    def test_default_config(self):
        """None config defaults to weak."""
        env = CoupledRegulation(seed=42)
        obs = env.reset(None)
        assert obs.shape == (16,)
        assert env.n_vars == 4
        assert env.get_context_label() == 0

    def test_deterministic_with_seed(self):
        """Same seed produces same initial state."""
        env1 = CoupledRegulation(seed=123)
        env2 = CoupledRegulation(seed=123)

        obs1 = env1.reset({"name": "moderate"})
        obs2 = env2.reset({"name": "moderate"})

        assert np.array_equal(obs1, obs2)

    def test_different_seeds_produce_different_states(self):
        """Different seeds produce different initial states."""
        env1 = CoupledRegulation(seed=123)
        env2 = CoupledRegulation(seed=456)

        obs1 = env1.reset({"name": "moderate"})
        obs2 = env2.reset({"name": "moderate"})

        # Should be different (with very high probability)
        assert not np.array_equal(obs1, obs2)

    def test_entropy_computation(self):
        """Entropy computation returns valid values."""
        env = CoupledRegulation(seed=42)
        env.reset({"name": "weak"})

        entropy = env.compute_entropy()
        assert entropy >= 0.0
        assert np.isfinite(entropy)

    def test_work_in_info(self):
        """Valid actions return work=1.0, invalid/no-op return work=0.0."""
        env = CoupledRegulation(seed=42)
        env.reset({"name": "weak"})

        # Valid increase action
        _, _, _, info = env.step(0)
        assert info["work"] == 1.0

        # Valid decrease action
        _, _, _, info = env.step(8)
        assert info["work"] == 1.0

        # No-op
        _, _, _, info = env.step(16)
        assert info["work"] == 0.0

        # Out of range action
        _, _, _, info = env.step(5)  # var 5 doesn't exist in weak config
        assert info["work"] == 0.0

    def test_heterogeneous_costs_disabled_by_default(self):
        """By default, heterogeneous_costs=False and work=1.0 for valid actions."""
        env = CoupledRegulation(seed=42)
        env.reset({"name": "strong"})

        # Valid action should have work=1.0
        _, _, _, info = env.step(0)
        assert info["work"] == 1.0

    def test_heterogeneous_costs_enabled(self):
        """With heterogeneous_costs=True, work depends on coupling strength."""
        env = CoupledRegulation(seed=42, heterogeneous_costs=True)
        env.reset({"name": "strong"})

        # Valid action should have work > 1.0 due to coupling
        _, _, _, info = env.step(0)
        assert info["work"] > 1.0

        # Work should vary by variable (different coupling sums)
        env.reset({"name": "strong"})
        _, _, _, info0 = env.step(0)

        env.reset({"name": "strong"})
        _, _, _, info1 = env.step(1)

        # Different variables likely have different coupling sums
        # (not guaranteed with random coupling, but highly likely with 7 other vars)
        work0 = info0["work"]
        work1 = info1["work"]
        assert work0 > 1.0
        assert work1 > 1.0

    def test_heterogeneous_costs_scales_with_coupling_strength(self):
        """Stronger coupling configs should have higher average costs."""
        env_weak = CoupledRegulation(seed=42, heterogeneous_costs=True)
        env_weak.reset({"name": "weak"})
        _, _, _, info_weak = env_weak.step(0)

        env_strong = CoupledRegulation(seed=42, heterogeneous_costs=True)
        env_strong.reset({"name": "strong"})
        _, _, _, info_strong = env_strong.step(0)

        # Strong config should have higher work cost
        assert info_strong["work"] > info_weak["work"]

    def test_heterogeneous_costs_noop_still_zero(self):
        """No-op action has work=0.0 even with heterogeneous costs."""
        env = CoupledRegulation(seed=42, heterogeneous_costs=True)
        env.reset({"name": "strong"})

        _, _, _, info = env.step(16)
        assert info["work"] == 0.0

    def test_heterogeneous_costs_out_of_range_still_zero(self):
        """Out-of-range actions have work=0.0 even with heterogeneous costs."""
        env = CoupledRegulation(seed=42, heterogeneous_costs=True)
        env.reset({"name": "weak"})  # n_vars=4

        _, _, _, info = env.step(5)  # var 5 doesn't exist
        assert info["work"] == 0.0
