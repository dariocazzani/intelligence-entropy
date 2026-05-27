"""Tests for the Sequence Sorting environment."""

import numpy as np
import pytest

from ie.environments.sequence_sorting import SequenceSorting


def test_reset_produces_shuffled_sequence():
    """After reset, sequence contains expected values (zero-padded to MAX_LENGTH)."""
    env = SequenceSorting(seed=42)

    # Test short config (length 15, padded to 25)
    obs = env.reset({"name": "short"})
    assert len(obs) == 25  # MAX_LENGTH
    assert set(obs[:15]) == set(range(15))
    assert all(obs[15:] == 0.0)  # Zero-padded

    # Test medium config (length 20, padded to 25)
    obs = env.reset({"name": "medium"})
    assert len(obs) == 25  # MAX_LENGTH
    assert set(obs[:20]) == set(range(20))
    assert all(obs[20:] == 0.0)  # Zero-padded

    # Test duplicates config (length 25, no padding)
    obs = env.reset({"name": "duplicates"})
    assert len(obs) == 25  # MAX_LENGTH
    assert all(val in [0, 1, 2, 3, 4] for val in obs)


def test_swap_action():
    """Swapping adjacent elements works correctly."""
    env = SequenceSorting(seed=42)
    env.reset({"name": "short"})

    # Manually set a known sequence (length 15 for short config)
    env.sequence = np.array([3, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)

    # Swap positions 0 and 1 (swap 3 and 1)
    obs, _, _, info = env.step(0)
    assert obs[0] == 1
    assert obs[1] == 3
    assert info["work"] == 1.0

    # Swap positions 2 and 3 (swap 2 and 0)
    obs, _, _, info = env.step(2)
    assert obs[2] == 0
    assert obs[3] == 2
    assert info["work"] == 1.0


def test_noop_doesnt_change_state():
    """No-op action leaves sequence unchanged."""
    env = SequenceSorting(seed=42)
    env.reset({"name": "short"})

    # Manually set a known sequence (length 15 for short config)
    env.sequence = np.array([3, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)
    before = env.sequence.copy()

    # Execute no-op (action MAX_LENGTH-1 = 24)
    obs, _, _, info = env.step(24)

    # Sequence should be unchanged (obs is zero-padded)
    assert np.array_equal(obs[:15], before.astype(np.float32))
    assert all(obs[15:] == 0.0)  # Zero-padded
    assert info["work"] == 0.0


def test_sorted_sequence_has_zero_entropy():
    """Manually set to sorted, entropy = 0."""
    env = SequenceSorting(seed=42)
    env.reset({"name": "short"})

    # Set to sorted sequence (length 15)
    env.sequence = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)

    entropy = env.compute_entropy()
    assert entropy == 0.0


def test_shuffled_sequence_has_positive_entropy():
    """Random sequence has entropy > 0."""
    env = SequenceSorting(seed=42)
    env.reset({"name": "short"})

    # Sequence is shuffled by reset
    entropy = env.compute_entropy()
    assert entropy > 0.0


def test_sorting_reduces_entropy():
    """Applying correct swaps reduces entropy."""
    env = SequenceSorting(seed=42)
    env.reset({"name": "short"})

    # Manually set a partially sorted sequence (length 15)
    env.sequence = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)

    # Initial entropy
    entropy_before = env.compute_entropy()
    assert entropy_before > 0.0

    # Swap positions 1 and 2 to sort the sequence
    env.step(1)

    # Final entropy should be 0 (fully sorted)
    entropy_after = env.compute_entropy()
    assert entropy_after == 0.0
    assert entropy_after < entropy_before


def test_work_accounting():
    """Swaps cost 1.0, no-ops cost 0.0."""
    env = SequenceSorting(seed=42)
    env.reset({"name": "short"})

    # Test swap action (should cost 1.0)
    _, _, _, info = env.step(0)
    assert info["work"] == 1.0

    # Test no-op action (should cost 0.0)
    _, _, _, info = env.step(24)  # Action MAX_LENGTH-1 = 24
    assert info["work"] == 0.0


def test_context_labels():
    """Each config returns correct label."""
    env = SequenceSorting(seed=42)

    # Short config
    env.reset({"name": "short"})
    assert env.get_context_label() == 0

    # Medium config
    env.reset({"name": "medium"})
    assert env.get_context_label() == 1

    # Duplicates config
    env.reset({"name": "duplicates"})
    assert env.get_context_label() == 2


def test_observation_shape():
    """Observation shape is always MAX_LENGTH regardless of config."""
    env = SequenceSorting(seed=42)

    # Short config
    env.reset({"name": "short"})
    assert env.observation_shape == (25,)

    # Medium config
    env.reset({"name": "medium"})
    assert env.observation_shape == (25,)

    # Duplicates config
    env.reset({"name": "duplicates"})
    assert env.observation_shape == (25,)


def test_action_space_size():
    """Returns MAX_LENGTH regardless of config."""
    env = SequenceSorting(seed=42)

    # Short config
    env.reset({"name": "short"})
    assert env.action_space_size == 25

    # Medium config
    env.reset({"name": "medium"})
    assert env.action_space_size == 25

    # Duplicates config
    env.reset({"name": "duplicates"})
    assert env.action_space_size == 25


def test_episode_length():
    """Done after 500 steps."""
    env = SequenceSorting(seed=42)
    env.reset({"name": "short"})

    # Execute 499 steps
    for _ in range(499):
        _, _, done, _ = env.step(24)  # No-op action (MAX_LENGTH-1)
        assert not done

    # Step 500 should set done=True
    _, _, done, _ = env.step(24)
    assert done


def test_duplicates_config():
    """Values drawn from [0,1,2,3,4], length 25 (no padding)."""
    env = SequenceSorting(seed=42)

    # Run multiple times to check consistency
    for _ in range(10):
        obs = env.reset({"name": "duplicates"})
        assert len(obs) == 25  # MAX_LENGTH
        assert all(val in [0, 1, 2, 3, 4] for val in obs)


def test_entropy_with_duplicates():
    """Entropy calculation handles duplicates correctly using stable sort."""
    env = SequenceSorting(seed=42)
    env.reset({"name": "duplicates"})

    # Set a sequence with duplicates in sorted order (length 25)
    env.sequence = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5, dtype=np.int32)
    entropy = env.compute_entropy()
    assert entropy == 0.0  # Should be 0 when sorted

    # Set a sequence with duplicates out of order
    out_of_order = [3, 1, 0, 2, 4, 3, 1, 0, 2, 4, 3, 1, 0, 2, 4, 3, 1, 0, 2, 4, 3, 1, 0, 2, 4]
    env.sequence = np.array(out_of_order, dtype=np.int32)
    entropy = env.compute_entropy()
    assert entropy > 0.0  # Should be positive when unsorted


def test_invalid_action_raises_error():
    """Invalid actions raise ValueError."""
    env = SequenceSorting(seed=42)
    env.reset({"name": "short"})

    # Action out of range (negative)
    with pytest.raises(ValueError):
        env.step(-1)

    # Action out of range (too large for MAX_LENGTH)
    with pytest.raises(ValueError):
        env.step(25)


def test_default_config():
    """None config defaults to short."""
    env = SequenceSorting(seed=42)

    obs = env.reset(None)
    assert len(obs) == 25  # MAX_LENGTH
    assert set(obs[:15]) == set(range(15))
    assert all(obs[15:] == 0.0)  # Zero-padded
    assert env.get_context_label() == 0


def test_unknown_config_raises_error():
    """Unknown config name raises ValueError."""
    env = SequenceSorting(seed=42)

    with pytest.raises(ValueError):
        env.reset({"name": "unknown"})


def test_deterministic_with_seed():
    """Same seed produces same sequence."""
    env1 = SequenceSorting(seed=123)
    env2 = SequenceSorting(seed=123)

    obs1 = env1.reset({"name": "short"})
    obs2 = env2.reset({"name": "short"})

    assert np.array_equal(obs1, obs2)


def test_different_seeds_produce_different_sequences():
    """Different seeds produce different sequences."""
    env1 = SequenceSorting(seed=123)
    env2 = SequenceSorting(seed=456)

    obs1 = env1.reset({"name": "short"})
    obs2 = env2.reset({"name": "short"})

    # Very unlikely to be equal (but not impossible)
    # Test that at least one element is different
    assert not np.array_equal(obs1, obs2)


class TestHeterogeneousCosts:
    """Tests for heterogeneous cost model in Sequence Sorting."""

    def test_uniform_cost_default(self):
        """Default (heterogeneous_costs=False) gives work=1.0 for all swaps."""
        env = SequenceSorting(seed=42, heterogeneous_costs=False)
        env.reset({"name": "short"})
        env.sequence = np.array([14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)

        _, _, _, info = env.step(0)
        assert info["work"] == 1.0

    def test_hetcost_swap_far_from_target(self):
        """Swapping an element far from its sorted position costs more than 1.0."""
        env = SequenceSorting(seed=42, heterogeneous_costs=True)
        env.reset({"name": "short"})
        # Element 14 is at position 0, sorted position is 14 → displacement = 14
        env.sequence = np.array([14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)

        _, _, _, info = env.step(0)  # swap position 0 and 1
        # work = 1 + displacement/N = 1 + 14/15
        expected_work = 1.0 + 14.0 / 15.0
        assert abs(info["work"] - expected_work) < 1e-10

    def test_hetcost_swap_near_target(self):
        """Swapping an element near its sorted position costs close to 1.0."""
        env = SequenceSorting(seed=42, heterogeneous_costs=True)
        env.reset({"name": "short"})
        # Element 1 is at position 0, sorted position is 1 → displacement = 1
        env.sequence = np.array([1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)

        _, _, _, info = env.step(0)  # swap position 0 and 1
        # work = 1 + displacement/N = 1 + 1/15
        expected_work = 1.0 + 1.0 / 15.0
        assert abs(info["work"] - expected_work) < 1e-10

    def test_hetcost_swap_at_target(self):
        """Swapping an element already at its sorted position costs 1.0."""
        env = SequenceSorting(seed=42, heterogeneous_costs=True)
        env.reset({"name": "short"})
        # Element 0 is at position 0 (sorted position) → displacement = 0
        env.sequence = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)

        _, _, _, info = env.step(0)  # swap position 0 and 1
        # work = 1 + 0/15 = 1.0
        assert info["work"] == 1.0

    def test_hetcost_noop_still_free(self):
        """No-op action costs 0.0 regardless of cost model."""
        env = SequenceSorting(seed=42, heterogeneous_costs=True)
        env.reset({"name": "short"})

        _, _, _, info = env.step(24)  # no-op
        assert info["work"] == 0.0

    def test_hetcost_with_duplicates(self):
        """Heterogeneous costs work correctly with duplicate values."""
        env = SequenceSorting(seed=42, heterogeneous_costs=True)
        env.reset({"name": "duplicates"})
        # All zeros → sorted is all zeros → all displacements = 0
        env.sequence = np.zeros(25, dtype=np.int32)

        _, _, _, info = env.step(0)  # swap position 0 and 1
        # displacement = 0, work = 1 + 0/25 = 1.0
        assert info["work"] == 1.0
