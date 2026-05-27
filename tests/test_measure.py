"""Tests for the core I_cd measure module."""

import numpy as np

from ie.measure import (
    context_diversity,
    efficiency,
    environmental_entropy,
    i_cd,
)


class TestEnvironmentalEntropy:
    """Tests for environmental_entropy function."""

    def test_environmental_entropy_known_values(self):
        """Uniform distribution over 4 categories should give ln(4)."""
        state = np.array([1, 1, 1, 1])
        result = environmental_entropy(state)
        expected = np.log(4)  # ln(4) ≈ 1.386
        assert np.isclose(result, expected), f"Expected {expected}, got {result}"

    def test_environmental_entropy_all_zeros(self):
        """All-zero state should return 0.0."""
        state = np.array([0, 0, 0, 0])
        result = environmental_entropy(state)
        assert result == 0.0

    def test_environmental_entropy_single_category(self):
        """All mass in one category should give 0.0 entropy."""
        state = np.array([10, 0, 0, 0])
        result = environmental_entropy(state)
        assert np.isclose(result, 0.0)

    def test_environmental_entropy_empty_array(self):
        """Empty array should return 0.0."""
        state = np.array([])
        result = environmental_entropy(state)
        assert result == 0.0

    def test_environmental_entropy_normalized_probs(self):
        """Should work with already-normalized probabilities."""
        state = np.array([0.25, 0.25, 0.25, 0.25])
        result = environmental_entropy(state)
        expected = np.log(4)
        assert np.isclose(result, expected)

    def test_environmental_entropy_non_uniform(self):
        """Non-uniform distribution should have lower entropy than uniform."""
        uniform = np.array([1, 1, 1, 1])
        non_uniform = np.array([3, 1, 1, 1])

        h_uniform = environmental_entropy(uniform)
        h_non_uniform = environmental_entropy(non_uniform)

        assert h_non_uniform < h_uniform


class TestEfficiency:
    """Tests for efficiency function."""

    def test_zero_work_gives_zero(self):
        """Work = 0 should return efficiency = 0.0."""
        result = efficiency(-1.0, 0.0)
        assert result == 0.0

    def test_entropy_reduction_positive_efficiency(self):
        """Negative delta_s (entropy reduction) should give positive efficiency."""
        result = efficiency(-2.0, 1.0)
        assert result == 2.0

    def test_entropy_increase_negative_efficiency(self):
        """Positive delta_s (entropy increase) should give negative efficiency."""
        result = efficiency(2.0, 1.0)
        assert result == -2.0

    def test_efficiency_calculation(self):
        """Test basic efficiency calculation."""
        delta_s = -1.5
        work = 3.0
        result = efficiency(delta_s, work)
        expected = -delta_s / work  # 1.5 / 3.0 = 0.5
        assert np.isclose(result, expected)


class TestContextDiversity:
    """Tests for context_diversity function."""

    def test_empty_contexts_gives_zero(self):
        """Empty context list should return 0.0."""
        result = context_diversity([])
        assert result == 0.0

    def test_single_context_gives_zero(self):
        """Single unique context should return 0.0."""
        result = context_diversity([0, 0, 0, 0])
        assert result == 0.0

    def test_uniform_contexts_max_entropy(self):
        """N equally frequent contexts should give H(C) = ln(N)."""
        contexts = [0, 1, 2, 3]
        result = context_diversity(contexts)
        expected = np.log(4)
        assert np.isclose(result, expected)

    def test_context_diversity_with_numpy_array(self):
        """Should work with numpy arrays as input."""
        contexts = np.array([0, 1, 2, 3])
        result = context_diversity(contexts)
        expected = np.log(4)
        assert np.isclose(result, expected)

    def test_non_uniform_contexts(self):
        """Non-uniform context distribution should have lower diversity."""
        uniform = [0, 1, 2, 3]
        non_uniform = [0, 0, 0, 1]

        h_uniform = context_diversity(uniform)
        h_non_uniform = context_diversity(non_uniform)

        assert h_non_uniform < h_uniform

    def test_context_diversity_order_invariant(self):
        """Context diversity should not depend on order of observations."""
        contexts1 = [0, 0, 1, 2, 1, 0]
        contexts2 = [0, 1, 0, 1, 2, 0]

        result1 = context_diversity(contexts1)
        result2 = context_diversity(contexts2)

        assert np.isclose(result1, result2)


class TestIcd:
    """Tests for the full I_cd measure."""

    def test_single_context_gives_zero_icd(self):
        """One unique context label should give I_cd = 0."""
        result = i_cd(-2.0, 1.0, [0, 0, 0, 0])
        assert result == 0.0

    def test_crystal_case(self):
        """High efficiency (large -ΔS, small work) but single context → I_cd = 0."""
        # A crystal: very efficient at reducing entropy, but zero context diversity
        delta_s = -10.0  # large entropy reduction
        work = 0.5  # small work
        contexts = [0, 0, 0, 0]  # single context

        result = i_cd(delta_s, work, contexts)
        assert result == 0.0

    def test_zero_work_gives_zero(self):
        """Work = 0 should give I_cd = 0 regardless of contexts."""
        result = i_cd(-2.0, 0.0, [0, 1, 2, 3])
        assert result == 0.0

    def test_uniform_contexts_calculation(self):
        """Test I_cd with uniform contexts."""
        delta_s = -2.0
        work = 1.0
        contexts = [0, 1, 2, 3]

        result = i_cd(delta_s, work, contexts)

        # Expected: efficiency = 2.0, diversity = ln(4)
        expected_efficiency = 2.0
        expected_diversity = np.log(4)
        expected_icd = expected_efficiency * expected_diversity

        assert np.isclose(result, expected_icd)

    def test_monotonicity_more_diversity_higher_icd(self):
        """Same efficiency, more diverse contexts → higher I_cd."""
        delta_s = -2.0
        work = 1.0

        # Two contexts
        contexts_2 = [0, 1]
        icd_2 = i_cd(delta_s, work, contexts_2)

        # Four contexts
        contexts_4 = [0, 1, 2, 3]
        icd_4 = i_cd(delta_s, work, contexts_4)

        assert icd_4 > icd_2

    def test_negative_icd_when_entropy_increases(self):
        """Positive delta_s (entropy increase) should give negative I_cd."""
        delta_s = 2.0  # entropy increased
        work = 1.0
        contexts = [0, 1, 2, 3]

        result = i_cd(delta_s, work, contexts)
        assert result < 0.0

    def test_empty_contexts_gives_zero(self):
        """Empty context list should give I_cd = 0."""
        result = i_cd(-2.0, 1.0, [])
        assert result == 0.0

    def test_icd_multiplicative_structure(self):
        """Verify I_cd = efficiency × diversity for known values."""
        delta_s = -3.0
        work = 2.0
        contexts = [0, 0, 1, 1, 2, 2]

        eff = efficiency(delta_s, work)
        div = context_diversity(contexts)
        expected = eff * div

        result = i_cd(delta_s, work, contexts)

        assert np.isclose(result, expected)
