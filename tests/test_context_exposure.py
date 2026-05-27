"""Tests for the context-exposure experiment.

The context-exposure experiment evaluates trained agents on subsets of 1, 2, and 3
environment configs to test whether the H(C) term in I_cd = eta × H(C) produces
meaningful discrimination across different context breadths.

This addresses the #1 reviewer concern: does the H(C) term actually contribute
meaningfully to the intelligence measure?
"""

import math

from ie.agents.heuristic_agent import SeqHeuristicAgent
from ie.agents.random_agent import RandomAgent
from ie.analysis.evaluate import evaluate_all_agents
from ie.environments.sequence_sorting import SequenceSorting


class TestContextExposure:
    """Tests for context-exposure experiment validating H(C) term contribution."""

    def test_single_config_gives_zero_icd(self):
        """Evaluating on a single config should give I_cd = 0.

        When only one context is evaluated, H(C) = 0 (Shannon entropy of a single
        category), so I_cd = efficiency × H(C) = efficiency × 0 = 0, regardless
        of the agent's efficiency.
        """
        env = SequenceSorting(seed=42)
        agent = RandomAgent(action_space_size=env.action_space_size, seed=42)

        # Single config evaluation
        configs = [{"name": "short"}]
        results = evaluate_all_agents({"random": agent}, env, configs, episodes_per_config=3)

        result = results["random"]
        assert result["i_cd"] == 0.0, "Single context should give I_cd = 0"
        assert result["context_diversity"] == 0.0, "Single context should give H(C) = 0"

    def test_two_configs_gives_nonzero_icd(self):
        """Evaluating on two configs should give nonzero I_cd for an effective agent.

        SeqHeuristicAgent reliably reduces entropy through bubble sort, so it should
        have positive efficiency. With 2 contexts, H(C) ≈ ln(2) ≈ 0.693, so
        I_cd should be positive.
        """
        env = SequenceSorting(seed=42)
        agent = SeqHeuristicAgent()

        # Two config evaluation
        configs = [{"name": "short"}, {"name": "medium"}]
        results = evaluate_all_agents({"heuristic": agent}, env, configs, episodes_per_config=3)

        result = results["heuristic"]
        assert result["context_diversity"] > 0, "Two contexts should give H(C) > 0"
        # H(C) for 2 equally-represented contexts is ln(2)
        expected_diversity = math.log(2)
        assert math.isclose(result["context_diversity"], expected_diversity, rel_tol=0.01), (
            f"Expected H(C) ≈ {expected_diversity:.3f}, got {result['context_diversity']:.3f}"
        )

        # Heuristic agent should have positive efficiency
        assert result["efficiency"] > 0, "SeqHeuristicAgent should reduce entropy"
        # Therefore I_cd should be positive
        assert result["i_cd"] > 0, "Effective agent on 2 contexts should have I_cd > 0"

    def test_icd_scales_with_context_breadth(self):
        """I_cd should scale with context breadth (number of configs).

        For an agent with consistent efficiency across contexts, I_cd should increase
        as the number of contexts increases because H(C) increases:
        - 1 config: H(C) = 0, so I_cd = 0
        - 2 configs: H(C) ≈ ln(2) ≈ 0.693
        - 3 configs: H(C) ≈ ln(3) ≈ 1.099

        Since ln(3) > ln(2) > 0, we expect I_cd(3) > I_cd(2) > I_cd(1).
        """
        env = SequenceSorting(seed=42)
        agent = SeqHeuristicAgent()

        # Evaluate on 1, 2, and 3 configs
        configs_1 = [{"name": "short"}]
        configs_2 = [{"name": "short"}, {"name": "medium"}]
        configs_3 = [{"name": "short"}, {"name": "medium"}, {"name": "duplicates"}]

        results_1 = evaluate_all_agents({"heuristic": agent}, env, configs_1, episodes_per_config=3)
        results_2 = evaluate_all_agents({"heuristic": agent}, env, configs_2, episodes_per_config=3)
        results_3 = evaluate_all_agents({"heuristic": agent}, env, configs_3, episodes_per_config=3)

        icd_1 = results_1["heuristic"]["i_cd"]
        icd_2 = results_2["heuristic"]["i_cd"]
        icd_3 = results_3["heuristic"]["i_cd"]

        # Assert scaling relationship
        assert icd_1 == 0.0, "1 config should give I_cd = 0"
        assert icd_2 > 0, "2 configs should give I_cd > 0"
        assert icd_3 > icd_2, (
            f"3 configs I_cd ({icd_3:.3f}) should exceed 2 configs I_cd ({icd_2:.3f})"
        )

        # Verify that H(C) values are as expected
        div_1 = results_1["heuristic"]["context_diversity"]
        div_2 = results_2["heuristic"]["context_diversity"]
        div_3 = results_3["heuristic"]["context_diversity"]

        assert div_1 == 0.0
        assert math.isclose(div_2, math.log(2), rel_tol=0.01)
        assert math.isclose(div_3, math.log(3), rel_tol=0.01)

    def test_efficiency_stable_across_context_subsets(self):
        """Efficiency should be approximately stable across different context subsets.

        When evaluating the same agent on different numbers of contexts, the
        efficiency (eta = -ΔS / W) should remain roughly similar because it
        measures the agent's per-unit-work effectiveness, which shouldn't change
        based on evaluation protocol breadth.

        This validates that changing the number of contexts in the evaluation doesn't
        artificially inflate or deflate the efficiency estimate.
        """
        env = SequenceSorting(seed=42)
        agent = SeqHeuristicAgent()

        # Evaluate on 1, 2, and 3 configs using the SAME agent
        configs_1 = [{"name": "short"}]
        configs_2 = [{"name": "short"}, {"name": "medium"}]
        configs_3 = [{"name": "short"}, {"name": "medium"}, {"name": "duplicates"}]

        results_1 = evaluate_all_agents({"heuristic": agent}, env, configs_1, episodes_per_config=5)
        results_2 = evaluate_all_agents({"heuristic": agent}, env, configs_2, episodes_per_config=5)
        results_3 = evaluate_all_agents({"heuristic": agent}, env, configs_3, episodes_per_config=5)

        eff_1 = results_1["heuristic"]["efficiency"]
        eff_2 = results_2["heuristic"]["efficiency"]
        eff_3 = results_3["heuristic"]["efficiency"]

        # All should be positive (heuristic reduces entropy)
        assert eff_1 > 0, "Heuristic should have positive efficiency"
        assert eff_2 > 0, "Heuristic should have positive efficiency"
        assert eff_3 > 0, "Heuristic should have positive efficiency"

        # Compute average efficiency
        avg_eff = (eff_1 + eff_2 + eff_3) / 3

        # Check that each efficiency is within 50% of the average
        # (allows for some variance due to stochastic config initialization)
        tolerance = 0.5
        assert abs(eff_1 - avg_eff) / avg_eff <= tolerance, (
            f"Efficiency on 1 config ({eff_1:.3f}) differs too much from average ({avg_eff:.3f})"
        )
        assert abs(eff_2 - avg_eff) / avg_eff <= tolerance, (
            f"Efficiency on 2 configs ({eff_2:.3f}) differs too much from average ({avg_eff:.3f})"
        )
        assert abs(eff_3 - avg_eff) / avg_eff <= tolerance, (
            f"Efficiency on 3 configs ({eff_3:.3f}) differs too much from average ({avg_eff:.3f})"
        )

    def test_random_agent_low_efficiency_but_hc_still_scales(self):
        """Random agent should have near-zero efficiency but H(C) should still scale.

        Even when efficiency is near zero (random agent doesn't systematically reduce
        entropy), the H(C) term should still increase with more contexts. This validates
        that H(C) is computed independently of agent performance.
        """
        env = SequenceSorting(seed=42)
        agent = RandomAgent(action_space_size=env.action_space_size, seed=42)

        configs_2 = [{"name": "short"}, {"name": "medium"}]
        configs_3 = [{"name": "short"}, {"name": "medium"}, {"name": "duplicates"}]

        results_2 = evaluate_all_agents({"random": agent}, env, configs_2, episodes_per_config=5)
        results_3 = evaluate_all_agents({"random": agent}, env, configs_3, episodes_per_config=5)

        # H(C) should scale as expected
        div_2 = results_2["random"]["context_diversity"]
        div_3 = results_3["random"]["context_diversity"]

        assert math.isclose(div_2, math.log(2), rel_tol=0.01)
        assert math.isclose(div_3, math.log(3), rel_tol=0.01)

        # Efficiency should be small (random agent is ineffective)
        eff_2 = results_2["random"]["efficiency"]
        eff_3 = results_3["random"]["efficiency"]
        assert abs(eff_2) < 0.1, "Random agent should have near-zero efficiency"
        assert abs(eff_3) < 0.1, "Random agent should have near-zero efficiency"
