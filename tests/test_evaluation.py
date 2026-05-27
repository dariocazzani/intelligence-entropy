"""Tests for the evaluation pipeline."""

from ie.agents.random_agent import RandomAgent
from ie.analysis.evaluate import (
    cobb_douglas_sensitivity,
    context_sensitivity_analysis,
    evaluate_agent,
    evaluate_all_agents,
)
from ie.environments.resource_grid import ResourceGrid
from ie.environments.sequence_sorting import SequenceSorting


class TestEvaluateAgent:
    def test_returns_expected_keys(self):
        env = ResourceGrid(seed=42)
        agent = RandomAgent(action_space_size=env.action_space_size, seed=42)
        configs = [{"name": "basic"}, {"name": "multi_type"}]
        result = evaluate_agent(agent, env, configs, episodes_per_config=2)

        expected_keys = {
            "i_cd",
            "efficiency",
            "context_diversity",
            "entropy_trajectories",
            "total_work",
            "total_entropy_change",
            "context_labels",
        }
        assert set(result.keys()) == expected_keys

    def test_random_agent_has_low_icd(self):
        """Random agent should have I_cd near zero (no systematic entropy reduction)."""
        env = ResourceGrid(seed=42)
        agent = RandomAgent(action_space_size=env.action_space_size, seed=42)
        configs = [{"name": "basic"}, {"name": "multi_type"}, {"name": "perturbation"}]
        result = evaluate_agent(agent, env, configs, episodes_per_config=3)

        # Random agent shouldn't systematically reduce entropy
        # I_cd should be small (could be slightly positive or negative by chance)
        assert abs(result["i_cd"]) < 1.0

    def test_context_labels_match_configs(self):
        env = ResourceGrid(seed=42)
        agent = RandomAgent(action_space_size=env.action_space_size, seed=42)
        configs = [{"name": "basic"}, {"name": "multi_type"}]
        result = evaluate_agent(agent, env, configs, episodes_per_config=2)

        # Should have context labels 0 and 1
        assert set(result["context_labels"]) == {0, 1}
        assert len(result["context_labels"]) == 4  # 2 configs * 2 episodes

    def test_evaluate_all_agents(self):
        env = ResourceGrid(seed=42)
        agents = {
            "random1": RandomAgent(action_space_size=env.action_space_size, seed=42),
            "random2": RandomAgent(action_space_size=env.action_space_size, seed=43),
        }
        configs = [{"name": "basic"}]
        results = evaluate_all_agents(agents, env, configs, episodes_per_config=2)

        assert "random1" in results
        assert "random2" in results
        assert "i_cd" in results["random1"]

    def test_works_with_sequence_sorting(self):
        env = SequenceSorting(seed=42)
        agent = RandomAgent(action_space_size=env.action_space_size, seed=42)
        configs = [{"name": "short"}, {"name": "medium"}]
        result = evaluate_agent(agent, env, configs, episodes_per_config=2)
        assert "i_cd" in result


class TestContextSensitivityAnalysis:
    def test_returns_expected_partitions(self):
        env = SequenceSorting(seed=42)
        agents = {
            "r1": RandomAgent(action_space_size=env.action_space_size, seed=42),
            "r2": RandomAgent(action_space_size=env.action_space_size, seed=99),
        }
        configs = [{"name": "short"}, {"name": "medium"}, {"name": "duplicates"}]
        partitions = context_sensitivity_analysis(agents, configs, env, episodes_per_config=1)

        assert "full" in partitions
        assert "coarsened" in partitions
        assert "single" in partitions

    def test_single_partition_has_zero_icd(self):
        """Single config → H(C) = 0 → I_cd = 0 for all agents."""
        env = SequenceSorting(seed=42)
        agents = {
            "r1": RandomAgent(action_space_size=env.action_space_size, seed=42),
        }
        configs = [{"name": "short"}, {"name": "medium"}]
        partitions = context_sensitivity_analysis(agents, configs, env, episodes_per_config=1)

        for entry in partitions["single"]:
            assert entry["i_cd"] == 0.0
            assert entry["context_diversity"] == 0.0

    def test_entries_sorted_by_icd_descending(self):
        env = SequenceSorting(seed=42)
        agents = {
            "r1": RandomAgent(action_space_size=env.action_space_size, seed=42),
            "r2": RandomAgent(action_space_size=env.action_space_size, seed=99),
        }
        configs = [{"name": "short"}, {"name": "medium"}]
        partitions = context_sensitivity_analysis(agents, configs, env, episodes_per_config=1)

        for partition_name, entries in partitions.items():
            for i in range(len(entries) - 1):
                assert entries[i]["i_cd"] >= entries[i + 1]["i_cd"]

    def test_entry_has_expected_keys(self):
        env = SequenceSorting(seed=42)
        agents = {"r1": RandomAgent(action_space_size=env.action_space_size, seed=42)}
        configs = [{"name": "short"}, {"name": "medium"}]
        partitions = context_sensitivity_analysis(agents, configs, env, episodes_per_config=1)

        for entry in partitions["full"]:
            assert set(entry.keys()) == {"agent", "i_cd", "efficiency", "context_diversity"}


class TestCobbDouglasSensitivity:
    def _make_aggregated(self):
        return {
            "AgentA": {
                "efficiency_mean": 0.5,
                "context_diversity_mean": 1.0,
                "i_cd_mean": 0.5,
            },
            "AgentB": {
                "efficiency_mean": 0.8,
                "context_diversity_mean": 0.6,
                "i_cd_mean": 0.48,
            },
            "AgentC": {
                "efficiency_mean": 0.1,
                "context_diversity_mean": 1.5,
                "i_cd_mean": 0.15,
            },
        }

    def test_default_exponents(self):
        agg = self._make_aggregated()
        results = cobb_douglas_sensitivity(agg)
        assert len(results) == 5
        assert (1.0, 1.0) in results

    def test_custom_exponents(self):
        agg = self._make_aggregated()
        results = cobb_douglas_sensitivity(agg, exponent_pairs=[(1.0, 1.0), (2.0, 2.0)])
        assert len(results) == 2

    def test_a1_b1_matches_standard_icd(self):
        """(1,1) exponents should produce eta * H(C)."""
        agg = self._make_aggregated()
        results = cobb_douglas_sensitivity(agg, exponent_pairs=[(1.0, 1.0)])
        ranking = results[(1.0, 1.0)]

        for name, score in ranking:
            expected = agg[name]["efficiency_mean"] * agg[name]["context_diversity_mean"]
            assert abs(score - expected) < 1e-10

    def test_results_sorted_descending(self):
        agg = self._make_aggregated()
        results = cobb_douglas_sensitivity(agg)
        for _, ranking in results.items():
            for i in range(len(ranking) - 1):
                assert ranking[i][1] >= ranking[i + 1][1]

    def test_negative_efficiency_handled(self):
        """Negative efficiency should produce negative scores."""
        agg = {
            "Bad": {"efficiency_mean": -0.5, "context_diversity_mean": 1.0, "i_cd_mean": -0.5},
            "Good": {"efficiency_mean": 0.5, "context_diversity_mean": 1.0, "i_cd_mean": 0.5},
        }
        results = cobb_douglas_sensitivity(agg, exponent_pairs=[(1.0, 1.0)])
        ranking = results[(1.0, 1.0)]
        assert ranking[0][0] == "Good"
        assert ranking[1][0] == "Bad"
        assert ranking[1][1] < 0
