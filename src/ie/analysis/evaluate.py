"""Evaluation pipeline for computing I_cd scores across agents and environments."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from ie.agents.base import Agent
from ie.environments.base import BaseEnvironment
from ie.measure import context_diversity, efficiency


def evaluate_agent(
    agent: Agent, env: BaseEnvironment, configs: list[dict], episodes_per_config: int = 5
) -> dict:
    """Evaluate an agent across multiple environment configurations.

    Runs the agent (using act, not act_deterministic — some agents like RandomAgent
    only have act) across all configs, collecting entropy trajectories and computing I_cd.

    Args:
        agent: Agent with .act(obs) -> int method
        env: Environment instance (BaseEnvironment subclass)
        configs: List of config dicts to evaluate on
        episodes_per_config: Number of episodes per config

    Returns dict with:
        'i_cd': float — overall I_cd score
        'efficiency': float — average -delta_s / work across all episodes
        'context_diversity': float — H(C) over observed context labels
        'entropy_trajectories': dict mapping config name to list of entropy-over-time lists
        'total_work': float
        'total_entropy_change': float
        'context_labels': list[int]
    """
    all_context_labels = []
    total_work = 0.0
    total_entropy_change = 0.0
    entropy_trajectories = {}

    for config in configs:
        config_name = config.get("name", "unknown")
        config_trajectories = []

        for ep in range(episodes_per_config):
            obs = env.reset(config)
            context_label = env.get_context_label()
            all_context_labels.append(context_label)

            prev_entropy = env.compute_entropy()
            trajectory = [prev_entropy]
            done = False

            while not done:
                action = agent.act(obs)
                obs, _, done, info = env.step(action)
                curr_entropy = env.compute_entropy()

                total_work += info["work"]
                total_entropy_change += curr_entropy - prev_entropy
                trajectory.append(curr_entropy)
                prev_entropy = curr_entropy

            config_trajectories.append(trajectory)

        entropy_trajectories[config_name] = config_trajectories

    # Compute I_cd components
    eff = efficiency(total_entropy_change, total_work) if total_work > 0 else 0.0
    div = context_diversity(all_context_labels)
    score = eff * div

    return {
        "i_cd": score,
        "efficiency": eff,
        "context_diversity": div,
        "entropy_trajectories": entropy_trajectories,
        "total_work": total_work,
        "total_entropy_change": total_entropy_change,
        "context_labels": all_context_labels,
    }


def evaluate_all_agents(
    agents: dict[str, Agent],
    env: BaseEnvironment,
    configs: list[dict],
    episodes_per_config: int = 5,
) -> dict:
    """Evaluate multiple agents and return comparative results.

    Args:
        agents: dict mapping agent name to agent instance
        env: Environment instance
        configs: List of config dicts
        episodes_per_config: Episodes per config

    Returns dict mapping agent name to evaluation result dict.
    """
    results = {}
    for name, agent in agents.items():
        results[name] = evaluate_agent(agent, env, configs, episodes_per_config)
    return results


def aggregate_seed_results(
    per_seed_results: list[dict[str, dict]],
) -> dict[str, dict]:
    """Aggregate evaluation results from multiple seeds into mean and std.

    Args:
        per_seed_results: list of dicts, each mapping agent name to eval result.
            Each eval result must have 'i_cd', 'efficiency', 'context_diversity' keys.

    Returns:
        dict mapping agent name to aggregated stats with keys:
            i_cd_mean, i_cd_std, efficiency_mean, efficiency_std,
            context_diversity_mean, context_diversity_std
    """
    if not per_seed_results:
        return {}

    agent_names = list(per_seed_results[0].keys())
    aggregated = {}

    for agent in agent_names:
        i_cd_values = np.array([seed[agent]["i_cd"] for seed in per_seed_results])
        efficiency_values = np.array([seed[agent]["efficiency"] for seed in per_seed_results])
        diversity_values = np.array([seed[agent]["context_diversity"] for seed in per_seed_results])

        aggregated[agent] = {
            "i_cd_mean": float(np.mean(i_cd_values)),
            "i_cd_std": float(np.std(i_cd_values)),
            "efficiency_mean": float(np.mean(efficiency_values)),
            "efficiency_std": float(np.std(efficiency_values)),
            "context_diversity_mean": float(np.mean(diversity_values)),
            "context_diversity_std": float(np.std(diversity_values)),
            "i_cd_values": i_cd_values.tolist(),
        }

    return aggregated


def context_sensitivity_analysis(
    agents: dict[str, Agent],
    configs: list[dict],
    env: BaseEnvironment,
    episodes_per_config: int = 5,
) -> dict[str, list[dict]]:
    """Re-evaluate agents under coarsened and refined context partitions.

    Tests how I_cd rankings change when context granularity varies:
    - Full: all configs (finest partition)
    - Coarsened: first two configs only (fewer contexts)
    - Single: one config only (H(C) = 0, I_cd = 0 by construction)

    Args:
        agents: Dict mapping agent name to agent instance.
        configs: Full list of config dicts (at least 2).
        env: Environment instance.
        episodes_per_config: Episodes per config for evaluation.

    Returns:
        Dict mapping partition name to list of dicts with keys:
            'agent', 'i_cd', 'efficiency', 'context_diversity'
        Sorted by I_cd descending within each partition.
    """
    partitions: dict[str, list[dict]] = {}

    # Full partition (all configs)
    full_results = evaluate_all_agents(
        agents, env, configs, episodes_per_config=episodes_per_config
    )
    partitions["full"] = _partition_summary(full_results)

    # Coarsened partition (first two configs)
    if len(configs) >= 2:
        coarsened_results = evaluate_all_agents(
            agents, env, configs[:2], episodes_per_config=episodes_per_config
        )
        partitions["coarsened"] = _partition_summary(coarsened_results)

    # Single partition (one config → H(C) = 0)
    single_results = evaluate_all_agents(
        agents, env, [configs[0]], episodes_per_config=episodes_per_config
    )
    partitions["single"] = _partition_summary(single_results)

    return partitions


def _partition_summary(results: dict[str, dict]) -> list[dict]:
    """Extract summary from evaluation results, sorted by I_cd descending."""
    summary = []
    for name, res in results.items():
        summary.append(
            {
                "agent": name,
                "i_cd": res["i_cd"],
                "efficiency": res["efficiency"],
                "context_diversity": res["context_diversity"],
            }
        )
    return sorted(summary, key=lambda x: x["i_cd"], reverse=True)


def cobb_douglas_sensitivity(
    aggregated: dict[str, dict],
    exponent_pairs: list[tuple[float, float]] | None = None,
) -> dict[tuple[float, float], list[tuple[str, float]]]:
    """Compute I_cd^{a,b} = eta^a * H(C)^b for multiple exponent pairs.

    Tests whether ordinal rankings are preserved across exponent choices.
    If rankings are stable, this supports the parsimony of a=b=1.

    Args:
        aggregated: Output of aggregate_seed_results. Each agent entry must have
            'efficiency_mean' and 'context_diversity_mean' keys.
        exponent_pairs: List of (a, b) tuples. Defaults to
            [(0.5, 1), (1, 0.5), (1, 1), (1, 2), (2, 1)].

    Returns:
        Dict mapping (a, b) to list of (agent_name, score) sorted by score descending.
    """
    if exponent_pairs is None:
        exponent_pairs = [(0.5, 1.0), (1.0, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]

    results: dict[tuple[float, float], list[tuple[str, float]]] = {}

    for a, b in exponent_pairs:
        scores: list[tuple[str, float]] = []
        for agent_name, agg in aggregated.items():
            eta = agg["efficiency_mean"]
            hc = agg["context_diversity_mean"]
            # Handle negative efficiency: sign(eta) * |eta|^a
            if eta >= 0:
                score = (eta**a) * (hc**b)
            else:
                score = -((-eta) ** a) * (hc**b)
            scores.append((agent_name, float(score)))
        results[(a, b)] = sorted(scores, key=lambda x: x[1], reverse=True)

    return results


def compute_pairwise_significance(
    aggregated: dict[str, dict],
    ranking_order: list[str] | None = None,
) -> list[dict]:
    """Compute Wilcoxon signed-rank tests for adjacent-rank agent pairs.

    For each pair of adjacent agents in the ranking, tests whether the
    difference in I_cd scores across seeds is statistically significant.

    Args:
        aggregated: Output of aggregate_seed_results (must include 'i_cd_values').
        ranking_order: Agent names in ascending I_cd order. If None, sorts by i_cd_mean.

    Returns:
        List of dicts, one per adjacent pair, with keys:
            'agent_a', 'agent_b': agent names (a < b in ranking)
            'mean_a', 'mean_b': mean I_cd values
            'p_value': two-sided Wilcoxon p-value (or NaN if test cannot be computed)
            'n_seeds': number of seeds
            'significant': bool (p < 0.05)
    """
    if ranking_order is None:
        ranking_order = sorted(aggregated.keys(), key=lambda k: aggregated[k]["i_cd_mean"])

    results = []
    for i in range(len(ranking_order) - 1):
        name_a = ranking_order[i]
        name_b = ranking_order[i + 1]

        values_a = np.array(aggregated[name_a]["i_cd_values"])
        values_b = np.array(aggregated[name_b]["i_cd_values"])
        n = len(values_a)

        diffs = values_b - values_a

        # Wilcoxon requires n >= 6 non-zero differences for a valid test
        non_zero = np.count_nonzero(diffs)
        if non_zero < 6:
            p_value = float("nan")
        else:
            _, p_value = wilcoxon(diffs)
            p_value = float(p_value)

        results.append(
            {
                "agent_a": name_a,
                "agent_b": name_b,
                "mean_a": aggregated[name_a]["i_cd_mean"],
                "mean_b": aggregated[name_b]["i_cd_mean"],
                "p_value": p_value,
                "n_seeds": n,
                "significant": not np.isnan(p_value) and p_value < 0.05,
            }
        )

    return results
