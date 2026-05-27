"""Master experiment script: train, evaluate (multi-seed), generate figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ie.agents.curiosity_agent import CuriosityAgent
from ie.agents.heuristic_agent import (
    RegulationHeuristicAgent,
    SeqHeuristicAgent,
)
from ie.agents.random_agent import RandomAgent
from ie.agents.reward_maximizer import RewardMaximizer
from ie.analysis.evaluate import (
    aggregate_seed_results,
    compute_pairwise_significance,
    evaluate_all_agents,
)
from ie.analysis.figures import plot_ablation, plot_entropy_curves
from ie.analysis.tables import generate_latex_tables
from ie.environments.base import BaseEnvironment
from ie.environments.coupled_regulation import CoupledRegulation
from ie.environments.sequence_sorting import SequenceSorting

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"
TABLES_DIR = Path(__file__).parent.parent / "paper" / "tables"
DATA_DIR = Path(__file__).parent.parent / "paper" / "data"

EVAL_EPISODES = 10
DEFAULT_SEEDS = list(range(42, 92))  # 50 seeds

# Per-environment training configuration
ENV_CONFIGS = {
    "Sequence Sorting": {
        "env_class": SequenceSorting,
        "configs": [{"name": "short"}, {"name": "medium"}, {"name": "duplicates"}],
        "heuristic_class": SeqHeuristicAgent,
        "train_episodes": 500,
        "hidden_size": 64,
        "seeds": DEFAULT_SEEDS,
    },
    "Coupled Regulation": {
        "env_class": CoupledRegulation,
        "configs": [{"name": "weak"}, {"name": "moderate"}, {"name": "strong"}],
        "heuristic_class": RegulationHeuristicAgent,
        "train_episodes": 1000,
        "hidden_size": 64,
        "seeds": DEFAULT_SEEDS,
    },
}


def train_agents(
    env: BaseEnvironment,
    configs: list[dict[str, str]],
    seed: int = 42,
    train_episodes: int = 200,
    hidden_size: int = 64,
) -> dict[str, RewardMaximizer | CuriosityAgent]:
    """Train reward maximizer and curiosity agents.

    Args:
        env: Environment instance
        configs: List of config dicts to train on (cycled)
        seed: Random seed
        train_episodes: Number of training episodes
        hidden_size: Hidden layer size for agent networks

    Returns:
        Dict mapping agent name to trained agent instance.
    """
    obs_size = env.observation_shape[0]
    action_size = env.action_space_size

    agents = {
        "reward_maximizer": RewardMaximizer(
            obs_size=obs_size, action_size=action_size, seed=seed, hidden_size=hidden_size
        ),
        "curiosity": CuriosityAgent(
            obs_size=obs_size, action_size=action_size, seed=seed + 1, hidden_size=hidden_size
        ),
    }

    for name, agent in agents.items():
        print(f"  Training {name}...")
        for ep in range(train_episodes):
            config = configs[ep % len(configs)]
            stats = agent.train_episode(env, config)
            if (ep + 1) % 100 == 0:
                print(
                    f"    Episode {ep + 1}/{train_episodes}: "
                    f"reward={stats['total_reward']:.3f}, "
                    f"entropy_change={stats['total_entropy_change']:.3f}"
                )

    return agents


def run_experiment(
    env_name: str,
    env: BaseEnvironment,
    configs: list[dict[str, str]],
    seed: int = 42,
    heuristic_agent: Any = None,
    train_episodes: int = 200,
    hidden_size: int = 64,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Run full experiment for one environment with one seed.

    Args:
        env_name: Name of the environment
        env: Environment instance
        configs: List of config dicts
        seed: Random seed
        heuristic_agent: Optional heuristic agent instance
        train_episodes: Number of training episodes
        hidden_size: Hidden layer size for agent networks

    Returns:
        Tuple of (results, all_agents) where all_agents is the dict of agent instances
        so they can be reused for ablation.
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment: {env_name} (seed={seed})")
    print(f"{'=' * 60}")

    # Train agents
    print("\nTraining agents...")
    trained_agents = train_agents(env, configs, seed, train_episodes, hidden_size)

    # Build agent dict in ascending expected I_cd order
    random_agent = RandomAgent(action_space_size=env.action_space_size, seed=seed + 2)

    all_agents: dict[str, Any] = {"Random": random_agent}
    if heuristic_agent is not None:
        all_agents["Heuristic"] = heuristic_agent
    all_agents["Curiosity"] = trained_agents["curiosity"]
    all_agents["Reward Maximizer"] = trained_agents["reward_maximizer"]

    # Evaluate all agents
    print("\nEvaluating agents...")
    results = evaluate_all_agents(all_agents, env, configs, episodes_per_config=EVAL_EPISODES)

    for name, res in results.items():
        print(
            f"  {name}: I_cd={res['i_cd']:.4f}, "
            f"efficiency={res['efficiency']:.4f}, "
            f"H(C)={res['context_diversity']:.4f}"
        )

    return results, all_agents


def export_pgfplots_data(aggregated: dict[str, dict[str, dict]]) -> None:
    """Export aggregated results to .dat file for pgfplots.

    Args:
        aggregated: dict mapping env_name to dict mapping agent_name to aggregated stats.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Get agent names from first environment
    agent_names = list(next(iter(aggregated.values())).keys())

    # Build header
    header = "Agent"
    for env_name in aggregated:
        safe_name = env_name.replace(" ", "")
        header += f" {safe_name}_Mean {safe_name}_Std"

    lines = [header]

    # Build data rows
    for agent in agent_names:
        safe_agent = agent.replace(" ", "_")
        line = safe_agent
        for env_name in aggregated:
            agg = aggregated[env_name][agent]
            line += f" {agg['i_cd_mean']:.6f} {agg['i_cd_std']:.6f}"
        lines.append(line)

    (DATA_DIR / "icd_ranking.dat").write_text("\n".join(lines))
    print(f"Exported pgfplots data to {DATA_DIR / 'icd_ranking.dat'}")


def run_context_exposure_experiment(
    last_agents: dict[str, dict[str, Any]],
    env_configs: dict[str, dict],
    eval_episodes: int,
) -> None:
    """Run context-exposure experiment to test I_cd = eta × H(C) discrimination.

    Evaluates the SAME trained agents on different context subsets:
    - 1 config → H(C) = 0 → I_cd = 0 (forced by math)
    - 2 configs → H(C) = ln(2) → I_cd = eta × ln(2)
    - 3 configs → H(C) = ln(3) → I_cd = eta × ln(3)

    Args:
        last_agents: Dict mapping env_name to dict of trained agent instances
        env_configs: ENV_CONFIGS dict with environment configurations
        eval_episodes: Number of episodes per config for evaluation
    """
    print(f"\n{'#' * 60}")
    print("CONTEXT-EXPOSURE EXPERIMENT")
    print(f"{'#' * 60}")
    print("\nEvaluating same agents on different context subsets to test H(C) discrimination...")

    # Store results for export
    all_results: list[dict[str, Any]] = []

    for env_name, env_cfg in env_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Environment: {env_name}")
        print(f"{'=' * 60}")

        # Get trained agents for this environment
        agents = last_agents[env_name]
        configs = env_cfg["configs"]

        # Create environment instance (use last seed for consistency)
        env = env_cfg["env_class"](seed=env_cfg["seeds"][-1])

        # Define context subsets
        context_subsets = [
            (1, [configs[0]], 0.0),  # 1 config → H(C) = 0
            (2, [configs[0], configs[1]], np.log(2)),  # 2 configs → H(C) = ln(2)
            (3, configs, np.log(3)),  # 3 configs → H(C) = ln(3)
        ]

        for num_contexts, config_subset, expected_hc in context_subsets:
            print(f"\n  Evaluating on {num_contexts} context(s) [H(C) = {expected_hc:.4f}]...")

            # Evaluate all agents on this subset
            results = evaluate_all_agents(
                agents, env, config_subset, episodes_per_config=eval_episodes
            )

            # Print and store results
            for agent_name, res in results.items():
                icd = res["i_cd"]
                efficiency = res["efficiency"]
                hc = res["context_diversity"]

                print(
                    f"    {agent_name}: I_cd={icd:.4f}, efficiency={efficiency:.4f}, H(C)={hc:.4f}"
                )

                # Store for export
                all_results.append(
                    {
                        "agent": agent_name,
                        "num_contexts": num_contexts,
                        "hc": hc,
                        "env": env_name,
                        "efficiency": efficiency,
                        "icd": icd,
                    }
                )

    # Export to .dat file for LaTeX/pgfplots
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / "icd_context_exposure.dat"

    lines = ["Agent NumContexts HC Env Efficiency Icd"]
    for result in all_results:
        # Replace spaces in agent/env names for easier parsing
        safe_agent = result["agent"].replace(" ", "_")
        safe_env = result["env"].replace(" ", "_")

        line = (
            f"{safe_agent} "
            f"{result['num_contexts']} "
            f"{result['hc']:.6f} "
            f"{safe_env} "
            f"{result['efficiency']:.6f} "
            f"{result['icd']:.6f}"
        )
        lines.append(line)

    output_file.write_text("\n".join(lines))
    print(f"\nExported context-exposure data to {output_file}")

    # Also save as JSON for downstream analysis
    json_output = RESULTS_DIR / "context_exposure_results.json"
    with open(json_output, "w") as f:
        json.dump(all_results, f, indent=2, default=_serialize_numpy)
    print(f"Saved JSON results to {json_output}")

    # Print summary showing how I_cd scales with H(C)
    print("\n" + "=" * 60)
    print("CONTEXT-EXPOSURE SUMMARY")
    print("=" * 60)
    print("\nHow I_cd scales with H(C) for each agent:\n")

    # Group by agent and environment
    from collections import defaultdict

    agent_env_results: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        agent_env_results[result["agent"]][result["env"]].append(result)

    for agent_name in sorted(agent_env_results.keys()):
        print(f"{agent_name}:")
        for env_name in sorted(agent_env_results[agent_name].keys()):
            env_results = sorted(
                agent_env_results[agent_name][env_name], key=lambda x: x["num_contexts"]
            )
            print(f"  {env_name}:")
            for res in env_results:
                print(
                    f"    {res['num_contexts']} context(s): "
                    f"H(C)={res['hc']:.4f} → I_cd={res['icd']:.4f}"
                )
        print()


def run_hetcost_experiment() -> None:
    """Run heterogeneous cost experiment comparing uniform vs heterogeneous costs.

    Evaluates all 4 agents on Coupled Regulation and Sequence Sorting with both
    cost models across multiple seeds, then exports results to paper/data/icd_hetcost.dat.
    """
    print(f"\n{'#' * 60}")
    print("HETEROGENEOUS COST EXPERIMENT")
    print(f"{'#' * 60}")

    # Environments that support heterogeneous costs
    hetcost_envs = {
        "Coupled Regulation": {
            "env_class": CoupledRegulation,
            "heuristic_class": RegulationHeuristicAgent,
        },
        "Sequence Sorting": {
            "env_class": SequenceSorting,
            "heuristic_class": SeqHeuristicAgent,
        },
    }

    # Collect per-environment aggregated results
    all_env_uniform_agg: dict[str, dict[str, dict]] = {}
    all_env_hetcost_agg: dict[str, dict[str, dict]] = {}

    for hetcost_env_name, hetcost_env_info in hetcost_envs.items():
        print(f"\n{'=' * 60}")
        print(f"Heterogeneous Cost: {hetcost_env_name}")
        print(f"{'=' * 60}")

        env_cfg = ENV_CONFIGS[hetcost_env_name]
        configs = env_cfg["configs"]
        hetcost_seeds = env_cfg["seeds"]

        uniform_results: list[dict[str, dict]] = []
        hetcost_results: list[dict[str, dict]] = []

        for idx, seed in enumerate(hetcost_seeds):
            print(f"\n--- Seed {seed} ({idx + 1}/{len(hetcost_seeds)}) ---")

            # Uniform cost environment
            env_uniform = hetcost_env_info["env_class"](seed=seed, heterogeneous_costs=False)
            heuristic = hetcost_env_info["heuristic_class"]()

            print("Training agents for UNIFORM cost model...")
            trained_uniform = train_agents(
                env_uniform, configs, seed, env_cfg["train_episodes"], env_cfg["hidden_size"]
            )

            agents_uniform: dict[str, Any] = {
                "Random": RandomAgent(
                    action_space_size=env_uniform.action_space_size, seed=seed + 2
                ),
                "Heuristic": heuristic,
                "Curiosity": trained_uniform["curiosity"],
                "Reward_Maximizer": trained_uniform["reward_maximizer"],
            }

            print("Evaluating agents on UNIFORM cost...")
            uniform_eval = evaluate_all_agents(
                agents_uniform, env_uniform, configs, episodes_per_config=EVAL_EPISODES
            )
            uniform_results.append(uniform_eval)

            for name, res in uniform_eval.items():
                print(f"  {name}: I_cd={res['i_cd']:.4f}, efficiency={res['efficiency']:.4f}")

            # Heterogeneous cost environment
            env_hetcost = hetcost_env_info["env_class"](seed=seed, heterogeneous_costs=True)

            print("Training agents for HETEROGENEOUS cost model...")
            trained_hetcost = train_agents(
                env_hetcost, configs, seed, env_cfg["train_episodes"], env_cfg["hidden_size"]
            )

            agents_hetcost: dict[str, Any] = {
                "Random": RandomAgent(
                    action_space_size=env_hetcost.action_space_size, seed=seed + 2
                ),
                "Heuristic": hetcost_env_info["heuristic_class"](),
                "Curiosity": trained_hetcost["curiosity"],
                "Reward_Maximizer": trained_hetcost["reward_maximizer"],
            }

            print("Evaluating agents on HETEROGENEOUS cost...")
            hetcost_eval = evaluate_all_agents(
                agents_hetcost, env_hetcost, configs, episodes_per_config=EVAL_EPISODES
            )
            hetcost_results.append(hetcost_eval)

            for name, res in hetcost_eval.items():
                print(f"  {name}: I_cd={res['i_cd']:.4f}, efficiency={res['efficiency']:.4f}")

        # Aggregate across seeds for this environment
        print(f"\nAggregating {hetcost_env_name} heterogeneous cost results...")
        all_env_uniform_agg[hetcost_env_name] = aggregate_seed_results(uniform_results)
        all_env_hetcost_agg[hetcost_env_name] = aggregate_seed_results(hetcost_results)

    # Export to .dat file (one row per agent per environment)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["Agent Env UniformEff UniformIcd HetEff HetIcd"]

    for env_name in hetcost_envs:
        uniform_agg = all_env_uniform_agg[env_name]
        hetcost_agg = all_env_hetcost_agg[env_name]
        safe_env = env_name.replace(" ", "_")

        for agent in uniform_agg:
            safe_agent = agent.replace(" ", "_")
            u_agg = uniform_agg[agent]
            h_agg = hetcost_agg[agent]

            line = (
                f"{safe_agent} {safe_env} "
                f"{u_agg['efficiency_mean']:.6f} "
                f"{u_agg['i_cd_mean']:.6f} "
                f"{h_agg['efficiency_mean']:.6f} "
                f"{h_agg['i_cd_mean']:.6f}"
            )
            lines.append(line)

    output_file = DATA_DIR / "icd_hetcost.dat"
    output_file.write_text("\n".join(lines))
    print(f"\nExported heterogeneous cost data to {output_file}")

    print("\nHeterogeneous Cost Summary:")
    for env_name in hetcost_envs:
        print(f"\n  {env_name}:")
        uniform_agg = all_env_uniform_agg[env_name]
        hetcost_agg = all_env_hetcost_agg[env_name]
        for agent in uniform_agg:
            u = uniform_agg[agent]
            h = hetcost_agg[agent]
            print(f"    {agent}:")
            print(f"      Uniform:       I_cd={u['i_cd_mean']:.4f}±{u['i_cd_std']:.4f}")
            print(f"      Heterogeneous: I_cd={h['i_cd_mean']:.4f}±{h['i_cd_std']:.4f}")


def _serialize_numpy(obj: object) -> object:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Multi-seed experiment loop (per-environment seed counts)
    all_env_seed_results: dict[str, list[dict[str, dict]]] = {}
    last_results: dict[str, dict[str, dict]] = {}
    last_agents: dict[str, dict[str, Any]] = {}

    for env_name, env_cfg in ENV_CONFIGS.items():
        env_seeds = env_cfg["seeds"]
        num_seeds = len(env_seeds)

        print(f"\n{'#' * 60}")
        print(f"ENVIRONMENT: {env_name} ({num_seeds} seeds)")
        print(f"{'#' * 60}")

        for idx, seed in enumerate(env_seeds):
            print(f"\n--- Seed {seed} ({idx + 1}/{num_seeds}) ---")

            env = env_cfg["env_class"](seed=seed)
            heuristic = env_cfg["heuristic_class"]()

            results, agents = run_experiment(
                env_name,
                env,
                env_cfg["configs"],
                seed,
                heuristic_agent=heuristic,
                train_episodes=env_cfg["train_episodes"],
                hidden_size=env_cfg["hidden_size"],
            )

            all_env_seed_results.setdefault(env_name, []).append(results)
            last_results[env_name] = results
            last_agents[env_name] = agents

    # Aggregate across seeds for each environment
    print("\nAggregating results across seeds...")
    aggregated: dict[str, dict[str, dict]] = {}
    for env_name, seed_results in all_env_seed_results.items():
        aggregated[env_name] = aggregate_seed_results(seed_results)

    for env_name, env_agg in aggregated.items():
        print(f"\n{env_name} ({len(ENV_CONFIGS[env_name]['seeds'])} seeds):")
        for name, agg in env_agg.items():
            print(f"  {name}: I_cd={agg['i_cd_mean']:.4f}±{agg['i_cd_std']:.4f}")

    # Pairwise significance tests for adjacent-rank agents
    print("\n" + "=" * 60)
    print("PAIRWISE SIGNIFICANCE TESTS (Wilcoxon signed-rank)")
    print("=" * 60)

    significance_results: dict[str, list[dict]] = {}
    for env_name, env_agg in aggregated.items():
        sig = compute_pairwise_significance(env_agg)
        significance_results[env_name] = sig
        n_seeds = len(ENV_CONFIGS[env_name]["seeds"])
        print(f"\n{env_name} ({n_seeds} seeds):")
        for pair in sig:
            p_str = f"p={pair['p_value']:.4f}" if not np.isnan(pair["p_value"]) else "p=N/A (n<6)"
            sig_str = "*" if pair["significant"] else ""
            print(
                f"  {pair['agent_a']} vs {pair['agent_b']}: "
                f"{pair['mean_a']:.4f} vs {pair['mean_b']:.4f}, "
                f"{p_str} {sig_str}"
            )

    # Build mean results dicts for figures (same shape as single-seed results)
    mean_results: dict[str, dict[str, dict]] = {}
    icd_stds: dict[str, dict[str, float]] = {}

    for env_name, env_agg in aggregated.items():
        mean_results[env_name] = {
            name: {
                "i_cd": agg["i_cd_mean"],
                "efficiency": agg["efficiency_mean"],
                "context_diversity": agg["context_diversity_mean"],
            }
            for name, agg in env_agg.items()
        }
        icd_stds[env_name] = {name: agg["i_cd_std"] for name, agg in env_agg.items()}

    # Export pgfplots data
    print("\nExporting pgfplots data...")
    export_pgfplots_data(aggregated)

    # Generate entropy curves for each environment (use last seed's full trajectories)
    print("\nGenerating entropy curves...")
    for env_name, results in last_results.items():
        safe_name = env_name.replace(" ", "_").lower()
        plot_entropy_curves(results, env_name, str(FIGURES_DIR / f"entropy_curves_{safe_name}.pdf"))

    # Ablation: evaluate TRAINED agents with single config (no context diversity)
    print("\nRunning ablation (single-config evaluation with last seed's agents)...")
    ablation_data: dict[str, dict[str, dict]] = {}

    for env_name, env_cfg in ENV_CONFIGS.items():
        last_env = env_cfg["env_class"](seed=env_cfg["seeds"][-1])
        single_config = [env_cfg["configs"][0]]  # first config only

        specialist = evaluate_all_agents(
            last_agents[env_name],
            last_env,
            single_config,
            episodes_per_config=EVAL_EPISODES,
        )

        # Generate ablation figure
        safe_name = env_name.replace(" ", "_").lower()
        plot_ablation(
            last_results[env_name],
            specialist,
            env_name,
            str(FIGURES_DIR / f"ablation_{safe_name}.pdf"),
        )

        # Store ablation data (Reward Maximizer only, for table generation)
        ablation_key = env_name.replace(" ", "_").lower()
        ablation_data[ablation_key] = {
            "specialist": specialist["Reward Maximizer"],
            "generalist": last_results[env_name]["Reward Maximizer"],
        }

    # Save raw results (aggregated means + per-seed data + significance tests)
    results_json: dict[str, Any] = {}

    for env_name, env_mean in mean_results.items():
        json_key = env_name.replace(" ", "_").lower()
        results_json[json_key] = {
            k: {kk: vv for kk, vv in v.items() if kk != "entropy_trajectories"}
            for k, v in env_mean.items()
        }

    # Save per-seed I_cd values and significance tests
    results_json["per_seed"] = {}
    results_json["significance_tests"] = {}
    for env_name, env_agg in aggregated.items():
        json_key = env_name.replace(" ", "_").lower()
        results_json["per_seed"][json_key] = {
            agent: agg.get("i_cd_values", []) for agent, agg in env_agg.items()
        }
        sig = compute_pairwise_significance(env_agg)
        results_json["significance_tests"][json_key] = sig

    results_json["ablation"] = {}
    for env_name, abl in ablation_data.items():
        results_json["ablation"][env_name] = {
            "specialist": {
                k: v for k, v in abl["specialist"].items() if k != "entropy_trajectories"
            },
            "generalist": {
                k: v for k, v in abl["generalist"].items() if k != "entropy_trajectories"
            },
        }

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=_serialize_numpy)

    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")

    env_stds = {
        env_name.replace(" ", "_").lower(): aggregated[env_name] for env_name in ENV_CONFIGS
    }

    generate_latex_tables(
        str(results_path),
        ablation_results=ablation_data,
        tables_dir=str(TABLES_DIR),
        env_stds=env_stds,
    )

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Figures saved to {FIGURES_DIR}/")
    print(f"Tables saved to {TABLES_DIR}/")
    print(f"Data files saved to {DATA_DIR}/")
    seed_counts = {name: len(cfg["seeds"]) for name, cfg in ENV_CONFIGS.items()}
    print(f"\nDone! (seeds per env: {seed_counts}, 4 agents per env)")

    # Run context-exposure experiment
    run_context_exposure_experiment(last_agents, ENV_CONFIGS, EVAL_EPISODES)

    # Run heterogeneous cost experiment
    run_hetcost_experiment()


if __name__ == "__main__":
    main()
