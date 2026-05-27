"""Sliding Puzzle joint I_cd test with restricted heuristic agents.

Demonstrates that I_cd = eta x H(C) jointly discriminates agents where
neither factor alone suffices. Restricted heuristics use the same greedy
Manhattan algorithm but are limited to certain grid sizes, creating agents
with naturally different eta and H(C) profiles.

Key result: Heuristic-Easy (w=2,3) has HIGHER eta than Heuristic-Full (all),
but LOWER I_cd because the product form correctly values broader H(C).

Agent types:
  1. Random: uniform random actions
  2. Heuristic-Narrow: greedy Manhattan on w=2 only, random on others
  3. Heuristic-Easy: greedy Manhattan on w=2,3 only, random on others
  4. Heuristic-Hard: greedy Manhattan on w=4,5 only, random on others
  5. Heuristic-Wide: greedy Manhattan on w=2,3,4, random on w=5
  6. Heuristic-Full: greedy Manhattan on all sizes

No training required. Pure evaluation.

Usage:
    uv run python experiments/run_sliding_puzzle_heuristic.py
    uv run python experiments/run_sliding_puzzle_heuristic.py --chunk 0 --num-chunks 5
    uv run python experiments/run_sliding_puzzle_heuristic.py --merge --num-chunks 5
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ie.agents.heuristic_agent import RestrictedSlidingPuzzleHeuristicAgent
from ie.agents.random_agent import RandomAgent
from ie.analysis.evaluate import compute_pairwise_significance
from ie.environments.sliding_puzzle import CONFIGS, SlidingPuzzle
from ie.measure import efficiency

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent.parent / "paper" / "data"

DEFAULT_SEEDS = list(range(42, 92))  # 50 seeds
EVAL_EPISODES = 10


def evaluate_with_performance_gating(
    agent: Any,
    env: SlidingPuzzle,
    all_configs: list[dict],
    episodes_per_config: int,
) -> dict:
    """Evaluate agent with performance-gated H(C).

    Evaluates on all 10 contexts, then computes H(C) only over contexts
    where eta > 0 (agent achieves net entropy reduction).
    """
    eta_per_context: dict[int, float] = {}

    for config in all_configs:
        context_id = config["context_id"]
        total_entropy_change = 0.0
        total_work = 0.0

        for _ in range(episodes_per_config):
            obs = env.reset(config)
            done = False
            prev_entropy = env.compute_entropy()

            while not done:
                action = agent.act(obs)
                obs, _, done, info = env.step(action)
                curr_entropy = env.compute_entropy()
                total_entropy_change += curr_entropy - prev_entropy
                total_work += info["work"]
                prev_entropy = curr_entropy

        eta = efficiency(total_entropy_change, total_work) if total_work > 0 else 0.0
        eta_per_context[context_id] = eta

    # Performance gating: only count contexts where eta > 0
    effective = [cid for cid, eta in eta_per_context.items() if eta > 0]
    k = len(effective)

    if k == 0:
        eta_avg = 0.0
        h_c = 0.0
    elif k == 1:
        eta_avg = eta_per_context[effective[0]]
        h_c = 0.0
    else:
        eta_avg = float(np.mean([eta_per_context[cid] for cid in effective]))
        h_c = float(np.log(k))

    i_cd = eta_avg * h_c

    return {
        "eta_per_context": eta_per_context,
        "effective_contexts": effective,
        "n_effective": k,
        "eta_avg": eta_avg,
        "h_c": h_c,
        "i_cd": i_cd,
    }


def run_single_seed(seed: int) -> dict[str, dict]:
    """Run the full restricted heuristic experiment for one seed."""
    print(f"\n{'=' * 60}")
    print(f"Sliding Puzzle Heuristic: seed={seed}")
    print(f"{'=' * 60}")

    env = SlidingPuzzle(seed=seed)

    agents: dict[str, Any] = {
        "Random": RandomAgent(action_space_size=env.action_space_size, seed=seed),
        "Heuristic-Narrow": RestrictedSlidingPuzzleHeuristicAgent(allowed_widths={2}, seed=seed),
        "Heuristic-Easy": RestrictedSlidingPuzzleHeuristicAgent(allowed_widths={2, 3}, seed=seed),
        "Heuristic-Hard": RestrictedSlidingPuzzleHeuristicAgent(allowed_widths={4, 5}, seed=seed),
        "Heuristic-Wide": RestrictedSlidingPuzzleHeuristicAgent(
            allowed_widths={2, 3, 4}, seed=seed
        ),
        "Heuristic-Full": RestrictedSlidingPuzzleHeuristicAgent(
            allowed_widths={2, 3, 4, 5}, seed=seed
        ),
    }

    results = {}
    for name, agent in agents.items():
        res = evaluate_with_performance_gating(agent, env, CONFIGS, EVAL_EPISODES)
        results[name] = res
        print(
            f"  {name:<20} eta={res['eta_avg']:.6f}  "
            f"H(C)={res['h_c']:.4f} (k={res['n_effective']})  "
            f"I_cd={res['i_cd']:.6f}"
        )

    return results


def aggregate_results(
    per_seed_results: list[dict[str, dict]],
) -> dict[str, dict]:
    """Aggregate results across seeds."""
    if not per_seed_results:
        return {}

    agent_names = list(per_seed_results[0].keys())
    aggregated: dict[str, dict] = {}

    for agent in agent_names:
        i_cd_values = [seed[agent]["i_cd"] for seed in per_seed_results]
        eta_values = [seed[agent]["eta_avg"] for seed in per_seed_results]
        hc_values = [seed[agent]["h_c"] for seed in per_seed_results]
        k_values = [seed[agent]["n_effective"] for seed in per_seed_results]

        aggregated[agent] = {
            "i_cd_mean": float(np.mean(i_cd_values)),
            "i_cd_std": float(np.std(i_cd_values)),
            "eta_mean": float(np.mean(eta_values)),
            "eta_std": float(np.std(eta_values)),
            "h_c_mean": float(np.mean(hc_values)),
            "h_c_std": float(np.std(hc_values)),
            "k_mean": float(np.mean(k_values)),
            "k_std": float(np.std(k_values)),
            "i_cd_values": i_cd_values,
        }

    return aggregated


def export_results(aggregated: dict[str, dict]) -> None:
    """Export results to .dat file for pgfplots."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    header = "Agent Eta_Mean Eta_Std HC_Mean HC_Std Icd_Mean Icd_Std K_Mean K_Std"
    lines = [header]

    for agent, agg in aggregated.items():
        safe = agent.replace(" ", "_").replace("(", "").replace(")", "")
        lines.append(
            f"{safe} "
            f"{agg['eta_mean']:.6f} {agg['eta_std']:.6f} "
            f"{agg['h_c_mean']:.6f} {agg['h_c_std']:.6f} "
            f"{agg['i_cd_mean']:.6f} {agg['i_cd_std']:.6f} "
            f"{agg['k_mean']:.1f} {agg['k_std']:.1f}"
        )

    output_file = DATA_DIR / "icd_sliding_puzzle_heuristic.dat"
    output_file.write_text("\n".join(lines))
    print(f"Exported data to {output_file}")


def _serialize_numpy(obj: object) -> object:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def _print_results(aggregated: dict[str, dict], n_seeds: int) -> None:
    """Print formatted results table."""
    print(f"\n{'#' * 60}")
    print(f"AGGREGATED RESULTS ({n_seeds} seeds)")
    print(f"{'#' * 60}")

    print(f"\n{'Agent':<22} {'eta':>10} {'H(C)':>8} {'I_cd':>10} {'k':>5}")
    print("-" * 59)
    for agent, agg in sorted(aggregated.items(), key=lambda x: x[1]["i_cd_mean"], reverse=True):
        print(
            f"{agent:<22} "
            f"{agg['eta_mean']:>10.6f} "
            f"{agg['h_c_mean']:>8.4f} "
            f"{agg['i_cd_mean']:>10.6f} "
            f"{agg['k_mean']:>5.1f}"
        )

    # Highlight the key comparison
    if "Heuristic-Easy" in aggregated and "Heuristic-Full" in aggregated:
        easy = aggregated["Heuristic-Easy"]
        full = aggregated["Heuristic-Full"]
        print(f"\n{'=' * 60}")
        print("KEY COMPARISON: Heuristic-Easy vs Heuristic-Full")
        print(f"{'=' * 60}")
        print(f"  eta:  Easy={easy['eta_mean']:.6f}  Full={full['eta_mean']:.6f}")
        print(f"  H(C): Easy={easy['h_c_mean']:.4f}      Full={full['h_c_mean']:.4f}")
        print(f"  I_cd: Easy={easy['i_cd_mean']:.6f}  Full={full['i_cd_mean']:.6f}")
        eta_easy_wins = easy["eta_mean"] > full["eta_mean"]
        icd_full_wins = full["i_cd_mean"] > easy["i_cd_mean"]
        if eta_easy_wins and icd_full_wins:
            print("  >>> JOINT DISCRIMINATION: Easy has higher eta but lower I_cd <<<")
        elif not eta_easy_wins:
            print("  WARNING: Expected Easy eta > Full eta, but it's not")
        elif not icd_full_wins:
            print("  WARNING: Expected Full I_cd > Easy I_cd, but it's not")


def run_chunk(chunk: int, num_chunks: int) -> None:
    """Run a chunk of seeds and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    seeds = DEFAULT_SEEDS
    chunk_size = len(seeds) // num_chunks
    remainder = len(seeds) % num_chunks

    if chunk < remainder:
        start = chunk * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = remainder * (chunk_size + 1) + (chunk - remainder) * chunk_size
        end = start + chunk_size

    chunk_seeds = seeds[start:end]

    print(f"\nCHUNK {chunk}/{num_chunks}: seeds {chunk_seeds[0]}-{chunk_seeds[-1]}")

    per_seed_results: list[dict] = []
    for idx, seed in enumerate(chunk_seeds):
        print(f"\n--- Seed {seed} ({idx + 1}/{len(chunk_seeds)}) ---")
        results = run_single_seed(seed)
        per_seed_results.append({"seed": seed, "results": results})

    chunk_path = RESULTS_DIR / f"sliding_puzzle_heuristic_chunk_{chunk}.json"
    with open(chunk_path, "w") as f:
        json.dump(per_seed_results, f, indent=2, default=_serialize_numpy)
    print(f"\nChunk {chunk} saved to {chunk_path}")


def merge_chunks(num_chunks: int) -> None:
    """Load all chunk files, aggregate, run significance tests, export."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_seed_results: list[dict[str, dict]] = []
    missing_chunks = []

    for chunk in range(num_chunks):
        chunk_path = RESULTS_DIR / f"sliding_puzzle_heuristic_chunk_{chunk}.json"
        if not chunk_path.exists():
            missing_chunks.append(chunk)
            continue
        with open(chunk_path) as f:
            chunk_data = json.load(f)
        for entry in chunk_data:
            all_seed_results.append(entry["results"])

    if missing_chunks:
        print(f"WARNING: Missing chunks: {missing_chunks}")
    print(f"Loaded {len(all_seed_results)} seeds.")

    if not all_seed_results:
        print("No results to merge!")
        return

    aggregated = aggregate_results(all_seed_results)
    _print_results(aggregated, len(all_seed_results))

    # Significance tests
    print(f"\n{'=' * 60}")
    print("PAIRWISE SIGNIFICANCE (Wilcoxon signed-rank)")
    print(f"{'=' * 60}")

    sig = compute_pairwise_significance(aggregated)
    for pair in sig:
        p_str = f"p={pair['p_value']:.4f}" if not np.isnan(pair["p_value"]) else "p=N/A"
        marker = "*" if pair["significant"] else ""
        print(
            f"  {pair['agent_a']} vs {pair['agent_b']}: "
            f"{pair['mean_a']:.6f} vs {pair['mean_b']:.6f}, {p_str} {marker}"
        )

    export_results(aggregated)

    json_path = RESULTS_DIR / "sliding_puzzle_heuristic_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "aggregated": {k: {kk: vv for kk, vv in v.items()} for k, v in aggregated.items()},
                "significance": sig,
                "n_seeds": len(all_seed_results),
                "eval_episodes": EVAL_EPISODES,
            },
            f,
            indent=2,
            default=_serialize_numpy,
        )
    print(f"\nSaved results to {json_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sliding Puzzle joint I_cd test with restricted heuristics"
    )
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_chunks(args.num_chunks)
    elif args.chunk is not None:
        if not 0 <= args.chunk < args.num_chunks:
            parser.error(f"--chunk must be 0 to {args.num_chunks - 1}")
        run_chunk(args.chunk, args.num_chunks)
    else:
        # No args: run everything sequentially
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        per_seed_results: list[dict[str, dict]] = []
        for idx, seed in enumerate(DEFAULT_SEEDS):
            print(f"\n--- Seed {seed} ({idx + 1}/{len(DEFAULT_SEEDS)}) ---")
            results = run_single_seed(seed)
            per_seed_results.append(results)

        aggregated = aggregate_results(per_seed_results)
        _print_results(aggregated, len(DEFAULT_SEEDS))

        sig = compute_pairwise_significance(aggregated)
        export_results(aggregated)

        json_path = RESULTS_DIR / "sliding_puzzle_heuristic_results.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "aggregated": {
                        k: {kk: vv for kk, vv in v.items()} for k, v in aggregated.items()
                    },
                    "significance": sig,
                    "n_seeds": len(DEFAULT_SEEDS),
                    "eval_episodes": EVAL_EPISODES,
                },
                f,
                indent=2,
                default=_serialize_numpy,
            )
        print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
