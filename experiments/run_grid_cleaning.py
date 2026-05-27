"""Grid Cleaning experiment: 6 agents x 10 layouts x 50 seeds.

This experiment jointly tests both factors of I_cd = eta x H(C).
Unlike CoupledReg (where H(C) is constant) or ARC (where eta is weak),
Grid Cleaning produces agents with varying eta AND varying H(C).

Agent types:
  1. Random: uniform random actions
  2. Heuristic: BFS nearest-dirty greedy
  3. Generalist PPO (Task): shaped reward on ALL 10 layouts
  4. Curiosity Generalist PPO: prediction-error reward on ALL 10 layouts
  5. Specialist PPO (Easy): shaped reward on layouts 0-1 ONLY
  6. Specialist PPO (Hard): shaped reward on layouts 7-9 ONLY

Reward shaping (per literature on grid-cleaning RL):
  +1.0 for successful clean (entropy decreased)
  -0.01 step penalty (encourages efficiency)
  This is a standard task reward, NOT the I_cd measure.

Performance-gated H(C): for each agent, H(C) is computed only over
layouts where eta > 0 (agent achieves net entropy reduction).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ie.agents.curiosity_agent import CuriosityAgent
from ie.agents.heuristic_agent import GridCleaningHeuristicAgent
from ie.agents.ppo import PPOAgent
from ie.agents.random_agent import RandomAgent
from ie.agents.training import run_training_episode
from ie.analysis.evaluate import (
    compute_pairwise_significance,
)
from ie.environments.grid_cleaning import GridCleaning
from ie.measure import efficiency

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent.parent / "paper" / "data"
FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"

DEFAULT_SEEDS = list(range(42, 92))  # 50 seeds
EVAL_EPISODES = 10
TRAIN_EPISODES = 3000  # 3000 x 200 steps = 600K timesteps (literature: 100K-500K for 8x8)
HIDDEN_SIZE = 64

# All 10 layout configs
ALL_LAYOUTS = [{"layout_id": i, "name": GridCleaning.LAYOUT_NAMES[i]} for i in range(10)]

# Specialist subsets
EASY_LAYOUTS = [cfg for cfg in ALL_LAYOUTS if cfg["layout_id"] in (0, 1)]
HARD_LAYOUTS = [cfg for cfg in ALL_LAYOUTS if cfg["layout_id"] in (7, 8, 9)]


def _nearest_dirty_manhattan(obs: np.ndarray) -> float:
    """Manhattan distance from agent to nearest dirty cell, decoded from obs.

    obs layout: [64 grid values, agent_row_norm, agent_col_norm]
    Grid values: 0.0=clean, 0.33=light, 0.67=medium, 1.0=heavy, -1.0=blocked
    """
    agent_row = round(obs[64] * (GRID_SIZE - 1))
    agent_col = round(obs[65] * (GRID_SIZE - 1))
    min_dist = float("inf")
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if obs[r * GRID_SIZE + c] > 0.01:  # dirty cell
                dist = abs(r - agent_row) + abs(c - agent_col)
                if dist < min_dist:
                    min_dist = dist
    return min_dist if min_dist < float("inf") else 0.0


GRID_SIZE = 8  # must match grid_cleaning.GRID_SIZE
SHAPING_SCALE = 0.1


def _task_reward(obs: np.ndarray, action: int, obs_next: np.ndarray, delta_entropy: float) -> float:
    """Shaped task reward for grid cleaning.

    Based on the RL grid-cleaning literature (Moon et al. 2022, Vo et al. 2025):
    dense reward signal with per-step penalty, cleaning bonus, and
    potential-based distance shaping (Ng et al. 1999).

    The distance potential rewards the agent for moving toward dirty cells,
    providing continuous navigation signal instead of sparse cleaning-only reward.

    This is a standard task reward, NOT the I_cd measure. Agents are trained
    to clean efficiently, then evaluated post-hoc with I_cd.
    """
    reward = -0.01  # step penalty: encourages efficiency
    if delta_entropy < -1e-8:
        reward += 1.0  # entropy decreased = successful cleaning
    # Potential-based shaping: reward for getting closer to dirt
    d_old = _nearest_dirty_manhattan(obs)
    d_new = _nearest_dirty_manhattan(obs_next)
    reward += SHAPING_SCALE * (d_old - d_new)
    return reward


def train_task_ppo(
    env: GridCleaning,
    train_configs: list[dict],
    seed: int,
    label: str = "",
) -> PPOAgent:
    """Train a PPO agent with shaped task reward.

    Returns a PPOAgent (has .act() method, compatible with evaluation).
    """
    ppo = PPOAgent(
        obs_size=env.observation_shape[0],
        action_size=env.action_space_size,
        seed=seed,
        hidden_size=HIDDEN_SIZE,
    )

    for ep in range(TRAIN_EPISODES):
        config = train_configs[ep % len(train_configs)]
        result = run_training_episode(ppo, env, _task_reward, config=config)
        if (ep + 1) % 500 == 0:
            print(
                f"    [{label}] Episode {ep + 1}/{TRAIN_EPISODES}: "
                f"reward={result.stats['total_reward']:.3f}, "
                f"entropy_change={result.stats['total_entropy_change']:.3f}"
            )

    return ppo


def train_curiosity_ppo(
    env: GridCleaning,
    train_configs: list[dict],
    seed: int,
    label: str = "",
) -> CuriosityAgent:
    """Train a Curiosity agent (prediction-error intrinsic reward)."""
    agent = CuriosityAgent(
        obs_size=env.observation_shape[0],
        action_size=env.action_space_size,
        seed=seed,
        hidden_size=HIDDEN_SIZE,
    )

    for ep in range(TRAIN_EPISODES):
        config = train_configs[ep % len(train_configs)]
        stats = agent.train_episode(env, config)
        if (ep + 1) % 500 == 0:
            print(
                f"    [{label}] Episode {ep + 1}/{TRAIN_EPISODES}: "
                f"reward={stats['total_reward']:.3f}, "
                f"entropy_change={stats['total_entropy_change']:.3f}"
            )

    return agent


def evaluate_with_performance_gating(
    agent: Any,
    env: GridCleaning,
    all_configs: list[dict],
    episodes_per_config: int,
) -> dict:
    """Evaluate agent with performance-gated H(C).

    Evaluates on all 10 layouts, then computes H(C) only over layouts
    where eta > 0 (agent achieves net entropy reduction).
    """
    eta_per_layout = {}

    for config in all_configs:
        layout_id = config["layout_id"]

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
        eta_per_layout[layout_id] = eta

    # Performance gating: only count layouts where eta > 0
    effective = [lid for lid, eta in eta_per_layout.items() if eta > 0]
    k = len(effective)

    if k == 0:
        eta_avg = 0.0
        h_c = 0.0
    elif k == 1:
        eta_avg = eta_per_layout[effective[0]]
        h_c = 0.0  # single context, H(C) = 0
    else:
        eta_avg = float(np.mean([eta_per_layout[lid] for lid in effective]))
        h_c = float(np.log(k))

    i_cd = eta_avg * h_c

    # Also compute ungated values for comparison
    all_etas = list(eta_per_layout.values())
    eta_all = float(np.mean(all_etas))
    h_c_all = float(np.log(10))
    i_cd_all = eta_all * h_c_all

    return {
        "eta_per_layout": eta_per_layout,
        "effective_layouts": effective,
        "n_effective": k,
        "eta_avg": eta_avg,
        "h_c": h_c,
        "i_cd": i_cd,
        "eta_all": eta_all,
        "h_c_all": h_c_all,
        "i_cd_all": i_cd_all,
    }


def run_single_seed(seed: int) -> dict[str, dict]:
    """Run the full grid cleaning experiment for one seed."""
    print(f"\n{'=' * 60}")
    print(f"Grid Cleaning: seed={seed}")
    print(f"{'=' * 60}")

    env = GridCleaning(seed=seed)

    # 1. Random agent
    print("  Building Random agent...")
    random_agent = RandomAgent(action_space_size=env.action_space_size, seed=seed)

    # 2. Heuristic agent
    print("  Building Heuristic agent...")
    heuristic = GridCleaningHeuristicAgent()

    # 3. Generalist PPO (Task: shaped reward, all 10 layouts)
    print("  Training Generalist PPO (Task)...")
    generalist_task = train_task_ppo(env, ALL_LAYOUTS, seed, label="Generalist-Task")

    # 4. Curiosity Generalist PPO (all 10 layouts)
    print("  Training Curiosity Generalist PPO...")
    curiosity = train_curiosity_ppo(env, ALL_LAYOUTS, seed + 1, label="Curiosity")

    # 5. Specialist PPO (Easy: layouts 0-1 only)
    print("  Training Specialist PPO (Easy)...")
    specialist_easy = train_task_ppo(env, EASY_LAYOUTS, seed + 2, label="Spec-Easy")

    # 6. Specialist PPO (Hard: layouts 7-9 only)
    print("  Training Specialist PPO (Hard)...")
    specialist_hard = train_task_ppo(env, HARD_LAYOUTS, seed + 3, label="Spec-Hard")

    agents = {
        "Random": random_agent,
        "Heuristic": heuristic,
        "Generalist (Task)": generalist_task,
        "Curiosity": curiosity,
        "Specialist (Easy)": specialist_easy,
        "Specialist (Hard)": specialist_hard,
    }

    # Evaluate all agents with performance-gated H(C)
    print("  Evaluating all agents...")
    results = {}
    for name, agent in agents.items():
        res = evaluate_with_performance_gating(agent, env, ALL_LAYOUTS, EVAL_EPISODES)
        results[name] = res
        print(
            f"    {name}: eta_avg={res['eta_avg']:.4f}, "
            f"H(C)={res['h_c']:.4f} (k={res['n_effective']}), "
            f"I_cd={res['i_cd']:.6f}"
        )

    return results


def aggregate_grid_results(
    per_seed_results: list[dict[str, dict]],
) -> dict[str, dict]:
    """Aggregate performance-gated results across seeds."""
    if not per_seed_results:
        return {}

    agent_names = list(per_seed_results[0].keys())
    aggregated = {}

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

    output_file = DATA_DIR / "icd_grid_cleaning.dat"
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


def run_chunk(chunk: int, num_chunks: int) -> None:
    """Run a chunk of seeds and save per-seed results to a chunk file."""
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

    print(f"\n{'#' * 60}")
    print(
        f"CHUNK {chunk}/{num_chunks}: seeds {chunk_seeds[0]}-{chunk_seeds[-1]} "
        f"({len(chunk_seeds)} seeds)"
    )
    print(f"{'#' * 60}")

    per_seed_results: list[dict[str, dict]] = []
    for idx, seed in enumerate(chunk_seeds):
        print(f"\n--- Seed {seed} ({idx + 1}/{len(chunk_seeds)}) ---")
        results = run_single_seed(seed)
        per_seed_results.append({"seed": seed, "results": results})

    chunk_path = RESULTS_DIR / f"grid_cleaning_chunk_{chunk}.json"
    with open(chunk_path, "w") as f:
        json.dump(per_seed_results, f, indent=2, default=_serialize_numpy)
    print(f"\nChunk {chunk} done. Saved {len(chunk_seeds)} seeds to {chunk_path}")


def merge_chunks(num_chunks: int) -> None:
    """Load all chunk files, aggregate, run significance tests, export."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_seed_results: list[dict[str, dict]] = []
    missing_chunks = []

    for chunk in range(num_chunks):
        chunk_path = RESULTS_DIR / f"grid_cleaning_chunk_{chunk}.json"
        if not chunk_path.exists():
            missing_chunks.append(chunk)
            continue
        with open(chunk_path) as f:
            chunk_data = json.load(f)
        for entry in chunk_data:
            all_seed_results.append(entry["results"])

    if missing_chunks:
        print(f"WARNING: Missing chunks: {missing_chunks}")
        print(f"Proceeding with {len(all_seed_results)} seeds from available chunks.")
    else:
        print(f"All {num_chunks} chunks found. {len(all_seed_results)} total seeds.")

    if not all_seed_results:
        print("No results to merge!")
        return

    print(f"\n{'#' * 60}")
    print(f"AGGREGATED RESULTS ({len(all_seed_results)} seeds)")
    print(f"{'#' * 60}")

    aggregated = aggregate_grid_results(all_seed_results)

    print(f"\n{'Agent':<20} {'eta':>8} {'H(C)':>8} {'I_cd':>10} {'k':>5}")
    print("-" * 55)
    for agent, agg in sorted(aggregated.items(), key=lambda x: x[1]["i_cd_mean"], reverse=True):
        print(
            f"{agent:<20} "
            f"{agg['eta_mean']:>8.4f} "
            f"{agg['h_c_mean']:>8.4f} "
            f"{agg['i_cd_mean']:>10.6f} "
            f"{agg['k_mean']:>5.1f}"
        )

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

    json_path = RESULTS_DIR / "grid_cleaning_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "aggregated": aggregated,
                "significance": sig,
                "n_seeds": len(all_seed_results),
                "train_episodes": TRAIN_EPISODES,
                "eval_episodes": EVAL_EPISODES,
            },
            f,
            indent=2,
            default=_serialize_numpy,
        )
    print(f"\nSaved merged results to {json_path}")
    print(f"Saved pgfplots data to {DATA_DIR / 'icd_grid_cleaning.dat'}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Grid Cleaning I_cd experiment")
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="Chunk index (0-based). Use with --num-chunks.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=5,
        help="Total number of chunks (default: 5).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all chunk files and produce final results.",
    )
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

        aggregated = aggregate_grid_results(per_seed_results)

        print(f"\n{'Agent':<20} {'eta':>8} {'H(C)':>8} {'I_cd':>10} {'k':>5}")
        print("-" * 55)
        for agent, agg in sorted(
            aggregated.items(),
            key=lambda x: x[1]["i_cd_mean"],
            reverse=True,
        ):
            print(
                f"{agent:<20} "
                f"{agg['eta_mean']:>8.4f} "
                f"{agg['h_c_mean']:>8.4f} "
                f"{agg['i_cd_mean']:>10.6f} "
                f"{agg['k_mean']:>5.1f}"
            )

        sig = compute_pairwise_significance(aggregated)
        export_results(aggregated)

        json_path = RESULTS_DIR / "grid_cleaning_results.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "aggregated": aggregated,
                    "significance": sig,
                    "n_seeds": len(DEFAULT_SEEDS),
                    "train_episodes": TRAIN_EPISODES,
                    "eval_episodes": EVAL_EPISODES,
                },
                f,
                indent=2,
                default=_serialize_numpy,
            )
        print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
