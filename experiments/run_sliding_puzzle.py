"""Sliding Puzzle experiment: 6 agents x 10 contexts x 50 seeds.

This experiment jointly tests both factors of I_cd = eta x H(C).
PPO converges on small puzzles (w=2,3) and degrades on larger ones (w=4,5),
creating natural variation in BOTH eta and the effective context set.

Agent types:
  1. Random: uniform random actions
  2. Heuristic: greedy Manhattan distance reduction
  3. Generalist PPO: trained on ALL 10 contexts
  4. Specialist PPO (Easy): trained on w=2,3 contexts ONLY (ids 0-3)
  5. Specialist PPO (Hard): trained on w=4,5 contexts ONLY (ids 4-9)
  6. Generalist PPO (Long): same as #3 but 2x training budget

Reward: -delta_entropy (dense for sliding puzzles since every valid move
changes tile positions). This is a standard task reward, NOT the I_cd measure.

Performance-gated H(C): for each agent, H(C) is computed only over
contexts where eta > 0 (agent achieves net entropy reduction).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from ie.agents.heuristic_agent import SlidingPuzzleHeuristicAgent
from ie.agents.random_agent import RandomAgent
from ie.agents.sb3_wrapper import BaseEnvGymnasiumWrapper, SB3Agent
from ie.analysis.evaluate import compute_pairwise_significance
from ie.environments.sliding_puzzle import CONFIGS, SlidingPuzzle
from ie.measure import efficiency

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent.parent / "paper" / "data"

DEFAULT_SEEDS = list(range(42, 92))  # 50 seeds
EVAL_EPISODES = 10
TRAIN_TIMESTEPS = 500_000
TRAIN_TIMESTEPS_LONG = 1_000_000

# Context subsets
ALL_CONTEXTS = CONFIGS  # 10 contexts
EASY_CONTEXTS = [c for c in CONFIGS if c["w"] in (2, 3)]  # ids 0-3
HARD_CONTEXTS = [c for c in CONFIGS if c["w"] in (4, 5)]  # ids 4-9


def _entropy_reward(
    _obs: np.ndarray,
    _action: int,
    _obs_next: np.ndarray,
    delta_entropy: float,
) -> float:
    """Reward = negative entropy change. Dense for sliding puzzles."""
    return -delta_entropy


def train_sb3_ppo(
    env: SlidingPuzzle,
    train_configs: list[dict],
    seed: int,
    total_timesteps: int = TRAIN_TIMESTEPS,
    label: str = "",
) -> SB3Agent:
    """Train SB3 PPO on sliding puzzle contexts."""
    gym_env = BaseEnvGymnasiumWrapper(
        env=env,
        configs=train_configs,
        reward_fn=_entropy_reward,
        seed=seed,
    )

    model = PPO(
        "MlpPolicy",
        gym_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        seed=seed,
        policy_kwargs={"net_arch": [64, 64]},
    )
    model.learn(total_timesteps=total_timesteps)
    if label:
        print(f"    [{label}] Training complete ({total_timesteps} steps)")
    return SB3Agent(model)


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

    # Ungated values for comparison
    all_etas = list(eta_per_context.values())
    eta_all = float(np.mean(all_etas))
    h_c_all = float(np.log(10))
    i_cd_all = eta_all * h_c_all

    return {
        "eta_per_context": eta_per_context,
        "effective_contexts": effective,
        "n_effective": k,
        "eta_avg": eta_avg,
        "h_c": h_c,
        "i_cd": i_cd,
        "eta_all": eta_all,
        "h_c_all": h_c_all,
        "i_cd_all": i_cd_all,
    }


def run_single_seed(seed: int) -> dict[str, dict]:
    """Run the full sliding puzzle experiment for one seed."""
    print(f"\n{'=' * 60}")
    print(f"Sliding Puzzle: seed={seed}")
    print(f"{'=' * 60}")

    env = SlidingPuzzle(seed=seed)

    # 1. Random agent
    print("  Building Random agent...")
    random_agent = RandomAgent(action_space_size=env.action_space_size, seed=seed)

    # 2. Heuristic agent
    print("  Building Heuristic agent...")
    heuristic = SlidingPuzzleHeuristicAgent()

    # 3. Generalist PPO (all 10 contexts)
    print("  Training Generalist PPO...")
    generalist = train_sb3_ppo(env, ALL_CONTEXTS, seed, TRAIN_TIMESTEPS, label="Generalist")

    # 4. Specialist PPO (Easy: w=2,3 only)
    print("  Training Specialist PPO (Easy)...")
    specialist_easy = train_sb3_ppo(
        env, EASY_CONTEXTS, seed + 1, TRAIN_TIMESTEPS, label="Spec-Easy"
    )

    # 5. Specialist PPO (Hard: w=4,5 only)
    print("  Training Specialist PPO (Hard)...")
    specialist_hard = train_sb3_ppo(
        env, HARD_CONTEXTS, seed + 2, TRAIN_TIMESTEPS, label="Spec-Hard"
    )

    # 6. Generalist PPO (Long: 2x training)
    print("  Training Generalist PPO (Long)...")
    generalist_long = train_sb3_ppo(
        env, ALL_CONTEXTS, seed + 3, TRAIN_TIMESTEPS_LONG, label="Gen-Long"
    )

    agents = {
        "Random": random_agent,
        "Heuristic": heuristic,
        "Generalist": generalist,
        "Specialist (Easy)": specialist_easy,
        "Specialist (Hard)": specialist_hard,
        "Generalist (Long)": generalist_long,
    }

    # Evaluate all agents with performance-gated H(C)
    print("  Evaluating all agents...")
    results = {}
    for name, agent in agents.items():
        res = evaluate_with_performance_gating(agent, env, ALL_CONTEXTS, EVAL_EPISODES)
        results[name] = res
        print(
            f"    {name}: eta_avg={res['eta_avg']:.4f}, "
            f"H(C)={res['h_c']:.4f} (k={res['n_effective']}), "
            f"I_cd={res['i_cd']:.6f}"
        )

    return results


def aggregate_results(
    per_seed_results: list[dict[str, dict]],
) -> dict[str, dict]:
    """Aggregate performance-gated results across seeds."""
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

    output_file = DATA_DIR / "icd_sliding_puzzle.dat"
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

    per_seed_results: list[dict] = []
    for idx, seed in enumerate(chunk_seeds):
        print(f"\n--- Seed {seed} ({idx + 1}/{len(chunk_seeds)}) ---")
        results = run_single_seed(seed)
        per_seed_results.append({"seed": seed, "results": results})

    chunk_path = RESULTS_DIR / f"sliding_puzzle_chunk_{chunk}.json"
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
        chunk_path = RESULTS_DIR / f"sliding_puzzle_chunk_{chunk}.json"
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

    aggregated = aggregate_results(all_seed_results)

    print(f"\n{'Agent':<22} {'eta':>8} {'H(C)':>8} {'I_cd':>10} {'k':>5}")
    print("-" * 57)
    for agent, agg in sorted(aggregated.items(), key=lambda x: x[1]["i_cd_mean"], reverse=True):
        print(
            f"{agent:<22} "
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

    json_path = RESULTS_DIR / "sliding_puzzle_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "aggregated": aggregated,
                "significance": sig,
                "n_seeds": len(all_seed_results),
                "train_timesteps": TRAIN_TIMESTEPS,
                "eval_episodes": EVAL_EPISODES,
            },
            f,
            indent=2,
            default=_serialize_numpy,
        )
    print(f"\nSaved merged results to {json_path}")
    print(f"Saved pgfplots data to {DATA_DIR / 'icd_sliding_puzzle.dat'}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sliding Puzzle I_cd experiment")
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

        aggregated = aggregate_results(per_seed_results)

        print(f"\n{'Agent':<22} {'eta':>8} {'H(C)':>8} {'I_cd':>10} {'k':>5}")
        print("-" * 57)
        for agent, agg in sorted(
            aggregated.items(),
            key=lambda x: x[1]["i_cd_mean"],
            reverse=True,
        ):
            print(
                f"{agent:<22} "
                f"{agg['eta_mean']:>8.4f} "
                f"{agg['h_c_mean']:>8.4f} "
                f"{agg['i_cd_mean']:>10.6f} "
                f"{agg['k_mean']:>5.1f}"
            )

        sig = compute_pairwise_significance(aggregated)
        export_results(aggregated)

        json_path = RESULTS_DIR / "sliding_puzzle_results.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "aggregated": aggregated,
                    "significance": sig,
                    "n_seeds": len(DEFAULT_SEEDS),
                    "train_timesteps": TRAIN_TIMESTEPS,
                    "eval_episodes": EVAL_EPISODES,
                },
                f,
                indent=2,
                default=_serialize_numpy,
            )
        print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
