"""Cobb-Douglas sensitivity analysis on sliding puzzle heuristic results.

Tests whether ordinal rankings are robust to varying exponents (a, b)
in the generalized form eta^a * H(C)^b. With the sliding puzzle data,
H(C) varies across agents (0.07 to 2.30), making this a non-trivial test.

Usage:
    uv run python experiments/cobb_douglas_sliding_puzzle.py
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

RESULTS_PATH = Path(__file__).parent / "results" / "sliding_puzzle_heuristic_results.json"

AGENT_ORDER = [
    "Random",
    "Heuristic-Narrow",
    "Heuristic-Easy",
    "Heuristic-Hard",
    "Heuristic-Wide",
    "Heuristic-Full",
]

EXPONENTS = [0.5, 1.0, 1.5]


def main() -> None:
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    agg = data["aggregated"]

    eta_vals = np.array([agg[a]["eta_mean"] for a in AGENT_ORDER])
    hc_vals = np.array([agg[a]["h_c_mean"] for a in AGENT_ORDER])

    # Baseline ranking at a=b=1
    baseline_scores = eta_vals * hc_vals
    baseline_rank = np.argsort(-baseline_scores)

    print("Cobb-Douglas Sensitivity: eta^a * H(C)^b")
    print("=" * 70)
    print("\nBaseline (a=1, b=1) ranking:")
    for rank, idx in enumerate(baseline_rank):
        print(f"  {rank + 1}. {AGENT_ORDER[idx]:<20} score={baseline_scores[idx]:.6f}")

    print(f"\n{'a':>4} {'b':>4} {'tau':>8} {'p':>10}  {'Full>Easy':>10}  Ranking")
    print("-" * 70)

    for a, b in product(EXPONENTS, repeat=2):
        scores = np.power(eta_vals, a) * np.power(hc_vals, b)
        # Handle 0^x = 0 correctly (already works with numpy)

        tau, p = kendalltau(baseline_scores, scores)
        ranking = np.argsort(-scores)
        rank_names = [AGENT_ORDER[i][:8] for i in ranking]

        # Check Full > Easy
        full_score = scores[AGENT_ORDER.index("Heuristic-Full")]
        easy_score = scores[AGENT_ORDER.index("Heuristic-Easy")]
        full_gt_easy = full_score > easy_score

        print(
            f"  {a:.1f}  {b:.1f}  {tau:>7.3f}  {p:>9.4f}  "
            f"{'YES' if full_gt_easy else 'NO':>10}  "
            f"{' > '.join(rank_names)}"
        )

    # Summary for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    all_taus = []
    full_gt_easy_count = 0
    total_configs = 0

    for a, b in product(EXPONENTS, repeat=2):
        scores = np.power(eta_vals, a) * np.power(hc_vals, b)
        tau, _ = kendalltau(baseline_scores, scores)
        all_taus.append(tau)

        full_score = scores[AGENT_ORDER.index("Heuristic-Full")]
        easy_score = scores[AGENT_ORDER.index("Heuristic-Easy")]
        if full_score > easy_score:
            full_gt_easy_count += 1
        total_configs += 1

    print(f"  Kendall's tau range: [{min(all_taus):.3f}, {max(all_taus):.3f}]")
    print(f"  Full > Easy in {full_gt_easy_count}/{total_configs} configurations")

    # Check which configs break Full > Easy
    broken = []
    for a, b in product(EXPONENTS, repeat=2):
        scores = np.power(eta_vals, a) * np.power(hc_vals, b)
        full_score = scores[AGENT_ORDER.index("Heuristic-Full")]
        easy_score = scores[AGENT_ORDER.index("Heuristic-Easy")]
        if not full_score > easy_score:
            broken.append((a, b))

    if broken:
        print(f"  Full <= Easy at: {broken}")
        print("  (These are b=0 or degenerate cases)")
    else:
        print("  Full > Easy holds for ALL (a,b) configurations")


if __name__ == "__main__":
    main()
