"""Compare product form vs alternative combination functions for I_cd.

Computes rankings under:
  - Product: eta * H(C)  (current I_cd)
  - Weighted sum: alpha*eta + beta*H(C) for several (alpha, beta)
  - Threshold: require H(C) > h0, then rank by eta

Reports Kendall's tau between product and each alternative for IPC,
sliding puzzle, and ARC experiments.

Usage:
    uv run python experiments/combination_alternatives.py
"""

import json
from pathlib import Path

from scipy.stats import kendalltau

RESULTS_DIR = Path(__file__).parent / "results"


def load_results():
    """Load IPC, sliding puzzle, and ARC results."""
    with open(RESULTS_DIR / "ipc_results.json") as f:
        ipc = json.load(f)

    with open(RESULTS_DIR / "sliding_puzzle_heuristic_results.json") as f:
        sp_raw = json.load(f)
    sp = sp_raw["aggregated"]

    with open(RESULTS_DIR / "arc_results.json") as f:
        arc = json.load(f)

    return ipc, sp, arc


def extract_eta_hc(results, eta_key="efficiency", hc_key="context_diversity"):
    """Extract (name, eta, H(C)) triples from a results dict."""
    agents = []
    for name, metrics in results.items():
        eta = metrics[eta_key]
        hc = metrics[hc_key]
        agents.append((name, eta, hc))
    return agents


def extract_sliding_puzzle(sp):
    """Extract mean eta and H(C) from sliding puzzle aggregated results."""
    agents = []
    for name, metrics in sp.items():
        eta = metrics["eta_mean"]
        hc = metrics["h_c_mean"]
        agents.append((name, eta, hc))
    return agents


def ranking(agents, score_fn):
    """Return list of (name, score) sorted by score descending."""
    scored = [(name, score_fn(eta, hc)) for name, eta, hc in agents]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def rank_vector(scored):
    """Convert scored list to a rank vector (name -> rank)."""
    return {name: i for i, (name, _) in enumerate(scored)}


def kendall_tau(ranking_a, ranking_b):
    """Compute Kendall's tau between two rankings."""
    names = sorted(ranking_a.keys())
    ranks_a = [ranking_a[n] for n in names]
    ranks_b = [ranking_b[n] for n in names]
    tau, p = kendalltau(ranks_a, ranks_b)
    return tau, p


def product_fn(eta, hc):
    return eta * hc


def weighted_sum_fn(alpha, beta):
    def fn(eta, hc):
        return alpha * eta + beta * hc

    return fn


def threshold_fn(h0):
    """Rank by eta, but only if H(C) > h0; otherwise score = -inf."""

    def fn(eta, hc):
        if hc > h0:
            return eta
        return float("-inf")

    return fn


def cobb_douglas_fn(a, b):
    def fn(eta, hc):
        if eta <= 0 or hc <= 0:
            return 0.0
        return (eta**a) * (hc**b)

    return fn


def analyze_experiment(name, agents):
    """Run all alternative combinations and report Kendall's tau vs product."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    # Reference: product form
    ref_ranked = ranking(agents, product_fn)
    ref_ranks = rank_vector(ref_ranked)
    print("\nProduct form (eta * H(C)) ranking:")
    for i, (n, s) in enumerate(ref_ranked):
        print(f"  {i + 1}. {n:25s}  score={s:.4f}")

    alternatives = {
        "Weighted sum (1.0, 1.0)": weighted_sum_fn(1.0, 1.0),
        "Weighted sum (0.5, 1.0)": weighted_sum_fn(0.5, 1.0),
        "Weighted sum (1.0, 0.5)": weighted_sum_fn(1.0, 0.5),
        "Weighted sum (2.0, 1.0)": weighted_sum_fn(2.0, 1.0),
        "Weighted sum (1.0, 2.0)": weighted_sum_fn(1.0, 2.0),
        "Threshold H(C) > 0.5": threshold_fn(0.5),
        "Threshold H(C) > 1.0": threshold_fn(1.0),
        "Cobb-Douglas (0.5, 1.5)": cobb_douglas_fn(0.5, 1.5),
        "Cobb-Douglas (1.5, 0.5)": cobb_douglas_fn(1.5, 0.5),
    }

    print(f"\n{'Alternative':<30s}  {'tau':>6s}  {'p':>8s}  {'Ranking changes?'}")
    print("-" * 70)

    for alt_name, alt_fn in alternatives.items():
        alt_ranked = ranking(agents, alt_fn)
        alt_ranks = rank_vector(alt_ranked)
        tau, p = kendall_tau(ref_ranks, alt_ranks)
        changed = "YES" if tau < 1.0 else "no"
        print(f"  {alt_name:<28s}  {tau:>6.3f}  {p:>8.4f}  {changed}")


def main():
    ipc, sp, arc = load_results()

    ipc_agents = extract_eta_hc(ipc)
    sp_agents = extract_sliding_puzzle(sp)
    arc_agents = extract_eta_hc(arc)

    analyze_experiment("IPC 2023 Learning Track (8 planners)", ipc_agents)
    analyze_experiment("Sliding Puzzle Heuristic (6 agents)", sp_agents)
    analyze_experiment("ARC Evaluation (10 solvers)", arc_agents)

    print(f"\n{'=' * 60}")
    print("Summary: If tau = 1.0 for all alternatives, the product form")
    print("produces the same ranking as simpler alternatives. If tau < 1.0,")
    print("the product form makes substantively different predictions.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
