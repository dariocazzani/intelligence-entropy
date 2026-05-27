"""Bootstrap confidence intervals for IPC planner rankings.

Resamples over domains (10 domains) to compute 95% CIs for eta, H(C),
and I_cd for each planner.

Usage:
    uv run python experiments/ipc_bootstrap.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
IPC_DATA_DIR = Path(__file__).parent.parent / "ipc_data"
N_BOOTSTRAP = 10000
SEED = 42


def load_ipc_per_domain():
    """Load per-domain IPC results.

    Returns dict: planner -> list of (domain, n_solved, total, avg_plan_length).
    """
    results_file = RESULTS_DIR / "ipc_per_domain.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    # If per-domain data doesn't exist, try to reconstruct from raw data
    print("Per-domain results not found. Looking for raw IPC data...")
    print(f"Expected: {results_file}")
    print("Please run run_ipc.py with --save-per-domain first,")
    print("or create ipc_per_domain.json with structure:")
    print('  {"PlannerName": [{"domain": str, "n_solved": int, "total": int,')
    print('    "avg_plan_length": float, "avg_goal_predicates": float}, ...]}')
    return None


def compute_eta_hc_icd(per_domain_data, domain_indices=None):
    """Compute eta, H(C), I_cd for a planner given per-domain data.

    If domain_indices is provided, only use those domains (bootstrap resample).
    """
    if domain_indices is not None:
        data = [per_domain_data[i] for i in domain_indices]
    else:
        data = per_domain_data

    total_solved = 0
    total_problems = 0
    total_delta_s = 0.0
    total_w = 0.0
    domain_counts = defaultdict(int)

    for d in data:
        n_solved = d["n_solved"]
        total = d["total"]
        domain = d["domain"]

        if n_solved > 0:
            # S(E) = unsatisfied goal predicates; S(goal) = 0 for solved
            total_delta_s += n_solved * d["avg_goal_predicates"]
            total_w += n_solved * d["avg_plan_length"]
            domain_counts[domain] += n_solved

        total_solved += n_solved
        total_problems += total

    if total_w == 0:
        return 0.0, 0.0, 0.0

    eta = total_delta_s / total_w

    # H(C) over domain distribution
    total_count = sum(domain_counts.values())
    if total_count == 0:
        return eta, 0.0, 0.0

    hc = 0.0
    for count in domain_counts.values():
        p = count / total_count
        if p > 0:
            hc -= p * math.log(p)

    return eta, hc, eta * hc


def bootstrap_planner(per_domain_data, n_domains, rng):
    """One bootstrap resample: resample domain indices with replacement."""
    indices = rng.integers(0, n_domains, size=n_domains)
    return compute_eta_hc_icd(per_domain_data, indices)


def main():
    per_domain = load_ipc_per_domain()
    if per_domain is None:
        return

    rng = np.random.default_rng(SEED)

    print(f"Bootstrap CIs for IPC planners ({N_BOOTSTRAP} resamples)")
    print(
        f"{'Planner':<20s}  {'eta':>8s}  {'95% CI':>16s}  "
        f"{'H(C)':>8s}  {'95% CI':>16s}  "
        f"{'I_cd':>8s}  {'95% CI':>16s}"
    )
    print("-" * 100)

    for planner_name, domains in per_domain.items():
        n_domains = len(domains)

        etas = []
        hcs = []
        icds = []

        for _ in range(N_BOOTSTRAP):
            eta, hc, icd = bootstrap_planner(domains, n_domains, rng)
            etas.append(eta)
            hcs.append(hc)
            icds.append(icd)

        etas = np.array(etas)
        hcs = np.array(hcs)
        icds = np.array(icds)

        eta_ci = np.percentile(etas, [2.5, 97.5])
        hc_ci = np.percentile(hcs, [2.5, 97.5])
        icd_ci = np.percentile(icds, [2.5, 97.5])

        print(
            f"{planner_name:<20s}  "
            f"{np.mean(etas):>8.3f}  [{eta_ci[0]:.3f}, {eta_ci[1]:.3f}]  "
            f"{np.mean(hcs):>8.3f}  [{hc_ci[0]:.3f}, {hc_ci[1]:.3f}]  "
            f"{np.mean(icds):>8.3f}  [{icd_ci[0]:.3f}, {icd_ci[1]:.3f}]"
        )


if __name__ == "__main__":
    main()
