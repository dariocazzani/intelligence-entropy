"""IPC 2023 Learning Track evaluation for I_cd.

Reads archival competition data (PDDL benchmarks + properties file)
and computes I_cd for all competing planners.

S(E) = unsatisfied goal predicates.  S(goal) = 0 by construction,
so eta is provably positive for every successful planner.
"""

from __future__ import annotations

import json
from pathlib import Path

from ie.environments.ipc_evaluation import (
    IPC_BENCHMARKS_DIR,
    IPC_PROPERTIES_PATH,
    compute_goal_entropies,
    evaluate_planners,
    load_properties,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "paper" / "data"


def save_per_domain(
    properties: dict[str, dict],
    goal_entropies: dict[tuple[str, str], int],
) -> None:
    """Save per-domain breakdown for bootstrap CI analysis."""
    from collections import defaultdict

    planner_domains: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(
            lambda: {"n_solved": 0, "total": 0, "sum_plan_length": 0.0, "sum_goal_preds": 0.0}
        )
    )

    for entry in properties.values():
        algo = entry["algorithm"]
        domain = entry["domain"]
        planner_domains[algo][domain]["total"] += 1

        if entry["coverage"] != 1:
            continue
        problem = entry["problem"]
        s_init = goal_entropies.get((domain, problem), 0)
        if s_init == 0 or entry["cost"] <= 0:
            continue

        planner_domains[algo][domain]["n_solved"] += 1
        planner_domains[algo][domain]["sum_plan_length"] += entry["cost"]
        planner_domains[algo][domain]["sum_goal_preds"] += s_init

    result = {}
    for planner, domains in sorted(planner_domains.items()):
        result[planner] = []
        for domain, stats in sorted(domains.items()):
            n = stats["n_solved"]
            result[planner].append(
                {
                    "domain": domain,
                    "n_solved": n,
                    "total": stats["total"],
                    "avg_plan_length": stats["sum_plan_length"] / n if n > 0 else 0.0,
                    "avg_goal_predicates": stats["sum_goal_preds"] / n if n > 0 else 0.0,
                }
            )

    out_path = Path(__file__).parent / "results" / "ipc_per_domain.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Per-domain data saved to {out_path}")


def export_results(results: dict[str, dict]) -> None:
    """Export results to .dat file for paper."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    header = "Planner Efficiency ContextDiversity Icd Solved Total Domains"
    lines = [header]

    for name, res in results.items():
        safe_name = name.replace(" ", "_").replace("-", "")
        lines.append(
            f"{safe_name} {res['efficiency']:.6f} {res['context_diversity']:.6f} "
            f"{res['i_cd']:.6f} {res['n_solved']} {res['n_total']} {res['n_domains']}"
        )

    outpath = DATA_DIR / "icd_ipc.dat"
    outpath.write_text("\n".join(lines))
    print(f"\nExported to {outpath}")


def main() -> None:
    """Run IPC evaluation."""
    if not IPC_BENCHMARKS_DIR.exists():
        print(f"ERROR: IPC benchmarks not found at {IPC_BENCHMARKS_DIR}")
        print(
            "Clone with: git clone https://github.com/"
            "ipc2023-learning/benchmarks.git data/ipc/benchmarks"
        )
        return

    if not IPC_PROPERTIES_PATH.exists():
        print(f"ERROR: IPC properties file not found at {IPC_PROPERTIES_PATH}")
        print(
            "Download from: https://raw.githubusercontent.com/ipc2023-learning/ipc2023-learning.github.io/main/properties"
        )
        return

    print("Computing S(E) from PDDL goal predicates...")
    goal_entropies = compute_goal_entropies()
    n_problems = len(goal_entropies)
    avg_s = sum(goal_entropies.values()) / n_problems if n_problems else 0
    print(f"  {n_problems} problems, avg S(E) = {avg_s:.1f} unsatisfied predicates")

    print("\nLoading competition results...")
    properties = load_properties()
    print(f"  {len(properties)} entries")

    print("\nComputing I_cd for each planner...")
    results = evaluate_planners(properties, goal_entropies)

    # Print results table
    print(
        f"\n{'Planner':<15} {'eta':>10} {'H(C)':>10} {'I_cd':>10} "
        f"{'Solved':>8} {'Total':>8} {'Domains':>8}"
    )
    print("-" * 75)
    for name, res in sorted(results.items(), key=lambda x: -x[1]["i_cd"]):
        print(
            f"{name:<15} {res['efficiency']:>10.4f} {res['context_diversity']:>10.4f} "
            f"{res['i_cd']:>10.4f} {res['n_solved']:>8} {res['n_total']:>8} "
            f"{res['n_domains']:>8}"
        )

    export_results(results)

    # Save full results as JSON
    results_path = Path(__file__).parent / "results" / "ipc_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {results_path}")

    # Save per-domain breakdown for bootstrap CI analysis
    save_per_domain(properties, goal_entropies)


if __name__ == "__main__":
    main()
