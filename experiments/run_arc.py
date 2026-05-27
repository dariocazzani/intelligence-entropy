"""ARC evaluation experiment for I_cd."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ie.environments.arc_evaluation import (
    ARC_DATA_DIR,
    DEFAULT_SOLVERS,
    LLM_SOLVER_CONFIGS,
    LLMSolver,
    compression_entropy,
    evaluate_solver,
    load_arc_tasks,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "paper" / "data"


def download_arc_data() -> None:
    """Download ARC training data if not present."""
    training_dir = ARC_DATA_DIR / "training"
    if training_dir.exists() and any(training_dir.glob("*.json")):
        print(f"ARC data already exists at {training_dir}")
        return

    print("Downloading ARC-AGI training data...")
    training_dir.mkdir(parents=True, exist_ok=True)

    # Clone just the data directory using git sparse checkout
    tmp_dir = PROJECT_ROOT / "data" / "_arc_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/fchollet/ARC-AGI.git",
                str(tmp_dir / "repo"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "data/training"],
            cwd=str(tmp_dir / "repo"),
            check=True,
            capture_output=True,
            text=True,
        )

        # Copy training data
        src = tmp_dir / "repo" / "data" / "training"
        if src.exists():
            for f in src.glob("*.json"):
                shutil.copy2(f, training_dir / f.name)
            print(f"Downloaded {len(list(training_dir.glob('*.json')))} ARC training tasks")
        else:
            raise FileNotFoundError(f"Expected training data at {src}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def export_results(results: dict[str, dict]) -> None:
    """Export ARC results to .dat file for paper."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    header = "Solver Efficiency ContextDiversity Icd Attempted Correct"
    lines = [header]

    for name, res in results.items():
        safe_name = name.replace(" ", "_").replace("-", "")
        lines.append(
            f"{safe_name} {res['efficiency']:.6f} {res['context_diversity']:.6f} "
            f"{res['i_cd']:.6f} {res['n_attempted']} {res['n_correct']}"
        )

    outpath = DATA_DIR / "icd_arc.dat"
    outpath.write_text("\n".join(lines))
    print(f"\nExported to {outpath}")


def main() -> None:
    """Run ARC evaluation experiment."""
    download_arc_data()

    tasks = load_arc_tasks()
    print(f"Loaded {len(tasks)} ARC tasks")

    # Rule-based solvers
    solvers = [cls() for cls in DEFAULT_SOLVERS]

    # LLM solvers (3 capability tiers)
    for cfg in LLM_SOLVER_CONFIGS:
        solvers.append(LLMSolver(model_id=cfg["model_id"], name=cfg["name"]))

    results = {}
    for solver in solvers:
        print(f"\nEvaluating {solver.name} ({len(tasks)} tasks)...")
        results[solver.name] = evaluate_solver(solver, tasks, entropy_fn=compression_entropy)
        res = results[solver.name]
        print(
            f"  Done: {res['n_attempted']} attempted, {res['n_correct']} correct, "
            f"I_cd={res['i_cd']:.6f}"
        )

    # Print results table
    print(f"\n{'Solver':<15} {'η':>10} {'H(C)':>10} {'I_cd':>12} {'Attempted':>10} {'Correct':>8}")
    print("-" * 70)
    for name, res in results.items():
        print(
            f"{name:<15} {res['efficiency']:>10.6f} {res['context_diversity']:>10.4f} "
            f"{res['i_cd']:>12.6f} {res['n_attempted']:>10} {res['n_correct']:>8}"
        )

    export_results(results)

    # Also save full results as JSON
    results_path = Path(__file__).parent / "results" / "arc_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove task_results for JSON (too large)
    json_results = {}
    for name, res in results.items():
        json_results[name] = {k: v for k, v in res.items() if k != "task_results"}

    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Full results saved to {results_path}")


if __name__ == "__main__":
    main()
