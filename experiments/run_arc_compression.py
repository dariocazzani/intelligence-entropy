"""Recompute ARC results with goal-relative compression S(E).

S(E) = compressed length of diff(grid, ground_truth). This parallels
CoupledReg's S(E) over deviations from target: the ground truth output
IS the environment's goal state.

Rule-based solvers are evaluated directly (fast).
LLM solvers use cached predictions only (no API calls).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ie.environments.arc_evaluation import (
    DEFAULT_SOLVERS,
    goal_compression_entropy,
    grid_edit_distance,
    load_arc_tasks,
    task_matches_output,
)
from ie.measure import context_diversity, efficiency

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "arc_llm"
DATA_DIR = PROJECT_ROOT / "paper" / "data"

# Cache dirs use underscores; Bedrock model IDs use colons
LLM_MODELS = {
    "LLM-Haiku-3.5": "us.anthropic.claude-3-5-haiku-20241022-v1_0",
    "LLM-Haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1_0",
    "LLM-Sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1_0",
}


def task_content_hash(task: dict) -> str:
    """Generate the same hash used by LLMSolver for cache lookups."""
    content = json.dumps(task, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_cached_predictions(model_dir: str) -> dict[str, list[list[int]] | None]:
    """Load all cached LLM predictions for a model."""
    cache_path = CACHE_DIR / model_dir
    predictions = {}
    for f in cache_path.glob("*.json"):
        task_hash = f.stem
        with open(f) as fp:
            data = json.load(fp)
        predictions[task_hash] = data.get("grid")
    return predictions


def evaluate_predictions(
    predictions: dict[str, list[list[int]] | None],
    tasks: dict[str, dict],
    task_id_lookup: dict[str, str] | None = None,
) -> dict:
    """Evaluate predictions using goal-relative compression S(E).

    S(E) = compression(diff(grid, ground_truth)). Parallels CoupledReg
    where S(E) measures deviation from the target state.

    Args:
        predictions: Maps task identifier -> predicted grid (or None).
        tasks: Maps task_id -> task dict.
        task_id_lookup: Optional hash->task_id mapping for cached predictions.
            If None, predictions keys are assumed to be task_ids directly.
    """
    total_delta_s = 0.0
    total_work = 0.0
    context_labels = []
    n_correct = 0
    n_attempted = 0

    for pred_key, predicted in predictions.items():
        if predicted is None:
            continue

        # Resolve task_id
        if task_id_lookup is not None:
            task_id = task_id_lookup.get(pred_key)
            if task_id is None:
                continue
        else:
            task_id = pred_key

        if task_id not in tasks:
            continue

        task = tasks[task_id]
        test_input = task["test"][0]["input"]
        test_output = task["test"][0]["output"]

        n_attempted += 1
        correct = task_matches_output(predicted, test_output)
        if correct:
            n_correct += 1

        # Goal-relative compression: S(E) = compression(diff(grid, goal))
        s_initial = goal_compression_entropy(test_input, test_output)
        s_final = goal_compression_entropy(predicted, test_output)
        delta_s = s_final - s_initial

        work = grid_edit_distance(test_input, predicted)
        context_label = hash(task_id) % (2**31)
        context_labels.append(context_label)

        if work == 0:
            continue  # No edits, skip entropy accumulation

        total_delta_s += delta_s
        total_work += work

    eff = efficiency(total_delta_s, total_work) if total_work > 0 else 0.0
    div = context_diversity(context_labels) if context_labels else 0.0
    score = eff * div

    return {
        "i_cd": score,
        "efficiency": eff,
        "context_diversity": div,
        "n_attempted": n_attempted,
        "n_correct": n_correct,
    }


def save_predictions(predictions: list[dict]) -> None:
    """Save per-prediction data for compressor sensitivity analysis."""
    out_path = Path(__file__).parent / "results" / "arc_predictions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f)


def export_results(results: dict[str, dict]) -> None:
    """Export results to .dat file for paper."""
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
    """Run ARC evaluation with goal-relative compression S(E)."""
    print("Loading ARC tasks...")
    tasks = load_arc_tasks()
    print(f"Loaded {len(tasks)} tasks")

    results: dict[str, dict] = {}
    all_predictions: list[dict] = []

    # Rule-based solvers: evaluate directly
    for solver_cls in DEFAULT_SOLVERS:
        solver = solver_cls()
        print(f"\nEvaluating {solver.name} ({len(tasks)} tasks)...")

        # Generate predictions for all tasks
        predictions: dict[str, list[list[int]] | None] = {}
        for task_id, task in tasks.items():
            pred = solver.solve(task)
            predictions[task_id] = pred
            if pred is not None and task_id in tasks:
                gt = task["test"][0]["output"]
                all_predictions.append(
                    {
                        "solver": solver.name,
                        "task_id": task_id,
                        "prediction": pred,
                        "ground_truth": gt,
                    }
                )

        results[solver.name] = evaluate_predictions(predictions, tasks)
        res = results[solver.name]
        acc = res["n_correct"] / res["n_attempted"] * 100 if res["n_attempted"] > 0 else 0
        print(
            f"  Done: {res['n_attempted']} attempted, {res['n_correct']} correct ({acc:.1f}%), "
            f"eta={res['efficiency']:.6f}, H(C)={res['context_diversity']:.4f}, "
            f"I_cd={res['i_cd']:.6f}"
        )

    # LLM solvers: evaluate from cache only
    # Build hash -> task_id mapping once
    hash_to_id = {}
    for task_id, task in tasks.items():
        h = task_content_hash(task)
        hash_to_id[h] = task_id

    for model_name, model_dir in LLM_MODELS.items():
        cache_path = CACHE_DIR / model_dir
        if not cache_path.exists():
            print(f"\nSkipping {model_name}: no cache at {cache_path}")
            continue

        print(f"\nEvaluating {model_name} (from cache)...")
        predictions = load_cached_predictions(model_dir)

        for pred_hash, pred_grid in predictions.items():
            if pred_grid is None:
                continue
            tid = hash_to_id.get(pred_hash)
            if tid and tid in tasks:
                gt = tasks[tid]["test"][0]["output"]
                all_predictions.append(
                    {
                        "solver": model_name,
                        "task_id": tid,
                        "prediction": pred_grid,
                        "ground_truth": gt,
                    }
                )

        results[model_name] = evaluate_predictions(predictions, tasks, task_id_lookup=hash_to_id)
        res = results[model_name]
        acc = res["n_correct"] / res["n_attempted"] * 100 if res["n_attempted"] > 0 else 0
        print(
            f"  Done: {res['n_attempted']} attempted, {res['n_correct']} correct ({acc:.1f}%), "
            f"eta={res['efficiency']:.6f}, H(C)={res['context_diversity']:.4f}, "
            f"I_cd={res['i_cd']:.6f}"
        )

    # Print results table
    print(
        f"\n{'Solver':<15} {'eta':>12} {'H(C)':>10} {'I_cd':>12} "
        f"{'Attempted':>10} {'Correct':>8} {'Acc%':>8}"
    )
    print("-" * 80)
    for name, res in results.items():
        acc = res["n_correct"] / res["n_attempted"] * 100 if res["n_attempted"] > 0 else 0
        print(
            f"{name:<15} {res['efficiency']:>12.6f} {res['context_diversity']:>10.4f} "
            f"{res['i_cd']:>12.6f} {res['n_attempted']:>10} {res['n_correct']:>8} "
            f"{acc:>7.1f}%"
        )

    export_results(results)

    # Save full results as JSON
    results_path = Path(__file__).parent / "results" / "arc_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to {results_path}")

    # Save per-prediction data for compressor sensitivity analysis
    save_predictions(all_predictions)
    print(f"Per-prediction data saved ({len(all_predictions)} entries)")


if __name__ == "__main__":
    main()
