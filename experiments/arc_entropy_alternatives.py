"""Explore alternative S(E) operationalizations for ARC grids.

Tests whether compression-based, co-occurrence, or diff-based entropy
measures produce positive η for capable LLM solvers (where cell-color
entropy produces negative η).

Uses cached LLM predictions — no API calls needed.
"""

from __future__ import annotations

import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from ie.environments.arc_evaluation import (
    grid_edit_distance,
    grid_entropy,
    load_arc_tasks,
    task_matches_output,
)
from ie.measure import environmental_entropy

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "arc_llm"

# LLM model IDs to analyze
MODELS = {
    "Haiku-3.5": "us.anthropic.claude-3-5-haiku-20241022-v1_0",
    "Haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1_0",
    "Sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1_0",
}


# --- Alternative S(E) measures ---


def serialize_grid(grid: list[list[int]]) -> bytes:
    """Serialize a grid to bytes (row-by-row, single byte per cell)."""
    return bytes(cell for row in grid for cell in row)


def compression_complexity(grid: list[list[int]]) -> float:
    """S(E) via zlib compressed size (Kolmogorov complexity approximation)."""
    raw = serialize_grid(grid)
    if len(raw) == 0:
        return 0.0
    compressed = zlib.compress(raw, level=9)
    return float(len(compressed))


def cooccurrence_entropy(grid: list[list[int]]) -> float:
    """S(E) via Shannon entropy over adjacent cell-pair distribution.

    Counts all horizontal and vertical adjacent pairs, then computes
    Shannon entropy over the pair frequency distribution. Captures
    spatial structure that color-histogram entropy misses.
    """
    pairs: list[tuple[int, int]] = []
    rows, cols = len(grid), len(grid[0]) if grid else 0

    for r in range(rows):
        for c in range(cols):
            # Horizontal neighbor
            if c + 1 < cols:
                pairs.append((grid[r][c], grid[r][c + 1]))
            # Vertical neighbor
            if r + 1 < rows:
                pairs.append((grid[r][c], grid[r + 1][c]))

    if not pairs:
        return 0.0

    counts = Counter(pairs)
    count_array = np.array(list(counts.values()), dtype=float)
    return environmental_entropy(count_array)


def diff_compression(grid_a: list[list[int]], grid_b: list[list[int]]) -> float:
    """Compression complexity of the element-wise diff between two grids.

    Captures transformation complexity: a simple, consistent rule produces
    a highly compressible diff. An arbitrary transformation produces an
    incompressible diff.

    If grids differ in dimension, returns compression of grid_b alone.
    """
    rows_a, cols_a = len(grid_a), len(grid_a[0]) if grid_a else 0
    rows_b, cols_b = len(grid_b), len(grid_b[0]) if grid_b else 0

    if rows_a != rows_b or cols_a != cols_b:
        # Dimension mismatch: diff is undefined, use output complexity
        return compression_complexity(grid_b)

    # Element-wise diff (mod 10 to stay in 0-9 range)
    diff = []
    for r in range(rows_a):
        for c in range(cols_a):
            diff.append((grid_b[r][c] - grid_a[r][c]) % 10)

    raw = bytes(diff)
    if len(raw) == 0:
        return 0.0
    compressed = zlib.compress(raw, level=9)
    return float(len(compressed))


# --- Analysis ---


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


def task_content_hash(task: dict) -> str:
    """Generate the same hash used by LLMSolver for cache lookups."""
    import hashlib

    content = json.dumps(task, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def analyze_model(
    model_name: str,
    model_dir: str,
    tasks: dict[str, dict],
) -> dict:
    """Analyze a model's cached predictions with multiple S(E) measures."""
    predictions = load_cached_predictions(model_dir)

    # Build hash→task_id mapping
    hash_to_id = {}
    for task_id, task in tasks.items():
        h = task_content_hash(task)
        hash_to_id[h] = task_id

    results = {
        "color": {"correct": [], "incorrect": [], "total_ds": 0.0, "total_w": 0.0},
        "compression": {"correct": [], "incorrect": [], "total_ds": 0.0, "total_w": 0.0},
        "cooccurrence": {"correct": [], "incorrect": [], "total_ds": 0.0, "total_w": 0.0},
        "diff_compress": {"correct": [], "incorrect": [], "total_ds": 0.0, "total_w": 0.0},
    }

    n_attempted = 0
    n_correct = 0
    n_skipped = 0

    for task_hash, predicted in predictions.items():
        if predicted is None:
            n_skipped += 1
            continue

        task_id = hash_to_id.get(task_hash)
        if task_id is None:
            n_skipped += 1
            continue

        task = tasks[task_id]
        test_input = task["test"][0]["input"]
        test_output = task["test"][0]["output"]
        correct = task_matches_output(predicted, test_output)

        n_attempted += 1
        if correct:
            n_correct += 1

        work = grid_edit_distance(test_input, predicted)
        if work == 0:
            continue  # No edit, skip

        bucket = "correct" if correct else "incorrect"

        # 1. Color entropy (current, broken)
        ds_color = grid_entropy(predicted) - grid_entropy(test_input)
        results["color"][bucket].append(ds_color)
        results["color"]["total_ds"] += ds_color
        results["color"]["total_w"] += work

        # 2. Compression complexity
        ds_compress = compression_complexity(predicted) - compression_complexity(test_input)
        results["compression"][bucket].append(ds_compress)
        results["compression"]["total_ds"] += ds_compress
        results["compression"]["total_w"] += work

        # 3. Co-occurrence entropy
        ds_coocc = cooccurrence_entropy(predicted) - cooccurrence_entropy(test_input)
        results["cooccurrence"][bucket].append(ds_coocc)
        results["cooccurrence"]["total_ds"] += ds_coocc
        results["cooccurrence"]["total_w"] += work

        # 4. Diff compression (transformation complexity)
        # Compare: diff(input, predicted) vs diff(input, ground_truth)
        diff_pred = diff_compression(test_input, predicted)
        diff_truth = diff_compression(test_input, test_output)
        ds_diff = diff_pred - diff_truth  # Positive if prediction is more complex transform
        results["diff_compress"][bucket].append(ds_diff)
        results["diff_compress"]["total_ds"] += ds_diff
        results["diff_compress"]["total_w"] += work

    return {
        "model": model_name,
        "n_attempted": n_attempted,
        "n_correct": n_correct,
        "n_skipped": n_skipped,
        "measures": results,
    }


def print_results(analysis: dict) -> None:
    """Print analysis results for a model."""
    m = analysis["model"]
    n = analysis["n_attempted"]
    nc = analysis["n_correct"]
    acc = nc / n * 100 if n > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  {m}: {nc}/{n} correct ({acc:.1f}%)")
    print(f"{'=' * 70}")

    for measure_name, data in analysis["measures"].items():
        total_ds = data["total_ds"]
        total_w = data["total_w"]
        eta = -total_ds / total_w if total_w > 0 else 0.0

        correct_ds = data["correct"]
        incorrect_ds = data["incorrect"]

        avg_correct = np.mean(correct_ds) if correct_ds else float("nan")
        avg_incorrect = np.mean(incorrect_ds) if incorrect_ds else float("nan")

        sign = "+" if eta > 0 else ""

        print(f"\n  {measure_name}:")
        print(f"    Overall eta:        {sign}{eta:.6f}")
        print(f"    Avg dS (correct):   {avg_correct:+.4f}  (n={len(correct_ds)})")
        print(f"    Avg dS (incorrect): {avg_incorrect:+.4f}  (n={len(incorrect_ds)})")


def main() -> None:
    """Run the analysis."""
    print("Loading ARC tasks...")
    tasks = load_arc_tasks()
    print(f"Loaded {len(tasks)} tasks")

    for model_name, model_dir in MODELS.items():
        cache_path = CACHE_DIR / model_dir
        if not cache_path.exists():
            print(f"\nSkipping {model_name}: no cache at {cache_path}")
            continue

        print(f"\nAnalyzing {model_name}...")
        analysis = analyze_model(model_name, model_dir, tasks)
        print_results(analysis)

    # Also analyze rule-based solvers for comparison
    print(f"\n{'=' * 70}")
    print("  Rule-based solver comparison (Geometric, 100% accuracy)")
    print(f"{'=' * 70}")

    from ie.environments.arc_evaluation import GeometricSolver

    geo = GeometricSolver()
    geo_correct_compress = []
    geo_correct_coocc = []

    for task_id, task in tasks.items():
        predicted = geo.solve(task)
        if predicted is None:
            continue

        test_input = task["test"][0]["input"]
        test_output = task["test"][0]["output"]

        if not task_matches_output(predicted, test_output):
            continue

        ds_compress = compression_complexity(predicted) - compression_complexity(test_input)
        ds_coocc = cooccurrence_entropy(predicted) - cooccurrence_entropy(test_input)
        ds_color = grid_entropy(predicted) - grid_entropy(test_input)

        geo_correct_compress.append(ds_compress)
        geo_correct_coocc.append(ds_coocc)

        print(
            f"  Task {task_id[:8]}: color_dS={ds_color:+.4f}  "
            f"compress_dS={ds_compress:+.1f}  coocc_dS={ds_coocc:+.4f}"
        )

    if geo_correct_compress:
        print(f"\n  Geometric avg compress_dS: {np.mean(geo_correct_compress):+.2f}")
        print(f"  Geometric avg coocc_dS:    {np.mean(geo_correct_coocc):+.4f}")


if __name__ == "__main__":
    main()
