"""Test sensitivity of ARC rankings to compression algorithm choice.

Re-computes S(E) using different compressors (zlib levels, bz2, lzma)
and reports whether ordinal rankings change.

Usage:
    uv run python experiments/arc_compressor_sensitivity.py
"""

import bz2
import json
import lzma
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

RESULTS_DIR = Path(__file__).parent / "results"
ARC_DATA_DIR = Path(__file__).parent.parent / "arc_data"


def compress_diff(diff_array, method="zlib", level=9):
    """Compress the element-wise difference array and return compressed length."""
    raw = diff_array.tobytes()

    if method == "zlib":
        return len(zlib.compress(raw, level))
    elif method == "bz2":
        return len(bz2.compress(raw, level))
    elif method == "lzma":
        return len(lzma.compress(raw))
    else:
        raise ValueError(f"Unknown compression method: {method}")


def load_arc_predictions():
    """Load ARC predictions and ground truths.

    Returns list of (solver_name, task_id, prediction, ground_truth).
    """
    pred_file = RESULTS_DIR / "arc_predictions.json"
    if pred_file.exists():
        with open(pred_file) as f:
            return json.load(f)

    print("ARC predictions not found.")
    print(f"Expected: {pred_file}")
    print("Please run run_arc_compression.py with --save-predictions first,")
    print("or create arc_predictions.json with structure:")
    print('  [{"solver": str, "task_id": str,')
    print('    "prediction": [[int]], "ground_truth": [[int]]}]')
    return None


def compute_rankings(predictions, compress_fn):
    """Compute eta, H(C), I_cd for each solver under a given compression."""
    import math
    from collections import defaultdict

    solver_stats = defaultdict(
        lambda: {
            "total_delta_s": 0.0,
            "total_w": 0,
            "task_counts": defaultdict(int),
        }
    )

    for entry in predictions:
        solver = entry["solver"]
        pred = np.array(entry["prediction"], dtype=np.uint8)
        gt = np.array(entry["ground_truth"], dtype=np.uint8)

        if pred.shape != gt.shape:
            continue

        diff = (pred != gt).astype(np.uint8) * (np.abs(pred.astype(int) - gt.astype(int))).astype(
            np.uint8
        )

        s_pred = compress_fn(diff)
        # S(E) for perfect match
        zeros = np.zeros_like(diff)
        s_goal = compress_fn(zeros)

        delta_s = s_goal - s_pred  # negative = entropy reduction
        w = int(np.sum(pred != gt))

        if w > 0:
            stats = solver_stats[solver]
            stats["total_delta_s"] += delta_s
            stats["total_w"] += w
            stats["task_counts"][entry["task_id"]] += 1

    # Compute eta, H(C), I_cd per solver
    results = {}
    for solver, stats in solver_stats.items():
        if stats["total_w"] == 0:
            results[solver] = (0.0, 0.0, 0.0)
            continue

        eta = -stats["total_delta_s"] / stats["total_w"]

        total = sum(stats["task_counts"].values())
        hc = 0.0
        for count in stats["task_counts"].values():
            p = count / total
            if p > 0:
                hc -= p * math.log(p)

        results[solver] = (eta, hc, eta * hc)

    return results


def main():
    predictions = load_arc_predictions()
    if predictions is None:
        print("\nAlternative: manually check by running run_arc_compression.py")
        print("with different compressor settings and comparing Table 5 output.")
        return

    compressors = {
        "zlib-1": lambda d: compress_diff(d, "zlib", 1),
        "zlib-6": lambda d: compress_diff(d, "zlib", 6),
        "zlib-9 (default)": lambda d: compress_diff(d, "zlib", 9),
        "bz2-9": lambda d: compress_diff(d, "bz2", 9),
        "lzma": lambda d: compress_diff(d, "lzma"),
    }

    ref_name = "zlib-9 (default)"
    ref_results = compute_rankings(predictions, compressors[ref_name])
    ref_order = sorted(ref_results.keys(), key=lambda s: ref_results[s][2], reverse=True)
    ref_ranks = {s: i for i, s in enumerate(ref_order)}

    print("ARC Compressor Sensitivity Analysis")
    print(f"Reference: {ref_name}")
    print(f"\n{'Compressor':<25s}  {'Kendall tau':>11s}  {'p-value':>8s}  {'Ranking changes?'}")
    print("-" * 65)

    for comp_name, comp_fn in compressors.items():
        if comp_name == ref_name:
            print(f"  {comp_name:<23s}  {'1.000':>11s}  {'---':>8s}  (reference)")
            continue

        alt_results = compute_rankings(predictions, comp_fn)
        alt_order = sorted(alt_results.keys(), key=lambda s: alt_results[s][2], reverse=True)
        alt_ranks = {s: i for i, s in enumerate(alt_order)}

        common = sorted(set(ref_ranks.keys()) & set(alt_ranks.keys()))
        r1 = [ref_ranks[s] for s in common]
        r2 = [alt_ranks[s] for s in common]
        tau, p = kendalltau(r1, r2)

        changed = "YES" if tau < 1.0 else "no"
        print(f"  {comp_name:<23s}  {tau:>11.3f}  {p:>8.4f}  {changed}")


if __name__ == "__main__":
    main()
