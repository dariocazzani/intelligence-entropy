"""ARC-AGI evaluation for I_cd demonstration.

Shows H(C) doing discriminatory work: solvers that handle more task types
get higher H(C) and thus different I_cd scores.
"""

from __future__ import annotations

import json
import os
import subprocess
import zlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ie.measure import context_diversity, efficiency, environmental_entropy

# ARC data directory
ARC_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "arc"


def load_arc_tasks(data_dir: Path | None = None, max_tasks: int | None = None) -> dict[str, dict]:
    """Load ARC training tasks from JSON files.

    Args:
        data_dir: Directory containing ARC task JSON files. Defaults to data/arc/training/
        max_tasks: Optional limit on number of tasks to load.

    Returns:
        Dict mapping task_id to task dict with 'train' and 'test' keys.
    """
    if data_dir is None:
        data_dir = ARC_DATA_DIR / "training"

    tasks = {}
    task_files = sorted(data_dir.glob("*.json"))
    if max_tasks is not None:
        task_files = task_files[:max_tasks]

    for f in task_files:
        task_id = f.stem
        with open(f) as fp:
            tasks[task_id] = json.load(fp)

    return tasks


def grid_entropy(grid: list[list[int]]) -> float:
    """Compute Shannon entropy over cell-color distribution of a grid.

    Args:
        grid: 2D list of integers (0-9 representing colors).

    Returns:
        Shannon entropy in nats over the color frequency distribution.
    """
    flat = [cell for row in grid for cell in row]
    counts = Counter(flat)
    count_array = np.array(list(counts.values()), dtype=float)
    return environmental_entropy(count_array)


def compression_entropy(grid: list[list[int]]) -> float:
    """Compute S(E) as lossless compressed length of the serialized grid.

    Estimates the empirical entropy rate via the source coding theorem:
    compressed length upper-bounds the Shannon entropy rate of the source.
    Captures spatial regularity that color-histogram entropy misses.

    Args:
        grid: 2D list of integers (0-9 representing colors).

    Returns:
        Compressed length in bytes (float). Returns 0.0 for empty grids.
    """
    flat = [cell for row in grid for cell in row]
    if not flat:
        return 0.0
    raw = bytes(flat)
    compressed = zlib.compress(raw, level=9)
    return float(len(compressed))


def goal_compression_entropy(grid: list[list[int]], goal: list[list[int]]) -> float:
    """Compute S(E) as compressed length of the element-wise diff from goal.

    Measures how far a grid is from the goal state using compression of
    the difference. When grid == goal, the diff is all zeros and compresses
    to near-minimum. When grid is far from goal, the diff retains spatial
    complexity and compresses poorly.

    If grids differ in dimensions, returns compression of the grid alone
    (maximally penalizing dimension mismatches).

    Args:
        grid: 2D list of integers (current/predicted state).
        goal: 2D list of integers (ground truth / goal state).

    Returns:
        Compressed length of the diff in bytes (float). Returns 0.0 for empty grids.
    """
    flat_g = [cell for row in grid for cell in row]
    flat_t = [cell for row in goal for cell in row]
    if not flat_g:
        return 0.0
    if len(flat_g) != len(flat_t):
        # Dimension mismatch: compress grid alone (high penalty)
        return float(len(zlib.compress(bytes(flat_g), level=9)))
    diff = [(g - t) % 256 for g, t in zip(flat_g, flat_t)]
    compressed = zlib.compress(bytes(diff), level=9)
    return float(len(compressed))


def grid_edit_distance(grid_a: list[list[int]], grid_b: list[list[int]]) -> int:
    """Count number of cells that differ between two grids.

    If grids have different dimensions, all cells count as different.

    Args:
        grid_a: First grid.
        grid_b: Second grid.

    Returns:
        Number of differing cells (used as work W).
    """
    rows_a, cols_a = len(grid_a), len(grid_a[0]) if grid_a else 0
    rows_b, cols_b = len(grid_b), len(grid_b[0]) if grid_b else 0

    if rows_a != rows_b or cols_a != cols_b:
        return rows_a * cols_a + rows_b * cols_b  # All cells differ

    diff = 0
    for r in range(rows_a):
        for c in range(cols_a):
            if grid_a[r][c] != grid_b[r][c]:
                diff += 1
    return diff


def task_matches_output(predicted: list[list[int]], expected: list[list[int]]) -> bool:
    """Check if predicted grid exactly matches expected output."""
    if len(predicted) != len(expected):
        return False
    return all(pred_row == exp_row for pred_row, exp_row in zip(predicted, expected))


# --- Solvers ---


class ArcSolver:
    """Base class for ARC solvers."""

    name: str = "base"

    def solve(self, task: dict) -> list[list[int]] | None:
        """Attempt to solve an ARC task.

        Args:
            task: ARC task dict with 'train' and 'test' keys.

        Returns:
            Predicted output grid for the first test example, or None if solver
            cannot handle this task.
        """
        raise NotImplementedError


class RandomSolver(ArcSolver):
    """Fill output grid with random colors."""

    name = "Random"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def solve(self, task: dict) -> list[list[int]] | None:
        """Generate random grid matching test input dimensions."""
        test_input = task["test"][0]["input"]
        rows, cols = len(test_input), len(test_input[0])
        grid = self.rng.integers(0, 10, size=(rows, cols))
        return [[int(cell) for cell in row] for row in grid]


class CopyInputSolver(ArcSolver):
    """Copy input grid to output (identity transform)."""

    name = "Copy-input"

    def solve(self, task: dict) -> list[list[int]] | None:
        """Return copy of test input."""
        return [row[:] for row in task["test"][0]["input"]]


class MostCommonColorSolver(ArcSolver):
    """Fill output with most common color from input."""

    name = "Most-common"

    def solve(self, task: dict) -> list[list[int]] | None:
        """Fill grid with most common color from test input."""
        test_input = task["test"][0]["input"]
        flat = [cell for row in test_input for cell in row]
        if not flat:
            return None
        most_common = Counter(flat).most_common(1)[0][0]
        rows, cols = len(test_input), len(test_input[0])
        return [[most_common] * cols for _ in range(rows)]


class ColorMappingSolver(ArcSolver):
    """Detect and apply simple color substitutions from training examples.

    Looks for tasks where the output is the input with some colors replaced.
    Only "solves" tasks where a consistent color mapping explains all training
    examples.
    """

    name = "Color-map"

    def solve(self, task: dict) -> list[list[int]] | None:
        """Apply learned color mapping to test input."""
        # Try to learn a color mapping from training examples
        mapping = self._learn_mapping(task["train"])
        if mapping is None:
            return None

        # Apply mapping to test input
        test_input = task["test"][0]["input"]
        result = []
        for row in test_input:
            new_row = [mapping.get(cell, cell) for cell in row]
            result.append(new_row)
        return result

    def _learn_mapping(self, examples: list[dict]) -> dict[int, int] | None:
        """Learn a consistent color mapping from training examples.

        Returns None if no consistent mapping exists or if grids have
        different dimensions.
        """
        mapping: dict[int, int] = {}

        for ex in examples:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out):
                return None
            for r in range(len(inp)):
                if len(inp[r]) != len(out[r]):
                    return None
                for c in range(len(inp[r])):
                    src, dst = inp[r][c], out[r][c]
                    if src in mapping:
                        if mapping[src] != dst:
                            return None  # Inconsistent mapping
                    else:
                        mapping[src] = dst

        # Verify the mapping is non-trivial (not all identity)
        if all(k == v for k, v in mapping.items()):
            return None

        return mapping


class GeometricSolver(ArcSolver):
    """Detect geometric transforms (rotation, reflection, transpose) from training examples.

    Tries all 7 non-identity elements of the dihedral group D4. If one transform
    consistently maps input to output across all training examples, applies it to
    the test input.
    """

    name = "Geometric"

    def solve(self, task: dict) -> list[list[int]] | None:
        """Find and apply a consistent geometric transform."""
        transforms = {
            "flip_lr": np.fliplr,
            "flip_ud": np.flipud,
            "rot90": lambda a: np.rot90(a, 1),
            "rot180": lambda a: np.rot90(a, 2),
            "rot270": lambda a: np.rot90(a, 3),
            "transpose": lambda a: a.T,
            "anti_transpose": lambda a: np.rot90(a, 2).T,
        }

        consistent: set[str] | None = None
        for ex in task["train"]:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])
            working = set()
            for name, fn in transforms.items():
                result = fn(inp)
                if result.shape == out.shape and np.array_equal(result, out):
                    working.add(name)
            consistent = working if consistent is None else consistent & working

        if not consistent:
            return None

        chosen = min(consistent)
        test_input = np.array(task["test"][0]["input"])
        predicted: list[list[int]] = transforms[chosen](test_input).tolist()
        return predicted


class TilingSolver(ArcSolver):
    """Detect when the output is the input tiled in a grid pattern.

    Checks whether the output dimensions are integer multiples of the input
    dimensions and the output equals np.tile(input, (rh, rw)). The tiling factor
    must be consistent across all training examples.
    """

    name = "Tiling"

    def solve(self, task: dict) -> list[list[int]] | None:
        """Find and apply a consistent tiling factor."""
        factor: tuple[int, int] | None = None

        for ex in task["train"]:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])

            in_h, in_w = inp.shape
            out_h, out_w = out.shape

            if in_h == 0 or in_w == 0:
                return None
            if out_h % in_h != 0 or out_w % in_w != 0:
                return None

            rh, rw = out_h // in_h, out_w // in_w
            if rh == 1 and rw == 1:
                return None  # Identity, not a tiling

            if not np.array_equal(np.tile(inp, (rh, rw)), out):
                return None

            if factor is None:
                factor = (rh, rw)
            elif factor != (rh, rw):
                return None  # Inconsistent tiling factors

        if factor is None:
            return None

        test_input = np.array(task["test"][0]["input"])
        tiled: list[list[int]] = np.tile(test_input, factor).tolist()
        return tiled


class BorderFillSolver(ArcSolver):
    """Detect border addition or removal patterns.

    Strategy 1 (addition): output is the input surrounded by a uniform-color
    border of consistent width.
    Strategy 2 (removal): output is the inner region of the input after
    stripping a uniform-color border.
    """

    name = "Border-fill"

    def solve(self, task: dict) -> list[list[int]] | None:
        """Try border addition, then border removal."""
        result = self._try_border_add(task)
        if result is not None:
            return result
        return self._try_border_remove(task)

    def _try_border_add(self, task: dict) -> list[list[int]] | None:
        """Detect output = input with uniform border added."""
        params: tuple[int, int, int] | None = None

        for ex in task["train"]:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])

            in_h, in_w = inp.shape
            out_h, out_w = out.shape

            dh, dw = out_h - in_h, out_w - in_w
            if dh <= 0 or dw <= 0 or dh % 2 != 0 or dw % 2 != 0:
                return None

            bh, bw = dh // 2, dw // 2
            inner = out[bh : bh + in_h, bw : bw + in_w]
            if not np.array_equal(inner, inp):
                return None

            border_color = int(out[0, 0])
            border_mask = np.ones(out.shape, dtype=bool)
            border_mask[bh : bh + in_h, bw : bw + in_w] = False
            if not np.all(out[border_mask] == border_color):
                return None

            cur = (bh, bw, border_color)
            if params is None:
                params = cur
            elif params != cur:
                return None

        if params is None:
            return None

        bh, bw, color = params
        test_input = np.array(task["test"][0]["input"])
        th, tw = test_input.shape
        arr = np.full((th + 2 * bh, tw + 2 * bw), color, dtype=int)
        arr[bh : bh + th, bw : bw + tw] = test_input
        bordered: list[list[int]] = arr.tolist()
        return bordered

    def _try_border_remove(self, task: dict) -> list[list[int]] | None:
        """Detect output = inner region of input after stripping uniform border."""
        params: tuple[int, int] | None = None

        for ex in task["train"]:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])

            in_h, in_w = inp.shape
            out_h, out_w = out.shape

            dh, dw = in_h - out_h, in_w - out_w
            if dh <= 0 or dw <= 0 or dh % 2 != 0 or dw % 2 != 0:
                return None

            bh, bw = dh // 2, dw // 2
            inner = inp[bh : bh + out_h, bw : bw + out_w]
            if not np.array_equal(inner, out):
                return None

            border_color = int(inp[0, 0])
            border_mask = np.ones(inp.shape, dtype=bool)
            border_mask[bh : bh + out_h, bw : bw + out_w] = False
            if not np.all(inp[border_mask] == border_color):
                return None

            cur = (bh, bw)
            if params is None:
                params = cur
            elif params != cur:
                return None

        if params is None:
            return None

        bh, bw = params
        test_input = np.array(task["test"][0]["input"])
        th, tw = test_input.shape
        stripped: list[list[int]] = test_input[bh : th - bh, bw : tw - bw].tolist()
        return stripped


class LLMSolver(ArcSolver):
    """Solve ARC tasks using an LLM via API.

    Calls a Claude model to predict ARC test outputs from training examples.
    Caches all responses to disk to avoid repeated API calls on reruns.

    Requires environment variables:
        LLM_BASE_URL: API endpoint (e.g., https://api.anthropic.com/v1)
        LLM_API_KEY: API authentication key
    """

    def __init__(
        self,
        model_id: str,
        name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.model_id = model_id
        self.name = name
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.cache_dir = cache_dir or (
            Path(__file__).parent.parent.parent.parent / ".cache" / "arc_llm"
        )
        self.max_tokens = max_tokens

    def _cache_path(self, task_id: str) -> Path:
        """Get cache file path for a task+model combination."""
        safe_model = self.model_id.replace("/", "_").replace(":", "_")
        cache_subdir = self.cache_dir / safe_model
        cache_subdir.mkdir(parents=True, exist_ok=True)
        return cache_subdir / f"{task_id}.json"

    def _load_cache(self, task_id: str) -> list[list[int]] | None:
        """Load cached prediction for a task, or None if not cached."""
        path = self._cache_path(task_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        grid = data.get("grid")
        if grid is None:
            return None
        return list(grid)

    def _save_cache(self, task_id: str, grid: list[list[int]] | None, raw: str) -> None:
        """Save prediction and raw response to cache."""
        path = self._cache_path(task_id)
        with open(path, "w") as f:
            json.dump({"grid": grid, "raw_response": raw}, f)

    @staticmethod
    def _format_grid(grid: list[list[int]]) -> str:
        """Format a grid as space-separated rows."""
        return "\n".join(" ".join(str(c) for c in row) for row in grid)

    def _build_prompt(self, task: dict) -> str:
        """Build the prompt for the LLM from training examples."""
        parts = [
            "You are solving an ARC (Abstraction and Reasoning Corpus) puzzle.",
            "Study the input-output examples below, find the pattern, "
            "and predict the output for the test input.",
            "Reply with ONLY the output grid, one row per line, "
            "cells separated by spaces. No explanation.",
            "",
        ]
        for i, ex in enumerate(task["train"], 1):
            parts.append(f"Example {i} input:")
            parts.append(self._format_grid(ex["input"]))
            parts.append(f"Example {i} output:")
            parts.append(self._format_grid(ex["output"]))
            parts.append("")

        parts.append("Test input:")
        parts.append(self._format_grid(task["test"][0]["input"]))
        parts.append("")
        parts.append("Test output:")
        return "\n".join(parts)

    @staticmethod
    def _parse_grid(text: str) -> list[list[int]] | None:
        """Parse LLM response text into a grid of integers."""
        lines = text.strip().split("\n")
        grid = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip lines that look like explanatory text
            if any(c.isalpha() for c in line):
                continue
            # Try to parse as space-separated integers
            try:
                row = [int(x) for x in line.split()]
                if row:
                    grid.append(row)
            except ValueError:
                continue
        if not grid:
            return None
        # Verify all rows have the same length
        row_len = len(grid[0])
        if not all(len(r) == row_len for r in grid):
            return None
        return grid

    def _call_api(self, prompt: str) -> str:
        """Call the LLM API via curl and return the response text."""
        url = f"{self.base_url}/messages"

        payload = json.dumps(
            {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        )

        result = subprocess.run(
            [
                "curl",
                "-s",
                "--max-time",
                "60",
                "-X",
                "POST",
                url,
                "-H",
                "Content-Type: application/json",
                "-H",
                f"x-api-key: {self.api_key}",
                "-H",
                "anthropic-version: 2023-06-01",
                "-d",
                payload,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=90,
        )

        data = json.loads(result.stdout)
        return str(data["content"][0]["text"])

    def solve(self, task: dict) -> list[list[int]] | None:
        """Solve an ARC task using the LLM, with caching."""
        # Derive a stable task_id from the task content
        task_id = _task_content_hash(task)

        # Check cache first
        cached = self._load_cache(task_id)
        if cached is not None:
            return cached

        self._task_count = getattr(self, "_task_count", 0) + 1
        if self._task_count % 20 == 1:
            print(f"  [{self.name}] calling API for task {self._task_count}...")

        # Build prompt and call API
        prompt = self._build_prompt(task)
        try:
            raw_response = self._call_api(prompt)
        except Exception as e:
            print(f"  [{self.name}] API error on {task_id[:8]}: {e}")
            self._save_cache(task_id, None, str(e))
            return None

        # Parse response
        grid = self._parse_grid(raw_response)
        self._save_cache(task_id, grid, raw_response)
        return grid


def _task_content_hash(task: dict) -> str:
    """Generate a stable hash for an ARC task based on its content."""
    import hashlib

    content = json.dumps(task, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# LLM solver configurations for three capability tiers
LLM_SOLVER_CONFIGS = [
    {
        "model_id": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "name": "LLM-Haiku-3.5",
    },
    {
        "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "name": "LLM-Haiku-4.5",
    },
    {
        "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "name": "LLM-Sonnet-4",
    },
]


DEFAULT_SOLVERS = [
    RandomSolver,
    CopyInputSolver,
    MostCommonColorSolver,
    ColorMappingSolver,
    GeometricSolver,
    TilingSolver,
    BorderFillSolver,
]


def evaluate_solver(
    solver: ArcSolver,
    tasks: dict[str, dict],
    entropy_fn: Callable[[list[list[int]]], float] | None = None,
) -> dict:
    """Evaluate a single solver across all ARC tasks, computing I_cd.

    Each task where the solver produces a non-None output counts as a context
    the solver "operates in". H(C) is computed over a categorical labeling of
    tasks: we assign context labels based on task characteristics.

    Args:
        solver: ARC solver instance.
        tasks: Dict mapping task_id to task dict.

    Returns:
        Dict with:
            'i_cd': float
            'efficiency': float (η = -ΔS/W)
            'context_diversity': float (H(C))
            'n_attempted': int — tasks where solver returned non-None
            'n_correct': int — tasks with exact match
            'task_results': list of per-task result dicts
    """
    if entropy_fn is None:
        entropy_fn = grid_entropy

    total_delta_s = 0.0
    total_work = 0.0
    context_labels = []
    task_results = []
    n_correct = 0

    for task_id, task in tasks.items():
        predicted = solver.solve(task)
        if predicted is None:
            continue  # Solver declined this task

        test_input = task["test"][0]["input"]
        test_output = task["test"][0]["output"]

        # Entropy of input (initial state) and predicted output (final state)
        s_initial = entropy_fn(test_input)
        s_final = entropy_fn(predicted)
        delta_s = s_final - s_initial

        # Work = number of cells edited from input to predicted output
        work = grid_edit_distance(test_input, predicted)

        # Context label = task_id hash to integer (unique per task)
        context_label = hash(task_id) % (2**31)
        context_labels.append(context_label)

        total_delta_s += delta_s
        total_work += work

        correct = task_matches_output(predicted, test_output)
        if correct:
            n_correct += 1

        task_results.append(
            {
                "task_id": task_id,
                "delta_s": delta_s,
                "work": work,
                "correct": correct,
                "context_label": context_label,
            }
        )

    # Compute I_cd
    eff = efficiency(total_delta_s, total_work) if total_work > 0 else 0.0
    div = context_diversity(context_labels) if context_labels else 0.0
    score = eff * div

    return {
        "i_cd": score,
        "efficiency": eff,
        "context_diversity": div,
        "n_attempted": len(context_labels),
        "n_correct": n_correct,
        "task_results": task_results,
    }


def evaluate_all_solvers(
    tasks: dict[str, dict],
    solvers: list[ArcSolver] | None = None,
) -> dict[str, dict]:
    """Evaluate multiple solvers and return comparative results.

    Args:
        tasks: Dict mapping task_id to task dict.
        solvers: List of solver instances. Defaults to one of each DEFAULT_SOLVERS.

    Returns:
        Dict mapping solver name to evaluation result dict.
    """
    if solvers is None:
        solvers = [cls() for cls in DEFAULT_SOLVERS]

    results = {}
    for solver in solvers:
        results[solver.name] = evaluate_solver(solver, tasks)

    return results
