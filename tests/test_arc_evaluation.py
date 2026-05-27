"""Tests for ARC evaluation module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from ie.environments.arc_evaluation import (
    BorderFillSolver,
    ColorMappingSolver,
    CopyInputSolver,
    GeometricSolver,
    LLMSolver,
    MostCommonColorSolver,
    RandomSolver,
    TilingSolver,
    compression_entropy,
    evaluate_all_solvers,
    evaluate_solver,
    goal_compression_entropy,
    grid_edit_distance,
    grid_entropy,
    task_matches_output,
)

# --- Test helpers ---


def make_task(
    train_pairs: list[tuple[list[list[int]], list[list[int]]]],
    test_input: list[list[int]],
    test_output: list[list[int]],
) -> dict:
    """Create an ARC task dict from grids."""
    return {
        "train": [{"input": inp, "output": out} for inp, out in train_pairs],
        "test": [{"input": test_input, "output": test_output}],
    }


# --- grid_entropy ---


class TestGridEntropy:
    """Tests for grid_entropy function."""

    def test_uniform_grid(self) -> None:
        """Test entropy of grid with uniform color distribution."""
        grid = [[0, 1], [2, 3]]  # 4 distinct colors, 1 each
        h = grid_entropy(grid)
        assert abs(h - np.log(4)) < 1e-6

    def test_single_color_grid(self) -> None:
        """Test entropy of uniform grid (all same color)."""
        grid = [[5, 5], [5, 5]]
        assert grid_entropy(grid) == 0.0

    def test_empty_grid(self) -> None:
        """Test entropy of empty grid."""
        assert grid_entropy([]) == 0.0

    def test_two_colors(self) -> None:
        """Test entropy of grid with two colors equally distributed."""
        grid = [[0, 0], [1, 1]]  # 2 colors, 2 each -> ln(2)
        h = grid_entropy(grid)
        assert abs(h - np.log(2)) < 1e-6


# --- compression_entropy ---


class TestCompressionEntropy:
    """Tests for compression_entropy function."""

    def test_uniform_grid_low_entropy(self) -> None:
        """Uniform grid compresses well, complex grid does not."""
        uniform = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        complex_grid = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        assert compression_entropy(complex_grid) > compression_entropy(uniform)

    def test_empty_grid(self) -> None:
        """Empty grid returns 0."""
        assert compression_entropy([]) == 0.0

    def test_single_cell(self) -> None:
        """Single cell grid returns positive value."""
        assert compression_entropy([[5]]) > 0.0

    def test_spatial_sensitivity(self) -> None:
        """Compression distinguishes grids with same color counts but different layout."""
        # Checkerboard: spatially complex
        checker = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]
        # Half-and-half: spatially simple (runs of same value compress well)
        half = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]
        # Both have identical color histograms (8 zeros, 8 ones)
        h_checker = compression_entropy(checker)
        h_half = compression_entropy(half)
        assert h_checker != h_half

    def test_larger_uniform_lower_than_random(self) -> None:
        """Larger uniform grid still compresses better than random-looking grid."""
        uniform = [[0] * 10 for _ in range(10)]
        diverse = [[(r * 10 + c) % 10 for c in range(10)] for r in range(10)]
        assert compression_entropy(uniform) < compression_entropy(diverse)


# --- goal_compression_entropy ---


class TestGoalCompressionEntropy:
    """Tests for goal_compression_entropy function."""

    def test_perfect_match_minimal_entropy(self) -> None:
        """Grid matching the goal should have minimal compressed diff."""
        goal = [[1, 2, 3], [4, 5, 6]]
        # diff is all zeros -> compresses very well
        assert goal_compression_entropy(goal, goal) < goal_compression_entropy(
            [[9, 9, 9], [9, 9, 9]], goal
        )

    def test_far_from_goal_high_entropy(self) -> None:
        """Grid far from goal should have higher compressed diff than one close."""
        # Use larger grids so zlib header doesn't dominate
        goal = [[0] * 20 for _ in range(20)]
        close = [[0] * 20 for _ in range(20)]
        close[0][0] = 1  # 1 cell differs
        far = [[(r * 20 + c) % 10 for c in range(20)] for r in range(20)]
        assert goal_compression_entropy(close, goal) < goal_compression_entropy(far, goal)

    def test_empty_grid(self) -> None:
        """Empty grid returns 0."""
        assert goal_compression_entropy([], []) == 0.0

    def test_single_cell_match(self) -> None:
        """Single cell matching goal."""
        s = goal_compression_entropy([[5]], [[5]])
        assert s > 0.0  # zlib header still produces bytes

    def test_single_cell_mismatch(self) -> None:
        """Single cell not matching goal has higher entropy than match."""
        s_match = goal_compression_entropy([[5]], [[5]])
        s_miss = goal_compression_entropy([[3]], [[5]])
        assert s_miss >= s_match

    def test_different_dimensions_fallback(self) -> None:
        """Mismatched dimensions fall back to raw compression of grid."""
        grid = [[1, 2, 3]]
        goal = [[1], [2], [3]]
        s = goal_compression_entropy(grid, goal)
        assert s > 0.0

    def test_varied_diff_above_zero_diff(self) -> None:
        """Varied diff compresses worse than zero diff (perfect match)."""
        goal = [[0] * 20 for _ in range(20)]
        # Grid with varied values -> diff has diverse bytes -> harder to compress
        varied = [[(r * 20 + c) % 10 for c in range(20)] for r in range(20)]
        s_varied = goal_compression_entropy(varied, goal)
        s_self = goal_compression_entropy(goal, goal)
        assert s_varied > s_self


# --- grid_edit_distance ---


class TestGridEditDistance:
    """Tests for grid_edit_distance function."""

    def test_identical_grids(self) -> None:
        """Test edit distance between identical grids."""
        grid = [[1, 2], [3, 4]]
        assert grid_edit_distance(grid, grid) == 0

    def test_all_different(self) -> None:
        """Test edit distance when all cells differ."""
        a = [[0, 0], [0, 0]]
        b = [[1, 1], [1, 1]]
        assert grid_edit_distance(a, b) == 4

    def test_some_different(self) -> None:
        """Test edit distance when some cells differ."""
        a = [[0, 1], [2, 3]]
        b = [[0, 1], [2, 4]]  # 1 cell differs
        assert grid_edit_distance(a, b) == 1

    def test_different_dimensions(self) -> None:
        """Test edit distance when grids have different dimensions."""
        a = [[1, 2]]
        b = [[1], [2]]
        dist = grid_edit_distance(a, b)
        assert dist == 2 + 2  # all cells in both grids

    def test_empty_grids(self) -> None:
        """Test edit distance with empty grids."""
        assert grid_edit_distance([], []) == 0

    def test_one_empty(self) -> None:
        """Test edit distance when one grid is empty."""
        a = [[1, 2]]
        b: list[list[int]] = []
        assert grid_edit_distance(a, b) == 2


# --- task_matches_output ---


class TestTaskMatches:
    """Tests for task_matches_output function."""

    def test_match(self) -> None:
        """Test exact match."""
        assert task_matches_output([[1, 2]], [[1, 2]])

    def test_no_match(self) -> None:
        """Test no match when values differ."""
        assert not task_matches_output([[1, 2]], [[1, 3]])

    def test_different_sizes(self) -> None:
        """Test no match when sizes differ."""
        assert not task_matches_output([[1]], [[1, 2]])

    def test_empty_grids(self) -> None:
        """Test match with empty grids."""
        assert task_matches_output([], [])

    def test_different_rows(self) -> None:
        """Test no match when number of rows differs."""
        assert not task_matches_output([[1, 2]], [[1, 2], [3, 4]])


# --- Solvers ---


class TestRandomSolver:
    """Tests for RandomSolver."""

    def test_returns_correct_shape(self) -> None:
        """Test that RandomSolver returns correct shape."""
        task = make_task(
            [([[1, 2], [3, 4]], [[4, 3], [2, 1]])],
            [[5, 6], [7, 8]],
            [[8, 7], [6, 5]],
        )
        solver = RandomSolver(seed=42)
        result = solver.solve(task)
        assert result is not None
        assert len(result) == 2
        assert len(result[0]) == 2

    def test_deterministic_with_seed(self) -> None:
        """Test that RandomSolver is deterministic with same seed."""
        task = make_task(
            [([[1]], [[2]])],
            [[3]],
            [[4]],
        )
        r1 = RandomSolver(seed=42).solve(task)
        r2 = RandomSolver(seed=42).solve(task)
        assert r1 == r2

    def test_colors_in_range(self) -> None:
        """Test that RandomSolver generates colors in valid range (0-9)."""
        task = make_task(
            [([[0]], [[1]])],
            [[0, 0], [0, 0]],
            [[1, 1], [1, 1]],
        )
        solver = RandomSolver(seed=42)
        result = solver.solve(task)
        assert result is not None
        for row in result:
            for cell in row:
                assert 0 <= cell < 10


class TestCopyInputSolver:
    """Tests for CopyInputSolver."""

    def test_copies_input(self) -> None:
        """Test that CopyInputSolver copies the input."""
        task = make_task(
            [([[1, 2]], [[3, 4]])],
            [[5, 6]],
            [[7, 8]],
        )
        solver = CopyInputSolver()
        result = solver.solve(task)
        assert result == [[5, 6]]

    def test_creates_new_list(self) -> None:
        """Test that CopyInputSolver creates a new list (not reference)."""
        task = make_task(
            [([[1]], [[2]])],
            [[3]],
            [[4]],
        )
        solver = CopyInputSolver()
        result = solver.solve(task)
        assert result is not None
        # Modify result and check original is unchanged
        result[0][0] = 999
        assert task["test"][0]["input"][0][0] == 3


class TestMostCommonColorSolver:
    """Tests for MostCommonColorSolver."""

    def test_fills_with_most_common(self) -> None:
        """Test that MostCommonColorSolver fills with most common color."""
        task = make_task(
            [([[0]], [[1]])],
            [[0, 0], [0, 1]],  # 0 appears 3 times, 1 appears 1 time
            [[0, 0], [0, 0]],
        )
        solver = MostCommonColorSolver()
        result = solver.solve(task)
        assert result == [[0, 0], [0, 0]]

    def test_empty_input(self) -> None:
        """Test that MostCommonColorSolver handles empty input."""
        task = make_task(
            [([[0]], [[1]])],
            [[]],
            [[1]],
        )
        solver = MostCommonColorSolver()
        result = solver.solve(task)
        assert result is None

    def test_single_cell(self) -> None:
        """Test with single cell input."""
        task = make_task(
            [([[0]], [[1]])],
            [[5]],
            [[1]],
        )
        solver = MostCommonColorSolver()
        result = solver.solve(task)
        assert result == [[5]]


class TestColorMappingSolver:
    """Tests for ColorMappingSolver."""

    def test_learns_simple_mapping(self) -> None:
        """Test that ColorMappingSolver learns simple color mapping."""
        # Training: 1->2, 0->0
        task = make_task(
            [([[0, 1], [1, 0]], [[0, 2], [2, 0]])],
            [[1, 0]],
            [[2, 0]],
        )
        solver = ColorMappingSolver()
        result = solver.solve(task)
        assert result == [[2, 0]]

    def test_returns_none_for_inconsistent_mapping(self) -> None:
        """Test that ColorMappingSolver returns None for inconsistent mapping."""
        task = make_task(
            [
                ([[1]], [[2]]),
                ([[1]], [[3]]),  # Contradicts: 1->2 vs 1->3
            ],
            [[1]],
            [[2]],
        )
        solver = ColorMappingSolver()
        assert solver.solve(task) is None

    def test_returns_none_for_identity_mapping(self) -> None:
        """Test that ColorMappingSolver returns None for identity mapping."""
        task = make_task(
            [([[1, 2]], [[1, 2]])],  # Identity mapping
            [[3, 4]],
            [[3, 4]],
        )
        solver = ColorMappingSolver()
        assert solver.solve(task) is None

    def test_returns_none_for_different_dimensions(self) -> None:
        """Test that ColorMappingSolver returns None for different dimensions."""
        task = make_task(
            [([[1, 2]], [[1], [2]])],  # Different dimensions
            [[3, 4]],
            [[3], [4]],
        )
        solver = ColorMappingSolver()
        assert solver.solve(task) is None

    def test_multi_color_mapping(self) -> None:
        """Test learning mapping with multiple colors."""
        task = make_task(
            [([[0, 1, 2]], [[3, 4, 5]])],
            [[0, 1, 2]],
            [[3, 4, 5]],
        )
        solver = ColorMappingSolver()
        result = solver.solve(task)
        assert result == [[3, 4, 5]]

    def test_consistent_across_examples(self) -> None:
        """Test that mapping is consistent across multiple training examples."""
        task = make_task(
            [
                ([[0, 1]], [[2, 3]]),
                ([[1, 0]], [[3, 2]]),  # Same mapping
            ],
            [[0, 1]],
            [[2, 3]],
        )
        solver = ColorMappingSolver()
        result = solver.solve(task)
        assert result == [[2, 3]]


# --- GeometricSolver ---


class TestGeometricSolver:
    """Tests for GeometricSolver."""

    def test_detects_rot90(self) -> None:
        """Test detection of 90-degree rotation."""
        # rot90 of [[1,2],[3,4]] = [[2,4],[1,3]]
        task = make_task(
            [([[1, 2], [3, 4]], [[2, 4], [1, 3]])],
            [[5, 6], [7, 8]],
            [[6, 8], [5, 7]],
        )
        solver = GeometricSolver()
        result = solver.solve(task)
        assert result == [[6, 8], [5, 7]]

    def test_detects_flip_lr(self) -> None:
        """Test detection of left-right flip."""
        task = make_task(
            [([[1, 2, 3]], [[3, 2, 1]])],
            [[4, 5, 6]],
            [[6, 5, 4]],
        )
        solver = GeometricSolver()
        result = solver.solve(task)
        assert result == [[6, 5, 4]]

    def test_detects_flip_ud(self) -> None:
        """Test detection of up-down flip."""
        task = make_task(
            [([[1, 2], [3, 4]], [[3, 4], [1, 2]])],
            [[5, 6], [7, 8]],
            [[7, 8], [5, 6]],
        )
        solver = GeometricSolver()
        result = solver.solve(task)
        assert result == [[7, 8], [5, 6]]

    def test_consistent_across_examples(self) -> None:
        """Test that transform must be consistent across all training examples."""
        task = make_task(
            [
                ([[1, 2], [3, 4]], [[2, 4], [1, 3]]),  # rot90
                ([[1, 2], [3, 4]], [[3, 4], [1, 2]]),  # flip_ud (inconsistent!)
            ],
            [[5, 6], [7, 8]],
            [[0, 0], [0, 0]],
        )
        solver = GeometricSolver()
        assert solver.solve(task) is None

    def test_returns_none_for_non_geometric(self) -> None:
        """Test that non-geometric tasks return None."""
        task = make_task(
            [([[1, 2], [3, 4]], [[9, 9], [9, 9]])],
            [[5, 6], [7, 8]],
            [[0, 0], [0, 0]],
        )
        solver = GeometricSolver()
        assert solver.solve(task) is None

    def test_non_square_rotation(self) -> None:
        """Test rotation of non-square grid (changes dimensions)."""
        # rot90 of [[1,2,3],[4,5,6]] (2x3) = [[3,6],[2,5],[1,4]] (3x2)
        task = make_task(
            [([[1, 2, 3], [4, 5, 6]], [[3, 6], [2, 5], [1, 4]])],
            [[7, 8, 9], [0, 1, 2]],
            [[9, 2], [8, 1], [7, 0]],
        )
        solver = GeometricSolver()
        result = solver.solve(task)
        assert result == [[9, 2], [8, 1], [7, 0]]


# --- TilingSolver ---


class TestTilingSolver:
    """Tests for TilingSolver."""

    def test_detects_2x2_tiling(self) -> None:
        """Test detection of 2x2 tiling."""
        inp = [[1, 2], [3, 4]]
        out = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]
        task = make_task([(inp, out)], [[5, 6], [7, 8]], [[0]])
        solver = TilingSolver()
        result = solver.solve(task)
        assert result == [[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]]

    def test_detects_1x3_horizontal_tiling(self) -> None:
        """Test detection of horizontal-only tiling."""
        inp = [[1, 2]]
        out = [[1, 2, 1, 2, 1, 2]]
        task = make_task([(inp, out)], [[3, 4]], [[0]])
        solver = TilingSolver()
        result = solver.solve(task)
        assert result == [[3, 4, 3, 4, 3, 4]]

    def test_returns_none_for_non_tiling(self) -> None:
        """Test that non-tiling tasks return None."""
        task = make_task(
            [([[1, 2], [3, 4]], [[9, 9], [9, 9]])],
            [[5, 6], [7, 8]],
            [[0, 0], [0, 0]],
        )
        solver = TilingSolver()
        assert solver.solve(task) is None

    def test_returns_none_for_identity(self) -> None:
        """Test that 1x1 tiling (identity) returns None."""
        task = make_task(
            [([[1, 2], [3, 4]], [[1, 2], [3, 4]])],
            [[5, 6], [7, 8]],
            [[5, 6], [7, 8]],
        )
        solver = TilingSolver()
        assert solver.solve(task) is None

    def test_inconsistent_factors_returns_none(self) -> None:
        """Test that inconsistent tiling factors return None."""
        task = make_task(
            [
                ([[1]], [[1, 1]]),  # 1x2 tiling
                ([[2]], [[2, 2, 2]]),  # 1x3 tiling (inconsistent!)
            ],
            [[3]],
            [[0]],
        )
        solver = TilingSolver()
        assert solver.solve(task) is None

    def test_non_divisible_dimensions_returns_none(self) -> None:
        """Test that non-divisible output dimensions return None."""
        task = make_task(
            [([[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]])],
            [[0]],
            [[0]],
        )
        solver = TilingSolver()
        assert solver.solve(task) is None


# --- BorderFillSolver ---


class TestBorderFillSolver:
    """Tests for BorderFillSolver."""

    def test_detects_border_addition(self) -> None:
        """Test detection of uniform border addition."""
        inp = [[1, 2], [3, 4]]
        out = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ]
        task = make_task([(inp, out)], [[5, 6], [7, 8]], [[0]])
        solver = BorderFillSolver()
        result = solver.solve(task)
        assert result == [
            [0, 0, 0, 0],
            [0, 5, 6, 0],
            [0, 7, 8, 0],
            [0, 0, 0, 0],
        ]

    def test_detects_border_removal(self) -> None:
        """Test detection of uniform border removal."""
        inp = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ]
        out = [[1, 2], [3, 4]]
        task = make_task(
            [(inp, out)],
            [
                [0, 0, 0, 0, 0],
                [0, 5, 6, 7, 0],
                [0, 8, 9, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [[0]],
        )
        solver = BorderFillSolver()
        result = solver.solve(task)
        assert result == [[5, 6, 7], [8, 9, 1]]

    def test_non_zero_border_color(self) -> None:
        """Test border addition with non-zero border color."""
        inp = [[0, 0], [0, 0]]
        out = [
            [5, 5, 5, 5],
            [5, 0, 0, 5],
            [5, 0, 0, 5],
            [5, 5, 5, 5],
        ]
        task = make_task([(inp, out)], [[1, 2], [3, 4]], [[0]])
        solver = BorderFillSolver()
        result = solver.solve(task)
        assert result == [
            [5, 5, 5, 5],
            [5, 1, 2, 5],
            [5, 3, 4, 5],
            [5, 5, 5, 5],
        ]

    def test_returns_none_for_non_uniform_border(self) -> None:
        """Test that non-uniform border returns None."""
        inp = [[1]]
        out = [
            [0, 1, 0],
            [2, 1, 0],  # Border not uniform (2 != 0)
            [0, 0, 0],
        ]
        task = make_task([(inp, out)], [[5]], [[0]])
        solver = BorderFillSolver()
        assert solver.solve(task) is None

    def test_returns_none_for_unrelated_task(self) -> None:
        """Test that unrelated tasks return None."""
        task = make_task(
            [([[1, 2], [3, 4]], [[9, 8], [7, 6]])],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
        )
        solver = BorderFillSolver()
        assert solver.solve(task) is None

    def test_consistent_border_across_examples(self) -> None:
        """Test that border params must be consistent across training examples."""
        task = make_task(
            [
                ([[1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # border=1, color=0
                ([[2]], [[5, 5, 5], [5, 2, 5], [5, 5, 5]]),  # border=1, color=5 (inconsistent!)
            ],
            [[3]],
            [[0]],
        )
        solver = BorderFillSolver()
        assert solver.solve(task) is None


# --- evaluate_solver ---


class TestEvaluateSolver:
    """Tests for evaluate_solver function."""

    def test_basic_evaluation(self) -> None:
        """Test basic solver evaluation."""
        tasks = {
            "task1": make_task(
                [([[0, 0], [0, 0]], [[1, 1], [1, 1]])],
                [[0, 0], [0, 0]],
                [[1, 1], [1, 1]],
            ),
            "task2": make_task(
                [([[1, 1], [1, 1]], [[2, 2], [2, 2]])],
                [[1, 1], [1, 1]],
                [[2, 2], [2, 2]],
            ),
        }
        solver = CopyInputSolver()
        result = evaluate_solver(solver, tasks)

        assert "i_cd" in result
        assert "efficiency" in result
        assert "context_diversity" in result
        assert result["n_attempted"] == 2

    def test_no_tasks_returns_zero(self) -> None:
        """Test that evaluating with no tasks returns zero."""
        result = evaluate_solver(CopyInputSolver(), {})
        assert result["i_cd"] == 0.0
        assert result["n_attempted"] == 0

    def test_solver_declines_task(self) -> None:
        """Test handling when solver returns None."""
        tasks = {
            "task1": make_task(
                [([[0]], [[0]])],  # Identity mapping — ColorMappingSolver returns None
                [[0]],
                [[0]],
            ),
        }
        solver = ColorMappingSolver()
        result = evaluate_solver(solver, tasks)
        assert result["n_attempted"] == 0

    def test_counts_correct_solutions(self) -> None:
        """Test that correct solutions are counted."""
        tasks = {
            "task1": make_task(
                [([[0]], [[1]])],
                [[5]],
                [[5]],  # CopyInputSolver will match this
            ),
        }
        solver = CopyInputSolver()
        result = evaluate_solver(solver, tasks)
        assert result["n_correct"] == 1

    def test_custom_entropy_fn(self) -> None:
        """evaluate_solver accepts a custom entropy function."""
        tasks = {
            "task1": make_task(
                [([[0, 0], [0, 0]], [[1, 1], [1, 1]])],
                [[0, 0], [0, 0]],
                [[1, 1], [1, 1]],
            ),
        }
        solver = CopyInputSolver()
        # Should run without error using compression_entropy
        result = evaluate_solver(solver, tasks, entropy_fn=compression_entropy)
        assert "efficiency" in result
        assert result["n_attempted"] == 1

    def test_custom_entropy_fn_changes_result(self) -> None:
        """Different entropy functions produce different efficiency values."""
        tasks = {
            "task1": make_task(
                [([[0, 1, 2], [3, 4, 5]], [[0, 0, 0], [0, 0, 0]])],
                [[0, 1, 2], [3, 4, 5]],
                [[0, 0, 0], [0, 0, 0]],
            ),
        }
        solver = MostCommonColorSolver()
        r_color = evaluate_solver(solver, tasks)
        r_compress = evaluate_solver(solver, tasks, entropy_fn=compression_entropy)
        # Both should run, but produce different efficiency values
        assert r_color["efficiency"] != r_compress["efficiency"]

    def test_task_results_structure(self) -> None:
        """Test that task_results have correct structure."""
        tasks = {
            "task1": make_task(
                [([[0]], [[1]])],
                [[0]],
                [[1]],
            ),
        }
        solver = CopyInputSolver()
        result = evaluate_solver(solver, tasks)
        assert len(result["task_results"]) == 1
        task_result = result["task_results"][0]
        assert "task_id" in task_result
        assert "delta_s" in task_result
        assert "work" in task_result
        assert "correct" in task_result
        assert "context_label" in task_result


# --- evaluate_all_solvers ---


class TestEvaluateAllSolvers:
    """Tests for evaluate_all_solvers function."""

    def test_runs_all_solvers(self) -> None:
        """Test that all default solvers are evaluated."""
        tasks = {
            "task1": make_task(
                [([[0, 1]], [[1, 0]])],
                [[0, 1]],
                [[1, 0]],
            ),
        }
        results = evaluate_all_solvers(tasks)
        assert len(results) == 7  # 7 default solvers
        for name in [
            "Random",
            "Copy-input",
            "Most-common",
            "Color-map",
            "Geometric",
            "Tiling",
            "Border-fill",
        ]:
            assert name in results

    def test_custom_solvers(self) -> None:
        """Test evaluation with custom solver list."""
        tasks = {
            "task1": make_task(
                [([[0]], [[1]])],
                [[0]],
                [[1]],
            ),
        }
        solvers = [CopyInputSolver(), RandomSolver(seed=42)]
        results = evaluate_all_solvers(tasks, solvers)
        assert len(results) == 2
        assert "Copy-input" in results
        assert "Random" in results

    def test_empty_tasks(self) -> None:
        """Test evaluation with no tasks."""
        results = evaluate_all_solvers({})
        assert len(results) == 7
        for result in results.values():
            assert result["i_cd"] == 0.0
            assert result["n_attempted"] == 0


# --- LLMSolver ---


class TestLLMSolverFormatGrid:
    """Tests for LLMSolver._format_grid static method."""

    def test_simple_grid(self) -> None:
        grid = [[1, 2, 3], [4, 5, 6]]
        assert LLMSolver._format_grid(grid) == "1 2 3\n4 5 6"

    def test_single_cell(self) -> None:
        assert LLMSolver._format_grid([[7]]) == "7"

    def test_single_row(self) -> None:
        assert LLMSolver._format_grid([[0, 9, 3]]) == "0 9 3"

    def test_empty_grid(self) -> None:
        assert LLMSolver._format_grid([]) == ""


class TestLLMSolverParseGrid:
    """Tests for LLMSolver._parse_grid static method."""

    def test_clean_response(self) -> None:
        text = "1 2 3\n4 5 6"
        assert LLMSolver._parse_grid(text) == [[1, 2, 3], [4, 5, 6]]

    def test_strips_whitespace(self) -> None:
        text = "  1 2  \n  3 4  \n"
        assert LLMSolver._parse_grid(text) == [[1, 2], [3, 4]]

    def test_skips_text_lines(self) -> None:
        text = "Here is the output:\n1 2\n3 4\nDone."
        assert LLMSolver._parse_grid(text) == [[1, 2], [3, 4]]

    def test_skips_empty_lines(self) -> None:
        text = "\n1 2\n\n3 4\n\n"
        assert LLMSolver._parse_grid(text) == [[1, 2], [3, 4]]

    def test_returns_none_for_all_text(self) -> None:
        assert LLMSolver._parse_grid("No grid here, just words.") is None

    def test_returns_none_for_empty(self) -> None:
        assert LLMSolver._parse_grid("") is None

    def test_returns_none_for_jagged_rows(self) -> None:
        text = "1 2 3\n4 5"
        assert LLMSolver._parse_grid(text) is None

    def test_single_value(self) -> None:
        assert LLMSolver._parse_grid("5") == [[5]]


class TestLLMSolverBuildPrompt:
    """Tests for LLMSolver._build_prompt."""

    def test_contains_training_examples(self) -> None:
        solver = LLMSolver(model_id="test-model", name="Test")
        task = make_task(
            [([[0, 1]], [[1, 0]])],
            [[2, 3]],
            [[3, 2]],
        )
        prompt = solver._build_prompt(task)
        assert "Example 1 input:" in prompt
        assert "0 1" in prompt
        assert "Example 1 output:" in prompt
        assert "1 0" in prompt
        assert "Test input:" in prompt
        assert "2 3" in prompt
        assert "Test output:" in prompt

    def test_multiple_training_examples(self) -> None:
        solver = LLMSolver(model_id="test-model", name="Test")
        task = make_task(
            [([[0]], [[1]]), ([[2]], [[3]])],
            [[4]],
            [[5]],
        )
        prompt = solver._build_prompt(task)
        assert "Example 1 input:" in prompt
        assert "Example 2 input:" in prompt


class TestLLMSolverCache:
    """Tests for LLMSolver caching."""

    def test_save_and_load_cache(self, tmp_path: Path) -> None:
        solver = LLMSolver(model_id="test-model", name="Test", cache_dir=tmp_path)
        grid = [[1, 2], [3, 4]]
        solver._save_cache("task_abc", grid, "1 2\n3 4")
        loaded = solver._load_cache("task_abc")
        assert loaded == grid

    def test_load_cache_miss(self, tmp_path: Path) -> None:
        solver = LLMSolver(model_id="test-model", name="Test", cache_dir=tmp_path)
        assert solver._load_cache("nonexistent") is None

    def test_save_none_grid(self, tmp_path: Path) -> None:
        solver = LLMSolver(model_id="test-model", name="Test", cache_dir=tmp_path)
        solver._save_cache("task_fail", None, "error text")
        assert solver._load_cache("task_fail") is None


class TestLLMSolverSolve:
    """Tests for LLMSolver.solve with mocked API."""

    def test_solve_calls_api_and_parses(self, tmp_path: Path) -> None:
        solver = LLMSolver(
            model_id="test-model",
            name="Test",
            base_url="https://fake.api",
            api_key="test-key",
            cache_dir=tmp_path,
        )
        task = make_task(
            [([[0, 1]], [[1, 0]])],
            [[2, 3]],
            [[3, 2]],
        )
        with patch.object(solver, "_call_api", return_value="3 2") as mock_api:
            result = solver.solve(task)
        assert result == [[3, 2]]
        mock_api.assert_called_once()

    def test_solve_uses_cache(self, tmp_path: Path) -> None:
        solver = LLMSolver(
            model_id="test-model",
            name="Test",
            cache_dir=tmp_path,
        )
        task = make_task(
            [([[0]], [[1]])],
            [[2]],
            [[3]],
        )
        # Pre-populate cache
        from ie.environments.arc_evaluation import _task_content_hash

        task_id = _task_content_hash(task)
        solver._save_cache(task_id, [[9, 9]], "cached")

        with patch.object(solver, "_call_api") as mock_api:
            result = solver.solve(task)
        assert result == [[9, 9]]
        mock_api.assert_not_called()

    def test_solve_handles_api_error(self, tmp_path: Path) -> None:
        solver = LLMSolver(
            model_id="test-model",
            name="Test",
            base_url="https://fake.api",
            api_key="test-key",
            cache_dir=tmp_path,
        )
        task = make_task(
            [([[0]], [[1]])],
            [[2]],
            [[3]],
        )
        with patch.object(solver, "_call_api", side_effect=ConnectionError("timeout")):
            result = solver.solve(task)
        assert result is None

    def test_solve_handles_unparseable_response(self, tmp_path: Path) -> None:
        solver = LLMSolver(
            model_id="test-model",
            name="Test",
            base_url="https://fake.api",
            api_key="test-key",
            cache_dir=tmp_path,
        )
        task = make_task(
            [([[0]], [[1]])],
            [[2]],
            [[3]],
        )
        with patch.object(solver, "_call_api", return_value="Sorry, I cannot solve this."):
            result = solver.solve(task)
        assert result is None
