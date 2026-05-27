"""Figure generation for the I_cd paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_icd_ranking(
    results: dict, env_name: str, output_path: str, stds: dict[str, float] | None = None
) -> None:
    """Bar chart of I_cd scores across agents.

    Args:
        results: dict mapping agent name to eval result dict (must have 'i_cd' key)
        env_name: Name of the environment (for title)
        output_path: Path to save the figure
        stds: Optional dict mapping agent name to standard deviation for error bars
    """
    names = list(results.keys())
    scores = [results[n]["i_cd"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    if stds is not None:
        yerr = [stds[n] for n in names]
        bars = ax.bar(
            names,
            scores,
            yerr=yerr,
            capsize=4,
            color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][: len(names)],
        )
    else:
        bars = ax.bar(
            names, scores, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][: len(names)]
        )
    ax.set_ylabel("I_cd Score")
    ax.set_title(f"I_cd Ranking — {env_name}")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Add value labels on bars
    for bar, score, name in zip(bars, scores, names):
        if stds is not None:
            label = f"{score:.3f}±{stds[name]:.3f}"
        else:
            label = f"{score:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_obj, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_curves(results: dict, env_name: str, output_path: str) -> None:
    """Plot entropy over time for each agent (averaged across episodes/configs).

    Args:
        results: dict mapping agent name to eval result dict
        env_name: Name of the environment
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for idx, (name, res) in enumerate(results.items()):
        # Average trajectory across all configs and episodes
        all_trajectories = []
        for config_trajs in res["entropy_trajectories"].values():
            all_trajectories.extend(config_trajs)

        if not all_trajectories:
            continue

        # Pad to same length and average
        max_len = max(len(t) for t in all_trajectories)
        padded = np.array([t + [t[-1]] * (max_len - len(t)) for t in all_trajectories])
        mean_traj = padded.mean(axis=0)

        ax.plot(mean_traj, label=name, color=colors[idx % len(colors)], linewidth=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Environmental Entropy (nats)")
    ax.set_title(f"Entropy Over Time — {env_name}")
    ax.legend()

    plt.tight_layout()
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_obj, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(
    results_with_hc: dict, results_without_hc: dict, env_name: str, output_path: str
) -> None:
    """Side-by-side comparison: I_cd with and without H(C) term.

    results_without_hc should have 'efficiency' values used as score
    (I_cd without H(C) = just efficiency).
    """
    names = list(results_with_hc.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # With H(C)
    scores_with = [results_with_hc[n]["i_cd"] for n in names]
    ax1.bar(names, scores_with, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][: len(names)])
    ax1.set_ylabel("Score")
    ax1.set_title(f"With H(C) — {env_name}")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Without H(C) — just efficiency
    scores_without = [results_without_hc[n]["efficiency"] for n in names]
    ax2.bar(names, scores_without, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"][: len(names)])
    ax2.set_ylabel("Score")
    ax2.set_title(f"Without H(C) (efficiency only) — {env_name}")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_obj, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_icd_ranking_combined(
    grid_results: dict,
    seq_results: dict,
    output_path: str,
    grid_stds: dict[str, float] | None = None,
    seq_stds: dict[str, float] | None = None,
) -> None:
    """Combined two-panel figure showing I_cd rankings for both environments.

    Args:
        grid_results: dict mapping agent name to eval result dict for Resource Grid
        seq_results: dict mapping agent name to eval result dict for Sequence Sorting
        output_path: Path to save the figure
        grid_stds: Optional dict mapping agent name to std for Resource Grid error bars
        seq_stds: Optional dict mapping agent name to std for Sequence Sorting error bars
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    # Grid results (left panel)
    grid_names = list(grid_results.keys())
    grid_scores = [grid_results[n]["i_cd"] for n in grid_names]
    if grid_stds is not None:
        grid_yerr = [grid_stds[n] for n in grid_names]
        bars1 = ax1.bar(
            grid_names, grid_scores, yerr=grid_yerr, capsize=4, color=colors[: len(grid_names)]
        )
    else:
        bars1 = ax1.bar(grid_names, grid_scores, color=colors[: len(grid_names)])
    ax1.set_ylabel("I_cd Score")
    ax1.set_title("Resource Grid")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    for bar, score, name in zip(bars1, grid_scores, grid_names):
        if grid_stds is not None:
            label = f"{score:.3f}±{grid_stds[name]:.3f}"
        else:
            label = f"{score:.3f}"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Sequence results (right panel)
    seq_names = list(seq_results.keys())
    seq_scores = [seq_results[n]["i_cd"] for n in seq_names]
    if seq_stds is not None:
        seq_yerr = [seq_stds[n] for n in seq_names]
        bars2 = ax2.bar(
            seq_names, seq_scores, yerr=seq_yerr, capsize=4, color=colors[: len(seq_names)]
        )
    else:
        bars2 = ax2.bar(seq_names, seq_scores, color=colors[: len(seq_names)])
    ax2.set_ylabel("I_cd Score")
    ax2.set_title("Sequence Sorting")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    for bar, score, name in zip(bars2, seq_scores, seq_names):
        if seq_stds is not None:
            label = f"{score:.3f}±{seq_stds[name]:.3f}"
        else:
            label = f"{score:.3f}"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_obj, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(results_by_granularity: dict, env_name: str, output_path: str) -> None:
    """Show I_cd rankings at different context discretization granularities.

    Args:
        results_by_granularity: dict mapping granularity label to results dict
            e.g., {"3 contexts": {...}, "6 contexts": {...}}
        env_name: Environment name
        output_path: Path to save
    """
    fig, axes = plt.subplots(
        1, len(results_by_granularity), figsize=(6 * len(results_by_granularity), 5)
    )
    if len(results_by_granularity) == 1:
        axes = [axes]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for ax, (granularity, results) in zip(axes, results_by_granularity.items()):
        names = list(results.keys())
        scores = [results[n]["i_cd"] for n in names]
        ax.bar(names, scores, color=colors[: len(names)])
        ax.set_title(f"{granularity}")
        ax.set_ylabel("I_cd Score")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.suptitle(f"Sensitivity Analysis — {env_name}", fontsize=14)
    plt.tight_layout()
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_obj, dpi=150, bbox_inches="tight")
    plt.close(fig)
