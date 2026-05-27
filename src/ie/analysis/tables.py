"""LaTeX table generation for the I_cd paper."""

from __future__ import annotations

import json
from pathlib import Path

SCALE = 1000  # Display values as ×10³


def _fmt(val: float, precision: int = 2) -> str:
    """Format a scaled value."""
    return f"{val * SCALE:.{precision}f}"


def _fmt_pm(mean: float, std: float, precision: int = 2) -> str:
    """Format mean ± std, both scaled."""
    return f"{mean * SCALE:.{precision}f} \\pm {std * SCALE:.{precision}f}"


def generate_context_exposure_table(
    results_path: str | Path,
    tables_dir: str | Path,
) -> None:
    """Generate LaTeX table for context-exposure experiment.

    Shows how I_cd scales with number of contexts (1, 2, 3) for each agent
    in one representative environment.

    Args:
        results_path: Path to context_exposure_results.json
        tables_dir: Directory to write .tex file
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    # Use the first environment as representative
    env_name = list(results.keys())[0]
    env_data = results[env_name]

    # Extract agent names from the first config
    config_key = list(env_data.keys())[0]
    agent_names = list(env_data[config_key].keys())

    # Config keys in order: 1_config, 2_configs, 3_configs
    config_keys = ["1_config", "2_configs", "3_configs"]

    # Find best agent per column (config)
    best_per_config: dict[str, str] = {}
    for config_key in config_keys:
        best_agent = max(agent_names, key=lambda a: env_data[config_key][a]["i_cd"])
        best_per_config[config_key] = best_agent

    lines = []
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Agent & $|C|=1$ & $|C|=2$ & $|C|=3$ \\")
    lines.append(r" & $H(C)=0$ & $H(C)=\ln 2$ & $H(C)=\ln 3$ \\")
    lines.append(r"\midrule")

    for agent in agent_names:
        row = agent
        for config_key in config_keys:
            icd = env_data[config_key][agent]["i_cd"]
            is_best = best_per_config[config_key] == agent

            cell = f"${_fmt(icd)}$"
            if is_best:
                cell = r"$\mathbf{" + cell[1:-1] + r"}$"

            row += f" & {cell}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    (tables_dir / "context_exposure.tex").write_text("\n".join(lines))


def generate_latex_tables(
    results_path: str | Path,
    ablation_results: dict,
    tables_dir: str | Path,
    env_stds: dict[str, dict[str, dict]] | None = None,
) -> None:
    """Generate LaTeX table fragments for the paper.

    Args:
        results_path: Path to results.json with full evaluation results
        ablation_results: dict mapping env_key to dicts with 'specialist' and
                         'generalist' results
        tables_dir: Directory to write .tex files
        env_stds: Optional dict mapping env_key to agent_name to dict with
                 'efficiency_std' and 'i_cd_std'
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    env_keys = [k for k in results if k != "ablation"]
    env_display_names = {
        "sequence_sorting": "Seq.~Sorting",
        "spatial_grid": "Spatial Grid",
        "coupled_regulation": "Coupled Reg.",
        "tile_puzzle": "Tile Puzzle",
    }

    agent_names = list(results[env_keys[0]].keys())

    # Find best agent per environment for bolding
    best_per_env: dict[str, str] = {}
    for env_key in env_keys:
        best_agent = max(agent_names, key=lambda a: results[env_key][a]["i_cd"])
        best_per_env[env_key] = best_agent

    ranking_lines = []
    ranking_lines.append(r"\begin{tabular}{l" + "r" * len(env_keys) + "}")
    ranking_lines.append(r"\toprule")

    header = "Agent"
    for env_key in env_keys:
        display_name = env_display_names.get(env_key, env_key.replace("_", " ").title())
        header += f" & {display_name}"
    header += r" \\"
    ranking_lines.append(header)
    ranking_lines.append(r"\midrule")

    for agent in agent_names:
        row = agent
        for env_key in env_keys:
            icd = results[env_key][agent]["i_cd"]
            is_best = best_per_env[env_key] == agent
            if env_stds is not None and env_key in env_stds and agent in env_stds[env_key]:
                icd_std = env_stds[env_key][agent]["i_cd_std"]
                cell = f"${_fmt_pm(icd, icd_std)}$"
            else:
                cell = f"${_fmt(icd)}$"
            if is_best:
                cell = r"$\mathbf{" + cell[1:-1] + r"}$"
            row += f" & {cell}"
        row += r" \\"
        ranking_lines.append(row)

    ranking_lines.append(r"\bottomrule")
    ranking_lines.append(r"\end{tabular}")

    (tables_dir / "ranking.tex").write_text("\n".join(ranking_lines))

    # Ablation table
    ablation_lines = []
    ablation_lines.append(r"\begin{tabular}{lrcrr}")
    ablation_lines.append(r"\toprule")
    ablation_lines.append(
        r"Agent & $\eta \times 10^3$ & $H(C)$ "
        r"& $I_{cd} \times 10^3$ & $I_{\text{ablated}} \times 10^3$ \\"
    )
    ablation_lines.append(r"\midrule")

    for env_key in ablation_results:
        abl = ablation_results[env_key]
        spec = abl["specialist"]
        gen = abl["generalist"]

        display_name = env_display_names.get(env_key, env_key.replace("_", " ").title()).replace(
            "~", "\\ "
        )

        ablation_lines.append(
            f"Specialist ({display_name}) "
            f"& ${_fmt(spec['efficiency'])}$ & "
            f"${spec['context_diversity']:.2f}$ & "
            f"${_fmt(spec['i_cd'])}$ & "
            f"${_fmt(spec['efficiency'])}$ \\\\"
        )
        ablation_lines.append(
            f"Generalist ({display_name}) "
            f"& ${_fmt(gen['efficiency'])}$ & "
            f"${gen['context_diversity']:.2f}$ & "
            f"${_fmt(gen['i_cd'])}$ & "
            f"${_fmt(gen['efficiency'])}$ \\\\"
        )

    ablation_lines.append(r"\bottomrule")
    ablation_lines.append(r"\end{tabular}")

    (tables_dir / "ablation.tex").write_text("\n".join(ablation_lines))
