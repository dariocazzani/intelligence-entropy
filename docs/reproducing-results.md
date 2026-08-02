# Reproducing the paper results

The paper evaluates the same conceptual measure in four domains, but each
domain defines environmental disorder, work, and contexts differently. Raw
scores are meaningful within one operationalization and should not be compared
directly across experiments.

## Environment

The project requires Python 3.11 or newer and uses
[`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --extra dev
```

The continuous-integration checks are:

```bash
uv run ruff check src/ tests/ experiments/
uv run ruff format --check src/ tests/ experiments/
uv run mypy src/
uv run pytest tests/ -v --tb=short
```

Experiment outputs are written primarily to `experiments/results/` and
`paper/data/`. The former is intentionally ignored by Git.

## Coupled regulation

```bash
uv run python experiments/run_all.py
```

This is a legacy master pipeline, not a quick example. It trains and evaluates
agents for both Sequence Sorting and Coupled Regulation over 50 seeds, then
runs the ablation, context-exposure, and heterogeneous-cost analyses. Sequence
Sorting is not one of the four evaluations reported in the published paper.
The script does not support the `--chunk` or `--merge` options.

Running it writes into `paper/figures/`, `paper/tables/`, and `paper/data/` as
well as `experiments/results/`. It can overwrite tracked paper artifacts such
as `paper/tables/ablation.tex`; inspect the resulting diff before retaining
those outputs.

In the Coupled Regulation experiment, every evaluated agent sees the same
three contexts. Consequently, $H(C)=\ln 3$ is fixed and the ranking isolates
entropy-reduction efficiency.

## Sliding puzzle

Run all 50 seeds sequentially:

```bash
uv run python experiments/run_sliding_puzzle_heuristic.py
```

Or split them into five independent chunks and merge the results:

```bash
uv run python experiments/run_sliding_puzzle_heuristic.py --chunk 0 --num-chunks 5
uv run python experiments/run_sliding_puzzle_heuristic.py --chunk 1 --num-chunks 5
uv run python experiments/run_sliding_puzzle_heuristic.py --chunk 2 --num-chunks 5
uv run python experiments/run_sliding_puzzle_heuristic.py --chunk 3 --num-chunks 5
uv run python experiments/run_sliding_puzzle_heuristic.py --chunk 4 --num-chunks 5
uv run python experiments/run_sliding_puzzle_heuristic.py --merge --num-chunks 5
```

The chunk commands may run on separate workers. Merge only after all five
result files are available under `experiments/results/`.

This experiment evaluates restricted versions of the same greedy
Manhattan-distance heuristic across ten grid-size and scramble-depth contexts.
It uses performance gating: $H(C)$ credits only contexts with positive
per-context efficiency, and aggregate efficiency is averaged across those
effective contexts.

## IPC 2023 Learning Track

The IPC evaluation requires both the benchmark PDDL files and the archival
competition properties:

```bash
mkdir -p data/ipc
git clone https://github.com/ipc2023-learning/benchmarks.git data/ipc/benchmarks
curl -L https://raw.githubusercontent.com/ipc2023-learning/ipc2023-learning.github.io/main/properties \
  -o data/ipc/properties.json
uv run python experiments/run_ipc.py
```

The expected layout is:

```text
data/ipc/
├── benchmarks/
│   └── <domain>/testing/<difficulty>/<problem>.pddl
└── properties.json
```

The evaluator counts initially unsatisfied goal predicates from each problem,
uses the archived plan cost as work, and treats the planning domain as context.
Only solved problems contribute trajectories.

## ARC

The 400 public ARC training tasks are included under `data/arc/training/`.
Recompute the goal-relative compression evaluation with:

```bash
uv run python experiments/run_arc_compression.py
```

This recomputes the rule-based solver results. It reads LLM predictions only
from `.cache/arc_llm/`; the published caches are not tracked, so the three LLM
rows are skipped when those directories are absent. Border-fill produces no
attempts; the evaluator emits zeros internally, while the paper displays its
undefined efficiency and $I_{cd}$ as dashes.

`experiments/run_arc.py` is an earlier pipeline that compresses grids directly
rather than using the published input-to-goal and prediction-to-goal residuals.
It can also make live API calls. Do not use it to reproduce the Section 4.4
table.

## Published values

The authoritative reported values and experimental definitions are in:

- [`paper/main.pdf`](../paper/main.pdf), the published manuscript;
- [`paper/main.tex`](../paper/main.tex), the manuscript source;
- [`paper/figures/`](../paper/figures/), the checked-in plot sources; and
- [`paper/tables/`](../paper/tables/), the checked-in table sources.

Because generated JSON outputs and LLM caches are not tracked, a clean checkout
does not contain every intermediate artifact used for the publication.
