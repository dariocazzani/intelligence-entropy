# intelligence-entropy

[![CI](https://github.com/dariocazzani/intelligence-entropy/actions/workflows/ci.yml/badge.svg)](https://github.com/dariocazzani/intelligence-entropy/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![AGI-26](https://img.shields.io/badge/AGI--26-accepted-brightgreen.svg)](https://agi-conf.org/)
[![Springer LNAI](https://img.shields.io/badge/Springer-LNAI-blue.svg)](https://www.springer.com/series/1244)

Pragmatic intelligence as efficient environmental entropy reduction: implementation and evaluation of I_cd = (-ΔS/W) × H(C).

## Paper

"Efficient Entropy Reduction as a Substrate-Independent Measure of Pragmatic Intelligence"
Dario Cazzani (Cisco Systems). Published in Proceedings of AGI-26, Springer LNAI, July 2026. Link forthcoming.

## Quick Start

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest tests/ -v
```

## Experiments

```bash
uv run python experiments/run_all.py              # coupled regulation (all agents, 50 seeds)
uv run python experiments/run_sliding_puzzle_heuristic.py  # variable-width sliding puzzle
uv run python experiments/run_arc.py              # ARC benchmark (rule-based + LLM solvers)
uv run python experiments/run_ipc.py              # IPC 2023 classical planning
```

For parallel multi-seed experiments: `--chunk N --num-chunks K` then `--merge` to aggregate.

## Data

**ARC tasks** are included in `data/arc/training/`.

**IPC 2023 benchmarks** (313 MB) are not included. Download from the [IPC 2023 Learning Track](https://ipc2023-learning.github.io/) and place in `data/ipc/benchmarks/`.

## Citation

```bibtex
@inproceedings{cazzani2026entropy,
  title={Efficient Entropy Reduction as a Substrate-Independent Measure of Pragmatic Intelligence},
  author={Cazzani, Dario},
  booktitle={Proceedings of the 18th International Conference on Artificial General Intelligence (AGI-26)},
  series={Lecture Notes in Artificial Intelligence},
  publisher={Springer},
  year={2026}
}
```

## License

MIT
