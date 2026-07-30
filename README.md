# intelligence-entropy

[![CI](https://github.com/dariocazzani/intelligence-entropy/actions/workflows/ci.yml/badge.svg)](https://github.com/dariocazzani/intelligence-entropy/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![AGI-26](https://img.shields.io/badge/AGI--26-accepted-brightgreen.svg)](https://agi-conf.org/)
[![Springer LNAI](https://img.shields.io/badge/Springer-LNAI-blue.svg)](https://www.springer.com/series/1244)

Pragmatic intelligence as efficient environmental entropy reduction: implementation and evaluation of

$$I_{cd} = \frac{-\Delta S}{W} \times H(C)$$

## Paper

["Efficient Entropy Reduction as a Substrate-Independent Measure of Pragmatic Intelligence"](https://doi.org/10.1007/978-3-032-33010-9_10)

Dario Cazzani (Cisco Systems). Published in *Artificial General Intelligence: 19th International Conference, AGI 2026, Proceedings, Part I*, Springer LNCS 16854, pp. 154–171, July 2026.

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
  author    = {Cazzani, Dario},
  editor    = {Iklé, Matthew and Franz, Arthur and Kemendo, Andrew and Lowy, Haley},
  title     = {Efficient Entropy Reduction as a Substrate-Independent Measure of Pragmatic Intelligence},
  booktitle = {Artificial General Intelligence},
  series    = {Lecture Notes in Computer Science},
  volume    = {16854},
  pages     = {154--171},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  year      = {2026},
  doi       = {10.1007/978-3-032-33010-9_10},
  isbn      = {978-3-032-33010-9},
  url       = {https://doi.org/10.1007/978-3-032-33010-9_10}
}
```

## License

MIT
