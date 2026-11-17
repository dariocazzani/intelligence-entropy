# Applying the measure to your own evaluation

Start with the [runnable walkthrough](../notebooks/measure_walkthrough.ipynb),
or [open it in Colab](https://colab.research.google.com/github/dariocazzani/intelligence-entropy/blob/main/notebooks/measure_walkthrough.ipynb).
It explains the published sliding-puzzle ranking and provides an editable,
illustrative evaluation. The complete experiment commands are in
[Reproducing the paper](reproducing-results.md).

## Define the evaluation before scoring

Choose an environmental disorder function $S$, an action-cost definition $W$,
and a set of contexts $C$. These choices are part of the measurement protocol.
Use the same choices, evaluation budget, and sampling policy for every agent
you compare.

For the sliding puzzle, $S$ is Shannon entropy over the histogram of tile
Manhattan-displacement distances, $W$ counts decisions (including invalid
slides), and a context is a grid-width/scramble-depth pair. A scramble depth
counts random legal moves from a solved board; moves can undo each other, so
it is not the shortest solution length.

An entropy value of zero does not by itself establish success: a histogram
concentrated at a nonzero displacement can also have zero entropy. Check that
the chosen $S$ tracks the kind of environmental progress you intend to study,
and report task success alongside it.

## Record outcomes and costs

For each agent, seed, context, and episode, record:

| Field | Meaning |
|---|---|
| `agent` | Agent identifier |
| `seed` | Evaluation repetition, not a new context |
| `context` | Label from the fixed context partition |
| `s_initial`, `s_final` | Disorder values under the same definition of $S$ |
| `work` | Environmental action cost over that episode |

Use natural logarithms when measuring Shannon entropy in nats. Keep raw
observations where possible so another evaluator can audit your definition of
$S$. Require finite inputs and positive work for scored episodes; a zero-cost
episode cannot establish an efficiency ratio. Declare any exclusion policy
before comparing agents.

## Choose and report the aggregation protocol

The core functions in [`ie.measure`](../src/ie/measure.py) compute efficiency
and Shannon diversity. They do not decide which contexts qualify, how episodes
are weighted, or whether an entropy function measures useful progress.

The sliding-puzzle experiment uses the following protocol **within each seed**:

1. Evaluate every agent on all ten contexts with ten episodes per context.
2. For each context, sum episode changes $\Delta S = S_{final}-S_{initial}$
   and episode work. Compute $\eta_c = -\sum\Delta S / \sum W$.
3. Retain contexts with $\eta_c > 0$. Let their count be $k$.
4. Average $\eta_c$ equally over those retained contexts. Use uniform context
   weights, giving $H(C)=\ln k$, then multiply the two factors.
5. If no contexts qualify, report zero for both factors and the score. If only
   one qualifies, $H(C)=0$ and the score is zero.

This performance gate credits demonstrated entropy reduction. It does not mean
the puzzle was solved, and qualifying contexts may differ between seeds.
Skipped contexts are missing observations, not measured failures. Report them
and do not silently remove them from the evaluation universe.

The paper uses different context conventions in other experiments. For
example, coupled regulation holds context exposure constant rather than
applying this performance gate. Do not silently transfer a convention from one
experiment to another.

For an exposure-weighted protocol, `context_diversity(labels)` uses empirical
label frequencies. Count labels according to the declared exposure unit;
episode counts and elapsed operational time can imply different weights.
One occurrence per retained context implements the uniform convention above.

Compute scores per seed before summarizing them. The mean of
$\eta H(C)$ need not equal mean efficiency multiplied by mean diversity.
Report variability, the number of repetitions, task success, both factors,
and the combined score.

## What comparisons are justified?

Comparing two puzzle agents under the same disorder function, decision cost,
context partition, and evaluation procedure is a within-protocol comparison.
Comparing a puzzle score with an ARC compression score is not: the disorder
function, units, action costs, and contexts differ.

Changing units or splitting one context into several can change scores without
changing behavior. Document these choices and test sensitivity before treating
a ranking as robust. The score describes observed pragmatic competence under
the stated evaluation; it is not a universal ranking of intelligence.
