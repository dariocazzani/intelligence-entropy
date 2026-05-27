"""Core I_cd measure implementation.

This module implements the intelligence-entropy measure I_cd = (-ΔS/W) × H(C),
which quantifies an agent's ability to reduce environmental entropy efficiently
across diverse contexts.
"""

from __future__ import annotations

import numpy as np


def environmental_entropy(state: np.ndarray) -> float:
    """Compute Shannon entropy over the state distribution.

    Args:
        state: 1D array representing either raw category counts or normalized
               probabilities. Will be normalized internally if needed.

    Returns:
        Shannon entropy H(S) = -Σ p(s) ln(p(s)) in nats (natural units).
        Returns 0.0 if all counts are zero or the array is empty.

    Examples:
        >>> environmental_entropy(np.array([1, 1, 1, 1]))  # uniform
        1.3862943611198906  # ln(4)
        >>> environmental_entropy(np.array([4, 0, 0, 0]))  # single state
        0.0
    """
    state = np.asarray(state, dtype=float)

    if state.size == 0 or np.sum(state) == 0:
        return 0.0

    probs = state / np.sum(state)

    # 0 * ln(0) = 0 by convention
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log(probs))

    return float(entropy)


def efficiency(delta_s: float, work: float) -> float:
    """Compute efficiency as -ΔS/W.

    Args:
        delta_s: Change in environmental entropy (S_final - S_initial).
                 Negative values indicate entropy reduction.
        work: Energy/resources expended. Must be non-negative.

    Returns:
        Efficiency metric -ΔS/W. Returns 0.0 if work is zero (no work means
        no intelligence, not infinity).

    Examples:
        >>> efficiency(-1.0, 2.0)  # reduced entropy by 1 nat with 2 units work
        0.5
        >>> efficiency(1.0, 2.0)  # increased entropy (negative efficiency)
        -0.5
        >>> efficiency(-1.0, 0.0)  # no work expended
        0.0
    """
    if work == 0:
        return 0.0

    return -delta_s / work


def context_diversity(contexts: list[int] | np.ndarray) -> float:
    """Compute context diversity as Shannon entropy H(C) over context labels.

    Args:
        contexts: List or array of context labels observed during evaluation.
                  E.g., [0, 0, 1, 2, 1, 0] indicates 3 obs in context 0,
                  2 obs in context 1, and 1 obs in context 2.

    Returns:
        Shannon entropy H(C) = -Σ p(c) ln(p(c)) in nats.
        Returns 0.0 if empty or single unique context.

    Examples:
        >>> context_diversity([0, 0, 0])  # single context
        0.0
        >>> context_diversity([0, 1, 2, 3])  # uniform over 4 contexts
        1.3862943611198906  # ln(4)
        >>> context_diversity([])  # no contexts
        0.0
    """
    if not isinstance(contexts, np.ndarray):
        contexts = np.array(contexts)

    if contexts.size == 0:
        return 0.0

    unique, counts = np.unique(contexts, return_counts=True)

    if len(unique) == 1:
        return 0.0

    return environmental_entropy(counts)


def i_cd(delta_s: float, work: float, contexts: list[int] | np.ndarray) -> float:
    """Compute the full I_cd measure: (-ΔS/W) × H(C).

    This is the main intelligence-entropy measure that combines efficiency
    (entropy reduction per unit work) with context diversity.

    Args:
        delta_s: Change in environmental entropy (S_final - S_initial).
        work: Energy/resources expended.
        contexts: List or array of context labels.

    Returns:
        I_cd = efficiency(delta_s, work) × context_diversity(contexts).
        Returns 0.0 if work is zero, contexts are empty, or single context.

    Examples:
        >>> i_cd(-2.0, 1.0, [0, 1, 2, 3])  # high efficiency, diverse contexts
        2.7725887222397812  # 2.0 * ln(4)
        >>> i_cd(-2.0, 1.0, [0, 0, 0, 0])  # high efficiency, single context
        0.0
        >>> i_cd(-2.0, 0.0, [0, 1, 2, 3])  # no work expended
        0.0
    """
    eff = efficiency(delta_s, work)
    diversity = context_diversity(contexts)

    return eff * diversity
