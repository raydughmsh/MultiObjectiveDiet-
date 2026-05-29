"""
Swap Mutation for permutation chromosomes.

For each part of the chromosome (breakfast and lunch+dinner), each gene
position is independently selected for mutation with probability p_m = 1/n
(where n is the length of that part). If a position is selected, it is
swapped with another randomly chosen position within the same part.

This preserves the permutation property within each part.
"""

import numpy as np
from typing import Optional
from pymoo.core.mutation import Mutation

from config import N_BREAKFAST, N_FOODS, P_MUTATION_B, P_MUTATION_LD


# ── Core swap mutation on a single part ──────────────────────────────────────

def _swap_mutate_single(
    part: np.ndarray,
    p_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply swap mutation to one permutation segment.

    For each position i, with probability p_m swap genes[i] with a random
    other position j ≠ i. Each swap is independent.

    Parameters
    ----------
    part : 1-D int array (a permutation segment)
    p_m  : mutation probability per gene
    rng  : numpy random Generator

    Returns
    -------
    mutated : 1-D int array, same length as part (permutation preserved)
    """
    mutated = part.copy()
    n = len(mutated)

    for i in range(n):
        if rng.random() < p_m:
            j = rng.integers(0, n)
            while j == i:
                j = rng.integers(0, n)
            mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated


# ── Split-chromosome swap mutation ───────────────────────────────────────────

def swap_mutation(
    chromosome: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    p_m_b: float = P_MUTATION_B,
    p_m_ld: float = P_MUTATION_LD,
) -> np.ndarray:
    """
    Apply swap mutation independently to each part of the split chromosome.

    Parameters
    ----------
    chromosome : 1-D int array of shape (N_FOODS,)
    rng        : numpy random Generator (created internally if None)
    p_m_b      : mutation probability for the breakfast part   (default 1/94)
    p_m_ld     : mutation probability for the lunch+dinner part (default 1/311)

    Returns
    -------
    mutated : 1-D int array of shape (N_FOODS,)
    """
    if rng is None:
        rng = np.random.default_rng()

    breakfast_part  = chromosome[:N_BREAKFAST]
    lunch_dinner_part = chromosome[N_BREAKFAST:]

    mutated_b  = _swap_mutate_single(breakfast_part,    p_m_b,  rng)
    mutated_ld = _swap_mutate_single(lunch_dinner_part, p_m_ld, rng)

    return np.concatenate([mutated_b, mutated_ld])


# ── pymoo-compatible Mutation class ──────────────────────────────────────────

class SplitSwapMutation(Mutation):
    """
    pymoo Mutation wrapper for swap mutation on split chromosomes.
    Plug this directly into NSGA-II / SPEA2 via the `mutation` argument.
    """

    def __init__(self, p_m_b: float = P_MUTATION_B, p_m_ld: float = P_MUTATION_LD):
        super().__init__()
        self.p_m_b  = p_m_b
        self.p_m_ld = p_m_ld

    def _do(self, problem, X, **kwargs):
        """
        X shape: (n_individuals, n_var)
        Returns mutated X of same shape.
        """
        rng = np.random.default_rng()
        X_mutated = X.copy()

        for i in range(len(X)):
            X_mutated[i] = swap_mutation(X[i], rng=rng, p_m_b=self.p_m_b, p_m_ld=self.p_m_ld)

        return X_mutated
