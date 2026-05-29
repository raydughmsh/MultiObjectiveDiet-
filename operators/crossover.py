"""
Order Crossover (OX) for permutation chromosomes.

The chromosome is split into two independent parts:
  - Breakfast    : genes [0 : N_BREAKFAST]       length = 94
  - Lunch+dinner : genes [N_BREAKFAST : N_FOODS]  length = 311

OX is applied separately to each part, then the two results are concatenated
to form the full offspring chromosome.

OX algorithm (for one part of length n):
  1. Pick two random cut points i < j.
  2. Copy segment parent1[i:j] directly into offspring at positions i:j.
  3. Fill remaining positions left-to-right (wrapping around from j)
     with genes from parent2, skipping genes already placed.
"""

import numpy as np
from typing import Optional
from config import N_BREAKFAST, N_FOODS, P_CROSSOVER


# ── Core OX on a single part ──────────────────────────────────────────────────

def _ox_single(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply OX to one permutation segment.

    Parameters
    ----------
    p1, p2 : 1-D int arrays of equal length (a permutation of some set of IDs)
    rng    : numpy random Generator

    Returns
    -------
    offspring : 1-D int array, same length as p1/p2
    """
    n = len(p1)
    offspring = np.full(n, -1, dtype=p1.dtype)

    # Step 1: choose cut points
    i, j = sorted(rng.choice(n, size=2, replace=False))

    # Step 2: copy the segment from parent1
    offspring[i:j] = p1[i:j]
    segment_set = set(p1[i:j])

    # Step 3: fill remaining positions with parent2 in order (starting after j)
    p2_filtered = [gene for gene in np.roll(p2, -(j % n)) if gene not in segment_set]

    fill_positions = [pos for pos in np.roll(np.arange(n), -(j % n)) if offspring[pos] == -1]

    for pos, gene in zip(fill_positions, p2_filtered):
        offspring[pos] = gene

    return offspring


# ── Split-chromosome OX (applied to each part independently) ─────────────────

def ox_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    p_c: float = P_CROSSOVER,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply OX crossover to a full split chromosome.

    The chromosome has shape (N_FOODS,) and is treated as two independent parts:
      breakfast    = chromosome[0 : N_BREAKFAST]
      lunch_dinner = chromosome[N_BREAKFAST : N_FOODS]

    Crossover is applied independently to each part with probability p_c.
    If crossover does not occur for a part, the parents' genes are passed through.

    Parameters
    ----------
    parent1, parent2 : 1-D int arrays of shape (N_FOODS,)
    rng              : numpy random Generator (created internally if None)
    p_c              : crossover probability

    Returns
    -------
    offspring1, offspring2 : two 1-D int arrays of shape (N_FOODS,)
    """
    if rng is None:
        rng = np.random.default_rng()

    p1_b  = parent1[:N_BREAKFAST]
    p1_ld = parent1[N_BREAKFAST:]
    p2_b  = parent2[:N_BREAKFAST]
    p2_ld = parent2[N_BREAKFAST:]

    # Breakfast part
    if rng.random() < p_c:
        off1_b = _ox_single(p1_b, p2_b, rng)
        off2_b = _ox_single(p2_b, p1_b, rng)
    else:
        off1_b = p1_b.copy()
        off2_b = p2_b.copy()

    # Lunch+dinner part
    if rng.random() < p_c:
        off1_ld = _ox_single(p1_ld, p2_ld, rng)
        off2_ld = _ox_single(p2_ld, p1_ld, rng)
    else:
        off1_ld = p1_ld.copy()
        off2_ld = p2_ld.copy()

    offspring1 = np.concatenate([off1_b, off1_ld])
    offspring2 = np.concatenate([off2_b, off2_ld])

    return offspring1, offspring2


# ── pymoo-compatible Crossover class ─────────────────────────────────────────

from pymoo.core.crossover import Crossover


class SplitOXCrossover(Crossover):
    """
    pymoo Crossover wrapper for OX on split chromosomes.
    Plug this directly into NSGA-II / SPEA2 via the `crossover` argument.
    """

    def __init__(self, p_c: float = P_CROSSOVER):
        # n_parents=2, n_offsprings=2
        super().__init__(n_parents=2, n_offsprings=2, prob=p_c)
        self._p_c = p_c   # store plain Python float separately

    def _do(self, problem, X, **kwargs):
        """
        X shape: (n_parents=2, n_matings, n_var)
        Returns: (n_offsprings=2, n_matings, n_var)
        """
        _, n_matings, n_var = X.shape
        offspring = np.empty_like(X)
        rng = np.random.default_rng()

        for k in range(n_matings):
            p1 = X[0, k, :]
            p2 = X[1, k, :]
            o1, o2 = ox_crossover(p1, p2, rng=rng, p_c=self._p_c)
            offspring[0, k, :] = o1
            offspring[1, k, :] = o2

        return offspring
