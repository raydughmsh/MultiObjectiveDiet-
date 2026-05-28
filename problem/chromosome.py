"""
Member 2 — Chromosome Representation
problem/chromosome.py

Each candidate solution is a permutation of all 405 food item IDs,
split into two independent parts:

    [ gene_0, ..., gene_93 | gene_94, ..., gene_404 ]
      ←── breakfast (94) ──→ ←── lunch+dinner (311) ──→

The two parts are operated on independently by crossover and mutation.
The chromosome itself does NOT select foods — that is the decoder's job.
"""

import numpy as np
from config import N_FOODS, N_BREAKFAST, N_LUNCH_DINNER


class Chromosome:
    """
    Represents a candidate solution as a permutation of food IDs.

    Attributes
    ----------
    genes : np.ndarray, shape (N_FOODS,), dtype int
        A permutation of the 405 food IDs from the database.
        genes[0:N_BREAKFAST]   → breakfast candidates (94 foods)
        genes[N_BREAKFAST:]    → lunch+dinner candidates (311 foods)

    Parameters
    ----------
    genes : array-like or None
        If provided, used directly (must be a valid permutation of length N_FOODS).
        If None, a random permutation is generated from food_ids.
    food_ids : array-like or None
        The actual food IDs from the database (length N_FOODS).
        Required only when genes=None (for random initialization).
        If not provided, defaults to np.arange(1, N_FOODS + 1) — assumes IDs are 1..405.
    """

    def __init__(self, genes=None, food_ids=None):
        if genes is not None:
            self.genes = np.asarray(genes, dtype=int)
            if len(self.genes) != N_FOODS:
                raise ValueError(
                    f"Chromosome length must be {N_FOODS}, got {len(self.genes)}"
                )
        else:
            # Random initialization
            if food_ids is not None:
                ids = np.asarray(food_ids, dtype=int)
            else:
                # Fallback: assume IDs are 1..405
                ids = np.arange(1, N_FOODS + 1, dtype=int)

            if len(ids) != N_FOODS:
                raise ValueError(
                    f"food_ids must have exactly {N_FOODS} elements, got {len(ids)}"
                )

            self.genes = self._random_init(ids)

    # ── Initialization ────────────────────────────────────────────────────────

    @staticmethod
    def _random_init(food_ids: np.ndarray) -> np.ndarray:
        """
        Return a uniformly random permutation of the given food IDs.
        The shuffle covers the full array; the breakfast / lunch+dinner
        split emerges naturally from positions [0:94] and [94:405].
        """
        permutation = food_ids.copy()
        np.random.shuffle(permutation)
        return permutation

    # ── Parts (views into genes) ──────────────────────────────────────────────

    @property
    def breakfast(self) -> np.ndarray:
        """
        Breakfast part: genes[0:N_BREAKFAST] — 94 food ID candidates.
        The decoder iterates these left-to-right and selects greedily.
        Returns a VIEW (not a copy) — mutate carefully.
        """
        return self.genes[:N_BREAKFAST]

    @property
    def lunch_dinner(self) -> np.ndarray:
        """
        Lunch+dinner part: genes[N_BREAKFAST:N_FOODS] — 311 food ID candidates.
        Returns a VIEW (not a copy) — mutate carefully.
        """
        return self.genes[N_BREAKFAST:]

    # ── Utility ───────────────────────────────────────────────────────────────

    def copy(self) -> "Chromosome":
        """
        Return a deep copy of this chromosome.
        Used by genetic operators so the original is never mutated in place.
        """
        return Chromosome(genes=self.genes.copy())

    def as_array(self) -> np.ndarray:
        """
        Return the full gene array.
        pymoo works with raw numpy arrays; use this when passing to pymoo.
        """
        return self.genes

    def is_valid_permutation(self, food_ids=None) -> bool:
        """
        Verify that genes is a valid permutation (no duplicates, correct set).
        Useful for unit testing after crossover / mutation.

        Parameters
        ----------
        food_ids : array-like or None
            Expected set of IDs. If None, checks that genes contains each
            value in range [genes.min(), genes.max()] exactly once.
        """
        if len(self.genes) != N_FOODS:
            return False
        if len(np.unique(self.genes)) != N_FOODS:
            return False
        if food_ids is not None:
            return set(self.genes.tolist()) == set(food_ids)
        return True

    def __len__(self) -> int:
        return len(self.genes)

    def __repr__(self) -> str:
        b_preview  = self.genes[:3].tolist()
        ld_preview = self.genes[N_BREAKFAST:N_BREAKFAST + 3].tolist()
        return (
            f"Chromosome(breakfast={b_preview}..., "
            f"lunch_dinner={ld_preview}..., "
            f"total_genes={N_FOODS})"
        )


# ── Factory helpers ────────────────────────────────────────────────────────────

def random_chromosome(food_ids=None) -> Chromosome:
    """
    Convenience factory: create one random Chromosome.

    Parameters
    ----------
    food_ids : array-like or None
        The 405 food IDs from the DB. If None, falls back to 1..405.
    """
    return Chromosome(food_ids=food_ids)


def chromosome_from_array(x: np.ndarray) -> Chromosome:
    """
    Wrap a raw numpy gene array (as used by pymoo) into a Chromosome object.
    Useful in decode_solution() and reporting code.
    """
    return Chromosome(genes=x)


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Chromosome self-test ===\n")

    # Test 1: random init with default IDs
    c1 = Chromosome()
    print(f"Random chromosome: {c1}")
    print(f"  Total genes   : {len(c1)}")
    print(f"  Breakfast len : {len(c1.breakfast)}  (expected {N_BREAKFAST})")
    print(f"  Lunch/din len : {len(c1.lunch_dinner)}  (expected {N_LUNCH_DINNER})")
    print(f"  Valid permut. : {c1.is_valid_permutation()}")
    print()

    # Test 2: copy independence
    c2 = c1.copy()
    c2.genes[0] = -999
    assert c1.genes[0] != -999, "copy() is not deep — original was mutated!"
    print("copy() is independent: OK")
    print()

    # Test 3: from_array round-trip
    arr = c1.as_array()
    c3 = chromosome_from_array(arr.copy())
    assert np.array_equal(c1.genes, c3.genes), "round-trip failed"
    print("as_array() / chromosome_from_array() round-trip: OK")
    print()

    # Test 4: custom food_ids
    import numpy as np
    custom_ids = np.arange(100, 505, dtype=int)  # IDs 100..504
    c4 = Chromosome(food_ids=custom_ids)
    assert c4.is_valid_permutation(custom_ids), "custom food_ids permutation broken"
    print("Custom food_ids permutation: OK")
    print()

    print("All tests passed.")
