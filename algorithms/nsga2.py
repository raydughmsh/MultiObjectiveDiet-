"""
NSGA-II runner for the Multi-Objective Diet Optimization Problem.

Uses pymoo's built-in NSGA2 algorithm with:
  - Custom SplitOXCrossover  (operators/crossover.py)
  - Custom SplitSwapMutation (operators/mutation.py)
  - Binary tournament selection (pymoo default for NSGA-II)

The chromosome and fitness evaluation are handled by DietProblem
(problem/diet_problem.py — implemented by Member 3).

Usage
-----
    from algorithms.nsga2 import run_nsga2
    from db.loader import load_data_for_user   # Member 1

    data = load_data_for_user(user_id=1)
    result, history = run_nsga2(data, user_id=1)
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.core.sampling import Sampling

from operators.crossover import SplitOXCrossover
from operators.mutation import SplitSwapMutation
from config import (
    POP_SIZE, N_GENERATIONS, P_CROSSOVER,
    P_MUTATION_B, P_MUTATION_LD, N_FOODS,
)


# ── Custom initialization — random permutation for each individual ────────────

class SplitPermutationSampling(Sampling):
    """
    Initializes each individual as a random permutation of N_FOODS food IDs.

    The food IDs passed at construction time (from the DB) are shuffled
    independently to create each chromosome.
    """

    def __init__(self, food_ids: np.ndarray):
        super().__init__()
        self.food_ids = food_ids  # shape (N_FOODS,), actual DB food IDs

    def _do(self, problem, n_samples, **kwargs):
        X = np.empty((n_samples, len(self.food_ids)), dtype=int)
        rng = np.random.default_rng()
        for i in range(n_samples):
            X[i] = rng.permutation(self.food_ids)
        return X


# ── NSGA-II runner ────────────────────────────────────────────────────────────

def run_nsga2(data: dict, user_id: int, seed: int = 42) -> tuple:
    """
    Run NSGA-II for a given user and return the result + per-generation history.

    Parameters
    ----------
    data    : dict returned by Member 1's load_data_for_user(user_id)
              Expected keys: 'foods_df', 'nutrient_matrix', 'dri', 'food_groups'
    user_id : int (1 = non-vegetarian, 2 = vegetarian)
    seed    : random seed for reproducibility

    Returns
    -------
    result  : pymoo Result object
                result.X  — Pareto-optimal chromosomes  shape (n_solutions, N_FOODS)
                result.F  — Pareto-optimal objective values shape (n_solutions, 3)
                            columns: [negated_preference, cost, time]
    history : list of dicts, one per generation, with keys:
                'gen', 'hypervolume', 'n_solutions'
    """
    # Import here to avoid circular dependency before Member 3 is ready
    from problem.diet_problem import DietProblem

    food_ids = np.array(data["foods"]["id"].tolist(), dtype=int)

    problem  = DietProblem(data=data, user_id=user_id)
    sampling = SplitPermutationSampling(food_ids)

    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=sampling,
        crossover=SplitOXCrossover(p_c=P_CROSSOVER),
        mutation=SplitSwapMutation(p_m_b=P_MUTATION_B, p_m_ld=P_MUTATION_LD),
        eliminate_duplicates=False,   # permutations — exact duplicates are rare
    )

    termination = get_termination("n_gen", N_GENERATIONS)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=True,
        verbose=True,
    )

    history = _extract_history(result)

    return result, history


# ── Extract per-generation hypervolume history ────────────────────────────────

def _extract_history(result) -> list[dict]:
    """
    Walk through the saved algorithm history and record:
      - generation number
      - number of non-dominated solutions
      - hypervolume (computed lazily once reference point is known)

    Actual HV values are filled in by experiments/hypervolume.py after all
    runs complete and the shared reference point is established.
    """
    history = []
    for gen, algo in enumerate(result.history):
        pop_F = algo.pop.get("F")
        history.append({
            "gen": gen + 1,
            "n_solutions": len(pop_F),
            "F": pop_F,           # raw objective matrix — HV computed later
            "hypervolume": None,  # filled in by hypervolume.py
        })
    return history


# ── Quick standalone test (run this file directly to verify the operators) ────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    print("NSGA-II operator wiring test")
    print(f"  Population size : {POP_SIZE}")
    print(f"  Generations     : {N_GENERATIONS}")
    print(f"  Crossover prob  : {P_CROSSOVER}")
    print(f"  Mutation prob B : {P_MUTATION_B:.5f}  (1/{round(1/P_MUTATION_B)})")
    print(f"  Mutation prob LD: {P_MUTATION_LD:.5f}  (1/{round(1/P_MUTATION_LD)})")
    print()
    print("To run a full experiment call run_nsga2(data, user_id) after")
    print("Member 1 (db/loader.py) and Member 3 (problem/diet_problem.py) are ready.")
