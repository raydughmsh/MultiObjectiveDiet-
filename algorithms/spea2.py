"""
Member 5 — SPEA2 runner for the Multi-Objective Diet Optimization Problem.

Uses pymoo's built-in SPEA2 algorithm with the same custom operators as NSGA-II:
  - SplitOXCrossover  (operators/crossover.py)
  - SplitSwapMutation (operators/mutation.py)

Usage
-----
    from algorithms.spea2 import run_spea2
    from db.loader import load_all_data

    data = load_all_data(user_id=1)
    result, history = run_spea2(data, user_id=1)
"""

import numpy as np
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.sampling import Sampling

from operators.crossover import SplitOXCrossover
from operators.mutation import SplitSwapMutation
from config import POP_SIZE, N_GENERATIONS, P_CROSSOVER, P_MUTATION_B, P_MUTATION_LD


class SplitPermutationSampling(Sampling):
    """
    Initializes each individual as a random permutation of food IDs.
    """

    def __init__(self, food_ids: np.ndarray):
        super().__init__()
        self.food_ids = food_ids

    def _do(self, problem, n_samples, **kwargs):
        n = len(self.food_ids)
        X = np.empty((n_samples, n), dtype=int)
        rng = np.random.default_rng()
        for i in range(n_samples):
            X[i] = rng.permutation(n)   # indices 0..N-1, not actual food IDs
        return X


def run_spea2(data: dict, user_id: int, seed: int = 42, use_diversity: bool = True) -> tuple:
    """
    Run SPEA2 for a given user and return the result + per-generation history.

    Parameters
    ----------
    data          : dict returned by load_all_data(user_id)
    user_id       : int (1 = non-vegetarian, 2 = vegetarian)
    seed          : random seed for reproducibility
    use_diversity : whether to enable Option-B diversity penalty in DietProblem

    Returns
    -------
    result  : pymoo Result object
                result.X — Pareto-optimal chromosomes  shape (n_solutions, N_FOODS)
                result.F — Pareto-optimal objective values shape (n_solutions, 3)
    history : list of dicts per generation: 'gen', 'n_solutions', 'F', 'hypervolume'
    """
    from problem.diet_problem import DietProblem

    food_ids = np.array(data["foods"]["id"].tolist(), dtype=int)

    problem  = DietProblem(data=data, use_diversity=use_diversity)
    sampling = SplitPermutationSampling(food_ids)

    algorithm = SPEA2(
        pop_size=POP_SIZE,
        sampling=sampling,
        crossover=SplitOXCrossover(p_c=P_CROSSOVER),
        mutation=SplitSwapMutation(p_m_b=P_MUTATION_B, p_m_ld=P_MUTATION_LD),
        eliminate_duplicates=False,
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


def _extract_history(result) -> list:
    """
    Walk through the saved algorithm history and record per-generation stats.
    Hypervolume values are filled in later by experiments/hypervolume.py.
    """
    history = []
    for gen, algo in enumerate(result.history):
        pop_F = algo.pop.get("F")
        history.append({
            "gen": gen + 1,
            "n_solutions": len(pop_F),
            "F": pop_F,
            "hypervolume": None,
        })
    return history
