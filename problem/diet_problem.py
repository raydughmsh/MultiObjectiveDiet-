"""
Member 3 — DietProblem
pymoo Problem wrapper that connects:
    Chromosome gene array  →  Decoder  →  Objectives + Penalty

Inherits from pymoo's ElementwiseProblem so each solution is
evaluated independently (easier to debug, fine for our pop size).

Usage (Members 4 & 5):
    from problem.diet_problem import DietProblem
    problem = DietProblem(data, use_diversity=True)
    # pass to NSGA-II / SPEA2 via pymoo minimize()
"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from problem.decoder import Decoder
from problem.objectives import evaluate_all_objectives
from problem.penalty import penalized_objective, compute_penalty


N_FOODS = 405          # total food items → gene length
N_OBJ   = 3            # f1 (pref, negated), f2 (cost), f3 (time)
N_CONSTR = 0           # hard constraints handled via penalty


class DietProblem(ElementwiseProblem):
    """
    Multi-Objective Diet Optimization Problem for pymoo.

    Decision variable:
        x — integer permutation of [0, N_FOODS-1] (indices into foods list)
        Mapped internally to food IDs by the decoder.

    Objectives (all minimized by pymoo):
        F[0] = -f1_preference  (negated → pymoo minimizes)
        F[1] =  f2_cost
        F[2] =  f3_time

    Penalty:
        Each objective gets λ·R added to it so infeasible solutions
        are pushed to worse positions in the Pareto front.

    Args:
        data          : dict returned by load_all_data() (Member 1)
        use_diversity : whether to include Option-B diversity penalty
    """

    def __init__(self, data: dict, use_diversity: bool = True, **kwargs):
        # Integer permutation: lower bound 0, upper bound N_FOODS-1
        super().__init__(
            n_var=N_FOODS,
            n_obj=N_OBJ,
            n_ieq_constr=N_CONSTR,
            xl=np.zeros(N_FOODS, dtype=int),
            xu=np.full(N_FOODS, N_FOODS - 1, dtype=int),
            **kwargs
        )

        self.foods_df         = data["foods"]
        self.food_nutrients_df = data["food_nutrients"]
        self.dri_df           = data["dri"]
        self.food_groups_df   = data["food_groups"]
        self.use_diversity    = use_diversity

        # Decoder instance (Member 2) — shared, stateless
        self._decoder = Decoder(
            foods_df=self.foods_df,
            food_nutrients_df=self.food_nutrients_df,
            dri_df=self.dri_df,
        )

        # Cache food IDs in the order the DB returns them
        self._food_ids = self.foods_df["id"].tolist()  # length = 405

    # ── pymoo entry point ────────────────────────────────────────────────────

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Called by pymoo for each individual.

        x : np.ndarray, shape (405,) — a permutation of integer indices
        """
        # 1. Map integer indices → food IDs
        gene_ids = [self._food_ids[int(i)] for i in x]

        # 2. Decode to an actual menu (selected food IDs)
        selected_foods = self._decoder.decode(gene_ids)

        # 3. Compute raw objectives
        raw_objs = evaluate_all_objectives(selected_foods, self.foods_df)

        # 4. Compute penalty (once, shared across all objectives)
        penalty = compute_penalty(
            selected_foods,
            self.food_nutrients_df,
            self.dri_df,
            self.foods_df,
            use_diversity=self.use_diversity,
        )

        # 5. Apply penalty: add λR to every objective
        #    (f1 is already negated, so adding penalty makes it worse → correct)
        penalized = raw_objs + penalty

        out["F"] = penalized

    # ── convenience: decode a solution for reporting ─────────────────────────

    def decode_solution(self, x: np.ndarray) -> dict:
        """
        Given a chromosome array x, return a rich dict with:
            - selected_food_ids
            - selected_food_names
            - objectives (raw, before penalty)
            - penalty
            - nutrient_totals vs DRI
        """
        from problem.penalty import constraint_violations

        gene_ids = [self._food_ids[int(i)] for i in x]
        selected_foods = self._decoder.decode(gene_ids)

        selected_df = self.foods_df[self.foods_df["id"].isin(selected_foods)]

        raw_objs = evaluate_all_objectives(selected_foods, self.foods_df)
        pen = compute_penalty(
            selected_foods,
            self.food_nutrients_df,
            self.dri_df,
            self.foods_df,
            use_diversity=self.use_diversity,
        )
        violations = constraint_violations(
            selected_foods, self.food_nutrients_df, self.dri_df
        )

        return {
            "selected_food_ids": selected_foods,
            "selected_food_names": selected_df["name"].tolist(),
            "n_foods_selected": len(selected_foods),
            "f1_preference_raw": -raw_objs[0],   # un-negate for display
            "f2_cost": raw_objs[1],
            "f3_time": raw_objs[2],
            "penalty": pen,
            "nutrient_violations": violations,
            "n_groups": selected_df["foodGroupId"].nunique()
                        if "foodGroupId" in selected_df.columns else None,
        }
