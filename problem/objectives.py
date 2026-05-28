"""
Member 3 — Objectives
Implements f1_preference (MAX), f2_cost (MIN), f3_time (MIN)
for the Multi-Objective Diet Optimization Problem.

NOTE: pymoo minimizes everything, so f1 is returned as -preference
      so that maximizing preference == minimizing (-preference).
"""

import numpy as np


def f1_preference(selected_foods: list, foods_df) -> float:
    """
    Objective 1 — Maximize user preference.
    Returns NEGATIVE value because pymoo minimizes.

    Args:
        selected_foods: list of food IDs chosen by the decoder
        foods_df: DataFrame with at least columns ['id', 'final_preference']
                  (use foods["final_preference"] as instructed by Member 1)

    Returns:
        float: -sum(final_preference) for selected foods
    """
    if not selected_foods:
        return 0.0

    selected = foods_df[foods_df["id"].isin(selected_foods)]
    total_preference = selected["final_preference"].sum()
    return -float(total_preference)  # negate → pymoo minimizes


def f2_cost(selected_foods: list, foods_df) -> float:
    """
    Objective 2 — Minimize cost.

    Args:
        selected_foods: list of food IDs chosen by the decoder
        foods_df: DataFrame with at least columns ['id', 'cost']

    Returns:
        float: sum(cost) for selected foods
    """
    if not selected_foods:
        return 0.0

    selected = foods_df[foods_df["id"].isin(selected_foods)]
    total_cost = selected["cost"].sum()
    return float(total_cost)


def f3_time(selected_foods: list, foods_df) -> float:
    """
    Objective 3 — Minimize total preparation + cooking time.

    Args:
        selected_foods: list of food IDs chosen by the decoder
        foods_df: DataFrame with at least columns ['id', 'preparingTime', 'cookingTime']
                  cookingTime may be NULL → treated as 0.

    Returns:
        float: sum(preparingTime + cookingTime) for selected foods
    """
    if not selected_foods:
        return 0.0

    selected = foods_df[foods_df["id"].isin(selected_foods)].copy()
    # cookingTime can be NULL in the DB; fill with 0
    selected["cookingTime"] = selected["cookingTime"].fillna(0.0)
    total_time = (selected["preparingTime"] + selected["cookingTime"]).sum()
    return float(total_time)


def evaluate_all_objectives(selected_foods: list, foods_df) -> np.ndarray:
    """
    Convenience: evaluate all 3 objectives at once.

    Returns:
        np.ndarray shape (3,): [f1, f2, f3]
        f1 is negated (pymoo minimizes), f2 and f3 are raw minimization values.
    """
    return np.array([
        f1_preference(selected_foods, foods_df),
        f2_cost(selected_foods, foods_df),
        f3_time(selected_foods, foods_df),
    ])
