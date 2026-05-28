"""
Member 3 — Penalty Function
Implements the nutritional constraint penalty R and the diversity
penalty term (Option B) for the Multi-Objective Diet Optimization Problem.

Penalty formula (from handout §4):
    viol_low_j  = max(0, RLL_j − v_j) / (RUL_j − RLL_j)
    viol_high_j = max(0, v_j − RUL_j) / (RUL_j − RLL_j)
    R = 0.7 × Σ viol_low_j  +  0.3 × Σ viol_high_j

Diversity penalty (Option B, §5):
    R_total = R  +  α × (1 / distinct_food_group_count)
"""

import numpy as np
from config import NUTRIENT_IDS, LAMBDA, ALPHA_DIVERSITY


# ─── helpers ──────────────────────────────────────────────────────────────────

def _compute_nutrient_totals(selected_foods: list, food_nutrients_df) -> dict:
    """
    Sum nutrient values for the 5 constrained nutrients over selected foods.

    Returns:
        dict {nutrient_id: total_value}
    """
    subset = food_nutrients_df[
        (food_nutrients_df["foodId"].isin(selected_foods)) &
        (food_nutrients_df["nutrientId"].isin(NUTRIENT_IDS.values()))
    ]
    totals = subset.groupby("nutrientId")["value"].sum().to_dict()
    # Ensure all 5 nutrients are present (default 0 if no data)
    return {nid: totals.get(nid, 0.0) for nid in NUTRIENT_IDS.values()}


def _get_dri_bounds(dri_df) -> dict:
    """
    Return {nutrient_id: (RLL, RUL)} from the pre-filtered DRI DataFrame.
    Member 1's load_all_data() already filters DRI for the correct user
    age/gender, so we just read the values.
    """
    bounds = {}
    for _, row in dri_df.iterrows():
        nid = int(row["nutrientId"])
        if nid in NUTRIENT_IDS.values():
            bounds[nid] = (float(row["RLL"]), float(row["RUL"]))
    return bounds


# ─── main penalty ─────────────────────────────────────────────────────────────

def compute_penalty(selected_foods: list,
                    food_nutrients_df,
                    dri_df,
                    foods_df=None,
                    use_diversity: bool = True) -> float:
    """
    Full penalty R (+ optional diversity term) for a decoded menu.

    Args:
        selected_foods   : list of selected food IDs (output of decoder)
        food_nutrients_df: DataFrame [foodId, nutrientId, value]
        dri_df           : DataFrame [nutrientId, RLL, RUL] — user-filtered
        foods_df         : DataFrame [id, foodGroupId] — needed only when
                           use_diversity=True
        use_diversity    : whether to add the Option-B diversity penalty

    Returns:
        float: λ × R_total  (ready to subtract from / add to objective)
    """
    if not selected_foods:
        # Empty menu → maximum under-nutrition penalty for all 5 nutrients
        # Each viol_low = 1.0 → R = 0.7 * 5 = 3.5
        R = 0.7 * len(NUTRIENT_IDS)
        return LAMBDA * R

    nutrient_totals = _compute_nutrient_totals(selected_foods, food_nutrients_df)
    dri_bounds = _get_dri_bounds(dri_df)

    viol_low_sum = 0.0
    viol_high_sum = 0.0

    for nid in NUTRIENT_IDS.values():
        if nid not in dri_bounds:
            continue  # skip if DRI missing for this user

        rll, rul = dri_bounds[nid]
        v = nutrient_totals.get(nid, 0.0)
        denom = rul - rll if (rul - rll) > 0 else 1.0  # safety guard

        viol_low_sum  += max(0.0, (rll - v) / denom)
        viol_high_sum += max(0.0, (v - rul) / denom)

    R = 0.7 * viol_low_sum + 0.3 * viol_high_sum

    # Diversity penalty (Option B)
    if use_diversity and foods_df is not None and len(selected_foods) > 0:
        selected_df = foods_df[foods_df["id"].isin(selected_foods)]
        distinct_groups = selected_df["foodGroupId"].nunique()
        if distinct_groups > 0:
            R += ALPHA_DIVERSITY * (1.0 / distinct_groups)

    return LAMBDA * R


# ─── penalized objective helper ───────────────────────────────────────────────

def penalized_objective(objective_value: float,
                        selected_foods: list,
                        food_nutrients_df,
                        dri_df,
                        foods_df=None,
                        use_diversity: bool = True,
                        is_maximization: bool = False) -> float:
    """
    Apply penalty to a single objective value.

    For pymoo (minimization):
        - minimization objectives (f2, f3): penalized = obj + λR
        - maximization objective (f1, already negated): penalized = obj + λR
          (adding penalty makes the negated value less negative → worse)

    Args:
        objective_value : raw objective (f1 already negated when passed in)
        is_maximization : unused here (kept for clarity), penalty is always added

    Returns:
        float: penalized objective value
    """
    penalty = compute_penalty(
        selected_foods, food_nutrients_df, dri_df, foods_df, use_diversity
    )
    return objective_value + penalty


# ─── violation summary (for reporting / menu table) ───────────────────────────

def constraint_violations(selected_foods: list,
                          food_nutrients_df,
                          dri_df) -> dict:
    """
    Returns a dict with per-nutrient compliance info for reporting.

    Keys: nutrient_id
    Values: dict with keys 'value', 'RLL', 'RUL', 'status'
            status ∈ {'OK', 'UNDER', 'OVER'}
    """
    from config import NUTRIENT_IDS  # local import to avoid circular deps

    nutrient_totals = _compute_nutrient_totals(selected_foods, food_nutrients_df)
    dri_bounds = _get_dri_bounds(dri_df)

    # Map id → name for readability
    id_to_name = {v: k for k, v in NUTRIENT_IDS.items()}

    result = {}
    for nid in NUTRIENT_IDS.values():
        if nid not in dri_bounds:
            continue
        rll, rul = dri_bounds[nid]
        v = nutrient_totals.get(nid, 0.0)
        if v < rll:
            status = "UNDER"
        elif v > rul:
            status = "OVER"
        else:
            status = "OK"
        result[id_to_name.get(nid, str(nid))] = {
            "value": round(v, 2),
            "RLL": rll,
            "RUL": rul,
            "status": status,
        }
    return result
