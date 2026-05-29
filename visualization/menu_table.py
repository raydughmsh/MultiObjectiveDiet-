"""
Member 5 — Sample Menu Tables

Builds human-readable tables for >= 3 Pareto-optimal solutions showing:
  - Selected foods
  - Objective values (preference, cost, time)
  - Nutrient totals vs DRI bounds with UNDER/OK/OVER status
"""

import os
import numpy as np
import pandas as pd


def build_menu_table(result, data: dict, n_solutions: int = 3) -> list:
    """
    Build a menu table from a pymoo result object.

    Parameters
    ----------
    result      : pymoo Result with .X (chromosomes) and .F (objectives)
    data        : dict from load_all_data() — keys: foods, food_nutrients, dri
    n_solutions : how many Pareto solutions to include (default 3)

    Returns
    -------
    list of dicts, one per solution
    """
    from problem.decoder import Decoder
    from problem.penalty import _compute_nutrient_totals, _get_dri_bounds
    from config import NUTRIENT_IDS

    foods_df          = data["foods"]
    food_nutrients_df = data["food_nutrients"]
    dri_df            = data["dri"]

    n_total = len(result.F)
    if n_total <= n_solutions:
        indices = list(range(n_total))
    else:
        indices = np.linspace(0, n_total - 1, n_solutions, dtype=int).tolist()

    food_ids = foods_df["id"].tolist()

    decoder = Decoder(
        foods_df=foods_df,
        food_nutrients_df=food_nutrients_df,
        dri_df=dri_df,
    )

    dri_bounds = _get_dri_bounds(dri_df)
    id_to_name = {v: k for k, v in NUTRIENT_IDS.items()}

    table = []
    for sol_idx, i in enumerate(indices):
        x = result.X[i]
        F = result.F[i]

        gene_ids = [food_ids[int(g)] for g in x]
        selected = decoder.decode(gene_ids)

        selected_df = foods_df[foods_df["id"].isin(selected)]
        food_names  = selected_df["name"].tolist()

        nutrient_totals = _compute_nutrient_totals(selected, food_nutrients_df)

        nutrients_info = {}
        for nid in NUTRIENT_IDS.values():
            if nid not in dri_bounds:
                continue
            rll, rul = dri_bounds[nid]
            v = nutrient_totals.get(nid, 0.0)
            status = "UNDER" if v < rll else ("OVER" if v > rul else "OK")
            nutrients_info[id_to_name.get(nid, str(nid))] = {
                "value": round(v, 2),
                "RLL": rll,
                "RUL": rul,
                "status": status,
            }

        table.append({
            "solution_id": sol_idx + 1,
            "preference": round(-float(F[0]), 4),
            "cost": round(float(F[1]), 4),
            "time": round(float(F[2]), 4),
            "n_foods": len(selected),
            "foods": food_names,
            "nutrients": nutrients_info,
        })

    return table


def save_menu_table(table: list, filename: str):
    """Save menu table to CSV — one row per solution."""
    rows = []
    for sol in table:
        row = {
            "solution_id": sol["solution_id"],
            "preference":  sol["preference"],
            "cost":        sol["cost"],
            "time":        sol["time"],
            "n_foods":     sol["n_foods"],
            "foods":       " | ".join(sol["foods"]),
        }
        for nutrient, info in sol["nutrients"].items():
            row[f"{nutrient}_value"]  = info["value"]
            row[f"{nutrient}_RLL"]    = info["RLL"]
            row[f"{nutrient}_RUL"]    = info["RUL"]
            row[f"{nutrient}_status"] = info["status"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"[saved] {os.path.basename(filename)}  ({len(df)} solutions)")


def print_menu_table(table: list):
    """Pretty-print the menu table to console."""
    for sol in table:
        print(f"\n{'─'*60}")
        print(f"  Solution #{sol['solution_id']}")
        print(f"  Preference: {sol['preference']:.2f}  |  "
              f"Cost: {sol['cost']:.2f}  |  "
              f"Time: {sol['time']:.0f} min")
        print(f"  Foods selected ({sol['n_foods']}):")
        for name in sol["foods"]:
            print(f"    - {name}")
        print(f"  Nutrient totals vs DRI:")
        for nutrient, info in sol["nutrients"].items():
            flag = "OK" if info["status"] == "OK" else info["status"]
            print(f"    [{flag}] {nutrient:20s}: {info['value']:8.2f}  "
                  f"[{info['RLL']:.1f} - {info['RUL']:.1f}]")
    print(f"\n{'─'*60}")
