"""
Member 5 — Experiments Runner

Runs all 3 required experiments and exports results to results/.

Experiment 1 — User Comparison
    NSGA-II on User 1 (non-vegetarian) vs User 2 (vegetarian).

Experiment 2 — Algorithm Comparison
    NSGA-II vs SPEA2 on User 1.

Experiment 3 — Diversity Impact
    NSGA-II on User 1 with diversity penalty vs without.

Usage
-----
    python experiments/run_experiments.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.loader import load_all_data
from algorithms.nsga2 import run_nsga2
from algorithms.spea2 import run_spea2
from experiments.hypervolume import (
    compute_reference_point, save_reference_point,
    fill_history_hv, compute_hv,
)
from visualization.pareto_plot import plot_pareto_2d, plot_pareto_overlay
from visualization.convergence import plot_convergence, plot_convergence_overlay
from visualization.menu_table import build_menu_table, save_menu_table

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_pareto_csv(result, filename: str, label: str = ""):
    F = result.F
    df = pd.DataFrame(F, columns=["neg_preference", "cost", "time"])
    df["preference"] = -df["neg_preference"]
    df.drop(columns=["neg_preference"], inplace=True)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[saved] {filename}  ({len(df)} Pareto solutions{' — ' + label if label else ''})")
    return df


def _save_history_csv(history: list, filename: str):
    rows = [{"gen": e["gen"], "hypervolume": e["hypervolume"],
             "n_solutions": e["n_solutions"]} for e in history]
    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[saved] {filename}")
    return df


def _save_summary_json(summary: dict, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {filename}")


# ── Experiment 1 — User Comparison ───────────────────────────────────────────

def experiment1_user_comparison():
    print("\n" + "="*60)
    print("Experiment 1 — User Comparison (NSGA-II: User1 vs User2)")
    print("="*60)

    data1 = load_all_data(user_id=1)
    data2 = load_all_data(user_id=2)

    result1, hist1 = run_nsga2(data1, user_id=1, seed=42)
    result2, hist2 = run_nsga2(data2, user_id=2, seed=42)

    ref = compute_reference_point([result1.F, result2.F])
    save_reference_point(ref)

    fill_history_hv(hist1, ref)
    fill_history_hv(hist2, ref)

    hv1 = compute_hv(result1.F, ref)
    hv2 = compute_hv(result2.F, ref)

    _save_pareto_csv(result1, "exp1_user1_nsga2_pareto.csv", "User1")
    _save_pareto_csv(result2, "exp1_user2_nsga2_pareto.csv", "User2")
    _save_history_csv(hist1, "exp1_user1_nsga2_history.csv")
    _save_history_csv(hist2, "exp1_user2_nsga2_history.csv")

    _save_summary_json({
        "experiment": "User Comparison",
        "algorithm": "NSGA-II",
        "user1": {"id": 1, "type": "non-vegetarian", "hypervolume": hv1, "n_pareto": len(result1.F)},
        "user2": {"id": 2, "type": "vegetarian",     "hypervolume": hv2, "n_pareto": len(result2.F)},
        "reference_point": ref.tolist(),
    }, "exp1_summary.json")

    plot_pareto_2d(result1.F,
                   title="Exp1: User1 (non-veg) NSGA-II Pareto Front",
                   filename=os.path.join(RESULTS_DIR, "exp1_user1_pareto.png"))
    plot_pareto_2d(result2.F,
                   title="Exp1: User2 (vegetarian) NSGA-II Pareto Front",
                   filename=os.path.join(RESULTS_DIR, "exp1_user2_pareto.png"))
    plot_pareto_overlay(
        [result1.F, result2.F],
        labels=["User1 (non-veg)", "User2 (veg)"],
        title="Exp1: User Comparison — NSGA-II Pareto Fronts",
        filename=os.path.join(RESULTS_DIR, "exp1_overlay_pareto.png"),
    )
    plot_convergence_overlay(
        [hist1, hist2],
        labels=["User1", "User2"],
        title="Exp1: User Comparison — HV Convergence",
        filename=os.path.join(RESULTS_DIR, "exp1_convergence.png"),
    )

    for user_id, result, data in [(1, result1, data1), (2, result2, data2)]:
        table = build_menu_table(result, data, n_solutions=3)
        save_menu_table(table, os.path.join(RESULTS_DIR, f"exp1_user{user_id}_menu_table.csv"))

    print(f"\nExp1 done — HV User1={hv1:.4f}  HV User2={hv2:.4f}")
    return result1, hist1, result2, hist2, ref


# ── Experiment 2 — Algorithm Comparison ──────────────────────────────────────

def experiment2_algorithm_comparison(ref_point=None):
    print("\n" + "="*60)
    print("Experiment 2 — Algorithm Comparison (User1: NSGA-II vs SPEA2)")
    print("="*60)

    data1 = load_all_data(user_id=1)

    result_nsga2, hist_nsga2 = run_nsga2(data1, user_id=1, seed=42)
    result_spea2, hist_spea2 = run_spea2(data1, user_id=1, seed=42)

    if ref_point is None:
        ref_point = compute_reference_point([result_nsga2.F, result_spea2.F])

    fill_history_hv(hist_nsga2, ref_point)
    fill_history_hv(hist_spea2, ref_point)

    hv_nsga2 = compute_hv(result_nsga2.F, ref_point)
    hv_spea2 = compute_hv(result_spea2.F, ref_point)

    _save_pareto_csv(result_nsga2, "exp2_user1_nsga2_pareto.csv", "NSGA-II")
    _save_pareto_csv(result_spea2, "exp2_user1_spea2_pareto.csv", "SPEA2")
    _save_history_csv(hist_nsga2, "exp2_user1_nsga2_history.csv")
    _save_history_csv(hist_spea2, "exp2_user1_spea2_history.csv")

    _save_summary_json({
        "experiment": "Algorithm Comparison",
        "user": {"id": 1, "type": "non-vegetarian"},
        "nsga2": {"hypervolume": hv_nsga2, "n_pareto": len(result_nsga2.F)},
        "spea2": {"hypervolume": hv_spea2, "n_pareto": len(result_spea2.F)},
        "reference_point": ref_point.tolist(),
    }, "exp2_summary.json")

    plot_pareto_overlay(
        [result_nsga2.F, result_spea2.F],
        labels=["NSGA-II", "SPEA2"],
        title="Exp2: Algorithm Comparison — User1 Pareto Fronts",
        filename=os.path.join(RESULTS_DIR, "exp2_overlay_pareto.png"),
    )
    plot_convergence_overlay(
        [hist_nsga2, hist_spea2],
        labels=["NSGA-II", "SPEA2"],
        title="Exp2: Algorithm Comparison — HV Convergence (User1)",
        filename=os.path.join(RESULTS_DIR, "exp2_convergence.png"),
    )

    table = build_menu_table(result_spea2, data1, n_solutions=3)
    save_menu_table(table, os.path.join(RESULTS_DIR, "exp2_spea2_menu_table.csv"))

    print(f"\nExp2 done — HV NSGA-II={hv_nsga2:.4f}  HV SPEA2={hv_spea2:.4f}")
    return result_nsga2, hist_nsga2, result_spea2, hist_spea2, ref_point


# ── Experiment 3 — Diversity Impact ──────────────────────────────────────────

def experiment3_diversity_impact(ref_point=None):
    print("\n" + "="*60)
    print("Experiment 3 — Diversity Impact (NSGA-II User1: with vs without)")
    print("="*60)

    data1 = load_all_data(user_id=1)

    result_div,   hist_div   = run_nsga2(data1, user_id=1, seed=42, use_diversity=True)
    result_nodiv, hist_nodiv = run_nsga2(data1, user_id=1, seed=42, use_diversity=False)

    if ref_point is None:
        ref_point = compute_reference_point([result_div.F, result_nodiv.F])

    fill_history_hv(hist_div,   ref_point)
    fill_history_hv(hist_nodiv, ref_point)

    hv_div   = compute_hv(result_div.F,   ref_point)
    hv_nodiv = compute_hv(result_nodiv.F, ref_point)

    _save_pareto_csv(result_div,   "exp3_diversity_on_pareto.csv",  "diversity=ON")
    _save_pareto_csv(result_nodiv, "exp3_diversity_off_pareto.csv", "diversity=OFF")
    _save_history_csv(hist_div,   "exp3_diversity_on_history.csv")
    _save_history_csv(hist_nodiv, "exp3_diversity_off_history.csv")

    _save_summary_json({
        "experiment": "Diversity Impact",
        "algorithm": "NSGA-II",
        "user": {"id": 1, "type": "non-vegetarian"},
        "with_diversity":    {"hypervolume": hv_div,   "n_pareto": len(result_div.F)},
        "without_diversity": {"hypervolume": hv_nodiv, "n_pareto": len(result_nodiv.F)},
        "reference_point": ref_point.tolist(),
    }, "exp3_summary.json")

    plot_pareto_overlay(
        [result_div.F, result_nodiv.F],
        labels=["With Diversity", "Without Diversity"],
        title="Exp3: Diversity Impact — NSGA-II User1 Pareto Fronts",
        filename=os.path.join(RESULTS_DIR, "exp3_overlay_pareto.png"),
    )
    plot_convergence_overlay(
        [hist_div, hist_nodiv],
        labels=["With Diversity", "Without Diversity"],
        title="Exp3: Diversity Impact — HV Convergence",
        filename=os.path.join(RESULTS_DIR, "exp3_convergence.png"),
    )

    print(f"\nExp3 done — HV with-div={hv_div:.4f}  HV no-div={hv_nodiv:.4f}")
    return result_div, hist_div, result_nodiv, hist_nodiv, ref_point


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting all experiments...")

    _, _, _, _, ref = experiment1_user_comparison()
    experiment2_algorithm_comparison(ref_point=ref)
    experiment3_diversity_impact(ref_point=ref)

    print("\n" + "="*60)
    print("All experiments complete. Results saved to results/")
    print("="*60)
