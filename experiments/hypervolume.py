"""
Hypervolume (HV) indicator for the Multi-Objective Diet Optimization Problem.

Since the true Pareto front is unknown, IGD cannot be used.
HV with a fixed reference point is used instead — the same reference point
is shared across ALL runs so that results are comparable.

Reference point rule (from the project spec):
  For each objective, take the WORST value observed across all runs
  and add a 10% margin.

  e.g. if worst negated_preference = -5, worst cost = 120, worst time = 180:
       ref = [-5 * 0.90, 120 * 1.10, 180 * 1.10]   (note: sign for negated obj)

Objectives (column order in result.F):
  col 0 — negated preference  (MIN, so lower = better; worst = highest number)
  col 1 — cost                (MIN)
  col 2 — preparation time    (MIN)
"""

import numpy as np
from pymoo.indicators.hv import HV

import config


# ── Reference point utilities ─────────────────────────────────────────────────

def compute_reference_point(all_F_matrices: list[np.ndarray], margin: float = 0.10) -> np.ndarray:
    """
    Compute a shared reference point from multiple runs / algorithms.

    Parameters
    ----------
    all_F_matrices : list of 2-D float arrays, each shape (n_solutions, n_obj)
                     Collect result.F from every run before calling this.
    margin         : fractional margin added on top of each worst value (default 10%)

    Returns
    -------
    ref_point : 1-D float array of shape (n_obj,)
    """
    combined = np.vstack(all_F_matrices)          # stack all solutions together
    worst    = np.max(combined, axis=0)           # worst (highest) value per objective

    # For objectives that could be zero or negative, use abs before applying margin
    ref_point = worst + margin * np.abs(worst)
    ref_point = np.where(worst == 0, margin, ref_point)  # guard against zero

    return ref_point


def save_reference_point(ref_point: np.ndarray) -> None:
    """
    Persist the computed reference point back into config.py so every
    module can read it consistently.
    """
    config.HV_REFERENCE_POINT = ref_point.tolist()
    print(f"[hypervolume] Reference point set: {ref_point.tolist()}")


# ── Hypervolume calculation ───────────────────────────────────────────────────

def compute_hv(F: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute hypervolume of a Pareto front approximation.

    Parameters
    ----------
    F         : 2-D float array of shape (n_solutions, n_obj)
                Objective values (all minimization — preference is already negated)
    ref_point : 1-D float array of shape (n_obj,)

    Returns
    -------
    hv_value : float
    """
    ind = HV(ref_point=ref_point)
    return float(ind(F))


def fill_history_hv(history: list[dict], ref_point: np.ndarray) -> list[dict]:
    """
    Fill in the 'hypervolume' field for each generation in an algorithm history.

    Call this after the reference point is finalized (post all runs).

    Parameters
    ----------
    history   : list of dicts from nsga2._extract_history() or spea2._extract_history()
                Each dict must have key 'F' (numpy array of objective values).
    ref_point : shared reference point

    Returns
    -------
    history with 'hypervolume' fields populated.
    """
    for entry in history:
        F = entry.get("F")
        if F is not None and len(F) > 0:
            try:
                entry["hypervolume"] = compute_hv(F, ref_point)
            except Exception:
                entry["hypervolume"] = float("nan")
    return history


# ── Convenience: compute HV curve for one algorithm run ──────────────────────

def hv_curve(history: list[dict], ref_point: np.ndarray) -> tuple[list[int], list[float]]:
    """
    Return (generations, hv_values) arrays suitable for plotting.

    Parameters
    ----------
    history   : filled history list (after fill_history_hv)
    ref_point : shared reference point

    Returns
    -------
    gens : list of int
    hvs  : list of float
    """
    filled = fill_history_hv(history, ref_point)
    gens = [e["gen"] for e in filled]
    hvs  = [e["hypervolume"] for e in filled]
    return gens, hvs


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic 3-objective minimization example
    rng  = np.random.default_rng(0)
    F1   = rng.uniform(0, 1, (50, 3))
    F2   = rng.uniform(0.2, 1.2, (50, 3))

    ref  = compute_reference_point([F1, F2])
    print(f"Reference point : {ref}")
    print(f"HV of F1        : {compute_hv(F1, ref):.6f}")
    print(f"HV of F2        : {compute_hv(F2, ref):.6f}")
