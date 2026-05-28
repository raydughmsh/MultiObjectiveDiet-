"""
Member 2 — Greedy Decoder
problem/decoder.py

Converts a chromosome (permutation of food IDs) into an actual daily menu
by scanning genes left-to-right and greedily selecting foods that keep
nutrient totals within the ε-relaxed DRI bounds.

Two-pass decoding
─────────────────
Pass 1 — Breakfast (genes[0:94]):
    • Only checks Energy (C1) and Protein (C2)
    • Target = 35% of daily DRI for those two nutrients
    • Upper bound: RUL_b = RUL × BREAKFAST_RATIO × EPS_RUL  (i.e. ×0.35×1.15)
    • Lower bound: RLL_b = RLL × BREAKFAST_RATIO × EPS_RLL  (i.e. ×0.35×0.90)
    • Skip a food if it would push either nutrient OVER its upper bound
    • Stop early once BOTH lower bounds are satisfied

Pass 2 — Lunch+dinner (genes[94:405]):
    • Checks all 5 nutrients: Energy, Protein, Carbohydrate, Fiber, Sodium
    • Carries over nutrient totals from breakfast
    • Upper bound: RUL_full = RUL × EPS_RUL  (×1.15)
    • Lower bound: RLL_full = RLL × EPS_RLL  (×0.90)
    • Same skip/stop logic as breakfast but for the full day

Diversity (Option B — penalty term):
    After decoding, count distinct food groups in the full menu.
    Returns α × (1 / distinct_group_count) as a penalty term.
    Member 3's penalty.py already handles this; the decoder exposes the
    distinct_group_count so the penalty can be applied externally if needed.

Interface
─────────
    decoder = Decoder(foods_df, food_nutrients_df, dri_df)
    selected_food_ids = decoder.decode(gene_ids)

    # gene_ids: list/array of food IDs in chromosome order (length 405)
    # returns:  list of food IDs that were actually selected for the daily menu
"""

import numpy as np
import pandas as pd
from config import (
    NUTRIENT_IDS,
    EPS_RUL,
    EPS_RLL,
    BREAKFAST_RATIO,
    N_BREAKFAST,
)


class Decoder:
    """
    Greedy decoder: chromosome gene order → selected daily menu.

    Parameters
    ----------
    foods_df : pd.DataFrame
        All 405 foods. Must have at least columns: ['id', 'foodGroupId'].
        (Loaded by Member 1's load_all_data → data["foods"])

    food_nutrients_df : pd.DataFrame
        Nutrient values per food.  Columns: ['foodId', 'nutrientId', 'value'].
        Filtered to only the 5 constrained nutrients by loader.py.
        (Loaded by Member 1 → data["food_nutrients"])

    dri_df : pd.DataFrame
        Daily Reference Intakes for the current user.  Columns: ['nutrientId', 'RLL', 'RUL'].
        Already filtered for the correct user age/gender by loader.py.
        (Loaded by Member 1 → data["dri"])
    """

    def __init__(self, foods_df: pd.DataFrame,
                 food_nutrients_df: pd.DataFrame,
                 dri_df: pd.DataFrame):

        self.foods_df          = foods_df
        self.food_nutrients_df = food_nutrients_df
        self.dri_df            = dri_df

        # Pre-build all lookup structures once (not per decode call)
        self._build_nutrient_lookup()
        self._build_dri_bounds()
        self._build_food_group_lookup()

    # ── Pre-computation ───────────────────────────────────────────────────────

    def _build_nutrient_lookup(self):
        """
        Build a fast dict: {food_id: {nutrient_id: value}} for the 5 nutrients.
        Only stores the nutrients we care about (NUTRIENT_IDS values).
        Missing (food_id, nutrient_id) pairs default to 0.0.
        """
        self._nutrient_lookup: dict[int, dict[int, float]] = {}

        required_nids = set(NUTRIENT_IDS.values())

        for _, row in self.food_nutrients_df.iterrows():
            fid = int(row["foodId"])
            nid = int(row["nutrientId"])
            if nid not in required_nids:
                continue
            if fid not in self._nutrient_lookup:
                self._nutrient_lookup[fid] = {}
            self._nutrient_lookup[fid][nid] = float(row["value"])

    def _build_dri_bounds(self):
        """
        Build {nutrient_id: (RLL, RUL)} from dri_df.
        These are the RAW DRI values — ε is applied during decode.
        """
        self._dri_bounds: dict[int, tuple[float, float]] = {}
        for _, row in self.dri_df.iterrows():
            nid = int(row["nutrientId"])
            if nid in NUTRIENT_IDS.values():
                self._dri_bounds[nid] = (float(row["RLL"]), float(row["RUL"]))

    def _build_food_group_lookup(self):
        """
        Build {food_id: food_group_id} for the diversity penalty calculation.
        """
        self._food_group: dict[int, int] = {}
        if "foodGroupId" in self.foods_df.columns:
            for _, row in self.foods_df.iterrows():
                fid = int(row["id"])
                gid = row["foodGroupId"]
                if pd.notna(gid):
                    self._food_group[fid] = int(gid)

    def _get_nutrient(self, food_id: int, nutrient_id: int) -> float:
        """Get a single nutrient value for a food (0.0 if not in DB)."""
        return self._nutrient_lookup.get(food_id, {}).get(nutrient_id, 0.0)

    # ── Main decode ───────────────────────────────────────────────────────────

    def decode(self, gene_ids) -> list:
        """
        Decode a chromosome (ordered list of food IDs) into a daily menu.

        Parameters
        ----------
        gene_ids : list or np.ndarray, length 405
            The chromosome's food IDs in gene order.
            gene_ids[0:94]   → breakfast candidates
            gene_ids[94:405] → lunch+dinner candidates

        Returns
        -------
        list of int
            Food IDs selected for the full day (breakfast + lunch+dinner),
            in selection order. Length varies depending on nutrient coverage.
        """
        energy_id  = NUTRIENT_IDS["energy"]
        protein_id = NUTRIENT_IDS["protein"]
        all_nids   = list(NUTRIENT_IDS.values())

        # Running nutrient totals across the whole day
        totals: dict[int, float] = {nid: 0.0 for nid in all_nids}

        selected: list[int] = []

        # ── Pass 1: Breakfast ─────────────────────────────────────────────────
        #
        # Check ONLY Energy + Protein against 35% of daily DRI.
        #
        # Effective breakfast bounds (ε applied here, NOT twice):
        #   upper: RUL × BREAKFAST_RATIO × EPS_RUL  = RUL × 0.35 × 1.15
        #   lower: RLL × BREAKFAST_RATIO × EPS_RLL  = RLL × 0.35 × 0.90

        breakfast_upper: dict[int, float] = {}
        breakfast_lower: dict[int, float] = {}

        for nid in [energy_id, protein_id]:
            if nid in self._dri_bounds:
                rll, rul = self._dri_bounds[nid]
                breakfast_upper[nid] = rul * BREAKFAST_RATIO * EPS_RUL
                breakfast_lower[nid] = rll * BREAKFAST_RATIO * EPS_RLL

        breakfast_genes = list(gene_ids[:N_BREAKFAST])

        for food_id in breakfast_genes:
            # Skip food if it would push Energy or Protein over breakfast upper bound
            skip = False
            for nid, upper in breakfast_upper.items():
                v = self._get_nutrient(food_id, nid)
                if totals[nid] + v > upper:
                    skip = True
                    break
            if skip:
                continue

            # Add food to menu and update totals
            selected.append(food_id)
            for nid in all_nids:
                totals[nid] += self._get_nutrient(food_id, nid)

            # Stop early if both breakfast lower bounds are satisfied
            if breakfast_lower and all(
                totals[nid] >= lower
                for nid, lower in breakfast_lower.items()
            ):
                break

        # ── Pass 2: Lunch + Dinner ────────────────────────────────────────────
        #
        # Check ALL 5 nutrients against the FULL daily DRI.
        # Nutrient totals carry over from breakfast.
        #
        # Effective full-day bounds:
        #   upper: RUL × EPS_RUL  = RUL × 1.15
        #   lower: RLL × EPS_RLL  = RLL × 0.90

        full_upper: dict[int, float] = {}
        full_lower: dict[int, float] = {}

        for nid in all_nids:
            if nid in self._dri_bounds:
                rll, rul = self._dri_bounds[nid]
                full_upper[nid] = rul * EPS_RUL
                full_lower[nid] = rll * EPS_RLL

        lunch_dinner_genes = list(gene_ids[N_BREAKFAST:])

        for food_id in lunch_dinner_genes:
            # Skip food if it would push ANY nutrient over its full-day upper bound
            skip = False
            for nid, upper in full_upper.items():
                v = self._get_nutrient(food_id, nid)
                if totals[nid] + v > upper:
                    skip = True
                    break
            if skip:
                continue

            # Add food to menu and update totals
            selected.append(food_id)
            for nid in all_nids:
                totals[nid] += self._get_nutrient(food_id, nid)

            # Stop early if ALL 5 full-day lower bounds are satisfied
            if full_lower and all(
                totals[nid] >= lower
                for nid, lower in full_lower.items()
            ):
                break

        return selected

    # ── Diversity helper ──────────────────────────────────────────────────────

    def count_distinct_food_groups(self, selected_foods: list) -> int:
        """
        Count the number of distinct food group IDs in a decoded menu.
        Used by penalty.py (Option B diversity term).

        Parameters
        ----------
        selected_foods : list of food IDs (output of decode())

        Returns
        -------
        int : number of distinct food groups (0 if menu is empty or no group data)
        """
        groups = {
            self._food_group[fid]
            for fid in selected_foods
            if fid in self._food_group
        }
        return len(groups)

    # ── Inspection helper (for reporting / menu tables) ───────────────────────

    def get_nutrient_totals(self, selected_foods: list) -> dict:
        """
        Return nutrient totals for a decoded menu.
        Useful for constraint_violations() and sample menu tables.

        Returns
        -------
        dict {nutrient_id: total_value}
        """
        totals = {nid: 0.0 for nid in NUTRIENT_IDS.values()}
        for food_id in selected_foods:
            for nid in NUTRIENT_IDS.values():
                totals[nid] += self._get_nutrient(food_id, nid)
        return totals


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to verify decoder logic against the live DB.
    Make sure config.py has your DB credentials before running.

        python -m problem.decoder
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from db.loader import load_all_data
    from problem.chromosome import Chromosome
    from config import NUTRIENT_IDS, USER1_ID, USER2_ID

    nutrient_names = {v: k for k, v in NUTRIENT_IDS.items()}

    for user_id in [USER1_ID, USER2_ID]:
        print(f"\n{'='*55}")
        print(f"  User {user_id}")
        print(f"{'='*55}")

        data = load_all_data(user_id)

        decoder = Decoder(
            foods_df=data["foods"],
            food_nutrients_df=data["food_nutrients"],
            dri_df=data["dri"],
        )

        # Test 5 random chromosomes
        food_ids = data["foods"]["id"].tolist()

        for trial in range(5):
            chrom = Chromosome(food_ids=food_ids)
            selected = decoder.decode(chrom.genes.tolist())
            totals   = decoder.get_nutrient_totals(selected)
            n_groups = decoder.count_distinct_food_groups(selected)

            print(f"\n  Trial {trial+1}: {len(selected)} foods selected, "
                  f"{n_groups} distinct food groups")

            for nid, val in totals.items():
                rll, rul = decoder._dri_bounds.get(nid, (0, 0))
                status = "OK" if rll * 0.9 <= val <= rul * 1.15 else "⚠ OUT"
                print(f"    {nutrient_names[nid]:15s}: {val:8.2f}  "
                      f"[{rll:.1f} – {rul:.1f}]  {status}")

    print("\n\nDecoder self-test complete.")
