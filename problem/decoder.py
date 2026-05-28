"""
decoder.py — نسخة مبسطة للاختبار فقط
سيتم استبدالها بكود Member 2 الكامل لاحقاً
"""

import pandas as pd
from config import (
    NUTRIENT_IDS, EPS_RUL, EPS_RLL,
    BREAKFAST_RATIO, N_BREAKFAST
)


class Decoder:
    def __init__(self, foods_df, food_nutrients_df, dri_df):
        self.foods_df          = foods_df
        self.food_nutrients_df = food_nutrients_df
        self.dri_df            = dri_df
        self._build_dri_bounds()

    def _build_dri_bounds(self):
        """Build {nutrient_id: (RLL, RUL)} from dri_df."""
        self.dri_bounds = {}
        for _, row in self.dri_df.iterrows():
            nid = int(row["nutrientId"])
            if nid in NUTRIENT_IDS.values():
                self.dri_bounds[nid] = (float(row["RLL"]), float(row["RUL"]))

    def _nutrient_value(self, food_id, nutrient_id):
        """Get nutrient value for a food item."""
        row = self.food_nutrients_df[
            (self.food_nutrients_df["foodId"] == food_id) &
            (self.food_nutrients_df["nutrientId"] == nutrient_id)
        ]
        return float(row["value"].values[0]) if len(row) > 0 else 0.0

    def decode(self, gene_ids: list) -> list:
        """
        Greedy decoder — simplified version for testing.

        Breakfast part (first N_BREAKFAST genes):
            Uses Energy + Protein only, targeting 35% of daily DRI.

        Lunch+dinner part (remaining genes):
            Uses all 5 nutrients, carries over from breakfast.

        Returns:
            list of selected food IDs
        """
        energy_id  = NUTRIENT_IDS["energy"]
        protein_id = NUTRIENT_IDS["protein"]

        selected = []
        nutrient_totals = {nid: 0.0 for nid in NUTRIENT_IDS.values()}

        # ── Breakfast pass ────────────────────────────────────────────────────
        breakfast_genes = gene_ids[:N_BREAKFAST]

        b_targets = {}
        for nid in [energy_id, protein_id]:
            if nid in self.dri_bounds:
                rll, rul = self.dri_bounds[nid]
                b_targets[nid] = (rll * BREAKFAST_RATIO, rul * BREAKFAST_RATIO)

        for food_id in breakfast_genes:
            # Check if adding this food exceeds breakfast upper bounds
            skip = False
            for nid, (b_rll, b_rul) in b_targets.items():
                v = self._nutrient_value(food_id, nid)
                if nutrient_totals[nid] + v > b_rul * EPS_RUL:
                    skip = True
                    break
            if skip:
                continue

            selected.append(food_id)
            for nid in NUTRIENT_IDS.values():
                nutrient_totals[nid] += self._nutrient_value(food_id, nid)

            # Stop if breakfast lower bounds satisfied
            if all(
                nutrient_totals[nid] >= b_rll * EPS_RLL
                for nid, (b_rll, _) in b_targets.items()
            ):
                break

        # ── Lunch+dinner pass ─────────────────────────────────────────────────
        lunch_dinner_genes = gene_ids[N_BREAKFAST:]

        for food_id in lunch_dinner_genes:
            skip = False
            for nid, (rll, rul) in self.dri_bounds.items():
                v = self._nutrient_value(food_id, nid)
                if nutrient_totals[nid] + v > rul * EPS_RUL:
                    skip = True
                    break
            if skip:
                continue

            selected.append(food_id)
            for nid in NUTRIENT_IDS.values():
                nutrient_totals[nid] += self._nutrient_value(food_id, nid)

            # Stop if all daily lower bounds satisfied
            if all(
                nutrient_totals[nid] >= rll * EPS_RLL
                for nid, (rll, _) in self.dri_bounds.items()
            ):
                break

        return selected
