# Multi-Objective Diet Optimization Problem
## Team Assignments — BLM20364E / BLM22332E Heuristic Optimization Algorithms

---

## Member 1 — Database & Data Layer

**Responsibilities:**
- Import `diet.sql` into MySQL, identify User 1 (non-vegetarian) and User 2 (vegetarian)
- Write all DB query functions: foods, nutrients, DRI (age/gender lookup), user preferences, food groups
- Handle NULL preferences in `user_foods` (fallback to `foods.preference`)
- `config.py` with DB credentials, nutrient IDs (C1–C5), and hyperparameters (λ, ε, p_c, p_m)

**Files to deliver:**
- `db/loader.py`
- `config.py`

---

## Member 2 — Chromosome & Decoder

**Responsibilities:**
- `Chromosome` class: 405-gene permutation split into `[0:94]` breakfast part and `[94:405]` lunch+dinner part
- Greedy decoder:
  - Breakfast pass: Energy + Protein only, 35% of daily DRI with ε tolerance (RUL×1.15, RLL×0.90)
  - Lunch+dinner pass: all 5 nutrients, carry over nutrient totals from breakfast
- Diversity mechanism (Option B): penalty term `α × (1 / distinct_food_group_count)`

**Files to deliver:**
- `problem/chromosome.py`
- `problem/decoder.py`

---

## Member 3 — Objectives & Penalty Function

**Responsibilities:**
- Implement 3 objective functions:
  - `f1_preference` → MAX (mandatory)
  - `f2_cost` → MIN
  - `f3_time` → MIN (preparingTime + cookingTime)
- Full penalty function R with 0.7 (under-nutrition) / 0.3 (over-nutrition) weighting
- Tune penalty weight λ experimentally (start with λ = 1.0)
- `DietProblem` pymoo wrapper class connecting decoder → objectives → penalty

**Files to deliver:**
- `problem/objectives.py`
- `problem/penalty.py`
- `problem/diet_problem.py`

---

## Member 4 — Genetic Operators & NSGA-II

**Responsibilities:**
- OX (Order Crossover) applied independently to each chromosome half (p_c = 0.9)
- Swap mutation within each chromosome half (p_m = 1/n)
- Binary tournament selection
- Implement and run **NSGA-II** with custom operators via pymoo for both users
- Hypervolume calculator with fixed reference point (worst observed values + 10% margin)

**Files to deliver:**
- `operators/crossover.py`
- `operators/mutation.py`
- `algorithms/nsga2.py`
- `experiments/hypervolume.py`

---

## Member 5 — SPEA2, Experiments & Visualization

**Responsibilities:**
- Implement and run **SPEA2** with the same custom operators via pymoo for both users
- Run all 3 required experiments:
  1. User comparison — User 1 vs. User 2, same algorithm
  2. Algorithm comparison — NSGA-II vs. SPEA2, same user
  3. Diversity impact — with vs. without diversity mechanism
- All visualizations:
  - Pareto front scatter plots (per algorithm + overlay comparison)
  - Hypervolume vs. generation convergence curves
  - Sample menu tables (≥3 Pareto solutions with nutrient totals vs. DRI bounds)
- Export all results to CSV/JSON

**Files to deliver:**
- `algorithms/spea2.py`
- `experiments/run_experiments.py`
- `visualization/` (pareto_plot.py, convergence.py, menu_table.py)
- `results/` (CSV/JSON output files)

---

## Report — Written Jointly

Each member writes the section covering their own component:

| Member | Report Section |
|--------|---------------|
| Member 1 | Problem Definition + Dataset Description |
| Member 2 | Chromosome Representation & Decoding |
| Member 3 | Objectives, Penalty Function, Mathematical Model |
| Member 4 | Algorithm Descriptions (NSGA-II), Genetic Operators |
| Member 5 | Experiments, Results, Discussion, Conclusion |

**Format:** PDF, max 15 pages.

---

## Shared Setup

Install dependencies (all members):

```bash
pip install pymoo PyMySQL pandas numpy matplotlib
```

Nutrient IDs to use (from the database):

| Constraint | Nutrient | DB `id` |
|------------|----------|---------|
| C1 | Energy | 5 |
| C2 | Protein | 15 |
| C3 | Carbohydrate | 8 |
| C4 | Fiber, total dietary | 4 |
| C5 | Sodium, Na | 17 |

---

## Project Structure

```
HeuristicProject/
├── config.py
├── db/
│   └── loader.py
├── problem/
│   ├── chromosome.py
│   ├── decoder.py
│   ├── objectives.py
│   ├── penalty.py
│   └── diet_problem.py
├── operators/
│   ├── crossover.py
│   └── mutation.py
├── algorithms/
│   ├── nsga2.py
│   └── spea2.py
├── experiments/
│   ├── hypervolume.py
│   └── run_experiments.py
├── visualization/
│   ├── pareto_plot.py
│   ├── convergence.py
│   └── menu_table.py
└── results/
```
