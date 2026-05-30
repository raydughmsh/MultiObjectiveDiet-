# ── Database ──────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": None,  # each member changes this to their own MySQL password (None = no password)
    "database": "diet",
}

# ── Users ─────────────────────────────────────────────────────────────────────
USER1_ID = 1   # Non-vegetarian 
USER2_ID = 2   # Vegetarian      

# ── Nutrient IDs (from `nutrients` table) ─────────────────────────────────────
NUTRIENT_IDS = {
    "energy":       5,   # C1 — kcal
    "protein":      15,  # C2 — g
    "carbohydrate": 8,   # C3 — g
    "fiber":        4,   # C4 — g
    "sodium":       17,  # C5 — mg
}

# ── Chromosome structure ───────────────────────────────────────────────────────
N_FOODS          = 405   # total food items (caseStudy filter)
N_BREAKFAST      = 94    # genes 0 .. 93   → breakfast candidates
N_LUNCH_DINNER   = 311   # genes 94 .. 404 → lunch+dinner candidates

# ── ε tolerances for the decoder ──────────────────────────────────────────────
EPS_RUL          = 1.15  # effective upper bound = RUL × 1.15
EPS_RLL          = 0.90  # effective lower bound = RLL × 0.90
BREAKFAST_RATIO  = 0.35  # breakfast share of daily DRI

# ── Penalty ────────────────────────────────────────────────────────────────────
LAMBDA           = 1.0   # penalty weight  (tune experimentally)
W_UNDER          = 0.7   # under-nutrition weight
W_OVER           = 0.3   # over-nutrition weight
ALPHA_DIVERSITY  = 1.0   # diversity penalty weight (tune experimentally)

# ── Genetic operators ──────────────────────────────────────────────────────────
P_CROSSOVER      = 0.9                          # crossover probability
P_MUTATION_B     = 1.0 / N_BREAKFAST            # swap mutation prob — breakfast part
P_MUTATION_LD    = 1.0 / N_LUNCH_DINNER         # swap mutation prob — lunch+dinner part

# ── NSGA-II / SPEA2 ───────────────────────────────────────────────────────────
POP_SIZE         = 100   # population size
N_GENERATIONS    = 200   # number of generations
N_OBJECTIVES     = 3     # f1=preference (MAX→negated), f2=cost (MIN), f3=time (MIN)

# ── Hypervolume reference point ────────────────────────────────────────────────
# Filled in by Member 4/5 after running all experiments.
# Each value = worst observed value across all runs + 10% margin.
HV_REFERENCE_POINT = None  # e.g. [worst_f1, worst_f2, worst_f3]