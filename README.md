# Multi-Objective Diet Optimization Problem (MODP)

**Course:** BLM20364E / BLM22332E — Heuristic Optimization Algorithms  
**University:** Fatih Sultan Mehmet Vakıf University — Faculty of Engineering, Dept. of Software Engineering  
**Instructor:** Asst. Prof. Dr. Cumali Türkmenoğlu  
**Academic Year:** 2024–2025  
**Repository:** https://github.com/raydughmsh/MultiObjectiveDiet-

---

## Team Members

| Student ID   | Name             |
|-------------|-----------------|
| 2221251391   | Rayan Dughmoush  |
| 2221251382   | Sedra Alshaar    |
| 2221251367   | Iman Saeid       |
| 2221251371   | Diaa Azrak       |
| 2221251376   | Hadal Kharouf    |

---

## Problem Description

This project solves a **Multi-Objective Diet Optimization Problem (MODP)**
modelled as a Multi-Objective Multidimensional Knapsack Problem (MOMKP).

**Three objectives:**
- **f1** — Maximize user food preference (negated for pymoo)
- **f2** — Minimize total meal cost
- **f3** — Minimize total preparation + cooking time

**Five nutritional constraints (DRI bounds):**
Energy (kcal), Protein (g), Carbohydrate (g), Fiber (g), Sodium (mg)

**Two algorithms compared:** NSGA-II and SPEA2

---

## Project Structure

```
MultiObjectiveDiet/
├── config.py                        # Central configuration
├── diet.sql                         # MySQL database dump
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── db/
│   └── loader.py                    # MySQL data loader
│
├── problem/
│   ├── chromosome.py                # Permutation chromosome
│   ├── decoder.py                   # Greedy two-pass decoder
│   ├── objectives.py                # f1, f2, f3 functions
│   ├── penalty.py                   # Nutritional penalty
│   └── diet_problem.py              # pymoo ElementwiseProblem
│
├── operators/
│   ├── crossover.py                 # Order Crossover (OX)
│   └── mutation.py                  # Swap Mutation
│
├── algorithms/
│   ├── nsga2.py                     # NSGA-II wrapper
│   └── spea2.py                     # SPEA2 wrapper
│
├── experiments/
│   ├── hypervolume.py               # HV indicator
│   └── run_experiments.py           # Exp 1, 2, 3
│
├── visualization/
│   ├── pareto_plot.py               # Pareto front plots
│   ├── convergence.py               # HV convergence curves
│   └── menu_table.py                # Decoded menu table builder
│
└── results/                         # Auto-generated output files
    ├── exp1_summary.json
    ├── exp2_summary.json
    ├── exp3_summary.json
    └── *.csv / *.png
```

---

## Setup

### 1. Prerequisites
- Python 3.9+
- MySQL Server running locally

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up the database
```bash
mysql -u root -p -e "CREATE DATABASE diet;"
mysql -u root -p diet < diet.sql
```

### 4. Configure your database password
Edit `config.py`:
```python
DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": None,   # <-- set your own password here (or None if no password)
    "database": "diet",
}
```

---

## Running the Experiments

```bash
python3 experiments/run_experiments.py
```

This runs all three experiments and saves results to the `results/` folder:

| Experiment | Description |
|-----------|-------------|
| Exp 1 | User comparison: Non-vegetarian (User 1) vs Vegetarian (User 2) |
| Exp 2 | Algorithm comparison: NSGA-II vs SPEA2 (User 1) |
| Exp 3 | Diversity penalty impact: ON vs OFF (User 1, NSGA-II) |

---

## Key Results

| Comparison | HV (NSGA-II) | HV (SPEA2) | Winner |
|-----------|-------------|-----------|--------|
| User 1 (non-veg) vs User 2 (veg) | 3,339,143 vs 5,018,704 | — | User 2 (+50%) |
| NSGA-II vs SPEA2 (User 1) | 3,210,377 | 3,156,261 | NSGA-II (+1.71%) |
| Diversity ON vs OFF | 3,424,788 | 3,337,781 | Diversity ON (+2.60%) |

---

## Algorithm Parameters

| Parameter | Value |
|-----------|-------|
| Population size | 100 |
| Generations | 200 |
| Crossover (OX) probability | 0.9 |
| Mutation rate (breakfast) | 1/94 |
| Mutation rate (lunch+dinner) | 1/311 |
| Random seed | 42 |

---

## Dependencies

```
pymoo>=0.6.1
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
PyMySQL>=1.1.0
```
