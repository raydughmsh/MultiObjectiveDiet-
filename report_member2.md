# Report Section — Member 2
## Chromosome Representation & Greedy Decoding

---

## 2.1 Chromosome Representation

### 2.1.1 Overview

Each candidate solution in the population is represented as a **permutation chromosome** — an ordered sequence of all 405 food item IDs drawn from the database. The chromosome encodes a priority ordering over foods rather than a direct binary selection; the actual daily menu is determined downstream by the greedy decoder. This indirect (permutation-based) encoding is well-suited to combinatorial diet planning because it naturally avoids duplicate food selections and allows the decoder to enforce complex nutritional constraints without requiring a repair operator.

### 2.1.2 Gene Structure

The 405-gene array is logically partitioned into two independent segments that correspond to the two meal groups:

```
[ gene_0, gene_1, ..., gene_93 | gene_94, gene_95, ..., gene_404 ]
  ←────────── breakfast (94) ──→ ←───────── lunch + dinner (311) ───→
```

| Segment | Index range | Length | Meal group |
|---------|-------------|--------|------------|
| Part 1  | `[0 : 94)`  | 94     | Breakfast  |
| Part 2  | `[94 : 405)`| 311    | Lunch + Dinner |

The split reflects the nutritional role of each meal: breakfast is allocated 35 % of the daily Dietary Reference Intake (DRI) and is constrained to only two nutrients (Energy and Protein), whereas lunch and dinner are jointly responsible for meeting all five nutrient constraints over the full day.

Crucially, the two parts are operated on **independently** by the crossover and mutation operators (Member 4). This preserves the semantic integrity of each meal segment: a crossover applied to the breakfast part cannot accidentally displace genes that belong to the lunch/dinner part.

### 2.1.3 `Chromosome` Class

The `Chromosome` class (in `problem/chromosome.py`) wraps the raw NumPy gene array and exposes two read-only views:

```python
@property
def breakfast(self) -> np.ndarray:
    return self.genes[:N_BREAKFAST]        # view into genes[0:94]

@property
def lunch_dinner(self) -> np.ndarray:
    return self.genes[N_BREAKFAST:]        # view into genes[94:405]
```

Random initialization is handled by `_random_init`, which produces a uniformly random permutation of the full food ID set:

```python
@staticmethod
def _random_init(food_ids: np.ndarray) -> np.ndarray:
    permutation = food_ids.copy()
    np.random.shuffle(permutation)
    return permutation
```

Returning a copy before shuffling ensures that the original food ID array (owned by the data loader) is never modified. The `copy()` method likewise performs a full deep copy of `self.genes`, which is required by the genetic operators so they do not mutate individuals in-place.

Validity checking is provided by `is_valid_permutation()`, which verifies both that the array contains exactly `N_FOODS = 405` elements and that all values are distinct. This is used in unit tests after every crossover and mutation to catch any accidental gene duplication or loss.

---

## 2.2 Greedy Decoder

### 2.2.1 Decoding Principle

The `Decoder` class (in `problem/decoder.py`) converts a chromosome into a concrete daily menu by scanning each segment from left to right and greedily selecting foods that do not violate the ε-relaxed DRI bounds. This left-to-right priority scanning means that genes appearing earlier in the chromosome are considered first; the chromosome ordering therefore acts as an implicit priority queue over foods for each meal group.

Three lookup structures are pre-built once at decoder initialisation to avoid repeated DataFrame queries during the evolutionary run:

- `_nutrient_lookup`: a nested dict `{food_id → {nutrient_id → value}}` for the five constrained nutrients.
- `_dri_bounds`: a dict `{nutrient_id → (RLL, RUL)}` holding the raw daily lower and upper bounds for the current user.
- `_food_group_lookup`: a dict `{food_id → food_group_id}` used by the diversity penalty.

### 2.2.2 ε-Relaxed DRI Bounds

To allow a degree of flexibility around the nutritional targets, both bounds are scaled by tolerance factors defined in `config.py`:

| Parameter | Symbol | Value | Meaning |
|-----------|--------|-------|---------|
| `EPS_RUL` | ε₊ | 1.15 | Effective upper bound = RUL × 1.15 |
| `EPS_RLL` | ε₋ | 0.90 | Effective lower bound = RLL × 0.90 |
| `BREAKFAST_RATIO` | β | 0.35 | Breakfast share of daily DRI |

Allowing the upper bound to be 15 % above the DRI maximum prevents excessive food rejection during greedy scanning (which would otherwise leave the menu nutritionally incomplete), while tolerating up to 10 % below the lower bound prevents the decoder from accepting marginally unsuitable foods purely to meet the minimum.

### 2.2.3 Pass 1 — Breakfast Decoding

The breakfast pass scans `gene_ids[0:94]` and checks only **Energy (C1)** and **Protein (C2)** against 35 % of the daily DRI.

Effective breakfast bounds for nutrient *k* ∈ {Energy, Protein}:

```
upper_k^B  =  RUL_k  ×  β  ×  ε₊   =  RUL_k  ×  0.35  ×  1.15
lower_k^B  =  RLL_k  ×  β  ×  ε₋   =  RLL_k  ×  0.35  ×  0.90
```

For each candidate food in the breakfast segment, the decoder applies the following logic:

1. **Skip** the food if adding its Energy or Protein would push the running total above `upper_k^B` for either nutrient.
2. Otherwise, **select** the food and update all five running nutrient totals (even though only two are checked in this pass — the totals carry forward into Pass 2).
3. **Early stop**: once both `totals[energy] ≥ lower_energy^B` and `totals[protein] ≥ lower_protein^B`, the breakfast pass terminates immediately, regardless of how many genes remain unvisited.

Restricting Pass 1 to two nutrients reflects the dietary reality that breakfast planning primarily targets energy and protein balance; micronutrients and sodium are managed at the day level.

### 2.2.4 Pass 2 — Lunch + Dinner Decoding

The lunch/dinner pass scans `gene_ids[94:405]` and checks **all five nutrients** against the full daily DRI. The nutrient totals accumulated during Pass 1 are carried over, so the lunch/dinner pass only needs to cover the remaining nutritional gap.

Effective full-day bounds for nutrient *k* ∈ {C1, …, C5}:

```
upper_k  =  RUL_k  ×  ε₊   =  RUL_k  ×  1.15
lower_k  =  RLL_k  ×  ε₋   =  RLL_k  ×  0.90
```

The selection logic mirrors Pass 1 but is generalised to all five nutrients:

1. **Skip** the food if adding it would push any of the five running totals above `upper_k`.
2. Otherwise, **select** and update all totals.
3. **Early stop**: once all five `totals[k] ≥ lower_k`, the pass terminates.

This two-pass structure guarantees that the breakfast portion is planned in isolation before the remaining nutritional requirements are filled in by lunch and dinner, closely mirroring how daily meal planning works in practice.

### 2.2.5 Two-Pass Decoding Algorithm (Pseudocode)

```
Input : gene_ids[0..404], DRI bounds, ε₊, ε₋, β
Output: selected_foods (ordered list of food IDs)

totals ← {C1: 0, C2: 0, C3: 0, C4: 0, C5: 0}
selected ← []

# ── Pass 1: Breakfast ──────────────────────────────────────
upper^B_k ← RUL_k × β × ε₊   for k ∈ {C1, C2}
lower^B_k ← RLL_k × β × ε₋   for k ∈ {C1, C2}

for food_id in gene_ids[0:94]:
    if totals[Ck] + value(food_id, Ck) > upper^B_k for any k ∈ {C1, C2}:
        skip
    else:
        selected.append(food_id)
        totals[Ck] += value(food_id, Ck)  for all k
        if totals[Ck] ≥ lower^B_k for all k ∈ {C1, C2}:
            break  # breakfast targets met

# ── Pass 2: Lunch + Dinner ─────────────────────────────────
upper_k ← RUL_k × ε₊   for k ∈ {C1, …, C5}
lower_k ← RLL_k × ε₋   for k ∈ {C1, …, C5}

for food_id in gene_ids[94:405]:
    if totals[Ck] + value(food_id, Ck) > upper_k for any k:
        skip
    else:
        selected.append(food_id)
        totals[Ck] += value(food_id, Ck)  for all k
        if totals[Ck] ≥ lower_k for all k ∈ {C1, …, C5}:
            break  # daily targets met

return selected
```

### 2.2.6 Diversity Mechanism — Option B

To discourage menus that repeatedly draw from the same food group, a diversity penalty term is applied after decoding (computed by `penalty.py`, Member 3). The decoder exposes a helper:

```python
def count_distinct_food_groups(self, selected_foods: list) -> int:
    groups = {self._food_group[fid]
              for fid in selected_foods if fid in self._food_group}
    return len(groups)
```

The penalty term added to the objective is:

```
diversity_penalty  =  α  ×  (1 / distinct_food_group_count)
```

where `α = ALPHA_DIVERSITY = 1.0` (configurable in `config.py`). A menu composed of foods from many different groups yields a small penalty (close to zero), whereas a menu concentrated in a single group yields a large penalty. This incentivises the evolutionary algorithm to explore diverse food combinations without hard-coding any food-group constraints into the chromosome structure itself.

### 2.2.7 Nutrient Total Inspector

An additional helper, `get_nutrient_totals(selected_foods)`, returns the summed nutrient values for a decoded menu. This is used by `penalty.py` to compute constraint violations and by the visualisation module to populate sample menu tables.

---

## 2.3 Design Decisions and Justifications

**Permutation encoding over binary encoding.** A binary string of length 405 with a repair operator was considered as an alternative. The permutation encoding was preferred because it implicitly prevents duplicate selections (no repair needed), and it integrates naturally with Order Crossover (OX), which is designed for permutation problems and preserves the relative ordering of genes in each half.

**Pre-built lookup tables.** The nutrient and DRI data are loaded into plain Python dicts at decoder construction time. This avoids repeated Pandas `.loc` or `.query` calls during the inner decode loop, reducing per-individual evaluation time significantly given a population of 100 evaluated over 200 generations (20 000 decode calls per run).

**Two-pass vs. single-pass decoding.** A single pass over all 405 genes checking all five nutrients at once was considered. The two-pass approach was chosen to enforce a nutritionally coherent meal structure (breakfast is planned to its own proportional target) and to give the chromosome's spatial structure a clear semantic meaning that the crossover and mutation operators can exploit — each operator acts on exactly one meal segment.

**ε tolerance.** Without the ε relaxation, the decoder would skip many nutritionally close but technically over-bound foods, often failing to meet lower bounds. The 15 %/10 % tolerance mirrors the flexible upper/lower margin commonly used in dietary planning and prevents overly sparse menus that would score poorly on all three objectives.
