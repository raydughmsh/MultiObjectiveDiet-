# Report Section — Member 4
## Algorithm Description: NSGA-II, Genetic Operators & Hypervolume Indicator

---

## 4.1 NSGA-II (Non-dominated Sorting Genetic Algorithm II)

### 4.1.1 Overview

NSGA-II is a Pareto-based multi-objective evolutionary algorithm introduced by Deb, Pratap, Agarwal, and Meyarivan (2002) [1]. It remains one of the most widely adopted MOEAs in the literature due to three key properties: (i) computational efficiency via fast non-dominated sorting in O(MN²) time, (ii) crowding distance-based diversity maintenance that requires no user-defined sharing parameters, and (iii) elitism that prevents loss of high-quality solutions across generations.

In this project, NSGA-II was selected as the primary algorithm because it is well-suited for 3-objective problems with a moderate population size. Its Pareto front is used to compare against SPEA2 in Experiment 2.

### 4.1.2 Non-dominated Sorting

The foundation of NSGA-II is the concept of Pareto dominance. A solution **x** dominates solution **y** (written **x** ≺ **y**) if and only if:

```
∀ k ∈ {1,...,M}: fk(x) ≤ fk(y)   AND   ∃ k: fk(x) < fk(y)
```

where M is the number of objectives and fk is the k-th objective function. The algorithm partitions the population into non-dominated fronts:

- **F₁**: solutions not dominated by any other solution in the population
- **F₂**: solutions dominated only by solutions in F₁
- **Fₖ**: solutions dominated only by solutions in F₁ ∪ ... ∪ Fₖ₋₁

Each solution is assigned a rank equal to its front index. Lower rank indicates better quality.

### 4.1.3 Crowding Distance

Within each front, solutions that are geometrically isolated in objective space are preferred to maintain diversity. The crowding distance of solution i in front Fₖ is defined as:

```
distance(i) = Σ_m [ f_m(i+1) - f_m(i-1) ] / [ f_m^max - f_m^min ]
```

where the sum is over all M objectives, and solutions are sorted by objective m before computing the difference. Boundary solutions (best and worst in any objective) are assigned infinite crowding distance to ensure they are always retained.

A solution i is preferred over solution j (the crowded comparison operator ≺_n) if:
- rank(i) < rank(j), OR
- rank(i) = rank(j) AND distance(i) > distance(j)

### 4.1.4 Main Loop

The NSGA-II generational procedure is as follows:

```
1.  Initialize population P₀ of size N (random permutations)
2.  Evaluate objectives f₁, f₂, f₃ for each individual
3.  For t = 1, 2, ..., T_max:
    a. Select parents using binary tournament with ≺_n
    b. Apply OX crossover (p_c = 0.9) to produce offspring Q_t
    c. Apply swap mutation (p_m = 1/n) to Q_t
    d. Evaluate objectives for all offspring in Q_t
    e. Combine: R_t = P_{t-1} ∪ Q_t  (size 2N)
    f. Sort R_t into fronts F₁, F₂, ...
    g. Fill P_t greedily: add fronts F₁, F₂, ... until size N is reached
       For the last partial front: select by descending crowding distance
4.  Return final non-dominated front F₁ of P_{T_max}
```

### 4.1.5 Implementation Details

NSGA-II was implemented using the **pymoo** library (v0.6.1+) with fully custom genetic operators for the split-chromosome representation. The `DietProblem` class (Member 3) was passed as the problem instance, encapsulating the decoder, objectives, and penalty function. All history (population objective values per generation) was saved for hypervolume convergence analysis.

```python
algorithm = NSGA2(
    pop_size=100,
    sampling=SplitPermutationSampling(),   # custom: random permutation of 0..404
    crossover=SplitOXCrossover(p_c=0.9),   # custom: OX on each chromosome part
    mutation=SplitSwapMutation(p_m_b=1/94, p_m_ld=1/311),  # custom: swap per part
    eliminate_duplicates=False,            # permutations rarely produce exact duplicates
)
result = minimize(problem, algorithm, ("n_gen", 200), seed=42, save_history=True)
```

The algorithm was executed independently for both users (User 1: non-vegetarian, User 2: vegetarian) using the same hyperparameters. A fixed random seed of 42 was used to ensure reproducibility.

### 4.1.6 Parameter Settings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Population size (N) | 100 | Standard baseline for 3-objective combinatorial problems |
| Generations (T) | 200 | Empirically sufficient for convergence; HV plateaus observed before generation 200 |
| Crossover probability (p_c) | 0.9 | Recommended default for OX crossover in permutation problems [2] |
| Mutation prob. — breakfast | 1/94 ≈ 0.0106 | Standard 1/n rule: expected one swap per individual per generation |
| Mutation prob. — lunch+dinner | 1/311 ≈ 0.0032 | Standard 1/n rule applied to the longer chromosome part |
| Random seed | 42 | Fixed for reproducibility across all runs |

---

## 4.2 Genetic Operators

The chromosome is a permutation of 405 food item indices (0 to 404), split into two structurally independent parts:

| Part | Gene Range | Length | Role |
|------|-----------|--------|------|
| Breakfast | [0 : 94] | 94 | Candidates for breakfast selection |
| Lunch + Dinner | [94 : 405] | 311 | Candidates for lunch+dinner selection |

**All operators are applied independently to each part.** This design is not optional — it is structurally required because the decoder processes the two parts in separate passes with different nutritional constraints. Applying a standard whole-chromosome operator would incorrectly mix breakfast and lunch+dinner candidates, invalidating the decoding logic.

### 4.2.1 Population Initialization

Each individual in the initial population is created as a uniformly random permutation of indices 0 to 404. The two chromosome parts emerge naturally from positions [0:94] and [94:405] of this permutation. The `DietProblem` class maps these indices to actual food IDs via its internal ordered food list.

```python
X[i] = rng.permutation(405)   # uniform random permutation of {0, 1, ..., 404}
```

This ensures the initial population uniformly covers the solution space without any bias toward specific food orderings.

### 4.2.2 Order Crossover (OX)

Order Crossover (OX) [3] is a permutation-preserving crossover operator designed for ordered representations. It preserves the relative ordering of elements inherited from each parent, which is semantically meaningful here because the decoder's greedy selection depends on gene order.

**Algorithm (applied to one part of length n):**

```
Input:  parent1, parent2 — two permutations of the same set
Output: offspring — a new permutation

1. Choose two cut points i, j uniformly at random, where i < j
2. offspring[i:j] ← parent1[i:j]                  (copy segment from P1)
3. Start scanning parent2 from position j (wrap around)
4. For each gene g in parent2 (in scan order):
       if g ∉ offspring:
           place g in the next empty position (scanning from j, wrapping)
```

**Worked Example** (n = 8, cut points i = 2, j = 5):

```
parent1:    [ 3,  8,  2 | 4,  1,  7 | 5,  6 ]
parent2:    [ 5,  6,  3,  8,  2,  4,  7,  1 ]

Step 2 — copy segment from parent1:
offspring:  [ _,  _,  2,  4,  1,  7,  _,  _ ]

Step 3-4 — scan parent2 from position 5 (wrap): 4(skip), 7(skip), 1(skip), 5, 6, 3, 8, 2(skip)
Remaining genes in order: [5, 6, 3, 8]
Empty positions from j=5: [5→skip, 6, 7, 0, 1]

offspring:  [ 6,  3,  2,  4,  1,  7,  8,  5 ]
```

OX is applied separately to the breakfast part (length 94) and the lunch+dinner part (length 311) with probability p_c = 0.9 each. Two offspring are produced per mating pair (roles of parent1 and parent2 are swapped for the second offspring).

**Justification for OX over PMX:** OX preserves global relative order while PMX only preserves local positional relationships. Since the decoder scans genes left-to-right and the order determines priority of selection, preserving global relative order is more semantically aligned with the problem.

### 4.2.3 Swap Mutation

Swap mutation introduces controlled random variation while strictly preserving the permutation structure. It is the simplest and most commonly used mutation operator for permutation chromosomes.

**Algorithm (applied to one part of length n):**

```
For each position i in {0, 1, ..., n-1}:
    with probability p_m:
        choose j ∈ {0, ..., n-1} uniformly at random, j ≠ i
        swap genes[i] and genes[j]
```

Each position is evaluated independently, so zero, one, or multiple swaps may occur in a single mutation call. Because only values within the array are swapped (never inserted or deleted), the permutation property is guaranteed to hold after mutation.

**Expected number of swaps per individual:**

| Part | n | p_m | Expected swaps |
|------|---|-----|----------------|
| Breakfast | 94 | 1/94 | ~1.0 |
| Lunch+dinner | 311 | 1/311 | ~1.0 |

Setting p_m = 1/n (the standard rule) means on average one gene per part is perturbed each generation. This provides sufficient exploration without excessively disrupting well-performing solutions.

---

## 4.3 Hypervolume Indicator

### 4.3.1 Definition

The Hypervolume (HV) indicator [4] measures the volume of the objective space that is dominated by the Pareto front approximation A and bounded above by a reference point **r**:

```
HV(A, r) = λ_M ( ∪_{a ∈ A} [a₁, r₁] × [a₂, r₂] × ... × [aM, rM] )
```

where λ_M denotes the M-dimensional Lebesgue measure. A larger HV value indicates better convergence (solutions closer to the true front) and better spread (solutions more evenly distributed). HV is the only widely-used unary indicator that simultaneously captures both properties.

### 4.3.2 Why HV and Not IGD?

The Inverted Generational Distance (IGD) metric requires knowledge of the true Pareto front, which is unknown for this problem since no analytical solution exists. HV requires only a reference point, which can be defined from the observed data. Therefore, HV is the appropriate performance indicator for this problem.

### 4.3.3 Reference Point Construction

To ensure all algorithms are compared on equal footing, a single shared reference point is used across all runs (NSGA-II and SPEA2, both users, all experiments). The reference point is computed after all experiments complete:

```
r_j = max{ f_j(x) : x ∈ A_NSGA2_U1 ∪ A_NSGA2_U2 ∪ A_SPEA2_U1 ∪ A_SPEA2_U2 } × 1.10
```

The 10% margin ensures the reference point is strictly dominated by no solution, which is a required property for HV to be well-defined. Using a shared reference point means differences in HV values are attributable to algorithm performance, not to different scales.

### 4.3.4 Convergence Tracking

HV is computed at every generation using the full population's objective values. This produces a convergence curve for each algorithm run. The curve is expected to be monotonically non-decreasing as the algorithm improves its Pareto front approximation over time.

```python
from pymoo.indicators.hv import HV

indicator = HV(ref_point=ref_point)
hv_per_generation = [indicator(gen["F"]) for gen in history]
```

---

## 4.4 Summary

| Component | Design Choice | Parameters |
|-----------|--------------|------------|
| Algorithm | NSGA-II (Deb et al., 2002) via pymoo | N=100, T=200, seed=42 |
| Initialization | Random permutation of indices 0..404 | — |
| Crossover | Order Crossover (OX), split-independent | p_c = 0.9 per part |
| Mutation | Swap mutation, split-independent | p_m = 1/94 (B), 1/311 (LD) |
| Selection | Binary tournament with crowded comparison | — |
| Performance metric | Hypervolume indicator | Shared reference point (worst + 10%) |
| Users evaluated | User 1 (non-vegetarian), User 2 (vegetarian) | Identical parameters |

---

## References

[1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

[2] Larranaga, P., Kuijpers, C. M. H., Murga, R. H., Inza, I., & Dizdarevic, S. (1999). Genetic algorithms for the travelling salesman problem: A review of representations and operators. *Artificial Intelligence Review*, 13(2), 129–170.

[3] Davis, L. (1985). Applying adaptive algorithms to epistatic domains. *Proceedings of the International Joint Conference on Artificial Intelligence*, 162–164.

[4] Zitzler, E., & Thiele, L. (1998). Multiobjective optimization using evolutionary algorithms — a comparative case study. *Proceedings of PPSN V*, 292–301.
