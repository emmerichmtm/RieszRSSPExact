# Exact Algorithms for Minimum Riesz s-Energy Subset Selection

This repository contains small, self-contained Python prototypes used in the
supplementary material of the paper

> *Minimum Riesz s-Energy Subset Selection in Geometric Settings:  
> Algorithms, Complexity, and Open Questions.*

The focus is on **exact** algorithms for the Riesz s-energy subset selection
problem:

- exhaustive enumeration with an incremental energy update,
- a one-dimensional dynamic programming prototype,
- and a 0–1 integer linear programming (ILP) formulation solved by SciPy's
  `milp` (HiGHS) backend.

The codes are intended for reproducibility and experimentation rather than as
production-quality solvers.

---

## Problem statement

Given a finite point set \(X = \{x_1,\dots,x_n\}\) in a metric space and an
exponent \(s > 0\), the discrete Riesz \(s\)-energy of a subset
\(S \subseteq X\) is

\[
E_s(S) = \sum_{\substack{i,j \in S \\ i < j}} \frac{1}{\|x_i - x_j\|^s}.
\]

The **Riesz s-Energy Subset Selection Problem (RSSP)** asks, for a given subset
size \(k\), to find a subset \(S \subseteq X\) of cardinality \(|S| = k\) that
minimizes \(E_s(S)\).

All implementations here work in Euclidean space (2D or 3D) and treat the
metric as the usual Euclidean distance.

---

## Repository structure

- `incrementalRiesz.py`  
  Incremental vs. brute-force exhaustive enumeration for RSSP instances in
  low dimension, together with a runtime comparison plot.

- `riesz1d.py`  
  Prototype dynamic programming algorithm for 1D instances (points on the real
  line), plus a brute-force checker for small examples.

- `ilpriesz.py`  
  0–1 ILP formulation of RSSP solved via SciPy's `milp` with the HiGHS
  backend, plus experiments on random 3D point sets.

- `incremental_vs_bruteforce_riesz_runtime.png`  
  Example output figure for the enumeration experiments.

- `rssp_ilp_cpu_times.png`  
  Example output figure for the ILP experiments.

- `supplement_exact_algorithms.tex`  
  LaTeX supplementary report describing the algorithms and experiments.

---

## Dependencies

The scripts assume a recent Python 3 (e.g. 3.10+).  Required packages:

- [NumPy](https://numpy.org/) (for the ILP script),
- [SciPy](https://scipy.org/) (for `scipy.optimize.milp` and sparse matrices),
- [pandas](https://pandas.pydata.org/) (for tabular outputs),
- [matplotlib](https://matplotlib.org/) (for plotting).

You can install them via:

```bash
pip install numpy scipy pandas matplotlib
```

The `riesz1d.py` script uses only the standard library (`itertools`) but is
typically run in the same environment.

---

## 1. Incremental enumeration vs. brute force

**Script:** `incrementalRiesz.py`

This script compares two exact enumeration strategies on random 2D Riesz
energy instances:

1. **Brute force**: enumerate all \(k\)-subsets and recompute the Riesz
   energy from scratch for each subset.
2. **Incremental**: maintain the current subset and its energy during a
   backtracking search, updating the energy incrementally when adding
   a new point.

### How it works

1. Generate `n` random points in `[0,1]^2` using the standard library RNG.
2. Precompute a symmetric matrix `mat[i][j] = 1 / ||x_i - x_j||^s`.
3. For each `k`:
   - run `incremental_best_subset(mat, k, time_limit=...)`,
   - run `bruteforce_best_subset(mat, k, time_limit=...)`,
   - record the best energy, subset, number of subsets visited, and runtime.
4. Aggregate everything into a `pandas.DataFrame`.
5. Plot runtime vs `n` for each `(method, k)` combination.

The default configuration in the `__main__` block is:

- `n_values = [20, 26, 32, 38]`,
- `k_values = [3, 4, 5]`,
- `s = 1.0`,
- `dim = 2`,
- `time_limit = 10.0` seconds per (n, k) pair.

### Running the script

```bash
python incrementalRiesz.py
```

This prints a timing table to the console and produces a plot

```text
incremental_vs_bruteforce_riesz_runtime.png
```

in the current directory.

---

## 2. Dynamic programming prototype in 1D

**Script:** `riesz1d.py`

This script explores the 1D case \(X \subset \mathbb{R}\) with a simple
dynamic programming (DP) scheme and compares its output to a brute-force
enumerator on small instances.

### `dp_subset(x, k, s)`

- Input:
  - `x`: sorted list of real coordinates,
  - `k`: desired subset size,
  - `s`: Riesz exponent.
- Output:
  - `min_energy`: minimal Riesz energy among all k-subsets,
  - `best_subset`: list of indices into `x` that attains `min_energy`.

The DP state `dp[r][i]` stores the best energy and a corresponding subset of
size `r` chosen from `x[0..i]` **with `x[i]` included**. The transition
tries all possible previous endpoints `p < i` and adds the incremental cost
of including `x[i]` against all points in the stored subset for `dp[r-1][p]`.

### `brute_force_subset(x, k, s)`

Enumerates all subsets of indices of size `k` using
`itertools.combinations` and computes the exact energy for each, returning
the minimum and the corresponding subset.

### Demo usage

When run as a script, the file executes a couple of small examples:

```bash
python riesz1d.py
```

For each example it prints the DP and brute-force solutions, allowing you to
check whether they agree and to explore where the simple DP fails to be
globally optimal (as discussed in the paper).

To adapt the examples, edit the vectors `x1`, `x2`, the subset sizes `k1`,
`k2`, and the exponent `s` in the `main()` function.

---

## 3. ILP formulation and experiments

**Script:** `ilpriesz.py`

This script formulates RSSP as a 0–1 ILP with additional continuous
variables for the linearization of products, and solves it using SciPy's
`milp` (HiGHS backend). It then measures solve time as a function of
problem size.

### `build_and_solve_ilp(points, k, s=3, time_limit=10.0)`

- Constructs all pairwise Riesz weights \(w_{ij} = 1/\|x_i-x_j\|^s\).
- Builds a linear 0–1 ILP with:
  - binary variables `z_i`,
  - continuous variables `y_ij` for each pair,
  - a cardinality constraint `sum z_i = k`,
  - McCormick inequalities relating `y_ij` and `z_i z_j`.
- Calls `scipy.optimize.milp` with the desired time limit.
- Returns `(elapsed_time, status, res)`.

### Experiment setup (default)

In `run_experiments()`:

- `n_values = [10, 100, 500, 1000]`,
- `fractions = [0.25, 0.5, 0.75]`, with `k = ceil(frac * n)`,
- points sampled uniformly in `[0,1]^3` using NumPy RNG,
- Riesz exponent `s = 3`,
- per-instance time limit `time_limit = 10.0` seconds.

Results are collected into a `pandas.DataFrame` with columns `(n, k, frac,
time, status)`.

### Plotting

`plot_results(df, filename="rssp_ilp_cpu_times.png")` creates a log–log
plot of CPU time vs `n` with a separate curve for each `k/n` fraction:

```text
rssp_ilp_cpu_times.png
```

### Running the script

```bash
python ilpriesz.py
```

This prints a small results table and saves the ILP runtime plot.

---

## Reproducibility tips

- To change the experimental grids (values of `n`, `k`, the ambient
  dimension, or the Riesz exponent `s`), edit the configuration values in
  the `__main__` section of each script.
- All scripts use fixed random seeds for reproducibility by default.
- The ILP script relies on SciPy's HiGHS interface. If you see solver
  errors, check that your SciPy installation is recent enough and that the
  HiGHS binaries are available for your platform.

---

## Citation

If you use this code in academic work, please cite the main paper on
Riesz s-energy subset selection (preprint / arXiv entry to be filled in
once available).
