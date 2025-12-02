#!/usr/bin/env python3
"""
RSSP ILP experiments for Riesz s-energy subset selection.

- Uses SciPy's milp (HiGHS backend) to solve the 0–1 ILP.
- Samples uniform points in the 3D unit cube.
- Runs 12 experiments: n in {10, 100, 500, 1000},
  k = ceil(alpha * n) for alpha in {0.25, 0.5, 0.75}.
- Saves a log–log plot of CPU time vs n as rssp_ilp_cpu_times.png.
"""

import math
import time

import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd


def build_and_solve_ilp(points, k, s=3, time_limit=10.0):
    """
    Build and solve the ILP formulation of RSSP for given points and subset size k.
    Uses SciPy's milp (HiGHS backend).

    Minimize sum_{i<j} w_ij * z_i z_j
    with z_i in {0,1}, sum_i z_i = k, and linearization y_ij = z_i z_j.

    Parameters
    ----------
    points : np.ndarray, shape (n, d)
        Point set in R^d.
    k : int
        Desired subset size.
    s : float
        Riesz exponent (E_s uses 1 / ||x_i - x_j||^s).
    time_limit : float
        Solver time limit in seconds.

    Returns
    -------
    elapsed : float
        Wall-clock time in seconds (model build + milp call).
    status : int
        SciPy milp status code (0 = optimal).
    result : scipy.optimize.OptimizeResult
        Full result object from milp.
    """
    n = points.shape[0]
    d = points.shape[1]

    # Build pair indices and weights
    m = n * (n - 1) // 2
    i_idx = np.empty(m, dtype=np.int32)
    j_idx = np.empty(m, dtype=np.int32)
    w = np.empty(m, dtype=float)

    p = 0
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            dist = np.linalg.norm(pi - pj)
            if dist == 0.0:
                val = 0.0  # should not happen in random sampling
            else:
                val = 1.0 / (dist ** s)
            w[p] = val
            i_idx[p] = i
            j_idx[p] = j
            p += 1

    # Variables: first n are z_i (binary), next m are y_ij (continuous in [0,1])
    n_z = n
    n_y = m
    n_vars = n_z + n_y

    # Objective: sum w_p * y_p
    c = np.zeros(n_vars, dtype=float)
    c[n_z:] = w

    # Integrality: 1 for z_i, 0 for y_ij
    integrality = np.zeros(n_vars, dtype=int)
    integrality[:n_z] = 1

    # Bounds: 0 <= z_i <= 1, 0 <= y_ij <= 1
    lb = np.zeros(n_vars, dtype=float)
    ub = np.ones(n_vars, dtype=float)
    bounds = opt.Bounds(lb, ub)

    # Constraints:
    # For each pair p with (i,j):
    #   y_p - z_i <= 0
    #   y_p - z_j <= 0
    # plus the equality sum_i z_i = k

    n_pair_constr = 2 * m
    n_constr = n_pair_constr + 1  # +1 for the cardinality constraint

    # Number of nonzeros: each pair constraint has 2 nonzeros, plus n for cardinality
    nnz = 4 * m + n_z
    row_inds = np.empty(nnz, dtype=np.int64)
    col_inds = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=float)

    # Fill pair constraints
    idx = 0
    row = 0
    for p in range(m):
        i = i_idx[p]
        j = j_idx[p]
        y_col = n_z + p

        # Constraint 1: y_p - z_i <= 0
        row_inds[idx] = row
        col_inds[idx] = y_col
        data[idx] = 1.0
        idx += 1

        row_inds[idx] = row
        col_inds[idx] = i
        data[idx] = -1.0
        idx += 1

        row += 1

        # Constraint 2: y_p - z_j <= 0
        row_inds[idx] = row
        col_inds[idx] = y_col
        data[idx] = 1.0
        idx += 1

        row_inds[idx] = row
        col_inds[idx] = j
        data[idx] = -1.0
        idx += 1

        row += 1

    # Cardinality constraint: sum_i z_i = k
    card_row = n_pair_constr
    for i in range(n_z):
        row_inds[idx] = card_row
        col_inds[idx] = i
        data[idx] = 1.0
        idx += 1

    assert idx == nnz

    A = sp.coo_matrix((data, (row_inds, col_inds)), shape=(n_constr, n_vars))

    # Bounds for constraints
    lb_constr = np.full(n_constr, -np.inf, dtype=float)
    ub_constr = np.zeros(n_constr, dtype=float)
    # Cardinality constraint as equality
    lb_constr[card_row] = float(k)
    ub_constr[card_row] = float(k)

    constraints = opt.LinearConstraint(A, lb_constr, ub_constr)

    options = {"time_limit": float(time_limit)}

    start = time.perf_counter()
    res = opt.milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    elapsed = time.perf_counter() - start
    return elapsed, res.status, res


def run_experiments():
    rng = np.random.default_rng(42)

    n_values = [10, 100, 500, 1000]
    fractions = [0.25, 0.5, 0.75]
    s = 3
    time_limit = 10.0

    records = []

    for n in n_values:
        points = rng.random((n, 3))  # uniform in [0,1]^3
        for frac in fractions:
            k = math.ceil(frac * n)
            print(f"Running experiment n={n}, k={k} (k/n≈{frac})...")
            try:
                elapsed, status, _ = build_and_solve_ilp(
                    points, k, s=s, time_limit=time_limit
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                elapsed = math.nan
                status = -1

            records.append(
                {
                    "n": n,
                    "k": k,
                    "frac": frac,
                    "time": elapsed,
                    "status": status,
                }
            )
            print(f"  time = {elapsed:.4g} s, status = {status}")

    df = pd.DataFrame(records)
    return df


def plot_results(df, filename="rssp_ilp_cpu_times.png"):
    plt.figure(figsize=(6, 4))

    for frac in sorted(df["frac"].unique()):
        sub = df[df["frac"] == frac].sort_values("n")
        label = f"k≈{frac}·n"
        plt.plot(sub["n"], sub["time"], marker="o", label=label)

    plt.xlabel("n (number of points)")
    plt.ylabel("CPU time (s)")
    plt.xscale("log")
    plt.yscale("log")
    #plt.title("ILP solve time for RSSP in 3D cube")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Plot saved as {filename}")


def main():
    df = run_experiments()
    print("\nResults table:")
    print(df.to_string(index=False))
    plot_results(df)


if __name__ == "__main__":
    main()
