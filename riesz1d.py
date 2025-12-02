

import itertools

def dp_subset(x, k, s):
    """
    Uses dynamic programming to choose k points from a sorted list x
    to minimize the Riesz s-energy:
    
        E(S) = sum_{p < q in S} 1/(x[q]-x[p])^s.
    
    We assume that x is sorted in increasing order.
    
    Returns:
        (min_energy, best_subset)
    where best_subset is a list of indices (into x) that achieves the minimal energy.
    """
    n = len(x)
    # dp[r][i] will store a tuple (energy, subset) meaning:
    #   "the best energy achieved by choosing r points from x[0:i+1] with x[i] chosen as the last point"
    # Here r will run from 1 to k and i from 0 to n-1.
    dp = [ [None]*n for _ in range(k+1) ]  # dp[0] is unused; dp[1] ... dp[k]
    
    # Base case: for r=1, each single point gives zero energy.
    for i in range(n):
        dp[1][i] = (0.0, [i])
    
    # For r>=2, build up the DP table.
    for r in range(2, k+1):
        # The smallest index that can be the last of r points is r-1 (0-indexed).
        for i in range(r-1, n):
            best = None
            best_subset = None
            # Try all possible previous endpoints p < i that have a valid (r-1)-point solution.
            for p in range(r-2, i):
                if dp[r-1][p] is None:
                    continue
                prev_energy, subset = dp[r-1][p]
                # Compute additional cost for adding x[i] to the subset.
                # That is, for each already chosen point in 'subset', add 1/(x[i]-x[q])^s.
                extra_cost = 0.0
                valid = True
                for q in subset:
                    gap = x[i] - x[q]
                    if gap <= 0:
                        valid = False
                        break
                    extra_cost += 1.0 / (gap ** s)
                if not valid:
                    continue
                candidate = prev_energy + extra_cost
                if best is None or candidate < best:
                    best = candidate
                    best_subset = subset + [i]
            dp[r][i] = (best, best_subset)
    
    # The answer is the minimum energy among all dp[k][i] for i=r-1,...,n-1.
    best_final = None
    best_subset_final = None
    for i in range(k-1, n):
        if dp[k][i] is None:
            continue
        energy, subset = dp[k][i]
        if best_final is None or energy < best_final:
            best_final = energy
            best_subset_final = subset
    return best_final, best_subset_final

def brute_force_subset(x, k, s):
    """
    Enumerate all subsets of indices of size k and return the minimal Riesz s-energy and subset.
    Energy for a given subset S (list of indices) is:
        E(S)= sum_{p<q in S} 1/(x[q]-x[p])^s.
    """
    n = len(x)
    best_energy = None
    best_subset = None
    for subset in itertools.combinations(range(n), k):
        energy = 0.0
        valid = True
        for i in range(k):
            for j in range(i+1, k):
                gap = x[subset[j]] - x[subset[i]]
                if gap <= 0:
                    valid = False
                    break
                energy += 1.0 / (gap ** s)
            if not valid:
                break
        if not valid:
            continue
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_subset = list(subset)
    return best_energy, best_subset

def print_solution(x, subset, energy, method=""):
    print(f"Method: {method}")
    print("Chosen indices:", subset)
    print("Chosen points: ", [x[i] for i in subset])
    print("Total energy: ", energy)
    print("-"*40)

def main():
    # Example 1: x = [0, 1, 3, 6], k = 3, s = 1
    x1 = [0, 1, 3, 6]
    k1 = 2
    s = 1
    print("Example 1: x =", x1, ", k =", k1, ", s =", s)
    
    dp_energy, dp_subset_indices = dp_subset(x1, k1, s)
    bf_energy, bf_subset_indices = brute_force_subset(x1, k1, s)
    
    print_solution(x1, dp_subset_indices, dp_energy, method="DP")
    print_solution(x1, bf_subset_indices, bf_energy, method="Brute Force")
    
    # Example 2: x = [0, 2, 4, 7], k = 3, s = 1
    x2 = [0,0.1,0.2,0.4, 2, 4, 7,8.1, 8.2,9]
    k2 = 7
    print("Example 2: x =", x2, ", k =", k2, ", s =", s)
    
    dp_energy2, dp_subset_indices2 = dp_subset(x2, k2, s)
    bf_energy2, bf_subset_indices2 = brute_force_subset(x2, k2, s)
    
    print_solution(x2, dp_subset_indices2, dp_energy2, method="DP")
    print_solution(x2, bf_subset_indices2, bf_energy2, method="Brute Force")

if __name__ == '__main__':
    main()
