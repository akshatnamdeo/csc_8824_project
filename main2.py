import os
import numpy as np
import time
from scipy.stats import binom
import numpy as np
import time
from scipy.stats import binom
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dataset_generation import (
    poly_add, poly_mul, matvec, random_poly, sample_cbd,
    generate_mlwe_sample, generate_uniform_sample, uniform_matrix
)
from evaluation import (
    coeff_variance, center_mod_q, variance_threshold,
    variance_distinguisher, cohens_d, binomial_ci, logreg_accuracy
)

import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

seed = 0
rng = np.random.default_rng(seed)

os.makedirs("graphs", exist_ok=True)

# Base configuration
config = {
    "n": 256,
    "q": 3329,
    "k": 3,
    "eta": 2,
    "rng": rng,
}


# NTT tables (Appendix A)
zetas = np.array([
    1,1729,2580,3289,2642,630,1897,848,
    1062,1919,193,797,2786,3260,569,1746,
    296,2447,1339,1476,3046,56,2240,1333,
    1426,2094,535,2882,2393,2879,1974,821,
    289,331,3253,1756,1197,2304,2277,2055,
    650,1977,2513,632,2865,33,1320,1915,
    2319,1435,807,452,1438,2868,1534,2402,
    2647,2617,1481,648,2474,3110,1227,910,
    17,2761,583,2649,1637,723,2288,1100,
    1409,2662,3281,233,756,2156,3015,3050,
    1703,1651,2789,1789,1847,952,1461,2687,
    939,2308,2437,2388,733,2337,268,641,
    1584,2298,2037,3220,375,2549,2090,1645,
    1063,319,2773,757,2099,561,2466,2594,
    2804,1092,403,1026,1143,2150,2775,886,
    1722,1212,1874,1029,2110,2935,885,2154
], dtype=np.int64)

gammas = np.array([
    17,-17,2761,-2761,583,-583,2649,-2649,
    1637,-1637,723,-723,2288,-2288,1100,-1100,
    1409,-1409,2662,-2662,3281,-3281,233,-233,
    756,-756,2156,-2156,3015,-3015,3050,-3050,
    1703,-1703,1651,-1651,2789,-2789,1789,-1789,
    1847,-1847,952,-952,1461,-1461,2687,-2687,
    939,-939,2308,-2308,2437,-2437,2388,-2388,
    733,-733,2337,-2337,268,-268,641,-641,
    1584,-1584,2298,-2298,2037,-2037,3220,-3220,
    375,-375,2549,-2549,2090,-2090,1645,-1645,
    1063,-1063,319,-319,2773,-2773,757,-757,
    2099,-2099,561,-561,2466,-2466,2594,-2594,
    2804,-2804,1092,-1092,403,-403,1026,-1026,
    1143,-1143,2150,-2150,2775,-2775,886,-886,
    1722,-1722,1212,-1212,1874,-1874,1029,-1029,
    2110,-2110,2935,-2935,885,-885,2154,-2154
], dtype=np.int64) % config["q"]

config["zetas"] = zetas
config["gammas"] = gammas

def sparse_random(config, nz):
    # Baseline: nz nonzero entries placed uniformly at random
    k, n = config["k"], config["n"]
    positions = config["rng"].choice(k*k, nz, replace=False)
    positions = set(positions)
    return [[random_poly(config) if i*k+j in positions
             else np.zeros(n, dtype=int)
             for j in range(k)] for i in range(k)]

def sparse_row_covered(config, nz):
    # Every row has at least one nonzero; remaining nz-k entries placed randomly
    k, n = config["k"], config["n"]
    assert nz >= k, "nz must be >= k to guarantee row coverage"

    # Guarantee one nonzero per row
    positions = set()
    for i in range(k):
        j = config["rng"].integers(0, k)
        positions.add(i*k + j)

    # Place remaining entries randomly among uncovered positions
    remaining = list(set(range(k*k)) - positions)
    extra = config["rng"].choice(remaining, nz - k, replace=False)
    positions.update(extra)

    return [[random_poly(config) if i*k+j in positions
             else np.zeros(n, dtype=int)
             for j in range(k)] for i in range(k)]

def sparse_row_col_covered(config, nz):
    # Every row and column has at least one nonzero
    k, n = config["k"], config["n"]
    assert nz >= k, "nz must be >= k for full row+col coverage"

    # Use a random permutation matrix as the coverage seed
    perm = config["rng"].permutation(k)
    positions = set(i*k + perm[i] for i in range(k))

    # Place remaining entries randomly
    remaining = list(set(range(k*k)) - positions)
    if nz - k > 0:
        extra = config["rng"].choice(remaining, nz - k, replace=False)
        positions.update(extra)

    return [[random_poly(config) if i*k+j in positions
             else np.zeros(n, dtype=int)
             for j in range(k)] for i in range(k)]

def sparse_balanced(config, nz):
    # Row weights differ by at most 1: floor(nz/k) or ceil(nz/k) per row
    k, n = config["k"], config["n"]
    base, extra = divmod(nz, k)

    positions = set()
    for i in range(k):
        row_nz = base + (1 if i < extra else 0)
        cols = config["rng"].choice(k, row_nz, replace=False)
        for j in cols:
            positions.add(i*k + j)

    return [[random_poly(config) if i*k+j in positions
             else np.zeros(n, dtype=int)
             for j in range(k)] for i in range(k)]

def sparse_diagonal(config, nz):
    # Nonzero entries on diagonal first, then random off-diagonal extras
    k, n = config["k"], config["n"]
    assert nz >= k, "nz must be >= k to fill diagonal"

    positions = set(i*k + i for i in range(k))
    remaining = list(set(range(k*k)) - positions)
    if nz - k > 0:
        extra = config["rng"].choice(remaining, nz - k, replace=False)
        positions.update(extra)

    return [[random_poly(config) if i*k+j in positions
             else np.zeros(n, dtype=int)
             for j in range(k)] for i in range(k)]
    
def generate_dataset_structured_sparse(config, A_fn, nz, N=5000):
    A = A_fn(config, nz)
    dataset = []
    for _ in range(N):
        if config["rng"].random() < 0.5:
            _, _, b = generate_mlwe_sample(A, config)
            dataset.append((b, 1))
        else:
            b = generate_uniform_sample(config)
            dataset.append((b, 0))
    return A, dataset

def matvec_sparse_aware(A, s, config):
    # Skip zero polynomial entries, reducing multiplications to nz
    k, n = config["k"], config["n"]
    b = [np.zeros(n, dtype=int) for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if not np.all(A[i][j] == 0):
                prod = poly_mul(A[i][j], s[j], config)
                b[i] = poly_add(b[i], prod, config)
    return b

matvec = matvec_sparse_aware

def time_matvec_fn(A, config, fn, n_trials=500):
    s = [sample_cbd(config) for _ in range(config["k"])]
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn(A, s, config)
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000

# Generate one matrix of each relevant type
A_uniform   = uniform_matrix(config)
A_rc_nz3    = sparse_row_covered(config, 3)
A_rc_nz4    = sparse_row_covered(config, 4)
A_rc_nz5    = sparse_row_covered(config, 5)
A_rc_nz6    = sparse_row_covered(config, 6)
A_rand_nz3  = sparse_random(config, 3)

configs_timing = [
    ("uniform (dense)",      A_uniform,  matvec),
    ("row-covered nz=3",     A_rc_nz3,   matvec_sparse_aware),
    ("row-covered nz=4",     A_rc_nz4,   matvec_sparse_aware),
    ("row-covered nz=5",     A_rc_nz5,   matvec_sparse_aware),
    ("row-covered nz=6",     A_rc_nz6,   matvec_sparse_aware),
    ("random sparse nz=3",   A_rand_nz3, matvec_sparse_aware),
]

baseline = None
timing_results = []
print(f"{'matrix type':<24} {'mean(ms)':>10} {'std(ms)':>8} {'speedup':>10} {'muls':>6}")

for label, A, fn in configs_timing:
    mean_t, std_t = time_matvec_fn(A, config, fn)
    if baseline is None:
        baseline = mean_t
    speedup = baseline / mean_t

    # Count actual nonzero entries
    nz = sum(1 for i in range(config["k"]) for j in range(config["k"])
             if not np.all(A[i][j] == 0))

    timing_results.append((label, mean_t, std_t, speedup, nz))
    print(f"{label:<24} {mean_t:>10.3f} {std_t:>8.3f} {speedup:>10.3f}x {nz:>6}")

labels = [x[0] for x in timing_results]
mean_times = [x[1] for x in timing_results]
std_times = [x[2] for x in timing_results]
speedups = [x[3] for x in timing_results]
muls = [x[4] for x in timing_results]

plt.figure(figsize=(10, 5))
plt.bar(labels, mean_times, yerr=std_times, capsize=5)
plt.xticks(rotation=0, ha="center")
plt.ylabel("Mean Time (ms)")
plt.title("Sparse-Aware Matvec Timing")
plt.tight_layout()
plt.savefig("graphs/sparse_aware_matvec_timing.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(labels, speedups)
plt.xticks(rotation=0, ha="center")
plt.ylabel("Speedup")
plt.title("Sparse-Aware Matvec Speedup")
plt.tight_layout()
plt.savefig("graphs/sparse_aware_matvec_speedup.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(labels, muls)
plt.xticks(rotation=0, ha="center")
plt.ylabel("Nonzero Entries / Multiplications")
plt.title("Actual Nonzero Entry Counts")
plt.tight_layout()
plt.savefig("graphs/sparse_aware_nonzero_counts.png", dpi=300, bbox_inches="tight")
plt.close()