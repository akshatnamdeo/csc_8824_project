import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

from dataset_generation import (
    poly_add, poly_mul, random_poly, sample_cbd,
    generate_mlwe_sample, generate_uniform_sample, uniform_matrix
)
from evaluation import (
    coeff_variance, center_mod_q, variance_threshold,
    variance_distinguisher, cohens_d, binomial_ci, logreg_accuracy
)

seed = 0
rng = np.random.default_rng(seed)
print(f"Random seed set to {seed}")

os.makedirs("graphs", exist_ok=True)

config = {
    "n": 256,
    "q": 3329,
    "k": 3,
    "eta": 2,
    "rng": rng,
}

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
    k, n = config["k"], config["n"]
    positions = set(config["rng"].choice(k*k, nz, replace=False))
    return [[random_poly(config) if i*k+j in positions
             else np.zeros(n, dtype=int)
             for j in range(k)] for i in range(k)]

def sparse_row_covered(config, nz):
    k, n = config["k"], config["n"]
    positions = set()
    for i in range(k):
        j = int(config["rng"].integers(0, k))
        positions.add(i*k + j)
    remaining = list(set(range(k*k)) - positions)
    if nz - k > 0:
        extra = config["rng"].choice(remaining, nz - k, replace=False)
        positions.update(extra)
    return [[random_poly(config) if i*k+j in positions
             else np.zeros(n, dtype=int)
             for j in range(k)] for i in range(k)]

def sparse_row_col_covered(config, nz):
    k, n = config["k"], config["n"]
    perm = config["rng"].permutation(k)
    positions = set(i*k + int(perm[i]) for i in range(k))
    remaining = list(set(range(k*k)) - positions)
    if nz - k > 0:
        extra = config["rng"].choice(remaining, nz - k, replace=False)
        positions.update(extra)
    return [[random_poly(config) if i*k+j in positions
             else np.zeros(n, dtype=int)
             for j in range(k)] for i in range(k)]

def sparse_balanced(config, nz):
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
    k, n = config["k"], config["n"]
    positions = set(i*k + i for i in range(k))
    remaining = list(set(range(k*k)) - positions)
    if nz - k > 0:
        extra = config["rng"].choice(remaining, nz - k, replace=False)
        positions.update(extra)
    return [[random_poly(config) if i*k+j in positions
             else np.zeros(n, dtype=int)
             for j in range(k)] for i in range(k)]

def count_uncovered_rows(A, config):
    # Number of rows where every entry is the zero polynomial
    k = config["k"]
    return sum(1 for i in range(k) if all(np.all(A[i][j] == 0) for j in range(k)))

def generate_dataset_from_fn(config, A_fn, nz, N=5000):
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


# Experiment 1: five-strategy placement comparison at nz = 3, 4, 5
# nz=3 is the critical budget where random sparse is distinguishable
# nz=4 is just above the threshold
# nz=5 serves as confirmation

strategies = [
    ("random",        sparse_random),
    ("row-covered",   sparse_row_covered),
    ("row+col-cov",   sparse_row_col_covered),
    ("balanced",      sparse_balanced),
    ("diagonal",      sparse_diagonal),
]

N = 5000
nz_values = [3, 4, 5]

print("Five-Strategy Placement Comparison")
print(f"{'strategy':<16} {'nz':>4} {'var_acc':>9} {'logreg_acc':>12} {'cohen_d':>10} {'ci_lo':>7} {'ci_hi':>7}")

placement_results = {}

for nz in nz_values:
    for name, fn in strategies:
        A, dataset = generate_dataset_from_fn(config, fn, nz, N=N)

        tau = variance_threshold(config)
        correct = sum(
            1 for b, label in dataset
            if variance_distinguisher(b, tau, config) == label
        )
        acc = correct / N
        lo, hi = binomial_ci(correct, N)
        d, _, _, _, _ = cohens_d(A, config, n_samples=300)
        lr_acc = logreg_accuracy(dataset, config)

        key = (nz, name)
        placement_results[key] = (acc, lr_acc, d, lo, hi)
        print(f"{name:<16} {nz:>4} {acc:>9.4f} {lr_acc:>12.4f} {d:>10.4f} {lo:>7.4f} {hi:>7.4f}")

# Plot: var_acc per strategy at each nz
fig, axes = plt.subplots(1, len(nz_values), figsize=(14, 5), sharey=True)

for ax, nz in zip(axes, nz_values):
    names = [s[0] for s in strategies]
    accs = [placement_results[(nz, n)][0] for n in names]
    ax.bar(names, accs)
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"nz = {nz}")
    ax.set_ylabel("Variance Accuracy") if nz == nz_values[0] else None
    ax.set_ylim(0.4, 0.85)
    ax.tick_params(axis="x", rotation=15)

plt.suptitle("Placement Strategy Comparison: Variance Distinguisher Accuracy")
plt.tight_layout()
plt.savefig("graphs/placement_strategy_comparison_var_acc.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot: cohen_d per strategy at each nz
fig, axes = plt.subplots(1, len(nz_values), figsize=(14, 5), sharey=True)

for ax, nz in zip(axes, nz_values):
    names = [s[0] for s in strategies]
    ds = [placement_results[(nz, n)][2] for n in names]
    ax.bar(names, ds)
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"nz = {nz}")
    ax.set_ylabel("Cohen's d") if nz == nz_values[0] else None
    ax.tick_params(axis="x", rotation=15)

plt.suptitle("Placement Strategy Comparison: Cohen's d")
plt.tight_layout()
plt.savefig("graphs/placement_strategy_comparison_cohens_d.png", dpi=300, bbox_inches="tight")
plt.close()


# Experiment 2: rho anomaly diagnostic
# For each rho value, generate 50 matrices and record what fraction
# have at least one uncovered row. Then correlate that with the
# observed accuracy from main.py.

print("\nRho Anomaly Diagnostic: Fraction of Matrices with Uncovered Rows")
print(f"{'rho':>6} {'p(uncov_row)':>14} {'expected_nz':>14}")

rho_values = np.round(np.arange(0.05, 1.0, 0.05), 2)
n_matrices = 200

rho_diag_results = []

for rho in rho_values:
    uncovered_count = 0
    for _ in range(n_matrices):
        A = sparse_random(config, max(1, round(rho * config["k"]**2)))
        if count_uncovered_rows(A, config) > 0:
            uncovered_count += 1
    p_uncov = uncovered_count / n_matrices
    expected_nz = round(rho * config["k"]**2, 2)
    rho_diag_results.append((rho, p_uncov, expected_nz))
    print(f"{rho:>6.2f} {p_uncov:>14.4f} {expected_nz:>14.2f}")

rhos = [x[0] for x in rho_diag_results]
p_uncovs = [x[1] for x in rho_diag_results]

plt.figure(figsize=(9, 5))
plt.plot(rhos, p_uncovs, marker="o")
plt.axhline(0.0, linestyle="--", color="gray", linewidth=1)
plt.xlabel("rho")
plt.ylabel("P(at least one uncovered row)")
plt.title("Fraction of Random Sparse Matrices with Uncovered Rows vs rho")
plt.tight_layout()
plt.savefig("graphs/rho_uncovered_row_probability.png", dpi=300, bbox_inches="tight")
plt.close()

# Overlay with observed accuracy from main.py to show the correlation
# Hard-coded from main.py logs
rho_acc_from_main = [
    (0.05, 0.7356), (0.10, 0.7430), (0.15, 0.7606), (0.20, 0.7446),
    (0.25, 0.7318), (0.30, 0.7408), (0.35, 0.4974), (0.40, 0.5002),
    (0.45, 0.7632), (0.50, 0.7584), (0.55, 0.4934), (0.60, 0.5006),
    (0.65, 0.5104), (0.70, 0.5008), (0.75, 0.4938), (0.80, 0.5046),
    (0.85, 0.4880), (0.90, 0.5096), (0.95, 0.5020),
]

rho_acc_rhos = [x[0] for x in rho_acc_from_main]
rho_accs = [x[1] for x in rho_acc_from_main]

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(rho_acc_rhos, rho_accs, marker="o", color="steelblue", label="Var. accuracy (main.py)")
ax2.plot(rhos, p_uncovs, marker="s", color="tomato", linestyle="--", label="P(uncovered row)")

ax1.set_xlabel("rho")
ax1.set_ylabel("Variance Accuracy", color="steelblue")
ax2.set_ylabel("P(at least one uncovered row)", color="tomato")
ax1.axhline(0.5, linestyle=":", color="gray", linewidth=1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Variance Accuracy vs Probability of Uncovered Row (rho sweep)")
plt.tight_layout()
plt.savefig("graphs/rho_accuracy_vs_uncovered_row_probability.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nDone. Graphs saved to graphs/")