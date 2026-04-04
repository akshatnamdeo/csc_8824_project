import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dataset_generation import (
    mod_q,
    center_mod_q,
    poly_add,
    poly_sub,
    sample_cbd,
    poly_mul_negacyclic,
    ntt,
    intt,
    base_mul,
    multiply_ntts,
    poly_mul,
    matvec,
    random_poly,
    uniform_matrix,
    low_rank_matrix,
    circulant_matrix,
    toeplitz_matrix,
    sparse_matrix,
    generate_mlwe_sample,
    generate_uniform_sample,
    generate_dataset,
    test_poly_mul_correctness,
    test_ntt_roundtrip,
    test_convolution_theorem,
    test_ntt_multiple_trials
)

from evaluation import (
    coeff_variance,
    variance_threshold,
    variance_distinguisher,
    evaluate_dataset,
    cohens_d,
    binomial_ci,
    run_matrix_family_experiment,
    run_cohens_d_experiment,
    sweep_sparse_rho,
    sparse_matrix_exact,
    generate_dataset_exact_sparse,
    sweep_exact_sparse,
    compare_var_vs_logreg_exact,
    logreg_accuracy,
    compare_all_methods,
    time_matvec,
    time_matrix_gen,
    run_timing_experiment
)

seed = 0
rng = np.random.default_rng(seed)
print(f"Random seed set to {seed}")

os.makedirs("graphs", exist_ok=True)


# Base configuration (no globals anymore)
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


# Polynomial / NTT correctness checks

a = random_poly(config)
b = random_poly(config)

c1 = poly_mul_negacyclic(a, b, config)
c2 = poly_mul(a, b, config)

print("Multiplication correct:", np.all(c1 == c2))


# Verify NTT inverse correctness
f = random_poly(config)
fh = ntt(f, config)
f2 = intt(fh, config)

print("roundtrip correct:", np.all(f % config["q"] == f2 % config["q"]))


# Verify convolution theorem
a = random_poly(config)
b = random_poly(config)

lhs = ntt(poly_mul_negacyclic(a, b, config), config)
rhs = multiply_ntts(ntt(a, config), ntt(b, config), config)

print("convolution theorem correct:", np.all(lhs % config["q"] == rhs % config["q"]))


# Extra randomized verification
for _ in range(100):
    a = random_poly(config)
    b = random_poly(config)
    assert np.all(poly_mul(a, b, config) == poly_mul_negacyclic(a, b, config))

print("NTT verified on 100 random tests")


# Sample + dataset sanity checks

A = circulant_matrix(config)
s, e, b = generate_mlwe_sample(A, config)

print("Example polynomial from b[0]:")
print(b[0][:10])


A, dataset = generate_dataset(config, N=10, matrix_type="lowrank", t=1)
num_mlwe = sum(l for _, l in dataset)

print(f"dataset size: {len(dataset)}, label distribution: {num_mlwe} MLWE / {len(dataset)-num_mlwe} uniform")


# Variance-based distinguisher

configs = [
    ("lowrank", dict(t=1)),
    ("lowrank", dict(t=2)),
    ("circulant", dict()),
    ("toeplitz", dict()),
    ("sparse", dict(rho=0.3)),
    ("sparse", dict(rho=0.5)),
    ("sparse", dict(rho=0.8)),
]

N = 5000

print(f"{'matrix type':<20} {'threshold':>12} {'accuracy':>10}")
print(f"{'':->20} {'':->12} {'':->10}")

variance_results = []

for mtype, mkwargs in configs:
    A, dataset = generate_dataset(config, N=N, matrix_type=mtype, **mkwargs)

    tau = variance_threshold(config)

    correct = 0
    for b, label in dataset:
        pred = variance_distinguisher(b, tau, config)
        if pred == label:
            correct += 1

    acc = correct / N

    label_str = mtype if not mkwargs else f"{mtype}({', '.join(f'{k}={v}' for k,v in mkwargs.items())})"
    variance_results.append((label_str, tau, acc))
    print(f"{label_str:<20} {tau:>12.1f} {acc:>10.4f}")

plt.figure(figsize=(10, 5))
plt.bar([x[0] for x in variance_results], [x[2] for x in variance_results])
plt.xticks(rotation=0, ha="center")
plt.ylabel("Accuracy")
plt.title("Variance Distinguisher Accuracy by Matrix Type")
plt.tight_layout()
plt.savefig("graphs/variance_distinguisher_accuracy_by_matrix_type.png", dpi=300, bbox_inches="tight")
plt.close()


# Cohen's d experiment

print(f"{'matrix type':<20} {'cohen d':>10} {'mlwe mean':>12} {'unif mean':>12} {'sep/sigma':>12}")

cohen_results = []

for mtype, mkwargs in configs:
    A, _ = generate_dataset(config, N=1, matrix_type=mtype, **mkwargs)

    d, m1, m2, s1, s2 = cohens_d(A, config)
    sep = (m2 - m1) / ((s1 + s2) / 2)

    label_str = mtype if not mkwargs else f"{mtype}({', '.join(f'{k}={v}' for k,v in mkwargs.items())})"
    cohen_results.append((label_str, d, m1, m2, sep))
    print(f"{label_str:<20} {d:>10.4f} {m1:>12.1f} {m2:>12.1f} {sep:>12.4f}")

plt.figure(figsize=(10, 5))
plt.bar([x[0] for x in cohen_results], [x[1] for x in cohen_results])
plt.xticks(rotation=0, ha="center")
plt.ylabel("Cohen's d")
plt.title("Effect Size by Matrix Type")
plt.tight_layout()
plt.savefig("graphs/effect_size_by_matrix_type.png", dpi=300, bbox_inches="tight")
plt.close()


# Sweep over rho for sparse matrices

rho_values = np.arange(0.05, 1.0, 0.05)
N_sweep = 5000

print(f"{'rho':>6} {'accuracy':>10} {'ci_lo':>8} {'ci_hi':>8} {'cohen_d':>10}")

rho_results = []

for rho in rho_values:
    rho = round(rho, 2)

    A, dataset = generate_dataset(config, N=N_sweep, matrix_type="sparse", rho=rho)

    tau = variance_threshold(config)

    correct = sum(
        1 for b, label in dataset
        if variance_distinguisher(b, tau, config) == label
    )

    acc = correct / N_sweep
    lo, hi = binomial_ci(correct, N_sweep)

    d, _, _, _, _ = cohens_d(A, config, n_samples=300)
    rho_results.append((rho, acc, lo, hi, d))

    print(f"{rho:>6.2f} {acc:>10.4f} {lo:>8.4f} {hi:>8.4f} {d:>10.4f}")

rhos = [x[0] for x in rho_results]
accs = [x[1] for x in rho_results]
ci_los = [x[2] for x in rho_results]
ci_his = [x[3] for x in rho_results]
d_vals = [x[4] for x in rho_results]

plt.figure(figsize=(8, 5))
plt.plot(rhos, accs, marker="o")
plt.fill_between(rhos, ci_los, ci_his, alpha=0.2)
plt.xlabel("rho")
plt.ylabel("Accuracy")
plt.title("Sparse Matrix Accuracy vs rho")
plt.tight_layout()
plt.savefig("graphs/sparse_matrix_accuracy_vs_rho.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(rhos, d_vals, marker="o")
plt.xlabel("rho")
plt.ylabel("Cohen's d")
plt.title("Sparse Matrix Effect Size vs rho")
plt.tight_layout()
plt.savefig("graphs/sparse_matrix_effect_size_vs_rho.png", dpi=300, bbox_inches="tight")
plt.close()


# Exact sparsity experiment

print(f"{'nonzero':>8} {'accuracy':>10} {'ci_lo':>8} {'ci_hi':>8} {'cohen_d':>10} {'mlwe_var':>12} {'unif_var':>12}")

exact_results = []

for nz in range(1, 10):
    A, dataset = generate_dataset_exact_sparse(config, N=5000, n_nonzero=nz)

    tau = variance_threshold(config)

    correct = sum(
        1 for b, label in dataset
        if variance_distinguisher(b, tau, config) == label
    )

    acc = correct / 5000
    lo, hi = binomial_ci(correct, 5000)

    d, m1, m2, _, _ = cohens_d(A, config, n_samples=300)

    exact_results.append((nz, acc, lo, hi, d, m1, m2))

    print(f"{nz:>8} {acc:>10.4f} {lo:>8.4f} {hi:>8.4f} {d:>10.4f} {m1:>12.1f} {m2:>12.1f}")

nz_vals = [x[0] for x in exact_results]
exact_acc_vals = [x[1] for x in exact_results]
exact_d_vals = [x[4] for x in exact_results]

plt.figure(figsize=(8, 5))
plt.plot(nz_vals, exact_acc_vals, marker="o")
plt.xlabel("Number of Nonzero Entries")
plt.ylabel("Accuracy")
plt.title("Variance Distinguisher Accuracy vs Exact Sparsity")
plt.tight_layout()
plt.savefig("graphs/variance_distinguisher_accuracy_vs_exact_sparsity.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(nz_vals, exact_d_vals, marker="o")
plt.xlabel("Number of Nonzero Entries")
plt.ylabel("Cohen's d")
plt.title("Effect Size vs Exact Sparsity")
plt.tight_layout()
plt.savefig("graphs/effect_size_vs_exact_sparsity.png", dpi=300, bbox_inches="tight")
plt.close()


# Logistic regression comparison

print(f"{'nonzero':>8} {'var_acc':>10} {'logreg_acc':>12}")

logreg_results = []

for nz, var_acc, *_ in exact_results:
    _, dataset = generate_dataset_exact_sparse(config, N=5000, n_nonzero=nz)
    lr_acc = logreg_accuracy(dataset, config)

    logreg_results.append((nz, var_acc, lr_acc))
    print(f"{nz:>8} {var_acc:>10.4f} {lr_acc:>12.4f}")

plt.figure(figsize=(8, 5))
plt.plot([x[0] for x in logreg_results], [x[1] for x in logreg_results], marker="o", label="Variance")
plt.plot([x[0] for x in logreg_results], [x[2] for x in logreg_results], marker="o", label="Logistic Regression")
plt.xlabel("Number of Nonzero Entries")
plt.ylabel("Accuracy")
plt.title("Variance vs Logistic Regression Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/variance_vs_logistic_regression_accuracy.png", dpi=300, bbox_inches="tight")
plt.close()


# Full comparison across matrix families

configs_all = [
    ("uniform", dict()),
    ("lowrank", dict(t=1)),
    ("lowrank", dict(t=2)),
    ("circulant", dict()),
    ("toeplitz", dict()),
    ("sparse", dict(rho=0.3)),
    ("sparse", dict(rho=0.5)),
    ("sparse", dict(rho=0.8)),
]

print(f"{'matrix type':<20} {'var_acc':>10} {'logreg_acc':>12} {'cohen_d':>10}")

comparison_results = []

for mtype, mkwargs in configs_all:
    A, dataset = generate_dataset(config, N=5000, matrix_type=mtype, **mkwargs)

    tau = variance_threshold(config)

    correct = sum(
        1 for b, label in dataset
        if variance_distinguisher(b, tau, config) == label
    )

    var_acc = correct / 5000
    lr_acc = logreg_accuracy(dataset, config)

    d, _, _, _, _ = cohens_d(A, config, n_samples=300)

    label_str = mtype if not mkwargs else f"{mtype}({', '.join(f'{k}={v}' for k,v in mkwargs.items())})"
    comparison_results.append((label_str, var_acc, lr_acc, d))

    print(f"{label_str:<20} {var_acc:>10.4f} {lr_acc:>12.4f} {d:>10.4f}")

labels = [x[0] for x in comparison_results]
var_accs = [x[1] for x in comparison_results]
lr_accs = [x[2] for x in comparison_results]
cohen_ds = [x[3] for x in comparison_results]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width / 2, var_accs, width, label="Variance")
plt.bar(x + width / 2, lr_accs, width, label="LogReg")
plt.xticks(x, labels, rotation=0, ha="center")
plt.ylabel("Accuracy")
plt.title("Method Comparison Across Matrix Families")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/method_comparison_across_matrix_families.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(labels, cohen_ds)
plt.xticks(rotation=0, ha="center")
plt.ylabel("Cohen's d")
plt.title("Cohen's d Across Matrix Families")
plt.tight_layout()
plt.savefig("graphs/cohens_d_across_matrix_families.png", dpi=300, bbox_inches="tight")
plt.close()


# Timing experiment

print(f"{'matrix type':<20} {'gen mean(ms)':>14} {'gen std':>10} {'matvec mean(ms)':>16} {'matvec std':>12} {'speedup':>10}")

baseline_matvec = None
timing_results = []

for mtype, mkwargs in configs_all:
    # Generate matrix
    A, _ = generate_dataset(config, N=1, matrix_type=mtype, **mkwargs)

    # Measure generation time
    gen_mean, gen_std = time_matrix_gen(config, mtype, mkwargs)

    # Measure matvec time (dominant ML-KEM operation)
    mv_mean, mv_std = time_matvec(A, config)

    # First entry = baseline
    if baseline_matvec is None:
        baseline_matvec = mv_mean

    speedup = baseline_matvec / mv_mean

    label_str = mtype if not mkwargs else f"{mtype}({', '.join(f'{k}={v}' for k,v in mkwargs.items())})"
    timing_results.append((label_str, gen_mean, gen_std, mv_mean, mv_std, speedup))

    print(f"{label_str:<20} {gen_mean:>14.3f} {gen_std:>10.3f} {mv_mean:>16.3f} {mv_std:>12.3f} {speedup:>10.3f}")

timing_labels = [x[0] for x in timing_results]
gen_means = [x[1] for x in timing_results]
matvec_means = [x[3] for x in timing_results]
speedups = [x[5] for x in timing_results]

plt.figure(figsize=(10, 5))
plt.bar(timing_labels, gen_means)
plt.xticks(rotation=0, ha="center")
plt.ylabel("Generation Time (ms)")
plt.title("Matrix Generation Time by Matrix Type")
plt.tight_layout()
plt.savefig("graphs/matrix_generation_time_by_matrix_type.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(timing_labels, matvec_means)
plt.xticks(rotation=0, ha="center")
plt.ylabel("Matvec Time (ms)")
plt.title("Matvec Time by Matrix Type")
plt.tight_layout()
plt.savefig("graphs/matvec_time_by_matrix_type.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(timing_labels, speedups)
plt.xticks(rotation=0, ha="center")
plt.ylabel("Speedup")
plt.title("Speedup by Matrix Type")
plt.tight_layout()
plt.savefig("graphs/speedup_by_matrix_type.png", dpi=300, bbox_inches="tight")
plt.close()