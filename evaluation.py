import numpy as np
import time
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dataset_generation import (
    center_mod_q,
    generate_uniform_sample,
    generate_mlwe_sample,
    generate_dataset,
    matvec,
    sample_cbd,
    uniform_matrix,
    low_rank_matrix,
    circulant_matrix,
    toeplitz_matrix,
    sparse_matrix,
    random_poly
)


def coeff_variance(b, config):
    coeffs = np.array(
        [center_mod_q(c, config) for poly in b for c in poly],
        dtype=float
    )
    return np.var(coeffs)


def variance_threshold(config, n_calib=200):
    vars_uniform = []

    for _ in range(n_calib):
        u = generate_uniform_sample(config)
        vars_uniform.append(coeff_variance(u, config))

    return np.median(vars_uniform)


def variance_distinguisher(b, tau, config):
    return 1 if coeff_variance(b, config) < tau else 0


def evaluate_dataset(dataset, tau, config):
    correct = sum(
        1 for b, label in dataset
        if variance_distinguisher(b, tau, config) == label
    )
    return correct / len(dataset)


def cohens_d(A, config, n_samples=500):
    vars_mlwe = []
    vars_unif = []

    for _ in range(n_samples):
        _, _, b = generate_mlwe_sample(A, config)
        vars_mlwe.append(coeff_variance(b, config))

        u = generate_uniform_sample(config)
        vars_unif.append(coeff_variance(u, config))

    m1, m2 = np.mean(vars_mlwe), np.mean(vars_unif)
    s1, s2 = np.std(vars_mlwe), np.std(vars_unif)

    s_pooled = np.sqrt((s1**2 + s2**2) / 2)
    return (m2 - m1) / s_pooled, m1, m2, s1, s2


def binomial_ci(correct, n, alpha=0.05):
    lo = binom.ppf(alpha/2, n, correct/n) / n if correct > 0 else 0.0
    hi = binom.ppf(1 - alpha/2, n, correct/n) / n if correct < n else 1.0
    return lo, hi


def run_matrix_family_experiment(config, configs, N=5000):
    results = []

    for mtype, mkwargs in configs:
        A, dataset = generate_dataset(
            config,
            N=N,
            matrix_type=mtype,
            **mkwargs
        )

        tau = variance_threshold(config)
        acc = evaluate_dataset(dataset, tau, config)

        label_str = mtype if not mkwargs else f"{mtype}({', '.join(f'{k}={v}' for k,v in mkwargs.items())})"
        results.append((label_str, tau, acc))

    return results


def run_cohens_d_experiment(config, configs):
    results = []

    for mtype, mkwargs in configs:
        A, _ = generate_dataset(
            config,
            N=1,
            matrix_type=mtype,
            **mkwargs
        )

        d, m1, m2, s1, s2 = cohens_d(A, config)
        sep = (m2 - m1) / ((s1 + s2) / 2)

        label_str = mtype if not mkwargs else f"{mtype}({', '.join(f'{k}={v}' for k,v in mkwargs.items())})"
        results.append((label_str, d, m1, m2, sep))

    return results


def sweep_sparse_rho(config, rho_values, N=5000):
    results = []

    for rho in rho_values:
        rho = round(rho, 2)

        A, dataset = generate_dataset(
            config,
            N=N,
            matrix_type="sparse",
            rho=rho
        )

        tau = variance_threshold(config)

        correct = sum(
            1 for b, label in dataset
            if variance_distinguisher(b, tau, config) == label
        )

        acc = correct / N
        lo, hi = binomial_ci(correct, N)

        d, _, _, _, _ = cohens_d(A, config, n_samples=300)

        results.append((rho, acc, lo, hi, d))

    return results

def sparse_matrix_exact(config, n_nonzero):
    k = config["k"]
    n = config["n"]
    rng = config["rng"]

    total = k * k
    positions = set(rng.choice(total, n_nonzero, replace=False))

    A = []

    for i in range(k):
        row = []
        for j in range(k):
            idx = i * k + j
            if idx in positions:
                row.append(random_poly(config))
            else:
                row.append(np.zeros(n, dtype=int))
        A.append(row)

    return A


def generate_dataset_exact_sparse(config, N=5000, n_nonzero=1):
    A = sparse_matrix_exact(config, n_nonzero)

    dataset = []

    for _ in range(N):
        if config["rng"].random() < 0.5:
            _, _, b = generate_mlwe_sample(A, config)
            dataset.append((b, 1))
        else:
            b = generate_uniform_sample(config)
            dataset.append((b, 0))

    return A, dataset


def sweep_exact_sparse(config, max_nonzero=9, N=5000):
    results = []

    for nz in range(1, max_nonzero + 1):
        A, dataset = generate_dataset_exact_sparse(
            config,
            N=N,
            n_nonzero=nz
        )

        tau = variance_threshold(config)

        correct = sum(
            1 for b, label in dataset
            if variance_distinguisher(b, tau, config) == label
        )

        acc = correct / N
        lo, hi = binomial_ci(correct, N)

        d, m1, m2, _, _ = cohens_d(A, config, n_samples=300)

        results.append((nz, acc, lo, hi, d, m1, m2))

    return results


def compare_var_vs_logreg_exact(config, exact_results, N=5000):
    results = []

    for nz, var_acc, *_ in exact_results:
        _, dataset = generate_dataset_exact_sparse(
            config,
            N=N,
            n_nonzero=nz
        )

        lr_acc = logreg_accuracy(dataset, config)

        results.append((nz, var_acc, lr_acc))

    return results

def logreg_accuracy(dataset, config, n_train=4000):
    X = np.array([
        [center_mod_q(c, config) for poly in b for c in poly]
        for b, _ in dataset
    ], dtype=float)

    y = np.array([label for _, label in dataset])

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)


def compare_all_methods(config, configs, N=5000):
    results = []

    for mtype, mkwargs in configs:
        A, dataset = generate_dataset(
            config,
            N=N,
            matrix_type=mtype,
            **mkwargs
        )

        tau = variance_threshold(config)

        correct = sum(
            1 for b, label in dataset
            if variance_distinguisher(b, tau, config) == label
        )

        var_acc = correct / N
        lr_acc = logreg_accuracy(dataset, config)
        d, _, _, _, _ = cohens_d(A, config, n_samples=300)

        label_str = mtype if not mkwargs else f"{mtype}({', '.join(f'{k}={v}' for k,v in mkwargs.items())})"
        results.append((label_str, var_acc, lr_acc, d))

    return results


def time_matvec(A, config, n_trials=200):
    s = [sample_cbd(config) for _ in range(config["k"])]

    times = []

    for _ in range(n_trials):
        t0 = time.perf_counter()
        matvec(A, s, config)
        times.append(time.perf_counter() - t0)

    return np.mean(times) * 1000, np.std(times) * 1000


def time_matrix_gen(config, matrix_type, mkwargs, n_trials=100):
    times = []

    for _ in range(n_trials):
        t0 = time.perf_counter()

        if matrix_type == "uniform":
            uniform_matrix(config)
        elif matrix_type == "lowrank":
            low_rank_matrix(config, **mkwargs)
        elif matrix_type == "circulant":
            circulant_matrix(config)
        elif matrix_type == "toeplitz":
            toeplitz_matrix(config)
        elif matrix_type == "sparse":
            sparse_matrix(config, **mkwargs)

        times.append(time.perf_counter() - t0)

    return np.mean(times) * 1000, np.std(times) * 1000


def run_timing_experiment(config, configs):
    results = []
    baseline = None

    for mtype, mkwargs in configs:
        A, _ = generate_dataset(
            config,
            N=1,
            matrix_type=mtype,
            **mkwargs
        )

        gen_mean, gen_std = time_matrix_gen(config, mtype, mkwargs)
        mv_mean, mv_std = time_matvec(A, config)

        if baseline is None:
            baseline = mv_mean

        speedup = baseline / mv_mean

        label_str = mtype if not mkwargs else f"{mtype}({', '.join(f'{k}={v}' for k,v in mkwargs.items())})"

        results.append((label_str, gen_mean, gen_std, mv_mean, mv_std, speedup))

    return results