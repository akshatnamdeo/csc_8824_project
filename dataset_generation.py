import numpy as np
import time
from scipy.stats import binom


def mod_q(x, config):
    """
    Reduce value modulo q.
    """
    return x % config["q"]


def center_mod_q(x, config):
    """
    Map coefficient into interval (-q/2, q/2].
    """
    q = config["q"]
    y = x % q
    if y > q // 2:
        y -= q
    return y


def poly_add(a, b, config):
    """
    Coordinate-wise polynomial addition in Z_q^n.
    """
    return (a + b) % config["q"]


def poly_sub(a, b, config):
    """
    Coordinate-wise polynomial subtraction in Z_q^n.
    """
    return (a - b) % config["q"]


def sample_cbd(config):
    """
    Sample polynomial from centered binomial distribution.
    """
    n = config["n"]
    q = config["q"]
    eta = config["eta"]
    rng = config["rng"]

    f = np.zeros(n, dtype=int)

    for i in range(n):
        a = rng.binomial(1, 0.5, eta).sum()
        b = rng.binomial(1, 0.5, eta).sum()
        f[i] = a - b

    return f % q


def poly_mul_negacyclic(a, b, config):
    """
    Reference negacyclic multiplication in R_q.
    """
    n = config["n"]
    q = config["q"]

    c = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < n:
                c[idx] += a[i] * b[j]
            else:
                c[idx - n] -= a[i] * b[j]

    return c % q


def ntt(f, config):
    """
    Forward NTT.
    """
    q = config["q"]
    zetas = config["zetas"]

    f = f.copy()
    k_ntt = 1
    length = config["n"] // 2

    while length >= 2:
        start = 0
        while start < config["n"]:
            z = zetas[k_ntt]
            k_ntt += 1

            for j in range(start, start + length):
                t = (z * f[j + length]) % q
                f[j + length] = (f[j] - t) % q
                f[j] = (f[j] + t) % q

            start += 2 * length

        length //= 2

    return f


def intt(fhat, config):
    """
    Inverse NTT.
    """
    q = config["q"]
    zetas = config["zetas"]

    f = fhat.copy().astype(np.int64)
    k_ntt = len(zetas) - 1
    length = 2

    while length <= config["n"] // 2:
        start = 0
        while start < config["n"]:
            z = zetas[k_ntt]
            k_ntt -= 1

            for j in range(start, start + length):
                t = f[j]
                f[j] = (t + f[j + length]) % q
                f[j + length] = (z * (f[j + length] - t)) % q

            start += 2 * length

        length *= 2

    inv_n = pow(config["n"] // 2, -1, q)
    return (f * inv_n) % q


def base_mul(a0, a1, b0, b1, gamma, config):
    """
    Base case multiplication in NTT domain.
    """
    q = config["q"]

    c0 = (a0 * b0 + gamma * a1 * b1) % q
    c1 = (a0 * b1 + a1 * b0) % q

    return c0, c1


def multiply_ntts(fhat, ghat, config):
    """
    Multiply two polynomials in NTT domain.
    """
    q = config["q"]
    gammas = config["gammas"]

    h = np.zeros(config["n"], dtype=np.int64)

    for i in range(0, config["n"], 2):
        gamma = gammas[i // 2]
        h[i], h[i + 1] = base_mul(
            fhat[i], fhat[i + 1],
            ghat[i], ghat[i + 1],
            gamma,
            config
        )

    return h


def poly_mul(a, b, config):
    """
    Fast polynomial multiplication using NTT.
    """
    ah = ntt(a, config)
    bh = ntt(b, config)
    ch = multiply_ntts(ah, bh, config)
    return intt(ch, config)


def matvec(A, s, config):
    """
    Matrix-vector multiplication over R_q.
    """
    k = config["k"]
    n = config["n"]

    b = []

    for i in range(k):
        acc = np.zeros(n, dtype=int)

        for j in range(k):
            prod = poly_mul(A[i][j], s[j], config)
            acc = poly_add(acc, prod, config)

        b.append(acc)

    return b


def random_poly(config):
    """
    Uniform polynomial in Z_q^n.
    """
    return config["rng"].integers(0, config["q"], size=config["n"])


def uniform_matrix(config):
    """
    Fully random matrix A ∈ R_q^(k×k).
    """
    k = config["k"]
    return [[random_poly(config) for _ in range(k)] for _ in range(k)]


def low_rank_matrix(config, t=1):
    """
    Generate matrix with rank ≤ t using A = U * V.
    """
    k = config["k"]
    n = config["n"]

    U = [[random_poly(config) for _ in range(t)] for _ in range(k)]
    V = [[random_poly(config) for _ in range(k)] for _ in range(t)]

    A = [[np.zeros(n, dtype=int) for _ in range(k)] for _ in range(k)]

    for i in range(k):
        for j in range(k):
            acc = np.zeros(n, dtype=int)
            for r in range(t):
                acc = poly_add(acc, poly_mul(U[i][r], V[r][j], config), config)
            A[i][j] = acc

    return A


def circulant_matrix(config):
    """
    Generate module-circulant matrix.
    """
    k = config["k"]

    a = [random_poly(config) for _ in range(k)]

    A = []

    for i in range(k):
        row = []
        for j in range(k):
            idx = (j - i) % k
            row.append(a[idx])
        A.append(row)

    return A


def toeplitz_matrix(config):
    """
    Generate module-Toeplitz matrix.
    """
    k = config["k"]

    t = [random_poly(config) for _ in range(2 * k - 1)]

    A = []

    for i in range(k):
        row = []
        for j in range(k):
            idx = j - i + (k - 1)
            row.append(t[idx])
        A.append(row)

    return A


def sparse_matrix(config, rho=0.3):
    """
    Generate sparse matrix with density rho.
    """
    k = config["k"]
    n = config["n"]
    rng = config["rng"]

    A = []
    total = k * k
    nonzero = int(rho * total)

    positions = set(rng.choice(total, nonzero, replace=False))

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


def generate_mlwe_sample(A, config):
    """
    Generate MLWE sample: b = A s + e
    """
    k = config["k"]

    s = [sample_cbd(config) for _ in range(k)]
    e = [sample_cbd(config) for _ in range(k)]

    b = matvec(A, s, config)

    for i in range(k):
        b[i] = poly_add(b[i], e[i], config)

    return s, e, b


def generate_uniform_sample(config):
    """
    Generate uniform random sample.
    """
    return [random_poly(config) for _ in range(config["k"])]


def generate_dataset(config, N=1000, matrix_type="uniform", t=1, rho=0.3):
    """
    Generate labeled dataset.

    label = 1 → MLWE
    label = 0 → uniform
    """

    if matrix_type == "uniform":
        A = uniform_matrix(config)
    elif matrix_type == "lowrank":
        A = low_rank_matrix(config, t=t)
    elif matrix_type == "circulant":
        A = circulant_matrix(config)
    elif matrix_type == "toeplitz":
        A = toeplitz_matrix(config)
    elif matrix_type == "sparse":
        A = sparse_matrix(config, rho=rho)
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

    dataset = []

    for _ in range(N):
        if config["rng"].random() < 0.5:
            _, _, b = generate_mlwe_sample(A, config)
            dataset.append((b, 1))
        else:
            b = generate_uniform_sample(config)
            dataset.append((b, 0))

    return A, dataset


def test_poly_mul_correctness(config):
    a = random_poly(config)
    b = random_poly(config)

    c1 = poly_mul_negacyclic(a, b, config)
    c2 = poly_mul(a, b, config)

    return np.all(c1 == c2)


def test_ntt_roundtrip(config):
    f = random_poly(config)
    fh = ntt(f, config)
    f2 = intt(fh, config)

    return np.all(f % config["q"] == f2 % config["q"])


def test_convolution_theorem(config):
    a = random_poly(config)
    b = random_poly(config)

    lhs = ntt(poly_mul_negacyclic(a, b, config), config)
    rhs = multiply_ntts(ntt(a, config), ntt(b, config), config)

    return np.all(lhs % config["q"] == rhs % config["q"])


def test_ntt_multiple_trials(config, num_trials=100):
    for _ in range(num_trials):
        a = random_poly(config)
        b = random_poly(config)

        if not np.all(
            poly_mul(a, b, config) ==
            poly_mul_negacyclic(a, b, config)
        ):
            return False

    return True