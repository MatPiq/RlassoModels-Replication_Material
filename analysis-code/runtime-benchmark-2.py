import time
from collections import defaultdict

import numpy as np
import pandas as pd
from rlassomodels import Rlasso, RlassoPDS
from scipy.linalg import toeplitz
from tqdm import tqdm


def DGP(seed, design=1, n=100, p=200, alpha=0.5, rho=0.5, R21=0.5, R22=0.5):

    assert design in (1, 2, 3, 4)

    # gererate toeplitz matrix
    rng = np.random.default_rng(seed=seed)
    cov_mat = toeplitz(rho ** np.arange(p))
    X = rng.multivariate_normal(np.zeros(p), cov_mat, n)

    # beta = np.zeros(p)

    # all quadratic decay
    if design == 1:
        beta = 1 / np.arange(1, p + 1) ** 2

    # quadratic j < 5, rest random with decaying var
    elif design == 2:
        beta = np.zeros(p)
        beta[:5] = 1 / np.arange(1, 6) ** 2
        beta[5:] = rng.normal(scale=1 / np.arange(1, p - 4), size=(p - 5))
    # constant coefs
    elif design == 3:
        beta = np.zeros(p)
        beta[1:2:40] = 1

    # linear + quadratic
    elif design == 4:
        beta = np.zeros(p)
        beta[:5] = 1 / np.arange(1, 6)
        beta[5:15] = 1 / np.arange(1, 11) ** 2

    sand = np.inner(beta, cov_mat) @ beta

    # c1 to achive desired R2 of first stage
    c1 = np.sqrt(R21 / ((1 - R21) * sand))

    # c2 to achive desired R2 of second stage
    a = (1 - R22) * sand
    b = 2 * (1 - R22) * alpha * c1 * sand
    c = (1 - R22) * ((alpha * c1) ** 2) * sand - R22 * (alpha**2 + 2)
    disc = b**2 - 4 * a * c

    if np.abs(disc) < 1e-12:
        disc = 0
    c2 = (-b + np.sqrt(disc)) / (2 * a)

    beta1 = c1 * beta
    beta2 = c2 * beta

    e = rng.normal(size=n)
    u = rng.normal(size=n)

    d = X @ beta1 + e
    y = alpha * d + X @ beta2 + u

    # cols = ["y", "d"] + [f"x{i}" for i in range(1,p+1)]

    return y, d, X


def runtime_benchmark():

    # define dimensions (n,p) for sims
    dims = [
        (100, 50),
        (100, 110),
        (250, 125),
        (250, 260),
        (500, 250),
        (500, 510),
        (1000, 500),
        (1000, 1010),
    ]

    models = {
        "Rlasso": Rlasso(),
        "Rlasso xdep": Rlasso(x_dependent=True, n_sim=1000, solver="cvxpy"),
        "Sqrt-Rlasso": Rlasso(sqrt=True, solver="cvxpy"),
        "Sqrt-Rlasso xdep": Rlasso(
            sqrt=True, x_dependent=True, n_sim=1000, solver="cvxpy"
        ),
        "PDS": RlassoPDS(solver="cvxpy"),
        "PDS xdep": RlassoPDS(x_dependent=True, n_sim=1000, solver="cvxpy"),
        "PDS Sqrt": RlassoPDS(sqrt=True, solver="cvxpy"),
        "PDS Sqrt xdep": RlassoPDS(
            sqrt=True, x_dependent=True, n_sim=1000, solver="cvxpy"
        ),
    }

    results = defaultdict(list)
    pbar = tqdm(total=len(dims))
    for i, (n, p) in enumerate(dims):
        pbar.update()

        y, d, X = DGP(seed=i, design=2, n=n, p=p, alpha=0.5, rho=0.5, R21=0.5, R22=0.5)

        for name, model in models.items():

            pbar.set_description(f"n={n}, p={p}, model={name}")

            if "PDS" in name:
                start = time.time()
                model.fit(X, y, d)
            else:
                start = time.time()
                model.fit(X, y)
            rt = time.time() - start
            results[f"{n},{p}"].append(rt)

    return pd.DataFrame(results, index=list(models)).round(3)


if __name__ == "__main__":
    results = runtime_benchmark()
    results.to_csv("runtime-benchmarks-2.csv")
