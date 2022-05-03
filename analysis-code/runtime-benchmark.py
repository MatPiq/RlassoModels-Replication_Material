import io
import sys
import time

import stata_setup

stata_setup.config("/Applications/Stata/", "be")
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
from pystata import stata
from rlassomodels import Rlasso
from rpy2.robjects.packages import importr
from tqdm import tqdm

hdm = importr("hdm")


class RuntimeBenchmark:
    def __init__(
        self,
        n_range,
        p_range,
        n_ticks,
        p_ticks,
        s=0.2,
        sigma=0.5,
        rho=0.9,
        seed=42,
        save_output=True,
    ):
        """
        Parameters
        ----------
        n_range : tuple
            Range of number of observations
        p_range : tuple
            Range of number of features
        s : float
            Support/Fraction of non-zero coefs.
        sigma : float
            Standard deviation of the noise.
        rho : float
            Correlation between features j-k.

        """

        self.n_range = n_range
        self.p_range = p_range
        self.n_ticks = n_ticks
        self.p_ticks = p_ticks
        self.s = s
        self.sigma = sigma
        self.rho = rho
        self.rng = np.random.default_rng(seed=seed)
        self.save_output = save_output

    def DGP(self, n, p):
        """
        Generate multivariate distribution with correlated features.

        Parameters
        ----------
        n : int
            Number of samples.
        p : int
            Number of features.
        """

        ii = np.arange(p)
        cov = self.rho ** np.abs(np.subtract.outer(ii, ii))
        X = self.rng.multivariate_normal(np.zeros(p), cov, n)

        beta = np.zeros(p)
        nonzero = [1 for _ in range(int(p * self.s + 1))]
        beta[: len(nonzero)] = nonzero

        y = X @ beta + self.sigma * self.rng.normal(size=n)

        return X, y

    def run_sims(self):

        A, B = self.n_ticks, self.p_ticks
        xn = np.linspace(*self.n_range, num=A).astype(int)
        xp = np.linspace(*self.p_range, num=B).astype(int)

        self.Xo, self.Yo = np.meshgrid(xn, xp)
        rlassopy_times = np.zeros((A, B))
        lassopack_times = np.zeros((A, B))
        hdm_times = np.zeros((A, B))

        # set same number of iterations for HDM
        hdm_iter = ro.r("list(numIter=2)")
        prog_bar = tqdm(xn)
        for i, n in enumerate(xn):
            prog_bar.update()

            for j, p in enumerate(xp):

                prog_bar.set_description(f"n={n}, p={p}")

                X, y = self.DGP(n, p)

                # rlassopy time
                rlasso = Rlasso()
                start = time.time()
                rlasso.fit(X, y)
                end = time.time()
                rlassopy_time = end - start
                rlassopy_times[i, j] = rlassopy_time

                # lassopack rlasso time
                df = pd.DataFrame(np.hstack([y[:, None], X]))
                stata.pdataframe_to_data(df, force=True)
                run_string = f"rlasso v1 v2-v{p+1}"

                # suppress output
                save_stdout = sys.stdout
                sys.stdout = io.StringIO()
                start = time.time()
                stata.run(run_string)
                end = time.time()
                sys.stdout = save_stdout

                lassopack_time = end - start
                lassopack_times[i, j] = lassopack_time
                stata.run("""clear""")

                # hdm rlasso time
                X_r = ro.r.matrix(X, nrow=n, ncol=p)
                ro.r.assign("X", X_r)
                y_r = ro.r.matrix(X, nrow=n, ncol=1)
                ro.r.assign("y", y_r)

                start = time.time()
                hdm.rlasso(x=X_r, y=y_r, controls=hdm_iter)
                end = time.time()

                hdm_time = end - start
                hdm_times[i, j] = hdm_time

        if self.save_output:
            np.save("output/hdm_runtime.npy", hdm_times)
            np.save("output/lassopack_runtime.npy", lassopack_times)
            np.save("output/rlassopy_runtime.npy", rlassopy_times)

        self.results_ = {
            "hdm": hdm_times,
            "lassopack": lassopack_times,
            "rlassopy": rlassopy_times,
        }


def main():
    n_range = (100, 10000)
    p_range = (10, 800)
    n_ticks = 20
    p_ticks = 20

    benchmark = RuntimeBenchmark(n_range, p_range, n_ticks, p_ticks)
    benchmark.run_sims()


if __name__ == "__main__":
    main()
