import parser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd
from rlassopy import Rlasso
from tqdm import tqdm


class OraclePerformance:
    def __init__(
        self, sigmas, n_sims: int = 100, n: int = 100, p: int = 500, save_results=True
    ):

        self.sigmas = sigmas
        self.n_sims = n_sims
        self.n = n
        self.p = p
        self.save_results = save_results

    def dgf(self, sigma):
        """
        Data-generating function following Belloni (2011).
        """
        # np.random.seed(234923)

        # Based on the example in the Belloni paper
        ii = np.arange(self.p)
        cx = 0.5 ** np.abs(np.subtract.outer(ii, ii))
        cxr = np.linalg.cholesky(cx)

        X = np.dot(np.random.normal(size=(self.n, self.p)), cxr.T)
        b = np.zeros(self.p)
        b[0:5] = [1, 1, 1, 1, 1]
        y = np.dot(X, b) + sigma * np.random.normal(size=self.n)

        return X, y, b, cx

    def run_sims(self):

        results = defaultdict(list)
        results["sigma"] = sigmas
        models = {
            "Rlasso": Rlasso(post=False),
            "Rlasso post": Rlasso(),
            "Sqrt-rlasso": Rlasso(sqrt=True, post=False),
            "Sqrt-rlasso post": Rlasso(sqrt=True),
        }

        for i, s in enumerate(tqdm(self.sigmas)):

            risks_tmp = np.empty((self.n_sims, 4))
            support_tmp = np.empty((self.n_sims, 4))

            for j in range(self.n_sims):

                # generate data
                X, y, b, cx = self.dgf(sigma=s)

                # get oracle
                X_oracle = X[:, :5]
                oracle = la.inv(X_oracle.T @ X_oracle) @ X_oracle.T @ y
                oracle_e = np.zeros(self.p)
                oracle_e[0:5] = oracle - b[0:5]
                denom = np.sqrt(np.dot(oracle_e, np.dot(cx, oracle_e)))

                for k, (m_name, m) in enumerate(models.items()):
                    res = m.fit(X, y)
                    # performance
                    e = res.coef_ - b
                    numer = np.sqrt(np.dot(e, np.dot(cx, e)))
                    risks_tmp[j, k] = numer / denom
                    # support
                    support_tmp[j, k] = np.nonzero(res.coef_)[0].size

            for k, m_name in enumerate(models):

                results[f"{m_name}_avg"].append(risks_tmp[:, k].mean())
                results[f"{m_name}_sd"].append(risks_tmp[:, k].std())
                results[f"{m_name}_support"].append(risks_tmp[:, k].mean())

        self.results = pd.DataFrame(results)
        if self.save_results:
            self.results.to_csv("outputs/tables/oracle-performance.csv")

        return self

    def plot_results(self):

        if not hasattr(self, "results"):
            raise AttributeError(
                """
            Simulations need to be completed"""
            )

        pass


if __name__ == "__main__":
    # get parser arguments (sigma, n_sims, n, p)
    sigmas = np.linspace(0.3, 3, 10)
    oracle_performance = OraclePerformance(
        sigmas=sigmas,
        n_sims=1000,
        n=100,
        p=500,
        save_results=True,
    )
    oracle_performance.run_sims()
