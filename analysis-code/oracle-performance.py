import parser
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd
from rlassomodels import Rlasso
from sklearn.linear_model import LassoCV, LassoLarsIC
from sklearn.metrics import f1_score
from tqdm import tqdm


class OraclePerformance:
    def __init__(
        self,
        sigmas,
        n_sims: int = 100,
        noise: str = "gaussian",
        n: int = 100,
        p: int = 500,
        save_results=True,
    ):
        assert noise in {"gaussian", "t", "exp"}
        self.sigmas = sigmas
        self.n_sims = n_sims
        self.noise = noise
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
        if self.noise == "gaussian":
            noise = np.random.normal(size=self.n)
        elif self.noise == "t":
            noise = np.random.standard_t(df=4, size=self.n) / np.sqrt(2)
        else:
            noise = np.random.exponential(size=self.n) - 1
        y = np.dot(X, b) + sigma * noise

        return X, y, b, cx

    def run_sims(self):

        results = defaultdict(list)
        results["sigma"] = self.sigmas
        models = {
            "Rlasso": Rlasso(post=False, fit_intercept=False),
            "Rlasso post": Rlasso(fit_intercept=False),
            "Sqrt-rlasso": Rlasso(sqrt=True, post=False, fit_intercept=False),
            "Sqrt-rlasso post": Rlasso(sqrt=True, fit_intercept=False),
            "Rlasso xdep": Rlasso(post=False, fit_intercept=False, x_dependent=True),
            "Rlasso post xdep": Rlasso(fit_intercept=False, x_dependent=True),
            "Sqrt-rlasso xdep": Rlasso(
                sqrt=True, post=False, fit_intercept=False, x_dependent=True
            ),
            "Sqrt-rlasso post xdep": Rlasso(
                sqrt=True, fit_intercept=False, x_dependent=True
            ),
            "CV": LassoCV(),
            "AIC": LassoLarsIC(criterion="aic", normalize=False),
            "BIC": LassoLarsIC(criterion="bic", normalize=False),
        }
        true_support = np.zeros(500).astype(bool)
        true_support[:5] = True

        pbar = tqdm(total=len(self.sigmas))

        for s in self.sigmas:

            pbar.update()
            risks_tmp = np.empty((self.n_sims, len(models)))
            support_tmp = np.empty((self.n_sims, len(models)))
            rmse_tmp = np.empty((self.n_sims, len(models)))

            # set sigma for aic and bic
            models["AIC"].noise_variance = s**2
            models["BIC"].noise_variance = s**2
            for j in range(self.n_sims):

                # generate data
                X, y, b, cx = self.dgf(sigma=s)

                # get oracle
                X_oracle = X[:, :5]
                oracle = la.solve(X_oracle.T @ X_oracle, X_oracle.T @ y)
                oracle_e = np.zeros(self.p)
                oracle_e[:5] = oracle - b[:5]
                denom = np.sqrt(np.sum(oracle_e**2))

                for k, (m_name, m) in enumerate(models.items()):
                    pbar.set_description(f"sigma={s}, sim={j}, model={m_name}")
                    res = m.fit(X, y)
                    # empirical risk
                    e = res.coef_ - b
                    numer = np.sqrt(np.sum(e**2))  #  np.sqrt(np.sum(e**2))
                    risks_tmp[j, k] = numer / denom
                    # support
                    support = res.coef_ != 0
                    support_tmp[j, k] = f1_score(true_support, support)

                    # root mean squared error
                    rmse_tmp[j, k] = np.sqrt(np.mean((y - X @ res.coef_) ** 2))

            for k, m_name in enumerate(models):

                results[f"{m_name} avg"].append(risks_tmp[:, k].mean())
                results[f"{m_name} f1"].append(support_tmp[:, k].mean())
                results[f"{m_name} rmse"].append(rmse_tmp[:, k].mean())

        self.results = pd.DataFrame(results)
        if self.save_results:
            self.results.to_csv("../outputs/tabs/oracle-performance.csv")

        return self

    def plot_results(self):

        if not hasattr(self, "results"):
            raise AttributeError("Simulations need to be completed")
        with plt.style.context(["science", "high-vis"]):

            fig, axs = plt.subplots(ncols=2, dpi=500, figsize=(6, 3))
            x = self.sigmas

            axs[0].plot(x, self.results["Rlasso post avg"], label="post-rlasso")
            axs[0].plot(x, self.results["Rlasso avg"], label="rlasso")
            axs[0].plot(x, self.results["CV avg"], label="CV")
            axs[0].plot(x, self.results["AIC avg"], label="AIC")
            axs[0].plot(x, self.results["BIC avg"], label="BIC")
            axs[0].set_ylim([0, 4.2])
            axs[0].set_xlabel("$\\sigma$")
            axs[0].set_ylabel("Empirical Risk")
            axs[0].legend(prop={"size": 6})

            axs[1].plot(x, self.results["Rlasso f1"], label="rlasso")
            axs[1].plot(x, self.results["CV f1"], label="CV")
            axs[1].plot(x, self.results["AIC f1"], label="AIC")
            axs[1].plot(x, self.results["BIC f1"], label="BIC")
            axs[1].set_ylim([0, 1.1])
            axs[1].set_xlabel("$\\sigma$")
            axs[1].set_ylabel("Support F1 Score")
            axs[1].legend(prop={"size": 6})

            plt.tight_layout()

            if self.save_results:
                plt.savefig("../outputs/figs/oracle-performance.pdf")
            else:
                plt.show()


def main():
    sigmas = np.linspace(0.1, 3.5, 10)
    oracle_performance = OraclePerformance(
        sigmas=sigmas,
        n_sims=500,
        noise="gaussian",
        n=100,
        p=500,
        save_results=True,
    )
    oracle_performance.run_sims()
    oracle_performance.plot_results()


if __name__ == "__main__":
    main()
