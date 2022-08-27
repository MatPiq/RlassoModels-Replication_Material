import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from linearmodels.iv import IV2SLS as OLS
from rlassomodels import Rlasso, RlassoPDS
from scipy.linalg import toeplitz
from tqdm import tqdm

warnings.filterwarnings(
    "ignore"
)  # Ignores rlasso warnings when no features are selected


def DGP(design=1, n=100, p=200, alpha=0.5, rho=0.5, R21=0.5, R22=0.5):

    assert design in (1, 2, 3, 4)

    # gererate toeplitz matrix
    cov_mat = toeplitz(rho ** np.arange(p))
    X = np.random.multivariate_normal(np.zeros(p), cov_mat, n)

    # beta = np.zeros(p)

    # all quadratic decay
    if design == 1:
        beta = 1 / np.arange(1, p + 1) ** 2

    # quadratic j < 5, rest random with decaying var
    elif design == 2:
        beta = np.zeros(p)
        beta[:5] = 1 / np.arange(1, p) ** 2
        beta[5:] = np.random.normal(scale=1 / np.arange(1, p - 4), size=(p - 5))
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

    e = np.random.normal(size=n)
    u = np.random.normal(size=n)

    d = X @ beta1 + e
    y = alpha * d + X @ beta2 + u

    # cols = ["y", "d"] + [f"x{i}" for i in range(1,p+1)]

    return y, d, X  # pd.DataFrame(np.c_[y,d,X], columns=cols)


def run_sims_1(R=100, save_results=True):

    # define models
    rlasso = Rlasso(gamma=0.05, lasso_psi=False, post=False)
    pds = RlassoPDS(gamma=0.05)
    alpha0 = 0.5

    R2s = [0.0, 0.2, 0.4, 0.6, 0.8]

    rlasso_coverage = np.empty((5, 5))
    rlasso_bias = np.empty((5, 5))
    rlasso_std = np.empty((5, 5))

    pds_coverage = np.empty((5, 5))
    pds_bias = np.empty((5, 5))
    pds_std = np.empty((5, 5))

    chs_coverage = np.empty((5, 5))
    chs_bias = np.empty((5, 5))
    chs_std = np.empty((5, 5))

    pbar = tqdm(total=len(R2s))
    for i, R21 in enumerate(R2s):

        pbar.update()

        for j, R22 in enumerate(R2s):

            # store temps
            rlasso_alphas = np.empty(R)
            rlasso_se = np.empty(R)

            pds_alphas = np.empty(R)
            pds_se = np.empty(R)

            chs_alphas = np.empty(R)
            chs_se = np.empty(R)

            for r in range(R):
                pbar.set_description(f"R21={R21}, R22={R22}, sim={r}")

                y, d, X = DGP(alpha=alpha0, rho=0.5, R21=R21, R22=R22)

                dX = np.c_[d, X]

                naive = rlasso.fit(dX, y, nopen_idx=[0]).nonzero_idx_
                ols = OLS(y, dX[:, naive], None, None).fit()
                rlasso_alphas[r] = ols.params[0]
                rlasso_se[r] = ols.std_errors[0]

                pds_fitted = pds.fit(X, y, d)
                pds_alphas[r] = pds_fitted.results_["PDS"].params[1]
                pds_se[r] = pds_fitted.results_["PDS"].std_errors[1]
                chs_alphas[r] = pds_fitted.results_["CHS"].params[0]
                chs_se[r] = pds_fitted.results_["CHS"].std_errors[0]

            # rlasso res
            # absZval = np.abs(rlasso_alphas - alpha0) /
            rlasso_bias[i, j] = np.mean(rlasso_alphas - alpha0)
            absZval = np.abs(rlasso_alphas - alpha0) / rlasso_se
            rlasso_coverage[i, j] = np.sum(absZval > 1.96) / R
            rlasso_std[i, j] = np.std(rlasso_alphas)

            # pds
            pds_bias[i, j] = np.mean(pds_alphas - alpha0)
            absZval = np.abs(pds_alphas - alpha0) / pds_se
            pds_coverage[i, j] = np.sum(absZval > 1.96) / R
            pds_std[i, j] = np.std(pds_alphas)

            # chs
            chs_bias[i, j] = np.mean(chs_alphas - alpha0)
            absZval = np.abs(chs_alphas - alpha0) / chs_se
            chs_coverage[i, j] = np.sum(absZval > 1.96) / R
            chs_std[i, j] = np.std(chs_alphas)

    results = {
        "rlasso_bias": rlasso_bias,
        "rlasso_cov": rlasso_coverage,
        "rlasso_std": rlasso_std,
        "pds_bias": pds_bias,
        "pds_cov": pds_coverage,
        "pds_std": pds_std,
        "chs_bias": chs_bias,
        "chs_cov": chs_coverage,
        "chs_std": chs_std,
    }

    if save_results:
        pass
    # pd.DataFrame(results).to_csv("../outputs/tabs/pds-performance.csv")

    return {k: v.T for k, v in results.items()}


def plot_sims_1(results, save_results=True):

    axs = [0, 0.2, 0.4, 0.6, 0.8]

    # ax2 = list(reversed(ax1))
    X, Y = np.meshgrid(axs, axs)

    with plt.style.context("science"):
        fig, axs = plt.subplots(ncols=3, nrows=4, dpi=300, figsize=(10, 8))

        axs = axs.flatten()
        cmap = plt.cm.plasma

        axs[0].contourf(
            X,
            Y,
            results["pds_bias"],
            levels=2,
            cmap=cmap,
            origin="upper",
            vmin=-0.5,
            vmax=0.5,
        )
        CS = axs[0].contour(
            X,
            Y,
            results["pds_bias"],
            levels=4,
            colors="black",
            linewidths=0.5,
            vmin=-0.5,
            vmax=0.5,
        )
        axs[0].clabel(CS, inline=True, fontsize=8, colors="black")
        axs[0].set_ylabel("$R^2$ stage 2")
        axs[0].set_title("PDS Bias")
        axs[0].set_xlabel("$R^2$ stage 1")
        # axs[0].set_xticklabels(xticks)
        cbar = plt.imshow(
            results["pds_bias"], cmap=cmap, interpolation="bicubic", vmin=-0.5, vmax=0.5
        )
        fig.colorbar(cbar, ax=axs[0], orientation="vertical", shrink=0.6)

        axs[1].contourf(
            X,
            Y,
            results["chs_bias"],
            100,
            cmap=cmap,
            origin="upper",
            vmin=-0.5,
            vmax=0.5,
        )
        CS = axs[1].contour(
            X,
            Y,
            results["chs_bias"],
            levels=4,
            colors="black",
            linewidths=0.5,
            vmin=-0.5,
            vmax=0.5,
        )
        axs[1].clabel(CS, inline=True, fontsize=8, colors="black")
        axs[1].set_title("PPO bias")
        axs[1].set_xlabel("$R^2$ stage 1")
        # axs[0].set_xticklabels(xticks)
        cbar = plt.imshow(
            results["chs_bias"], cmap=cmap, interpolation="bicubic", vmin=-0.5, vmax=0.5
        )
        fig.colorbar(cbar, ax=axs[1], orientation="vertical", shrink=0.6)

        axs[2].contourf(
            X,
            Y,
            results["rlasso_bias"],
            100,
            cmap=cmap,
            origin="upper",
            vmin=-0.5,
            vmax=0.5,
        )
        CS = axs[2].contour(
            X,
            Y,
            results["rlasso_bias"],
            levels=4,
            colors="black",
            linewidths=0.5,
            vmin=-0.5,
            vmax=0.5,
        )
        axs[2].clabel(CS, inline=True, fontsize=8, colors="black")
        axs[2].set_title("Naive Post-Lasso Bias")
        axs[2].set_xlabel("$R^2$ stage 1")
        # axs[0].set_xticklabels(xticks)
        cbar = plt.imshow(
            results["rlasso_bias"],
            cmap=cmap,
            interpolation="bicubic",
            vmin=-0.5,
            vmax=0.5,
        )
        fig.colorbar(cbar, ax=axs[2], orientation="vertical", shrink=0.6, label="bias")

        cmap = plt.cm.viridis

        axs[3].contourf(
            X, Y, results["pds_std"], 100, cmap=cmap, origin="upper", vmin=0, vmax=0.3
        )
        CS = axs[3].contour(
            X,
            Y,
            results["pds_std"],
            4,
            colors="black",
            linewidths=0.5,
            vmin=0,
            vmax=0.3,
        )
        axs[3].clabel(CS, inline=True, fontsize=8, colors="black")
        axs[3].set_ylabel("$R^2$ stage 2")
        axs[3].set_title("PDS Std")
        axs[3].set_xlabel("$R^2$ stage 1")
        # axs[0].set_xticklabels(xticks)
        cbar = plt.imshow(
            results["pds_std"], cmap=cmap, interpolation="bicubic", vmin=0, vmax=0.3
        )
        fig.colorbar(cbar, ax=axs[3], orientation="vertical", shrink=0.6)

        axs[4].contourf(
            X, Y, results["chs_std"], 100, cmap=cmap, origin="upper", vmin=0, vmax=0.3
        )
        CS = axs[4].contour(
            X,
            Y,
            results["chs_std"],
            4,
            colors="black",
            linewidths=0.5,
            vmin=0,
            vmax=0.3,
        )
        axs[4].clabel(CS, inline=True, fontsize=8, colors="black")
        axs[4].set_title("PPO Std")
        axs[4].set_xlabel("$R^2$ stage 1")
        # axs[0].set_xticklabels(xticks)
        cbar = plt.imshow(
            results["chs_std"], cmap=cmap, interpolation="bicubic", vmin=0, vmax=0.3
        )
        fig.colorbar(cbar, ax=axs[4], orientation="vertical", shrink=0.6)

        axs[5].contourf(
            X,
            Y,
            results["rlasso_std"],
            100,
            cmap=cmap,
            origin="upper",
            vmin=0,
            vmax=0.3,
        )
        CS = axs[5].contour(
            X,
            Y,
            results["rlasso_std"],
            4,
            colors="black",
            linewidths=0.5,
            vmin=0,
            vmax=0.3,
        )
        axs[5].clabel(CS, inline=True, fontsize=8, colors="black")
        axs[5].set_title("Naive Post-Lasso Std")
        axs[5].set_xlabel("$R^2$ stage 1")
        # axs[0].set_xticklabels(xticks)
        cbar = plt.imshow(
            results["rlasso_std"], cmap=cmap, interpolation="bicubic", vmin=0, vmax=0.3
        )
        fig.colorbar(cbar, ax=axs[5], orientation="vertical", shrink=0.6, label="std")

        cmap = plt.cm.coolwarm
        axs[6].contourf(
            X, Y, results["pds_cov"], 100, cmap=cmap, origin="upper", vmin=0, vmax=0.4
        )
        CS = axs[6].contour(
            X,
            Y,
            results["pds_cov"],
            4,
            colors="black",
            linewidths=0.5,
            vmin=0,
            vmax=0.4,
        )
        axs[6].clabel(CS, inline=True, fontsize=8, colors="black")
        axs[6].set_ylabel("$R^2$ stage 2")
        axs[6].set_title("PDS Coverage")
        axs[6].set_xlabel("$R^2$ stage 1")
        # axs[0].set_xticklabels(xticks)
        cbar = plt.imshow(
            results["pds_cov"], cmap=cmap, interpolation="bicubic", vmin=0, vmax=0.4
        )
        fig.colorbar(cbar, ax=axs[6], orientation="vertical", shrink=0.6)

        axs[7].contourf(
            X, Y, results["chs_cov"], 100, cmap=cmap, origin="upper", vmin=0, vmax=0.4
        )
        CS = axs[7].contour(
            X,
            Y,
            results["chs_cov"],
            4,
            colors="black",
            linewidths=0.5,
            vmin=0,
            vmax=0.4,
        )
        axs[7].clabel(CS, inline=True, fontsize=8, colors="black")
        axs[7].set_title("PPO Coverage")
        axs[7].set_xlabel("$R^2$ stage 1")
        # axs[0].set_xticklabels(xticks)
        cbar = plt.imshow(
            results["chs_cov"], cmap=cmap, interpolation="bicubic", vmin=0, vmax=0.4
        )
        fig.colorbar(cbar, ax=axs[7], orientation="vertical", shrink=0.6)

        axs[8].contourf(
            X,
            Y,
            results["rlasso_cov"],
            100,
            cmap=cmap,
            origin="upper",
            vmin=0,
            vmax=0.4,
        )
        CS = axs[8].contour(
            X,
            Y,
            results["rlasso_cov"],
            4,
            colors="black",
            linewidths=0.5,
            vmin=0,
            vmax=0.4,
        )
        axs[8].clabel(CS, inline=True, fontsize=8, colors="black")
        axs[8].set_title("Naive Post-Lasso Coverage")
        axs[8].set_xlabel("$R^2$ stage 1")
        # axs[0].set_xticklabels(xticks)
        cbar = plt.imshow(
            results["rlasso_cov"], cmap=cmap, interpolation="bicubic", vmin=0, vmax=0.4
        )
        fig.colorbar(cbar, ax=axs[8], orientation="vertical", shrink=0.6)
        fig.tight_layout()

        # bugfix
        fig.delaxes(axs[9])
        fig.delaxes(axs[10])
        fig.delaxes(axs[11])

        if save_results:
            plt.savefig("../outputs/figs/pds-performance.pdf")

        else:
            plt.show()


def run_sims_2(rho=0.5, R=500, R21=0.5, R22=0.5):

    rlasso = Rlasso(gamma=0.05, lasso_psi=False, post=False)
    pds = RlassoPDS(gamma=0.05)
    alpha0 = 0.5

    rlasso_alphas = np.empty(R)
    pds_alphas = np.empty(R)
    pbar = tqdm(total=R)
    for r in range(R):
        pbar.update()
        y, d, X = DGP(alpha=alpha0, rho=rho, R21=R21, R22=R22, design=4)

        dX = np.c_[d, X]
        naive = rlasso.fit(dX, y, nopen_idx=[0]).nonzero_idx_
        ols = OLS(y, dX[:, naive], None, None).fit()
        rlasso_alphas[r] = ols.params[0]

        pds_fitted = pds.fit(X, y, d)
        pds_alphas[r] = pds_fitted.results_["PDS"].params[1]

    return rlasso_alphas, pds_alphas


def plot_sims_2(rlasso_alphas, pds_alphas, save_results=True):

    norm_rlasso = (rlasso_alphas - 0.5) / rlasso_alphas.std()
    norm_pds = (pds_alphas - 0.5) / pds_alphas.std()

    # normal dist
    mu, sigma = 0, 1
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    with plt.style.context("science"):
        fig, axs = plt.subplots(ncols=2, dpi=300, figsize=(10, 5))

        sns.histplot(norm_pds, ax=axs[0], stat="density")
        axs[0].plot(x, st.norm.pdf(x, mu, sigma), "r")
        axs[0].set_title("Post-Double-Selection")
        axs[0].set_xlabel("$\\hat{\\alpha} - \\alpha_0 / \\hat{\\sigma}$")

        sns.histplot(norm_rlasso, ax=axs[1], stat="density")
        axs[1].plot(x, st.norm.pdf(x, mu, sigma), "r")
        axs[1].set_title("Naive Post-Lasso")
        axs[1].set_xlabel("$\\hat{\\alpha} - \\alpha_0 / \\hat{\\sigma}$")

        if save_results:
            fig.savefig("../outputs/figs/pds-density.pdf")
        else:
            plt.show()


def main():

    results_sim_1 = run_sims_1(save_results=True)
    plot_sims_1(results_sim_1, save_results=True)
    rlasso_alphas, pds_alphas = run_sims_2()
    plot_sims_2(rlasso_alphas, pds_alphas, save_results=True)


if __name__ == "__main__":
    main()
