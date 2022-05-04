import io
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import stata_setup

stata_setup.config("/Applications/Stata/", "be")
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from scipy.ndimage.interpolation import zoom
from sklearn.linear_model import LassoCV, LassoLarsIC

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
        rlassomodels_times = np.zeros((A, B))
        lassopack_times = np.zeros((A, B))
        hdm_times = np.zeros((A, B))
        lasso_cv_times = np.zeros((A, B))
        lasso_lars_times = np.zeros((A, B))

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
                rlassomodels_times[i, j] = rlassopy_time

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

                # lasso CV time
                cv = LassoCV(cv=5, max_iter=1000)
                start = time.time()
                cv.fit(X, y)
                end = time.time()
                lasso_cv_times[i, j] = end - start

                # lasso lars times
                lars = LassoLarsIC(normalize=False, noise_variance=0.5**2)
                start = time.time()
                lars.fit(X, y)
                end = time.time()
                lasso_lars_times[i, j] = end - start

        if self.save_output:
            np.save("output/hdm_runtime.npy", hdm_times)
            np.save("output/lassopack_runtime.npy", lassopack_times)
            np.save("output/rlassopy_runtime.npy", rlassomodels_times)
            np.save("output/lasso_cv_runtime.npy", lasso_cv_times)
            np.save("output/lasso_lars_runtime.npy", lasso_lars_times)

        self.results_ = {
            "hdm": hdm_times,
            "lassopack": lassopack_times,
            "rlassomodels": rlassomodels_times,
            "lasso_cv": lasso_cv_times,
            "lasso_lars": lasso_lars_times,
        }

    def plot_hdm_lassopack(self):

        Xsm = zoom(self.Xo, 3)
        Ysm = zoom(self.Yo, 3)
        Z_rlassopy = zoom(self.results_["rlassomodels"], 3)
        Z_lassopack = zoom(self.results_["lassopack"], 3)
        Z_hdm = zoom(self.results_["hdm"], 3)

        Z_rlassopy_vs_lassopack = Z_lassopack / Z_rlassopy
        Z_rlassopy_vs_hdm = Z_hdm / Z_rlassopy

        with plt.style.context("science"):
            fig = plt.figure(figsize=(9, 9), dpi=300)
            gs = gridspec.GridSpec(2, 6)

            ax1a = plt.subplot(gs[0, :2])
            ax1b = plt.subplot(gs[0, 2:4])
            ax1c = plt.subplot(gs[0, 4:])
            ax2a = plt.subplot(gs[1, 1:3])
            ax2b = plt.subplot(gs[1, 3:5])
            axdel = plt.subplot(gs[1, 5:6])

            ax1a.contourf(
                Xsm, Ysm, Z_rlassopy, 100, cmap=plt.cm.Spectral, origin="upper"
            )
            ax1a.contour(Xsm, Ysm, Z_rlassopy, 10, colors="black", linewidths=0.5)
            ax1a.set_ylabel("Dimensions")
            ax1a.set_title("rlassomodels runtime")
            ax1a.set_xlabel("Observations")
            # axs[0].set_xticklabels(xticks)
            cbar = plt.imshow(Z_rlassopy, cmap=plt.cm.Spectral, interpolation="bicubic")
            fig.colorbar(
                cbar, ax=ax1a, orientation="vertical", shrink=0.6, label="Seconds"
            )

            ax1b.contourf(
                Xsm, Ysm, Z_lassopack, 100, cmap=plt.cm.Spectral, origin="upper"
            )
            ax1b.contour(Xsm, Ysm, Z_lassopack, 10, colors="black", linewidths=0.5)
            ax1b.set_title("lassopack runtime")
            ax1b.set_xlabel("Observations")
            # axs[1].set_xticklabels(xticks)
            cbar = plt.imshow(
                Z_lassopack, cmap=plt.cm.Spectral, interpolation="bicubic"
            )
            fig.colorbar(
                cbar, ax=ax1b, orientation="vertical", shrink=0.6, label="Seconds"
            )

            ax1c.contourf(Xsm, Ysm, Z_hdm, 100, cmap=plt.cm.Spectral, origin="upper")
            ax1c.contour(Xsm, Ysm, Z_hdm, 10, colors="black", linewidths=0.5)
            ax1c.set_title("hdm runtime")
            ax1c.set_xlabel("Observations")
            # axs[2].set_xticklabels(xticks)
            cbar = plt.imshow(Z_hdm, cmap=plt.cm.Spectral, interpolation="bicubic")
            fig.colorbar(
                cbar, ax=ax1c, orientation="vertical", shrink=0.6, label="Seconds"
            )

            ax2a.contourf(
                Xsm,
                Ysm,
                Z_rlassopy_vs_lassopack,
                100,
                cmap=plt.cm.Spectral,
                origin="upper",
            )
            ax2a.contour(
                Xsm,
                Ysm,
                Z_rlassopy_vs_lassopack,
                10,
                colors="black",
                linewidths=0.5,
            )
            ax2a.set_title("rlassomodels vs. lassopack")
            ax2a.set_ylabel("Dimensions")
            ax2a.set_xlabel("Observations")
            cbar = plt.imshow(
                Z_rlassopy_vs_lassopack,
                cmap=plt.cm.Spectral,
                interpolation="bicubic",
            )
            fig.colorbar(
                cbar,
                ax=ax2a,
                orientation="vertical",
                shrink=0.6,
                label="Speedup factor",
            )

            ax2b.contourf(
                Xsm,
                Ysm,
                Z_rlassopy_vs_hdm,
                100,
                cmap=plt.cm.Spectral,
                origin="upper",
            )
            ax2b.contour(
                Xsm, Ysm, Z_rlassopy_vs_hdm, 10, colors="black", linewidths=0.5
            )
            ax2b.set_title("rlassomodels vs. hdm")
            ax2b.set_xlabel("Observations")
            cbar = plt.imshow(
                Z_rlassopy_vs_hdm, cmap=plt.cm.Spectral, interpolation="bicubic"
            )
            fig.colorbar(
                cbar,
                ax=ax2b,
                orientation="vertical",
                shrink=0.6,
                label="Speedup factor",
            )
            # axs[4].set_xticklabels(xticks)

            # delete axis (bug fix)
            fig.delaxes(axdel)

            plt.tight_layout()
            if self.save_output:
                plt.savefig("../outputs/figs/runtime-benchmark.pdf")
            else:
                plt.show()

    def plot_sklearn(self):

        Xsm = zoom(self.Xo, 3)
        Ysm = zoom(self.Yo, 3)
        Z_rlassomodels = zoom(self.results_["rlassomodels"], 3)
        Z_lasso_cv = zoom(self.results_["lasso_cv"], 3)
        Z_lasso_lars = zoom(self.results_["lasso_lars"], 3)

        Z_rlassomodels_vs_lasso_cv = Z_lasso_cv / Z_rlassomodels
        Z_rlassomodels_vs_lasso_lars = Z_lasso_lars / Z_rlassomodels

        with plt.style.context("science"):
            fig = plt.figure(figsize=(9, 9), dpi=300)
            gs = gridspec.GridSpec(2, 6)

            ax1a = plt.subplot(gs[0, :2])
            ax1b = plt.subplot(gs[0, 2:4])
            ax1c = plt.subplot(gs[0, 4:])
            ax2a = plt.subplot(gs[1, 1:3])
            ax2b = plt.subplot(gs[1, 3:5])
            axdel = plt.subplot(gs[1, 5:6])

            ax1a.contourf(
                Xsm, Ysm, Z_rlassomodels, 100, cmap=plt.cm.Spectral, origin="upper"
            )
            ax1a.contour(Xsm, Ysm, Z_rlassomodels, 10, colors="black", linewidths=0.5)
            ax1a.set_ylabel("Dimensions")
            ax1a.set_title("rlassomodels runtime")
            ax1a.set_xlabel("Observations")
            # axs[0].set_xticklabels(xticks)
            cbar = plt.imshow(
                Z_rlassomodels, cmap=plt.cm.Spectral, interpolation="bicubic"
            )
            fig.colorbar(
                cbar, ax=ax1a, orientation="vertical", shrink=0.6, label="Seconds"
            )

            ax1b.contourf(
                Xsm, Ysm, Z_lasso_cv, 100, cmap=plt.cm.Spectral, origin="upper"
            )
            ax1b.contour(Xsm, Ysm, Z_lasso_cv, 10, colors="black", linewidths=0.5)
            ax1b.set_title("sklearn lassocv runtime")
            ax1b.set_xlabel("Observations")
            # axs[1].set_xticklabels(xticks)
            cbar = plt.imshow(Z_lasso_cv, cmap=plt.cm.Spectral, interpolation="bicubic")
            fig.colorbar(
                cbar, ax=ax1b, orientation="vertical", shrink=0.6, label="Seconds"
            )

            ax1c.contourf(
                Xsm, Ysm, Z_lasso_lars, 100, cmap=plt.cm.Spectral, origin="upper"
            )
            ax1c.contour(Xsm, Ysm, Z_lasso_lars, 10, colors="black", linewidths=0.5)
            ax1c.set_title("sklearn lassolars runtime")
            ax1c.set_xlabel("Observations")
            # axs[2].set_xticklabels(xticks)
            cbar = plt.imshow(
                Z_lasso_lars, cmap=plt.cm.Spectral, interpolation="bicubic"
            )
            fig.colorbar(
                cbar, ax=ax1c, orientation="vertical", shrink=0.6, label="Seconds"
            )

            ax2a.contourf(
                Xsm,
                Ysm,
                Z_rlassomodels_vs_lasso_cv,
                100,
                cmap=plt.cm.Spectral,
                origin="upper",
            )
            ax2a.contour(
                Xsm,
                Ysm,
                Z_rlassomodels_vs_lasso_cv,
                10,
                colors="black",
                linewidths=0.5,
            )
            ax2a.set_title("rlassomodels vs. sklearn lassocv")
            ax2a.set_ylabel("Dimensions")
            ax2a.set_xlabel("Observations")
            cbar = plt.imshow(
                Z_rlassomodels_vs_lasso_cv,
                cmap=plt.cm.Spectral,
                interpolation="bicubic",
            )
            fig.colorbar(
                cbar,
                ax=ax2a,
                orientation="vertical",
                shrink=0.6,
                label="Speedup factor",
            )

            ax2b.contourf(
                Xsm,
                Ysm,
                Z_rlassomodels_vs_lasso_lars,
                100,
                cmap=plt.cm.Spectral,
                origin="upper",
            )
            ax2b.contour(
                Xsm,
                Ysm,
                Z_rlassomodels_vs_lasso_lars,
                10,
                colors="black",
                linewidths=0.5,
            )
            ax2b.set_title("rlassomodels vs. sklearn lassolars")
            ax2b.set_xlabel("Observations")
            cbar = plt.imshow(
                Z_rlassomodels_vs_lasso_lars,
                cmap=plt.cm.Spectral,
                interpolation="bicubic",
            )
            fig.colorbar(
                cbar,
                ax=ax2b,
                orientation="vertical",
                shrink=0.6,
                label="Speedup factor",
            )
            # axs[4].set_xticklabels(xticks)

            # delete axis (bug fix)
            fig.delaxes(axdel)

            plt.tight_layout()
            if self.save_output:
                plt.savefig("../outputs/figs/runtime-sklearn-benchmark.pdf")
            else:
                plt.show()


def main():
    n_range = (100, 10000)
    p_range = (10, 800)
    n_ticks = 20
    p_ticks = 20

    benchmark = RuntimeBenchmark(n_range, p_range, n_ticks, p_ticks)
    benchmark.run_sims()
    benchmark.plot_hdm_lassopack()
    benchmark.plot_sklearn()


if __name__ == "__main__":
    main()
