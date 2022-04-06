import time

import stata_setup

stata_setup.config("/Applications/Stata/", "be")
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from pystata import stata
from rlassopy import Rlasso
from rpy2.robjects.packages import importr
from sklearn.datasets import make_regression

rpy2.robjects.numpy2ri.activate()

hdm = importr("hdm")


X, y = make_regression(n_samples=100, n_features=50, n_informative=50, random_state=0)

n, p = X.shape

X_r = ro.r.matrix(X, nrow=n, ncol=p)
ro.r.assign("X", X_r)
y_r = ro.r.matrix(X, nrow=n, ncol=1)
ro.r.assign("y", y_r)
print("done!")

hdm.rlasso(x=X_r, y=y_r)
