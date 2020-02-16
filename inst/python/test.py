# python 2
# edward 1.3.5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import edward as ed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy
import scipy.io as sio
import math
import os
from datetime import *
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from edward.models import Normal, Gamma, Dirichlet, InverseGamma, \
    Poisson, PointMass, Empirical, ParamMixture, \
    MultivariateNormalDiag, Categorical, Laplace, \
    MultivariateNormalTriL, Bernoulli, TransformedDistribution,\
    Binomial
from edward.util import Progbar
from scipy import sparse, stats
from scipy.special import expit, logit
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split


import tensorflow as tf
import edward as ed
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import numpy as np
import numpy.random as npr
import pandas as pd
import os
from datetime import *
import argparse

import utils

reload(utils)
import utils_multi

reload(utils_multi)
from utils import *
from utils_multi import *

def load_data(path, data_filename, create_mask = True):
    path = os.path.expanduser(path)
    filepath = os.path.join(path, data_filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError

    data = sio.mmread(filepath)
    if create_mask:
        row, col = data.nonzero()
        mask = scipy.sparse.csr_matrix((np.ones((row.shape[0])), (row, col)), dtype=int).toarray()
        return data.toarray(), mask
    else:
        return data.toarray()

data_dir = "/phi/proj/deconfounder/multivariate_medical_deconfounder/dat/PackageTest/"
out_dir = "/phi/proj/deconfounder/multivariate_medical_deconfounder/res/52297372nitr100000regnitr100000/"
Y, mask = load_data(data_dir, "measChangeSparseMat.txt", create_mask=True)
n_outcomes = Y.shape[1]

G = load_data(data_dir, 'drugSparseMat.txt', create_mask=False)
n_units = G.shape[0]
n_causes = G.shape[1]
pmf_z_post_np = np.loadtxt(os.path.join(out_dir, "pmf_z_post_np.txt"))
pmf_z_post_np = pmf_z_post_np[:n_units, :]

pmf_no_ctrl = np.zeros([n_causes, n_outcomes])

pmf_dcf = np.zeros([n_causes, n_outcomes])

print(n_units, n_causes, n_outcomes)
for j in range(n_causes):
    print("cause {}".format(j))
    for o in range(n_outcomes):
        row_bool = mask[:, o] == 1
        if sum(row_bool)>=2:
            X = np.column_stack([G[row_bool, j][:, np.newaxis], pmf_z_post_np[row_bool, :]])
            y = Y[row_bool, o]
            reg_no_ctrl = fit_outcome_linear(G[row_bool,j][:,np.newaxis], y, 1, CV=False)
            pmf_no_ctrl[j][o] = reg_no_ctrl.coef_
            reg_dcf = fit_outcome_linear(X, y, 1, CV=False)
            pmf_dcf[j][o] = reg_dcf.coef_[0]



np.savetxt(os.path.join("/phi/proj/deconfounder/multivariate_medical_deconfounder/res", "pmf_no_ctrl.txt"), pmf_no_ctrl)
np.savetxt(os.path.join("/phi/proj/deconfounder/multivariate_medical_deconfounder/res", "pmf_dcf.txt"), pmf_dcf)