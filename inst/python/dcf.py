# python 2.7.12 (default, Dec  4 2017, 14:50:18) 
# [GCC 5.4.0 20160609]
# tensorflow 1.5.0
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
import os
from datetime import *
import argparse
import scipy.io as sio

import utils
reload(utils)
from utils import *


#############################################################
# set random seed
#############################################################

import random
import time
randseed = 52744889
# randseed = int(time.time()*1000000%100000000)
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)
tf.set_random_seed(randseed)

#############################################################
# set the scale of simulation
#############################################################

parser = argparse.ArgumentParser()
parser.add_argument('-nc', '--numcauses', \
    type=int, default=1000)
parser.add_argument('-nu', '--numunits', \
    type=int, default=1000)
parser.add_argument('-sim', '--simset', \
    choices=['BN','TGP','PSD','SP'], default="BN")
parser.add_argument('-cv', '--cv', \
    type=int, default=0)
parser.add_argument('-snpsig', '--snpsig', \
    type=int, default=40)
parser.add_argument('-confint', '--confint', \
    type=int, default=40)
parser.add_argument('-cp', '--causalprop', \
    type=int, default=1)
parser.add_argument('-nitr', '--niter', \
    type=int, default=20000) # 20000 in yixin's code
parser.add_argument('-data_dir', '--data_dir', \
    type=str, default="/phi/proj/deconfounder/multivariate_medical_deconfounder/dat/1FullCohort")
parser.add_argument('-out_dir', '--out_dir', \
    type=str, default="/phi/proj/deconfounder/MvDeconfounder/res/pca")

args, unknown = parser.parse_known_args()

simset = args.simset
CV = args.cv
a = args.snpsig/100.
b = args.confint/100.
causalprop = args.causalprop/100.
n_iter = args.niter
data_dir = args.data_dir
out_dir = args.out_dir

if not os.path.exists(out_dir):
    os.makedirs(out_dir)


G = load_data(data_dir, 'drugSparseMat.txt', create_mask=False)
n_units = G.shape[0]
n_causes = G.shape[1]
x_train, x_vad, holdout_mask = holdout_data(G)

#############################################################
# estimate causal effects: ppca
#############################################################
print("\nppca control\n")

# the stochastic vi code subsamples on columns. we pass in the
# transpose of x_train to subsampling on rows.
for K in [10,20,30,50,100]:
    print("K: {}".format(K))
    ppca_x_post, ppca_w_post, ppca_z_post, ppca_x_post_np, ppca_z_post_np = fit_ppca(x_train.T, stddv_datapoints=1.0, M=100, K=10, n_iter=n_iter, optimizer="adam")
    np.savetxt(os.path.join(out_dir, "ppca_x_post_np_{}.txt".format(K)), ppca_x_post_np)
    np.savetxt(os.path.join(out_dir, "ppca_z_post_np_{}.txt".format(K)), ppca_z_post_np)

    print("check PPCA fit")

    print("trivial mse", ((G-0)**2).mean())

    print("PPCA mse", ((G-ppca_x_post_np.T)**2).mean())

    ppca_pval = ppca_predictive_check_subsample(x_train.T, x_vad.T, holdout_mask.T, ppca_x_post, ppca_w_post, ppca_z_post)

    print("PPCA predictive check", ppca_pval)

