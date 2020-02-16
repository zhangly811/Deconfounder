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
    type=str, default="/phi/proj/deconfounder/MvDeconfounder/res/20200116_fitDef")

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

# # simulate outcome
# y, true_betas = sim_single_outcome(G, n_causes, causalprop=causalprop)
#
#
# trivial_rmse = np.sqrt(((true_betas - 0)**2).mean())
#
# print("trivial", trivial_rmse)


############################################################
# estimate causal effects: no control
############################################################
# print("\nno control\n")
#
#
# nctrl_linear_rmse_sing = np.zeros(n_causes)
#
# # nctrl_logistic_rmse_sing = np.zeros(n_causes)
#
# for j in range(n_causes):
#     X = np.column_stack([G[:,j][:,np.newaxis]])
#     _, nctrl_linear_rmse_sing[j] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False)
#
#
# print("nctrl_linear_rmse_sing.mean()", nctrl_linear_rmse_sing.mean())
#
# X = G
#
# nctrl_linear_reg, nctrl_linear_rmse = fit_outcome_linear(X, y, true_betas, n_causes, CV=CV, verbose=False)
#
# print("nctrl_linear_rmse", nctrl_linear_rmse)


# #############################################################
# # estimate causal effects: oracle
# #############################################################
# print("\noracle control\n")
#
#
#
# oracle_linear_rmse_sing = np.zeros(n_causes)
#
#
# for j in range(n_causes):
#     X = np.column_stack([G[:,j][:,np.newaxis], lambdas])
#     # X = np.column_stack([G[:,j].toarray(), lambdas])
#     _, oracle_linear_rmse_sing[j] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False)
#
#
# print("oracle_linear_rmse_sing.mean()", oracle_linear_rmse_sing.mean())
#
#
#
# X = np.column_stack([G, lambdas])
#
#
# oracle_linear_reg, oracle_linear_rmse = fit_outcome_linear(X, y, true_betas, n_causes, CV=CV, verbose=False)
# print("oracle_linear_rmse", oracle_linear_rmse)

x_train, x_vad, holdout_mask = holdout_data(G)

#############################################################
# estimate causal effects: def
#############################################################
print("\ndef control\n")

# the stochastic vi code subsamples on columns. we pass in the
# transpose of x_train to subsampling on rows. also need to return a
# different def_z_post_np in utils.py. please see explanation in
# utils.py in the fit_def() function.

# in training def, please make sure the loss is negative, otherwise it
# is offer thanks to optimization failure that makes the learning of
# def fail and cannot deconfound.

def_x_post, def_z3_post, def_z2_post, def_z1_post, def_W0_post, def_x_post_np, def_z_post_np = fit_def(x_train.T, K=[100,30,5], prior_a=0.1, prior_b=0.3, optimizer=tf.train.RMSPropOptimizer(1e-3), n_iter=n_iter) #n_iter*100 in yixin's code
np.savetxt(os.path.join(out_dir, "x_post_np_100_30_5.txt"), def_x_post_np)
np.savetxt(os.path.join(out_dir, "z_post_np_100_30_5.txt"), def_z_post_np)
print("def_x_post", def_x_post.shape) #148,100
print("def_z3_post", def_z3_post.shape) #148,5
print("def_z2_post", def_z2_post.shape) #148,40
print("def_z1_post", def_z1_post.shape) #148,99
print("def_W0_post", def_W0_post.shape) #99,100
print("def_x_post_np", def_x_post_np.shape) #100,148
print("def_z_post_np", def_z_post_np.shape) #100,99


print("check DEF fit")

print("trivial mse", ((G-0)**2).mean())

print("DEF mse", ((G-def_x_post_np.T)**2).mean())

subsample_pvals = def_predictive_check_subsample(x_train.T, x_vad.T, holdout_mask.T, def_x_post, def_z1_post, def_W0_post)
def_pval = np.mean(subsample_pvals)

print("DEF predictive check", def_pval)

# X = G - def_x_post_np


# def_linear_rmse_sing = np.zeros(n_causes)
#
#
# for j in range(n_causes):
#     # X = np.column_stack([G[:,j][:,np.newaxis], def_x_post_np[:,j][:,np.newaxis]])
#     # X = np.column_stack([G[:,j][:,np.newaxis], def_x_post_np.T])
#     X = np.column_stack([G[:,j][:,np.newaxis], def_z_post_np])
#     _, def_linear_rmse_sing[j] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False, verbose=False)
#
#
# print("def_linear_rmse_sing.mean()", def_linear_rmse_sing.mean())
#
#
# X = np.column_stack([G, def_z_post_np])
#
# def_linear_reg, def_linear_rmse = fit_outcome_linear(X, y, true_betas, n_causes, CV=CV, verbose=False)
#
# print("def_linear_rmse", def_linear_rmse)

#############################################################
# print all results
#############################################################
# print("\n\n\n###############################")
# print("########## print all results")
#
# print("randomseed", randseed)
#
# print("n_causes", n_causes, "n_units", n_units)
#
# print("genotype simulation setting", simset, "n_iter", n_iter)
#
# print("causal prop", causalprop)
#
# print("########## estimate single SNP effect")
#
# print("trivial", trivial_rmse)
#
# print("\nno control\n")
# print("nctrl_linear_rmse_sing.mean()", nctrl_linear_rmse_sing.mean())
#
#
# print("\noracle control\n")
# print("oracle_linear_rmse_sing.mean()", oracle_linear_rmse_sing.mean())
#
# print("\npca control\n")
# print("pca_linear_rmse_sing.mean()", pca_linear_rmse_sing.mean())
# print("pca_logistic_rmse_sing.mean()", pca_logistic_rmse_sing.mean())
#
# print("\nppca control\n")
# print("ppca_linear_rmse_sing.mean()", ppca_linear_rmse_sing.mean())
# print("ppca_logistic_rmse_sing.mean()", ppca_logistic_rmse_sing.mean())
#
# print("\npmf control\n")
# print("pmf_linear_rmse_sing.mean()", pmf_linear_rmse_sing.mean())
# print("pmf_logistic_rmse_sing.mean()", pmf_logistic_rmse_sing.mean())
#
# print("\ndef control\n")
# print("def_linear_rmse_sing.mean()", def_linear_rmse_sing.mean())
#
# print("\nlfa control\n")
# print("lfa_linear_rmse_sing.mean()", lfa_linear_rmse_sing.mean())
# print("lfa_logistic_rmse_sing.mean()", lfa_logistic_rmse_sing.mean())
#
#
# print("########## estimate all SNPs effect")
#
# print("trivial", trivial_rmse)
#
# print("\nno control\n")
# print("nctrl_linear_rmse", nctrl_linear_rmse)
#
# print("\noracle control\n")
# print("oracle_linear_rmse", oracle_linear_rmse)
#
# print("\nppca control\n")
# print("ppca_linear_rmse", ppca_linear_rmse)
# print("ppca_logistic_rmse", ppca_logistic_rmse)
#
# print("\npca control\n")
# print("pca_linear_rmse", pca_linear_rmse)
# print("pca_logistic_rmse", pca_logistic_rmse)
#
#
# print("\npmf control\n")
# print("pmf_linear_rmse", pmf_linear_rmse)
# print("pmf_logistic_rmse", pmf_logistic_rmse)
#
# print("\ndef control\n")
# print("def_linear_rmse", def_linear_rmse)
#
# print("\nlfa control\n")
# print("lfa_linear_rmse", lfa_linear_rmse)
# print("lfa_logistic_rmse", lfa_logistic_rmse)
#
#
# res = pd.DataFrame({"randomseed": randseed,
#     "n_causes": n_causes,
#     "n_units": n_units,
#     "simset": simset,
#     "n_iter": n_iter,
#     "cv": CV,
#     "causalprop": causalprop,
#     "SNPeffect": args.snpsig,
#     "groupintercept": args.confint,
#     "grouprandomeffect": 100-args.snpsig-args.confint,
#     "trivial_rmse": trivial_rmse,
#     "nctrl_linear_rmse_sing_mean": nctrl_linear_rmse_sing.mean(),
    # "nctrl_logistic_rmse_sing_mean": nctrl_logistic_rmse_sing.mean(),
    # "oracle_linear_rmse_sing_mean": oracle_linear_rmse_sing.mean(),
    # "oracle_logistic_rmse_sing_mean": oracle_logistic_rmse_sing.mean(),
#     "pca_linear_rmse_sing_mean": pca_linear_rmse_sing.mean(),
#     "pca_logistic_rmse_sing_mean": pca_logistic_rmse_sing.mean(),
#     "ppca_linear_rmse_sing_mean": ppca_linear_rmse_sing.mean(),
#     "ppca_logistic_rmse_sing_mean": ppca_logistic_rmse_sing.mean(),
#     "pmf_linear_rmse_sing_mean": pmf_linear_rmse_sing.mean(),
#     "pmf_logistic_rmse_sing_mean": pmf_logistic_rmse_sing.mean(),
#     "def_linear_rmse_sing_mean": def_linear_rmse_sing.mean(),
    # "def_logistic_rmse_sing_mean": def_logistic_rmse_sing.mean(),
#     "lfa_linear_rmse_sing_mean": lfa_linear_rmse_sing.mean(),
#     "lfa_logistic_rmse_sing_mean": lfa_logistic_rmse_sing.mean(),
#     "nctrl_linear_rmse": nctrl_linear_rmse,
    # "nctrl_logistic_rmse": nctrl_logistic_rmse,
    # "oracle_linear_rmse": oracle_linear_rmse,
    # "oracle_logistic_rmse": oracle_logistic_rmse,
#     "pca_linear_rmse": pca_linear_rmse,
#     "pca_logistic_rmse": pca_logistic_rmse,
#     "ppca_linear_rmse": ppca_linear_rmse,
#     "ppca_logistic_rmse": ppca_logistic_rmse,
#     "pmf_linear_rmse": pmf_linear_rmse,
#     "pmf_logistic_rmse": pmf_logistic_rmse,
#     "def_linear_rmse": def_linear_rmse,
    # "def_logistic_rmse": def_logistic_rmse,
#     "lfa_linear_rmse": lfa_linear_rmse,
#     "lfa_logistic_rmse": lfa_logistic_rmse
#                         }, index=[0])
#
# res = res.T
# #
# filename = "dcf"+"_ncauses"+str(n_causes)+"_nunits"+str(n_units)+"_simset"+simset+"_cv"+str(CV)+"_causalprop"+str(args.causalprop)+"_snpsig"+str(args.snpsig)+"_confint"+str(args.confint)+"_nitr"+str(n_iter)+"_randseed"+str(randseed)+".csv"
#
# if os.path.isfile(os.path.join(outdir, filename)):
#     filename = "dcf"+"_ncauses"+str(n_causes)+"_nunits"+str(n_units)+"_simset"+simset+"_cv"+str(CV)+"_causalprop"+str(args.causalprop)+"_snpsig"+str(args.snpsig)+"_confint"+str(args.confint)+"_nitr"+str(n_iter)+"_randseed"+str(randseed+1)+".csv"
#
# res.to_csv(os.path.join(outdir, filename))
