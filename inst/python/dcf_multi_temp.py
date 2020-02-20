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


#############################################################
# set random seed
#############################################################

import random
import time

randseed = int(time.time() * 1000000 % 100000000)
print("random seed: ", randseed)
random.seed(randseed)
np.random.seed(randseed)
tf.set_random_seed(randseed)

#############################################################
# set the scale of simulation
#############################################################

parser = argparse.ArgumentParser()
parser.add_argument('-nc', '--numcauses', \
                    type=int, default=10)
parser.add_argument('-nu', '--numunits', \
                    type=int, default=20000)
parser.add_argument('-sim', '--simset', \
                    choices=['BN', 'TGP', 'PSD', 'SP'], default="BN")
parser.add_argument('-cv', '--cv', \
                    type=int, default=0)
parser.add_argument('-snpsig', '--snpsig', \
                    type=int, default=98)
parser.add_argument('-confint', '--confint', \
                    type=int, default=1)
parser.add_argument('-cp', '--causalprop', \
                    type=int, default=100)
parser.add_argument('-nitr', '--niter', \
                    type=int, default=20000)#20000
parser.add_argument('-no', '--numoutcomes', \
                    type=int, default=50)
parser.add_argument('-ld', '--lowdim', \
                    type=int, default=1)
parser.add_argument('-K', '--K', \
                    type=int, default=2)
parser.add_argument('-rn', '--reg_nitr', \
                    type=int, default=100000)#100000
parser.add_argument('-data_dir', '--data_dir', \
    type=str, default="/phi/proj/deconfounder/multivariate_medical_deconfounder/dat/1FullCohort")
parser.add_argument('-out_dir', '--out_dir', \
    type=str, default="/phi/proj/deconfounder/multivariate_medical_deconfounder/res")

args, unknown = parser.parse_known_args()

n_causes = args.numcauses
n_units = args.numunits
simset = args.simset
CV = args.cv
a = args.snpsig / 100.
b = args.confint / 100.
causalprop = args.causalprop / 100.
n_iter = args.niter
n_outcomes = args.numoutcomes
lowdim = args.lowdim
K = args.K
reg_nitr = args.reg_nitr
data_dir = args.data_dir
# out_dir = os.path.join(args.out_dir, str(randseed)+"nitr"+str(n_iter)+"regnitr"+str(reg_nitr))
#
#
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
#############################################################
# load preprocessed genetics data
# to preprocess the data, run clean_hapmap.py
#############################################################

# Fs = np.loadtxt("hapmap_mimno_genes_clean_Fs.csv")
# ps = np.loadtxt("hapmap_mimno_genes_clean_ps.csv")
# hapmap_gene_clean = pd.read_csv("hapmap_mimno_genes_clean.csv")
# n_hapmapgenes = hapmap_gene_clean.shape[1]
#
# #############################################################
# # simulate genes (causes) and traits (outcomes)
# #############################################################
#
# D = 3
# if simset == "BN":
#     G, lambdas = sim_genes_BN(Fs, ps, n_hapmapgenes, n_causes, n_units, D)
# elif simset == "TGP":
#     G, lambdas = sim_genes_TGP(Fs, ps, n_hapmapgenes, n_causes, n_units, hapmap_gene_clean, D)
# elif simset == "PSD":
#     G, lambdas = sim_genes_PSD(Fs, ps, n_hapmapgenes, n_causes, n_units, D)
# elif simset == "SP":
#     G, lambdas = sim_genes_SP(Fs, ps, n_hapmapgenes, n_causes, n_units, D)
#
# # remove genes that take the same value on all individuals
# const_cols = np.where(np.var(G, axis=0) < 0.001)[0]
# print(const_cols)
# if len(const_cols) > 0:
#     G = G[:, list(set(range(n_causes)) - set(const_cols))]
#     n_causes -= len(const_cols)
#
# # simulate outcome
# y, y_bin, true_betas, true_lambdas, betas, gamma, lambdacoefs = sim_multiple_traits(lambdas, G, \
#                                                                                     a=a, b=b, n_outcomes=n_outcomes,
#                                                                                     K=K, causalprop=causalprop)
#
# trivial_rmse = np.sqrt(((true_betas - 0) ** 2).mean())
#
# print("trivial", trivial_rmse)


G = load_data(data_dir, 'drugSparseMat.txt', create_mask=False)
n_units = G.shape[0]
n_causes = G.shape[1]

# #############################################################
# # estimate causal effects: no control
# #############################################################
# print("\nno control\n")
#
# nctrl_linear_rmse_sing = np.zeros([n_causes, n_outcomes])
#
# nctrl_logistic_rmse_sing = np.zeros([n_causes, n_outcomes])
#
# for j in range(n_causes):
#     X = np.column_stack([G[:, j][:, np.newaxis]])
#     # X = np.column_stack([G[:,j].toarray()])
#     for o in range(n_outcomes):
#         _, nctrl_linear_rmse_sing[j][o] = fit_outcome_linear(X, y[:, o], true_betas[j][o], 1, CV=False)
#         _, nctrl_logistic_rmse_sing[j][o] = fit_outcome_logistic(X, y_bin[:, o], true_betas[j][o], 1, CV=False)
#
# print("nctrl_linear_rmse_sing.mean()", nctrl_linear_rmse_sing.mean())
#
# print("nctrl_logistic_rmse_sing.mean()", nctrl_logistic_rmse_sing.mean())
#
# X = G
#
# nctrl_linear_reg, nctrl_linear_rmse = fit_multiple_outcome_linear(X, y, true_betas, n_causes, CV=CV, lowdim=lowdim, K=K,
#                                                                   n_iter=reg_nitr, verbose=False)
#
# print("nctrl_linear_rmse", nctrl_linear_rmse)
#
# nctrl_logistic_reg, nctrl_logistic_rmse = fit_multiple_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV,
#                                                                         lowdim=lowdim, K=K, n_iter=reg_nitr,
#                                                                         verbose=False)
#
# print("nctrl_logistic_rmse", nctrl_logistic_rmse)
#
# #############################################################
# # estimate causal effects: oracle
# #############################################################
# print("\noracle control\n")
#
# oracle_linear_rmse_sing = np.zeros([n_causes, n_outcomes])
#
# oracle_logistic_rmse_sing = np.zeros([n_causes, n_outcomes])
#
# for j in range(n_causes):
#     X = np.column_stack([G[:, j][:, np.newaxis], lambdas])
#     # X = np.column_stack([G[:,j].toarray(), lambdas])
#     for o in range(n_outcomes):
#         _, oracle_linear_rmse_sing[j][o] = fit_outcome_linear(X, y[:, o], true_betas[j][o], 1, CV=False)
#         _, oracle_logistic_rmse_sing[j][o] = fit_outcome_logistic(X, y_bin[:, o], true_betas[j][o], 1, CV=False)
#
# print("oracle_linear_rmse_sing.mean()", oracle_linear_rmse_sing.mean())
#
# print("oracle_logistic_rmse_sing.mean()", oracle_logistic_rmse_sing.mean())
#
# X = np.column_stack([G, lambdas])
#
# oracle_linear_reg, oracle_linear_rmse = fit_multiple_outcome_linear(X, y, true_betas, n_causes, CV=CV, lowdim=lowdim,
#                                                                     K=K, n_iter=reg_nitr, verbose=False)
# print("oracle_linear_rmse", oracle_linear_rmse)
#
# oracle_logistic_reg, oracle_logistic_rmse = fit_multiple_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV,
#                                                                           lowdim=lowdim, K=K, n_iter=reg_nitr,
#                                                                           verbose=False)
# print("oracle_logistic_rmse", oracle_logistic_rmse)
#

x_train, x_vad, holdout_mask = holdout_data(G)

#############################################################
# estimate causal effects: ppca
#############################################################
# print("\nppca control\n")
#
# # the stochastic vi code subsamples on columns. we pass in the
# # transpose of x_train to subsampling on rows.
#
# ppca_x_post, ppca_w_post, ppca_z_post, ppca_x_post_np, ppca_z_post_np = fit_ppca(x_train.T, stddv_datapoints=1.0, M=100,
#                                                                                  K=5, n_iter=n_iter, optimizer="adam")
# # np.savetxt(os.path.join(out_dir, "ppca_x_post_np.txt"), ppca_x_post_np)
# # np.savetxt(os.path.join(out_dir, "ppca_z_post_np.txt"), ppca_z_post_np)
#
# print("check PPCA fit")
#
# print("trivial mse", ((G - 0) ** 2).mean())
#
# print("PPCA mse", ((G - ppca_x_post_np.T) ** 2).mean())
#
# ppca_pval = ppca_predictive_check_subsample(x_train.T, x_vad.T, holdout_mask.T, ppca_x_post, ppca_w_post, ppca_z_post)
#
# print("PPCA predictive check", ppca_pval)

#
# ppca_linear_rmse_sing = np.zeros([n_causes, n_outcomes])
#
# ppca_logistic_rmse_sing = np.zeros([n_causes, n_outcomes])
#
# for j in range(n_causes):
#     # X = np.column_stack([G[:,j][:,np.newaxis], ppca_x_post_np[:,j][:,np.newaxis]])
#     X = np.column_stack([G[:, j][:, np.newaxis], ppca_z_post_np])
#     for o in range(n_outcomes):
#         _, ppca_linear_rmse_sing[j][o] = fit_outcome_linear(X, y[:, o], true_betas[j][o], 1, CV=False)
#         _, ppca_logistic_rmse_sing[j][o] = fit_outcome_logistic(X, y_bin[:, o], true_betas[j][o], 1, CV=False)
#
# print("ppca_linear_rmse_sing.mean()", ppca_linear_rmse_sing.mean())
#
# print("ppca_logistic_rmse_sing.mean()", ppca_logistic_rmse_sing.mean())
#
# X = np.column_stack([G, ppca_z_post_np])
#
# ppca_linear_reg, ppca_linear_rmse = fit_multiple_outcome_linear(X, y, true_betas, n_causes, CV=CV, lowdim=lowdim, K=K,
#                                                                 n_iter=reg_nitr, verbose=False)
# print("ppca_linear_rmse", ppca_linear_rmse)
#
# ppca_logistic_reg, ppca_logistic_rmse = fit_multiple_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV,
#                                                                       lowdim=lowdim, K=K, n_iter=reg_nitr,
#                                                                       verbose=False)
# print("ppca_logistic_rmse", ppca_logistic_rmse)

#############################################################
# estimate causal effects: pmf
#############################################################
print("\npmf control\n")

# the stochastic vi code subsamples on columns. we pass in the
# transpose of x_train to subsampling on rows.
#
# pmf_x_post, pmf_z_post, pmf_w_post, pmf_x_post_np, pmf_z_post_np = fit_pmf(x_train.T, M=100, K=10, n_iter=n_iter)
# np.savetxt(os.path.join(out_dir, "pmf_x_post_np.txt"), pmf_x_post_np)
# np.savetxt(os.path.join(out_dir, "pmf_z_post_np.txt"), pmf_z_post_np)
# print("check PMF fit")
#
# print("trivial mse", ((G - 0) ** 2).mean())
#
# print("PMF mse", ((G - pmf_x_post_np.T) ** 2).mean())
#
# pmf_pval = pmf_predictive_check_subsample(x_train.T, x_vad.T, holdout_mask.T, pmf_x_post, pmf_w_post, pmf_z_post)
#
# print("PMF predictive check", pmf_pval)

Y, mask = load_data(data_dir, "measChangeSparseMat.txt", create_mask=True)
n_outcomes = Y.shape[1]
out_dir = "/phi/proj/deconfounder/multivariate_medical_deconfounder/res/52297372nitr100000regnitr100000"
pmf_z_post_np = np.loadtxt(os.path.join(out_dir, "pmf_z_post_np.txt"))

pmf_no_ctrl = np.zeros([n_causes, n_outcomes])
pmf_dcf = np.zeros([n_causes, n_outcomes])

for o in range(n_outcomes):
    print("Outcome {}".format(o))
    row_bool = mask[:, o] == 1
    col_bool = G[row_bool,:].sum(axis=0) >= 0.001*mask[:,o].sum()
    if sum(row_bool) >= 10:
        X = np.column_stack([G[row_bool,:][:,col_bool], pmf_z_post_np[row_bool, :]])
        y = Y[row_bool, o]
        reg_no_ctrl = fit_outcome_linear(G[row_bool,:][:,col_bool], y, sum(col_bool), CV=False, verbose=True)
        pmf_no_ctrl[col_bool, o] = reg_no_ctrl.coef_
        pmf_no_ctrl[~col_bool, o] = np.nan
        reg_dcf = fit_outcome_linear(X, y, sum(col_bool), CV=False, verbose=True)
        pmf_dcf[col_bool, o] = reg_dcf.coef_[:sum(col_bool)]
        pmf_dcf[~col_bool, o] = np.nan
    else:
        pmf_no_ctrl[:,o] = np.nan
        pmf_dcf[:,o] = np.nan
        print("Less than 10 samples, skip outcome {}".format(o))



np.savetxt(os.path.join(out_dir, "pmf_no_ctrl_indept_threshold_coef.txt"), pmf_no_ctrl)
np.savetxt(os.path.join(out_dir, "pmf_dcf_indept_threshold_coef.txt"), pmf_dcf)

# Y[mask==0] = np.nan
# col_mean = np.nanmean(Y, axis=0)
# inds = np.where(np.isnan(Y))
# Y[inds] = np.take(col_mean, inds[1])
#
# pmf_linear_reg = fit_multiple_outcome_linear(G, Y, n_causes, CV=False, lowdim=False, K=K,
#                                                               n_iter=reg_nitr, verbose=False)

# for j in range(n_causes):
#     # X = np.column_stack([G[:,j][:,np.newaxis], pmf_x_post_np[:,j][:,np.newaxis]])
#     X = np.column_stack([G[:, j][:, np.newaxis], pmf_z_post_np])
#     for o in range(n_outcomes):
#         reg_no_ctrl = fit_outcome_linear(G[:,j][:,np.newaxis], y[:, o], 1, CV=False)
#         pmf_no_ctrl[j][o] = reg_no_ctrl.coef_
#         reg_dcf = fit_outcome_linear(X, y[:, o], 1, CV=False)
#         pmf_dcf[j][o] = reg.coef_
#
# X = np.column_stack([G, pmf_z_post_np])
#

#
# pmf_logistic_reg, pmf_logistic_rmse = fit_multiple_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV,
#                                                                     lowdim=lowdim, K=K, n_iter=reg_nitr, verbose=False)
# print("pmf_logistic_rmse", pmf_logistic_rmse)

#############################################################
# estimate causal effects: def
#############################################################
# print("\ndef control\n")
#
# # the stochastic vi code subsamples on columns. we pass in the
# # transpose of x_train to subsampling on rows.
#
# def_x_post, def_z3_post, def_z2_post, def_z1_post, def_W0_post, def_x_post_np, def_z_post_np = fit_def(x_train.T,
#                                                                                                        K=[100, 30, 5],
#                                                                                                        prior_a=0.1,
#                                                                                                        prior_b=0.3,
#                                                                                                        optimizer=tf.train.AdamOptimizer(
#                                                                                                            1e-2),
#                                                                                                        n_iter=n_iter)
# # np.savetxt(os.path.join(out_dir, "def_x_post_np.txt"), def_x_post_np)
# # np.savetxt(os.path.join(out_dir, "def_z_post_np.txt"), def_z_post_np)
# print("check DEF fit")
#
# print("trivial mse", ((G - 0) ** 2).mean())
#
# print("DEF mse", ((G - def_x_post_np.T) ** 2).mean())
#
# def_pval = def_predictive_check_subsample(x_train.T, x_vad.T, holdout_mask.T, def_x_post, def_z1_post, def_W0_post)
#
# print("DEF predictive check", def_pval)

# X = G - def_x_post_np

#
# def_linear_rmse_sing = np.zeros([n_causes, n_outcomes])
#
# def_logistic_rmse_sing = np.zeros([n_causes, n_outcomes])
#
# for j in range(n_causes):
#     # X = np.column_stack([G[:,j][:,np.newaxis], def_x_post_np[:,j][:,np.newaxis]])
#     X = np.column_stack([G[:, j][:, np.newaxis], def_z_post_np])
#     for o in range(n_outcomes):
#         _, def_linear_rmse_sing[j][o] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False, verbose=False)
#         _, def_logistic_rmse_sing[j][o] = fit_outcome_logistic(X, y_bin[:, o], true_betas[j][o], 1, CV=False,
#                                                                verbose=False)
#
# print("def_linear_rmse_sing.mean()", def_linear_rmse_sing.mean())
#
# print("def_logistic_rmse_sing.mean()", def_logistic_rmse_sing.mean())
#
# X = np.column_stack([G, def_z_post_np])
#
# def_linear_reg, def_linear_rmse = fit_multiple_outcome_linear(X, y, true_betas, n_causes, CV=CV, lowdim=lowdim, K=K,
#                                                               n_iter=reg_nitr, verbose=False)
#
# print("def_linear_rmse", def_linear_rmse)
#
# def_logistic_reg, def_logistic_rmse = fit_multiple_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV,
#                                                                     lowdim=lowdim, K=K, n_iter=reg_nitr, verbose=False)
#
# print("def_logistic_rmse", def_logistic_rmse)

#############################################################
# estimate causal effects: lfa
#############################################################
# print("\nlfa control\n")
#
# # the stochastic vi code subsamples on columns. we pass in the
# # transpose of x_train to subsampling on rows.
#
# lfa_x_post, lfa_z_post, lfa_w_post, lfa_x_post_np, lfa_z_post_np = fit_lfa(x_train.T, M=100, K=10, n_iter=n_iter)
#
# print("check LFA fit")
#
# print("trivial mse", ((G - 0) ** 2).mean())
#
# print("LFA mse", ((G - lfa_x_post_np.T) ** 2).mean())

# lfa_pval = lfa_predictive_check(x_train.T, x_vad.T, holdout_mask.T, lfa_x_post, lfa_w_post, lfa_z_post)

# "LFA predictive check", lfa_pval)

# "PPCA predictive check", lfa_pval)

#
# lfa_linear_rmse_sing = np.zeros([n_causes, n_outcomes])
#
# lfa_logistic_rmse_sing = np.zeros([n_causes, n_outcomes])
#
# for j in range(n_causes):
#     # X = np.column_stack([G[:,j][:,np.newaxis], lfa_x_post_np[:,j][:,np.newaxis]])
#     X = np.column_stack([G[:, j][:, np.newaxis], lfa_z_post_np])
#     for o in range(n_outcomes):
#         _, lfa_linear_rmse_sing[j][o] = fit_outcome_linear(X, y[:, o], true_betas[j][o], 1, CV=False)
#         _, lfa_logistic_rmse_sing[j][o] = fit_outcome_logistic(X, y_bin[:, o], true_betas[j][o], 1, CV=False)
#
# print("lfa_linear_rmse_sing.mean()", lfa_linear_rmse_sing.mean())
#
# print("lfa_logistic_rmse_sing.mean()", lfa_logistic_rmse_sing.mean())
#
# X = np.column_stack([G, lfa_z_post_np])
#
# lfa_linear_reg, lfa_linear_rmse = fit_multiple_outcome_linear(X, y, true_betas, n_causes, CV=CV, lowdim=lowdim, K=K,
#                                                               n_iter=reg_nitr, verbose=False)
# print("lfa_linear_rmse", lfa_linear_rmse)
#
# lfa_logistic_reg, lfa_logistic_rmse = fit_multiple_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV,
#                                                                     lowdim=lowdim, K=K, n_iter=reg_nitr, verbose=False)
# print("lfa_logistic_rmse", lfa_logistic_rmse)

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
# print("genotype simulation setting", simset)
#
# print("causal prop", causalprop)
#
# print("SNP effect", a, "group intercept", b, "group-specific random effect", 100 - a - b)
#
# print("########## estimate single SNP effect")
#
# print("trivial", trivial_rmse)
#
# print("\nno control\n")
# print("nctrl_linear_rmse_sing.mean()", nctrl_linear_rmse_sing.mean())
# print("nctrl_logistic_rmse_sing.mean()", nctrl_logistic_rmse_sing.mean())
#
# print("\noracle control\n")
# print("oracle_linear_rmse_sing.mean()", oracle_linear_rmse_sing.mean())
# print("oracle_logistic_rmse_sing.mean()", oracle_logistic_rmse_sing.mean())
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
# print("def_logistic_rmse_sing.mean()", def_logistic_rmse_sing.mean())
#
# print("\nlfa control\n")
# print("lfa_linear_rmse_sing.mean()", lfa_linear_rmse_sing.mean())
# print("lfa_logistic_rmse_sing.mean()", lfa_logistic_rmse_sing.mean())
#
# print("########## estimate all SNPs effect")
#
# print("trivial", trivial_rmse)
#
# print("\nno control\n")
# print("nctrl_linear_rmse", nctrl_linear_rmse)
# print("nctrl_logistic_rmse", nctrl_logistic_rmse)
#
# print("\noracle control\n")
# print("oracle_linear_rmse", oracle_linear_rmse)
# print("oracle_logistic_rmse", oracle_logistic_rmse)
#
# print("\nppca control\n")
# print("ppca_linear_rmse", ppca_linear_rmse)
# print("ppca_logistic_rmse", ppca_logistic_rmse)
#
# print("\npmf control\n")
# print("pmf_linear_rmse", pmf_linear_rmse)
# print("pmf_logistic_rmse", pmf_logistic_rmse)
#
# print("\ndef control\n")
# print("def_linear_rmse", def_linear_rmse)
# print("def_logistic_rmse", def_logistic_rmse)
#
# print("\nlfa control\n")
# print("lfa_linear_rmse", lfa_linear_rmse)
# print("lfa_logistic_rmse", lfa_logistic_rmse)
#
# res = pd.DataFrame({"randomseed": randseed,
#                     "n_causes": n_causes,
#                     "n_units": n_units,
#                     "simset": simset,
#                     "causalprop": causalprop,
#                     "SNPeffect": a,
#                     "groupintercept": b,
#                     "grouprandomeffect": 100 - a - b,
#                     "trivial_mse": trivial_rmse,
#                     "nctrl_linear_rmse_sing_mean": nctrl_linear_rmse_sing.mean(),
#                     "nctrl_logistic_rmse_sing_mean": nctrl_logistic_rmse_sing.mean(),
#                     "oracle_linear_rmse_sing_mean": oracle_linear_rmse_sing.mean(),
#                     "oracle_logistic_rmse_sing_mean": oracle_logistic_rmse_sing.mean(),
#                     "ppca_linear_rmse_sing_mean": ppca_linear_rmse_sing.mean(),
#                     "ppca_logistic_rmse_sing_mean": ppca_logistic_rmse_sing.mean(),
#                     "pmf_linear_rmse_sing_mean": pmf_linear_rmse_sing.mean(),
#                     "pmf_logistic_rmse_sing_mean": pmf_logistic_rmse_sing.mean(),
#                     "def_linear_rmse_sing_mean": def_linear_rmse_sing.mean(),
#                     "def_logistic_rmse_sing_mean": def_logistic_rmse_sing.mean(),
#                     "lfa_linear_rmse_sing_mean": lfa_linear_rmse_sing.mean(),
#                     "lfa_logistic_rmse_sing_mean": lfa_logistic_rmse_sing.mean(),
#                     "nctrl_linear_rmse": nctrl_linear_rmse,
#                     "nctrl_logistic_rmse": nctrl_logistic_rmse,
#                     "oracle_linear_rmse": oracle_linear_rmse,
#                     "oracle_logistic_rmse": oracle_logistic_rmse,
#                     "ppca_linear_rmse": ppca_linear_rmse,
#                     "ppca_logistic_rmse": ppca_logistic_rmse,
#                     "pmf_linear_rmse": pmf_linear_rmse,
#                     "pmf_logistic_rmse": pmf_logistic_rmse,
#                     "def_linear_rmse": def_linear_rmse,
#                     "def_logistic_rmse": def_logistic_rmse,
#                     "lfa_linear_rmse": lfa_linear_rmse,
#                     "lfa_logistic_rmse": lfa_logistic_rmse}, index=[0])
#
# res = res.T
#
# filename = "dcf" + "_ncauses" + str(n_causes) + "_nunits" + str(n_units) + "_simset" + simset + "_cv" + str(
#     CV) + "_causalprop" + str(causalprop) + "_snpsig" + str(args.snpsig) + "_confint" + str(
#     args.confint) + "_randseed" + str(randseed) + ".csv"
#
# if os.path.isfile(os.path.join(out_dir, filename)):
#     filename = filename = "dcf" + "_ncauses" + str(n_causes) + "_nunits" + str(
#         n_units) + "_simset" + simset + "_cv" + str(CV) + "_causalprop" + str(causalprop) + "_snpsig" + str(
#         args.snpsig) + "_confint" + str(args.confint) + "_randseed" + str(randseed + 1) + ".csv"
#
# res.to_csv(os.path.join(out_dir, filename))
