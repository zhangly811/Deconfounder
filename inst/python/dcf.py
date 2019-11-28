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

import utils
reload(utils)
from utils import *


if not os.path.exists("res"):
    os.makedirs("res")

outdir = "res"

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
    type=int, default=100) #20000

args, unknown = parser.parse_known_args()

n_causes = args.numcauses
n_units = args.numunits
simset = args.simset
CV = args.cv
a = args.snpsig/100.
b = args.confint/100.
causalprop = args.causalprop/100.
n_iter = args.niter

#############################################################
# load preprocessed genetics data
# to preprocess the data, run clean_hapmap.py
#############################################################

Fs = np.loadtxt("hapmap_mimno_genes_clean_Fs.csv")
ps = np.loadtxt("hapmap_mimno_genes_clean_ps.csv")
hapmap_gene_clean = pd.read_csv("hapmap_mimno_genes_clean.csv")
n_hapmapgenes = hapmap_gene_clean.shape[1]

#############################################################
# simulate genes (causes) and traits (outcomes)
#############################################################

D = 3
if simset == "BN":
    G, lambdas = sim_genes_BN(Fs, ps, n_hapmapgenes, n_causes, n_units, D)
elif simset == "TGP":
    G, lambdas = sim_genes_TGP(Fs, ps, n_hapmapgenes, n_causes, n_units, hapmap_gene_clean, D)
elif simset == "PSD":
    G, lambdas = sim_genes_PSD(Fs, ps, n_hapmapgenes, n_causes, n_units, D)
elif simset == "SP":
    G, lambdas = sim_genes_SP(Fs, ps, n_hapmapgenes, n_causes, n_units, D)

# remove genes that take the same value on all individuals
const_cols = np.where(np.var(G,axis=0)<0.001)[0]
print(const_cols)
if len(const_cols) > 0:
    G = G[:,list(set(range(n_causes))-set(const_cols))]
    n_causes -= len(const_cols)
    

# simulate outcome
y, y_bin, true_betas, true_lambdas = sim_single_traits(lambdas, G, \
    a=a, b=b, causalprop=causalprop)


trivial_rmse = np.sqrt(((true_betas - 0)**2).mean())

print("trivial", trivial_rmse)


#############################################################
# estimate causal effects: no control
#############################################################
print("\nno control\n")


nctrl_linear_rmse_sing = np.zeros(n_causes)

nctrl_logistic_rmse_sing = np.zeros(n_causes)

for j in range(n_causes):
    X = np.column_stack([G[:,j][:,np.newaxis]])
    # X = np.column_stack([G[:,j].toarray()])
    _, nctrl_linear_rmse_sing[j] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False)
    _, nctrl_logistic_rmse_sing[j] = fit_outcome_logistic(X, y_bin, true_betas[j], 1, CV=False)


print("nctrl_linear_rmse_sing.mean()", nctrl_linear_rmse_sing.mean())

print("nctrl_logistic_rmse_sing.mean()", nctrl_logistic_rmse_sing.mean())


X = G


nctrl_linear_reg, nctrl_linear_rmse = fit_outcome_linear(X, y, true_betas, n_causes, CV=CV, verbose=False)

print("nctrl_linear_rmse", nctrl_linear_rmse)


nctrl_logistic_reg, nctrl_logistic_rmse = fit_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV, verbose=False)

print("nctrl_logistic_rmse", nctrl_logistic_rmse)


#############################################################
# estimate causal effects: oracle
#############################################################
print("\noracle control\n")



oracle_linear_rmse_sing = np.zeros(n_causes)

oracle_logistic_rmse_sing = np.zeros(n_causes)

for j in range(n_causes):
    X = np.column_stack([G[:,j][:,np.newaxis], lambdas])
    # X = np.column_stack([G[:,j].toarray(), lambdas])
    _, oracle_linear_rmse_sing[j] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False)
    _, oracle_logistic_rmse_sing[j] = fit_outcome_logistic(X, y_bin, true_betas[j], 1, CV=False)


print("oracle_linear_rmse_sing.mean()", oracle_linear_rmse_sing.mean())

print("oracle_logistic_rmse_sing.mean()", oracle_logistic_rmse_sing.mean())


X = np.column_stack([G, lambdas])


oracle_linear_reg, oracle_linear_rmse = fit_outcome_linear(X, y, true_betas, n_causes, CV=CV, verbose=False)
print("oracle_linear_rmse", oracle_linear_rmse)


oracle_logistic_reg, oracle_logistic_rmse = fit_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV, verbose=False)
print("oracle_logistic_rmse", oracle_logistic_rmse)

x_train, x_vad, holdout_mask = holdout_data(G)



#############################################################
# estimate causal effects: pca
#############################################################
print("\npca control\n")

# the stochastic vi code subsamples on columns. we pass in the
# transpose of x_train to subsampling on rows.

pca = PCA(n_components=10)
pca.fit(x_train)  
pca_z_post_np = pca.fit_transform(G)
pca_x_post_np = pca.inverse_transform(pca_z_post_np)

# pca_x_post, pca_w_post, pca_z_post, pca_x_post_np, pca_z_post_np = fit_pca(x_train.T, stddv_datapoints=1.0, M=100, K=5, n_iter=n_iter, optimizer="adam")

print("check pca fit")

print("trivial mse", ((G-0)**2).mean())

print("pca mse", ((G-pca_x_post_np)**2).mean())

# pca_pval = pca_predictive_check(x_train.T, x_vad.T, holdout_mask.T, pca_x_post, pca_w_post, pca_z_post)

# "pca predictive check", pca_pval)



pca_linear_rmse_sing = np.zeros(n_causes)

pca_logistic_rmse_sing = np.zeros(n_causes)

for j in range(n_causes):
    # X = np.column_stack([G[:,j][:,np.newaxis], pca_x_post_np[:,j][:,np.newaxis]])
    X = np.column_stack([G[:,j][:,np.newaxis], pca_z_post_np])
    _, pca_linear_rmse_sing[j] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False)
    _, pca_logistic_rmse_sing[j] = fit_outcome_logistic(X, y_bin, true_betas[j], 1, CV=False)


print("pca_linear_rmse_sing.mean()", pca_linear_rmse_sing.mean())

print("pca_logistic_rmse_sing.mean()", pca_logistic_rmse_sing.mean())



# X = np.column_stack([G[:,:nsub], pca_z_post_np])

# pca_linear_reg, pca_linear_rmse = fit_outcome_linear(X, y, true_betas[:nsub], nsub, CV=CV, verbose=False)
pca_linear_reg, pca_linear_rmse = -100, -100
print("pca_linear_rmse", pca_linear_rmse)

# pca_logistic_reg, pca_logistic_rmse = fit_outcome_logistic(X, y_bin, true_betas[:nsub], nsub, CV=CV, verbose=False)


pca_logistic_reg, pca_logistic_rmse = -100, -100
print("pca_logistic_rmse", pca_logistic_rmse)



#############################################################
# estimate causal effects: ppca
#############################################################
print("\nppca control\n")

# the stochastic vi code subsamples on columns. we pass in the
# transpose of x_train to subsampling on rows.

ppca_x_post, ppca_w_post, ppca_z_post, ppca_x_post_np, ppca_z_post_np = fit_ppca(x_train.T, stddv_datapoints=1.0, M=100, K=5, n_iter=n_iter, optimizer="adam")

print("check PPCA fit")

print("trivial mse", ((G-0)**2).mean())

print("PPCA mse", ((G-ppca_x_post_np.T)**2).mean())

# ppca_pval = ppca_predictive_check(x_train.T, x_vad.T, holdout_mask.T, ppca_x_post, ppca_w_post, ppca_z_post)

# "PPCA predictive check", ppca_pval)



ppca_linear_rmse_sing = np.zeros(n_causes)

ppca_logistic_rmse_sing = np.zeros(n_causes)

for j in range(n_causes):
    # X = np.column_stack([G[:,j][:,np.newaxis], ppca_x_post_np[:,j][:,np.newaxis]])
    X = np.column_stack([G[:,j][:,np.newaxis], ppca_z_post_np])
    _, ppca_linear_rmse_sing[j] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False)
    _, ppca_logistic_rmse_sing[j] = fit_outcome_logistic(X, y_bin, true_betas[j], 1, CV=False)


print("ppca_linear_rmse_sing.mean()", ppca_linear_rmse_sing.mean())

print("ppca_logistic_rmse_sing.mean()", ppca_logistic_rmse_sing.mean())


X = np.column_stack([G, ppca_z_post_np])

ppca_linear_reg, ppca_linear_rmse = fit_outcome_linear(X, y, true_betas, n_causes, CV=CV, verbose=False)
print("ppca_linear_rmse", ppca_linear_rmse)

ppca_logistic_reg, ppca_logistic_rmse = fit_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV, verbose=False)
print("ppca_logistic_rmse", ppca_logistic_rmse)


#############################################################
# estimate causal effects: pmf
#############################################################
print("\npmf control\n")

# the stochastic vi code subsamples on columns. we pass in the
# transpose of x_train to subsampling on rows.

pmf_x_post, pmf_z_post, pmf_w_post, pmf_x_post_np, pmf_z_post_np = fit_pmf(x_train.T, M=100, K=10, n_iter=n_iter, optimizer="adam")

print("check PMF fit")

print("trivial mse", ((G-0)**2).mean())

print("PMF mse", ((G-pmf_x_post_np.T)**2).mean())


# pmf_pval = pmf_predictive_check(x_train.T, x_vad.T, holdout_mask.T, pmf_x_post, pmf_w_post, pmf_z_post)

# "PMF predictive check", pmf_pval)


pmf_linear_rmse_sing = np.zeros(n_causes)

pmf_logistic_rmse_sing = np.zeros(n_causes)

for j in range(n_causes):
    # X = np.column_stack([G[:,j][:,np.newaxis], pmf_x_post_np[:,j][:,np.newaxis]])
    X = np.column_stack([G[:,j][:,np.newaxis], pmf_z_post_np])
    _, pmf_linear_rmse_sing[j] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False)
    _, pmf_logistic_rmse_sing[j] = fit_outcome_logistic(X, y_bin, true_betas[j], 1, CV=False)


print("pmf_linear_rmse_sing.mean()", pmf_linear_rmse_sing.mean())

print("pmf_logistic_rmse_sing.mean()", pmf_logistic_rmse_sing.mean())



X = np.column_stack([G, pmf_z_post_np])

pmf_linear_reg, pmf_linear_rmse = fit_outcome_linear(X, y, true_betas, n_causes, CV=CV, verbose=False)
print("pmf_linear_rmse", pmf_linear_rmse)

pmf_logistic_reg, pmf_logistic_rmse = fit_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV, verbose=False)
print("pmf_logistic_rmse", pmf_logistic_rmse)



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

def_x_post, def_z3_post, def_z2_post, def_z1_post, def_W0_post, def_x_post_np, def_z_post_np = fit_def(x_train.T, K=[100,30,5], prior_a=0.1, prior_b=0.3, optimizer=tf.train.RMSPropOptimizer(1e-3), n_iter=n_iter*100)

# optimizer=tf.train.AdamOptimizer(1e-2), 

print("check DEF fit")

print("trivial mse", ((G-0)**2).mean())

print("DEF mse", ((G-def_x_post_np.T)**2).mean())

# def_pval = def_predictive_check(x_train.T, x_vad.T, holdout_mask.T, def_x_post, def_z1_post, def_W0_post)

# "DEF predictive check", def_pval)

# X = G - def_x_post_np


def_linear_rmse_sing = np.zeros(n_causes)

def_logistic_rmse_sing = np.zeros(n_causes)

for j in range(n_causes):
    # X = np.column_stack([G[:,j][:,np.newaxis], def_x_post_np[:,j][:,np.newaxis]])
    # X = np.column_stack([G[:,j][:,np.newaxis], def_x_post_np.T])
    X = np.column_stack([G[:,j][:,np.newaxis], def_z_post_np])
    _, def_linear_rmse_sing[j] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False, verbose=False)
    _, def_logistic_rmse_sing[j] = fit_outcome_logistic(X, y_bin, true_betas[j], 1, CV=False, verbose=False)


print("def_linear_rmse_sing.mean()", def_linear_rmse_sing.mean())

print("def_logistic_rmse_sing.mean()", def_logistic_rmse_sing.mean())



X = np.column_stack([G, def_z_post_np])


def_linear_reg, def_linear_rmse = fit_outcome_linear(X, y, true_betas, n_causes, CV=CV, verbose=False)

print("def_linear_rmse", def_linear_rmse)


def_logistic_reg, def_logistic_rmse = fit_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV, verbose=False)

print("def_logistic_rmse", def_logistic_rmse)


#############################################################
# estimate causal effects: lfa
#############################################################
print("\nlfa control\n")

# the stochastic vi code subsamples on columns. we pass in the
# transpose of x_train to subsampling on rows.

lfa_x_post, lfa_z_post, lfa_w_post, lfa_x_post_np, lfa_z_post_np = fit_lfa(x_train.T, M=100, K=10, n_iter=n_iter)

print("check LFA fit")

print("trivial mse", ((G-0)**2).mean())

print("LFA mse", ((G-lfa_x_post_np.T)**2).mean())


# lfa_pval = lfa_predictive_check(x_train.T, x_vad.T, holdout_mask.T, lfa_x_post, lfa_w_post, lfa_z_post)

# "LFA predictive check", lfa_pval)

# "PPCA predictive check", lfa_pval)


lfa_linear_rmse_sing = np.zeros(n_causes)

lfa_logistic_rmse_sing = np.zeros(n_causes)

for j in range(n_causes):
    # X = np.column_stack([G[:,j][:,np.newaxis], lfa_x_post_np[:,j][:,np.newaxis]])
    X = np.column_stack([G[:,j][:,np.newaxis], lfa_z_post_np])
    _, lfa_linear_rmse_sing[j] = fit_outcome_linear(X, y, true_betas[j], 1, CV=False)
    _, lfa_logistic_rmse_sing[j] = fit_outcome_logistic(X, y_bin, true_betas[j], 1, CV=False)


print("lfa_linear_rmse_sing.mean()", lfa_linear_rmse_sing.mean())

print("lfa_logistic_rmse_sing.mean()", lfa_logistic_rmse_sing.mean())



X = np.column_stack([G, lfa_z_post_np])

lfa_linear_reg, lfa_linear_rmse = fit_outcome_linear(X, y, true_betas, n_causes, CV=CV, verbose=False)
print("lfa_linear_rmse", lfa_linear_rmse)

lfa_logistic_reg, lfa_logistic_rmse = fit_outcome_logistic(X, y_bin, true_betas, n_causes, CV=CV, verbose=False)
print("lfa_logistic_rmse", lfa_logistic_rmse)


#############################################################
# print all results
#############################################################
print("\n\n\n###############################")
print("########## print all results")

print("randomseed", randseed)

print("n_causes", n_causes, "n_units", n_units)

print("genotype simulation setting", simset, "n_iter", n_iter)

print("causal prop", causalprop)

print("SNP effect", a, "group intercept", b, "group-specific random effect", 100-a-b)

print("########## estimate single SNP effect")

print("trivial", trivial_rmse)

print("\nno control\n")
print("nctrl_linear_rmse_sing.mean()", nctrl_linear_rmse_sing.mean())
print("nctrl_logistic_rmse_sing.mean()", nctrl_logistic_rmse_sing.mean())

print("\noracle control\n")
print("oracle_linear_rmse_sing.mean()", oracle_linear_rmse_sing.mean())
print("oracle_logistic_rmse_sing.mean()", oracle_logistic_rmse_sing.mean())

print("\npca control\n")
print("pca_linear_rmse_sing.mean()", pca_linear_rmse_sing.mean())
print("pca_logistic_rmse_sing.mean()", pca_logistic_rmse_sing.mean())

print("\nppca control\n")
print("ppca_linear_rmse_sing.mean()", ppca_linear_rmse_sing.mean())
print("ppca_logistic_rmse_sing.mean()", ppca_logistic_rmse_sing.mean())

print("\npmf control\n")
print("pmf_linear_rmse_sing.mean()", pmf_linear_rmse_sing.mean())
print("pmf_logistic_rmse_sing.mean()", pmf_logistic_rmse_sing.mean())

print("\ndef control\n")
print("def_linear_rmse_sing.mean()", def_linear_rmse_sing.mean())
print("def_logistic_rmse_sing.mean()", def_logistic_rmse_sing.mean())

print("\nlfa control\n")
print("lfa_linear_rmse_sing.mean()", lfa_linear_rmse_sing.mean())
print("lfa_logistic_rmse_sing.mean()", lfa_logistic_rmse_sing.mean())


print("########## estimate all SNPs effect")

print("trivial", trivial_rmse)

print("\nno control\n")
print("nctrl_linear_rmse", nctrl_linear_rmse)
print("nctrl_logistic_rmse", nctrl_logistic_rmse)

print("\noracle control\n")
print("oracle_linear_rmse", oracle_linear_rmse)
print("oracle_logistic_rmse", oracle_logistic_rmse)

print("\nppca control\n")
print("ppca_linear_rmse", ppca_linear_rmse)
print("ppca_logistic_rmse", ppca_logistic_rmse)

print("\npca control\n")
print("pca_linear_rmse", pca_linear_rmse)
print("pca_logistic_rmse", pca_logistic_rmse)


print("\npmf control\n")
print("pmf_linear_rmse", pmf_linear_rmse)
print("pmf_logistic_rmse", pmf_logistic_rmse)

print("\ndef control\n")
print("def_linear_rmse", def_linear_rmse)
print("def_logistic_rmse", def_logistic_rmse)

print("\nlfa control\n")
print("lfa_linear_rmse", lfa_linear_rmse)
print("lfa_logistic_rmse", lfa_logistic_rmse)


res = pd.DataFrame({"randomseed": randseed, 
    "n_causes": n_causes, 
    "n_units": n_units, 
    "simset": simset, 
    "n_iter": n_iter,
    "cv": CV,
    "causalprop": causalprop, 
    "SNPeffect": args.snpsig, 
    "groupintercept": args.confint, 
    "grouprandomeffect": 100-args.snpsig-args.confint, 
    "trivial_rmse": trivial_rmse, 
    "nctrl_linear_rmse_sing_mean": nctrl_linear_rmse_sing.mean(),
    "nctrl_logistic_rmse_sing_mean": nctrl_logistic_rmse_sing.mean(), 
    "oracle_linear_rmse_sing_mean": oracle_linear_rmse_sing.mean(),
    "oracle_logistic_rmse_sing_mean": oracle_logistic_rmse_sing.mean(),
    "pca_linear_rmse_sing_mean": pca_linear_rmse_sing.mean(),
    "pca_logistic_rmse_sing_mean": pca_logistic_rmse_sing.mean(),
    "ppca_linear_rmse_sing_mean": ppca_linear_rmse_sing.mean(),
    "ppca_logistic_rmse_sing_mean": ppca_logistic_rmse_sing.mean(),
    "pmf_linear_rmse_sing_mean": pmf_linear_rmse_sing.mean(),
    "pmf_logistic_rmse_sing_mean": pmf_logistic_rmse_sing.mean(),
    "def_linear_rmse_sing_mean": def_linear_rmse_sing.mean(), 
    "def_logistic_rmse_sing_mean": def_logistic_rmse_sing.mean(), 
    "lfa_linear_rmse_sing_mean": lfa_linear_rmse_sing.mean(), 
    "lfa_logistic_rmse_sing_mean": lfa_logistic_rmse_sing.mean(), 
    "nctrl_linear_rmse": nctrl_linear_rmse,
    "nctrl_logistic_rmse": nctrl_logistic_rmse,
    "oracle_linear_rmse": oracle_linear_rmse,
    "oracle_logistic_rmse": oracle_logistic_rmse,
    "pca_linear_rmse": pca_linear_rmse,
    "pca_logistic_rmse": pca_logistic_rmse,
    "ppca_linear_rmse": ppca_linear_rmse,
    "ppca_logistic_rmse": ppca_logistic_rmse,
    "pmf_linear_rmse": pmf_linear_rmse,
    "pmf_logistic_rmse": pmf_logistic_rmse,
    "def_linear_rmse": def_linear_rmse,
    "def_logistic_rmse": def_logistic_rmse,
    "lfa_linear_rmse": lfa_linear_rmse,
    "lfa_logistic_rmse": lfa_logistic_rmse}, index=[0])

res = res.T

filename = "dcf"+"_ncauses"+str(n_causes)+"_nunits"+str(n_units)+"_simset"+simset+"_cv"+str(CV)+"_causalprop"+str(args.causalprop)+"_snpsig"+str(args.snpsig)+"_confint"+str(args.confint)+"_nitr"+str(n_iter)+"_randseed"+str(randseed)+".csv"

if os.path.isfile(os.path.join(outdir, filename)):
    filename = "dcf"+"_ncauses"+str(n_causes)+"_nunits"+str(n_units)+"_simset"+simset+"_cv"+str(CV)+"_causalprop"+str(args.causalprop)+"_snpsig"+str(args.snpsig)+"_confint"+str(args.confint)+"_nitr"+str(n_iter)+"_randseed"+str(randseed+1)+".csv"

res.to_csv(os.path.join(outdir, filename))
