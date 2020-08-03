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
import os
import itertools
import math
from datetime import *
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score
from scipy.special import expit
from statistics import mode, mean
from edward.models import Normal, Gamma, Dirichlet, InverseGamma, \
    Poisson, PointMass, Empirical, ParamMixture, \
    MultivariateNormalDiag, Categorical, Laplace, \
    MultivariateNormalTriL, Bernoulli, TransformedDistribution, \
    Binomial
from edward.util import Progbar
from scipy import sparse, stats
from scipy.special import expit, logit
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split


def next_batch_row(x_train, M):
    # subsample M rows
    D, N = x_train.shape
    idx_batch = np.random.choice(D, M)
    return x_train[idx_batch, :], idx_batch


def sim_multiple_traits(lambdas, G, a, b, n_outcomes, K=10, causalprop=0.05, bin_scale=1):
    tau = 1. / npr.gamma(3, 1, size=[3, n_outcomes])
    sigmasqs = tau[lambdas]
    epsilons = npr.normal(0, sigmasqs)
    n_causes = G.shape[1]
    betas = npr.normal(0, 1.0, size=[n_causes, K])
    gamma = npr.normal(0, 1.0, size=[n_outcomes, K])
    causal_snps = int(causalprop * n_causes)
    betas[causal_snps:] = 0.0
    coefs = betas.dot(gamma.T) + npr.normal(0, 0.1, size=[n_causes, n_outcomes])

    lambdabetas = npr.normal(0, 1.0, size=[1, n_outcomes])
    confs = lambdas[:, np.newaxis].dot(lambdabetas)

    c = 1 - a - b

    raw_y = G.dot(coefs)
    raw_y_std = raw_y.std(axis=0)

    y = raw_y + \
        np.sqrt(b) * raw_y_std / np.sqrt(a) * confs / confs.std(axis=0) + \
        np.sqrt(c) * raw_y_std / np.sqrt(a) * epsilons / epsilons.std(axis=0)

    # no confounding for testing
    # y = raw_y + \
    # np.sqrt(c)*raw_y_std/np.sqrt(a)*epsilons/epsilons.std(axis=0)

    # rescale the confounding and error components to keep the MSE
    # invariant to the number of causes

    y_binpred = y
    # y_binpred = raw_y + \
    #     np.sqrt(b)*raw_y_std/np.sqrt(a)*lambdas/lambdas.std()
    y_bin = npr.binomial(1, expit((y_binpred - y_binpred.mean()) * bin_scale))

    # we do y_binpred - y_binpred.mean() to get balanced data

    true_lambdas = \
        np.sqrt(b) * raw_y_std / np.sqrt(a) * (confs) / confs.std(axis=0)

    true_betas = coefs

    print("confounding strength np.corrcoef(y, true_lambdas)", \
          [np.corrcoef(y[:, j], true_lambdas[:, j])[0, 1] for j in range(y.shape[1])])

    return y, y_bin, true_betas, true_lambdas, betas, gamma, lambdacoefs


def fit_multiple_outcome_linear(X, y, n_causes, true_betas=None, CV=False, lowdim=False, K=10, n_iter=1000, verbose=False):
    if CV == True:
        wval, bval, vadacc, trainacc = multiple_regression_CV(X, y, n_causes, outtype="linear", lowdim=lowdim, K=K,
                                                              n_iter=n_iter, verbose=verbose)
    else:
        wval, bval, vadacc, trainacc = multiple_regression_noCV(X, y, n_causes, outtype="linear", lowdim=lowdim, K=K,
                                                                n_iter=n_iter, verbose=verbose)
    if true_betas!=None:
        linear_rmse = np.sqrt(((true_betas - wval[:n_causes]) ** 2).mean())
        trivial_rmse = np.sqrt(((true_betas - 0) ** 2).mean())
        if verbose:
            print("linear outcome rmse", linear_rmse, "\nlinear - trivial", linear_rmse - trivial_rmse)
        linear_reg = (wval, bval, vadacc, trainacc)
        return linear_reg, linear_rmse
    else:
        linear_reg = (wval, bval, vadacc, trainacc)
        return linear_reg


def fit_multiple_outcome_logistic(X, y_bin, true_betas, n_causes, CV=False, lowdim=False, K=10, n_iter=1000,
                                  verbose=False):
    if CV == True:
        wval, bval, vadacc, trainacc = multiple_regression_CV(X, y_bin, n_causes, outtype="logistic", lowdim=lowdim,
                                                              K=K, n_iter=n_iter, verbose=verbose)
    else:
        wval, bval, vadacc, trainacc = multiple_regression_noCV(X, y_bin, n_causes, outtype="logistic", lowdim=lowdim,
                                                                K=K, n_iter=n_iter, verbose=verbose)
    logistic_rmse = np.sqrt(((true_betas - wval[:n_causes]) ** 2).mean())
    trivial_rmse = np.sqrt(((true_betas - 0) ** 2).mean())
    if verbose:
        print("logistic outcome rmse: ", logistic_rmse, \
              "\nlogistic - trivial: ", logistic_rmse - trivial_rmse)
    logistic_reg = (wval, bval, vadacc, trainacc)
    return logistic_reg, logistic_rmse


def evalacc(Y_vad, yvadpredmean, outtype):
    if outtype == "logistic":
        yvadpredmean = expit(yvadpredmean)
        yvadpred = npr.binomial(n=1, p=yvadpredmean)
    else:
        yvadpred = yvadpredmean

    yvadpred, yvadpredmean = np.squeeze(np.array(yvadpred)), np.squeeze(np.array(yvadpredmean))

    if outtype != "logistic":
        print("trivial accuracy", r2_score(Y_vad, np.array(
            [np.ones_like(Y_vad[:, j]) * mean(Y_vad[:, j]) for j in range(Y_vad.shape[1])]).T))
        print("realized accuracy (hand)", r2_score(Y_vad, yvadpredmean))
        vadacc = r2_score(Y_vad, yvadpredmean)
    else:
        print("trivial accuracy", accuracy_score(Y_vad, np.array(
            [np.ones_like(Y_vad[:, j]) * stats.mode(Y_vad[:, j])[0][0] for j in range(Y_vad.shape[1])]).T))
        print("realized accuracy (hand)", accuracy_score(Y_vad, yvadpred))
        vadacc = accuracy_score(Y_vad, yvadpred)

    return vadacc


def multiple_regression_CV(X, y, n_causes, outtype="linear", lowdim=False, K=10, var_priorsd=0.1, y_priorsd=1.,
                           n_iter=1000, vad_n_iter=1000, optimizer=tf.train.RMSPropOptimizer(1e-3), verbose=False):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=123)

    var_priorsd_sweep = np.logspace(-5, 0, 5)
    bestvadacc = 1e16
    bestvar_priorsd = None
    for var_priorsd in var_priorsd_sweep:
        if lowdim:
            wval, bval, vadacc, trainacc = multiple_regression_lowdim_noCV(X, y, n_causes, outtype=outtype, K=K,
                                                                           n_iter=n_iter, var_priorsd=var_priorsd,
                                                                           verbose=verbose)
        else:
            wval, bval, vadacc, trainacc = multiple_regression_noCV(X, y, \
                                                                    outtype=outtype, n_iter=n_iter,
                                                                    var_priorsd=var_priorsd, verbose=verbose)
        if vadacc < bestvadacc:
            bestvadacc = vadacc
            bestvar_priorsd = var_priorsd

    print("best var_priorsd", bestvar_priorsd)

    if lowdim:
        wval, bval, vadacc, trainacc = multiple_regression_lowdim_noCV(X, y, n_causes, outtype=outtype, K=K,
                                                                       var_priorsd=bestvar_priorsd, verbose=False)
    else:
        wval, bval, vadacc, trainacc = multiple_regression_noCV(X, y, \
                                                                outtype=outtype, var_priorsd=bestvar_priorsd,
                                                                verbose=False)

    if verbose:
        print("training score", trainacc)
        print("predictive score", vadacc)

    return wval, bval, vadacc, trainacc


def multiple_regression_noCV(X, y, n_causes, outtype="linear", lowdim=False, K=10, var_priorsd=1., y_priorsd=1.,
                             n_iter=1000, vad_n_iter=1000, verbose=False):
    if lowdim:
        wval, bval, vadacc, trainacc = multiple_regression_lowdim_noCV(X, y, n_causes, outtype=outtype, K=K,
                                                                       n_iter=n_iter, var_priorsd=var_priorsd,
                                                                       verbose=verbose)
    else:
        wval, bval, vadacc, trainacc = multiple_regression_vanilla_noCV(X, y, \
                                                                        outtype=outtype, n_iter=n_iter,
                                                                        var_priorsd=var_priorsd, verbose=verbose)
    return wval, bval, vadacc, trainacc


def multiple_regression_vanilla_noCV(X, y, outtype="linear", var_priorsd=1., y_priorsd=1., n_iter=1000, vad_n_iter=1000,
                                     optimizer=tf.train.RMSPropOptimizer(1e-4), verbose=False):
    X_train, X_vad, Y_train, Y_vad = \
        train_test_split(X, y, test_size=0.2, random_state=123)

    num_vad = len(Y_vad)

    D = X_train.shape[0]
    N = X_train.shape[1]
    M = 100  # batch size
    ydim = Y_train.shape[1]
    assert D == Y_train.shape[0]

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x_ph = tf.placeholder(tf.float32, [M, N])

    if outtype == "logistic":
        y_ph = tf.placeholder(tf.int32, [M, ydim])
    else:
        y_ph = tf.placeholder(tf.float32, [M, ydim])

    idx_ph = tf.placeholder(tf.int32, M)

    b = Normal(loc=tf.zeros([ydim]), scale=10. * tf.ones([ydim]))
    w = Normal(loc=tf.zeros([N, ydim]), scale=var_priorsd * tf.ones([N, ydim]))

    ymean = tf.matmul(x_ph, w) + b

    if outtype == "logistic":
        y = Bernoulli(logits=ymean)
    else:
        y = Normal(loc=ymean, scale=y_priorsd * tf.ones(ymean.shape))

    qw_variables = [tf.Variable(-1. * tf.random_normal([N, ydim]))]
    qw = PointMass(params=qw_variables[0])

    qb_variables = [tf.Variable(-1. * tf.random_normal([ydim]))]
    qb = PointMass(params=qb_variables[0])

    all_variables = list(itertools.chain(qw_variables, qb_variables))

    # optimizer = tf.train.RMSPropOptimizer(1e-3)
    # optimizer = tf.train.AdamOptimizer(lr)
    # optimizer=tf.train.RMSPropOptimizer(lr)
    # optimizer='rmsprop'
    # optimizer = 'adam'

    scale_factor = float(D) / M

    inference = ed.MAP({w: qw, b: qb}, data={y: y_ph})

    inference.initialize( \
        scale={y: scale_factor}, \
        var_list=all_variables, optimizer=optimizer)

    # validation variables
    xvad_ph = tf.placeholder(tf.float32, [M, N])

    tf.global_variables_initializer().run()

    loss = []

    for i in range(n_iter):
        x_batch, idx_batch = next_batch_row(X_train, M)
        y_batch = Y_train[idx_batch].astype('float32')

        # x_batch = x_batch.todense()
        info_dict = inference.update( \
            feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                       y_ph: y_batch})
        inference.print_progress(info_dict)
        loss.append(info_dict["loss"])
        if i % 1000 == 0:
            print("Itr", i, "loss", info_dict["loss"])
        if math.isnan(info_dict["loss"]):
            break

        if (i + 0) % 1000 == 0:
            wval = qw_variables[0].eval()
            bval = qb_variables[0].eval()

            print("#####################")
            print("Itr", i, "validation accuracy")

            y_vadpredmean = bval + np.dot(X_vad, wval)
            evalacc(Y_vad, y_vadpredmean, outtype)

            print("#####################")
            print("Itr", i, "training accuracy")

            y_trainpredmean = bval + np.dot(X_train, wval)
            evalacc(Y_train, y_trainpredmean, outtype)

    wval = qw_variables[0].eval()
    bval = qb_variables[0].eval()

    loss = np.array(loss)

    print("#####################")
    print("final ", "validation accuracy")

    y_vadpredmean = bval + np.dot(X_vad, wval)
    vadacc = evalacc(Y_vad, y_vadpredmean, outtype)

    print("#####################")
    print("final ", "training accuracy")

    y_trainpredmean = bval + np.dot(X_train, wval)
    trainacc = evalacc(Y_train, y_trainpredmean, outtype)

    return wval, bval, vadacc, trainacc


def multiple_regression_lowdim_noCV_naive(X, y, outtype="linear", K=10, var_priorsd=1., y_priorsd=0.1, n_iter=1000,
                                          vad_n_iter=1000, optimizer=tf.train.RMSPropOptimizer(1e-4), verbose=False):
    # this is doing multiple linear regression with multiple outcomes.
    # the coefficient matrix (n_covariates * n_outcomes) is factorized
    # into K-dim latents, i.e.
    # y = (theta1 * beta) causes + (theta2 * beta) confounders + error

    # note that n_covariates include both causes
    # and (substitute) confounders. that is both the coefficient of
    # the confounders and the causes are sharing the same K-dim
    # latents. We probably should treat the causes and the outcomes
    # differently, where the coefficient of confounders does not need
    # to be factorized, i.e.
    # y = (theta1 * beta) causes + theta2 * confounders + error

    X_train, X_vad, Y_train, Y_vad = \
        train_test_split(X, y, test_size=0.2, random_state=123)

    num_vad = len(Y_vad)

    D = X_train.shape[0]
    N = X_train.shape[1]
    M = 100  # batch size
    ydim = Y_train.shape[1]
    assert D == Y_train.shape[0]

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x_ph = tf.placeholder(tf.float32, [M, N])

    if outtype == "logistic":
        y_ph = tf.placeholder(tf.int32, [M, ydim])
    else:
        y_ph = tf.placeholder(tf.float32, [M, ydim])

    idx_ph = tf.placeholder(tf.int32, M)

    b = Normal(loc=tf.zeros([ydim]), scale=10. * tf.ones([ydim]))
    wtheta = Normal(loc=tf.zeros([N, K]), scale=var_priorsd * tf.ones([N, K]))
    wbeta = Normal(loc=tf.zeros([K, ydim]), scale=tf.ones([K, ydim]))
    w = Normal(loc=tf.matmul(wtheta, wbeta), scale=0.1 * tf.ones([N, ydim]))

    ymean = tf.matmul(x_ph, w) + b

    if outtype == "logistic":
        y = Bernoulli(logits=ymean)
    else:
        y = Normal(loc=ymean, scale=y_priorsd * tf.ones(ymean.shape))

    qwtheta_variables = [tf.Variable(-0.1 * tf.random_normal([N, K]))]
    qwtheta = PointMass(params=qwtheta_variables[0])

    qwbeta_variables = [tf.Variable(-0.1 * tf.random_normal([K, ydim]))]
    qwbeta = PointMass(params=qwbeta_variables[0])

    qw_variables = [tf.Variable(-1. * tf.random_normal([N, ydim]))]
    qw = PointMass(params=qw_variables[0])

    qb_variables = [tf.Variable(-1. * tf.random_normal([ydim]))]
    qb = PointMass(params=qb_variables[0])

    all_variables = list(itertools.chain(qwtheta_variables, qwbeta_variables, qw_variables, qb_variables))

    # optimizer = tf.train.RMSPropOptimizer(1e-3)
    # optimizer=tf.train.RMSPropOptimizer(lr)
    # optimizer='rmsprop'
    # optimizer = 'adam'

    scale_factor = float(D) / M

    inference = ed.MAP({wtheta: qwtheta, wbeta: qwbeta, w: qw, b: qb}, data={y: y_ph})

    inference.initialize( \
        scale={y: scale_factor}, \
        var_list=all_variables, optimizer=optimizer)

    # validation variables
    xvad_ph = tf.placeholder(tf.float32, [M, N])

    tf.global_variables_initializer().run()

    loss = []

    for i in range(n_iter):
        x_batch, idx_batch = next_batch_row(X_train, M)
        y_batch = Y_train[idx_batch].astype('float32')

        # x_batch = x_batch.todense()
        info_dict = inference.update( \
            feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                       y_ph: y_batch})
        inference.print_progress(info_dict)
        loss.append(info_dict["loss"])
        if i % 1000 == 0:
            print("Itr", i, "loss", info_dict["loss"])
        if math.isnan(info_dict["loss"]):
            break

        if (i + 0) % 1000 == 0:
            wthetaval = qwtheta_variables[0].eval()
            wbetaval = qwbeta_variables[0].eval()
            wval = qw_variables[0].eval()
            bval = qb_variables[0].eval()

            print("#####################")
            print("Itr", i, "validation accuracy")

            y_vadpredmean = bval + np.dot(X_vad, wval)
            evalacc(Y_vad, y_vadpredmean, outtype)

            print("#####################")
            print("Itr", i, "training accuracy")

            y_trainpredmean = bval + np.dot(X_train, wval)
            evalacc(Y_train, y_trainpredmean, outtype)

    wthetaval = qwtheta_variables[0].eval()
    wbetaval = qwbeta_variables[0].eval()
    wval = qw_variables[0].eval()
    bval = qb_variables[0].eval()

    loss = np.array(loss)

    print("#####################")
    print("final ", "validation accuracy")

    y_vadpredmean = bval + np.dot(X_vad, wval)
    vadacc = evalacc(Y_vad, y_vadpredmean, outtype)

    print("#####################")
    print("final ", "training accuracy")

    y_trainpredmean = bval + np.dot(X_train, wval)
    trainacc = evalacc(Y_train, y_trainpredmean, outtype)

    return wval, bval, vadacc, trainacc


def multiple_regression_lowdim_noCV(X, y, n_causes, outtype="linear", K=10, var_priorsd=1., y_priorsd=0.1, n_iter=1000,
                                    vad_n_iter=1000, optimizer=tf.train.RMSPropOptimizer(1e-4), verbose=False):
    # y = (theta1 * beta) causes + theta2 * confounders + error

    X_train, X_vad, Y_train, Y_vad = \
        train_test_split(X, y, test_size=0.2, random_state=123)

    num_vad = len(Y_vad)

    D = X_train.shape[0]
    N = n_causes
    confdim = X_train.shape[1] - n_causes
    M = 100  # batch size
    ydim = Y_train.shape[1]
    assert D == Y_train.shape[0]

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x_ph = tf.placeholder(tf.float32, [M, N])
    A_ph = tf.placeholder(tf.float32, [M, confdim])

    if outtype == "logistic":
        y_ph = tf.placeholder(tf.int32, [M, ydim])
    else:
        y_ph = tf.placeholder(tf.float32, [M, ydim])

    idx_ph = tf.placeholder(tf.int32, M)

    b = Normal(loc=tf.zeros([ydim]), scale=10. * tf.ones([ydim]))
    wtheta = Normal(loc=tf.zeros([N, K]), scale=var_priorsd * tf.ones([N, K]))
    wbeta = Normal(loc=tf.zeros([K, ydim]), scale=tf.ones([K, ydim]))
    w = Normal(loc=tf.matmul(wtheta, wbeta), scale=0.1 * tf.ones([N, ydim]))
    v = Normal(loc=tf.zeros([confdim, ydim]), scale=tf.ones([confdim, ydim]))

    ymean = tf.matmul(x_ph, w) + tf.matmul(A_ph, v) + b

    if outtype == "logistic":
        y = Bernoulli(logits=ymean)
    else:
        y = Normal(loc=ymean, scale=y_priorsd * tf.ones(ymean.shape))

    qwtheta_variables = [tf.Variable(-0.1 * tf.random_normal([N, K]))]
    qwtheta = PointMass(params=qwtheta_variables[0])

    qwbeta_variables = [tf.Variable(-0.1 * tf.random_normal([K, ydim]))]
    qwbeta = PointMass(params=qwbeta_variables[0])

    qw_variables = [tf.Variable(-1. * tf.random_normal([N, ydim]))]
    qw = PointMass(params=qw_variables[0])

    qv_variables = [tf.Variable(-1. * tf.random_normal([confdim, ydim]))]
    qv = PointMass(params=qv_variables[0])

    qb_variables = [tf.Variable(-1. * tf.random_normal([ydim]))]
    qb = PointMass(params=qb_variables[0])

    all_variables = list(itertools.chain(qwtheta_variables, qwbeta_variables, qw_variables, qb_variables, qv_variables))

    # optimizer = tf.train.RMSPropOptimizer(1e-3)
    # optimizer=tf.train.RMSPropOptimizer(lr)
    # optimizer='rmsprop'
    # optimizer = 'adam'

    scale_factor = float(D) / M

    inference = ed.MAP({wtheta: qwtheta, wbeta: qwbeta, w: qw, b: qb, v: qv}, data={y: y_ph})

    inference.initialize( \
        scale={y: scale_factor}, \
        var_list=all_variables, optimizer=optimizer)

    # validation variables
    xvad_ph = tf.placeholder(tf.float32, [M, N])

    tf.global_variables_initializer().run()

    loss = []

    for i in range(n_iter):
        x_batch, idx_batch = next_batch_row(X_train[:, :n_causes], M)
        Ahat_batch = X_train[idx_batch, n_causes:]
        y_batch = Y_train[idx_batch].astype('float32')

        # x_batch = x_batch.todense()
        info_dict = inference.update( \
            feed_dict={x_ph: x_batch, idx_ph: idx_batch, \
                       y_ph: y_batch, A_ph: Ahat_batch})
        inference.print_progress(info_dict)
        loss.append(info_dict["loss"])
        if i % 1000 == 0:
            print("Itr", i, "loss", info_dict["loss"])
        if math.isnan(info_dict["loss"]):
            break

        if (i + 0) % 1000 == 0:
            wthetaval = qwtheta_variables[0].eval()
            wbetaval = qwbeta_variables[0].eval()
            wval = qw_variables[0].eval()
            vval = qv_variables[0].eval()
            bval = qb_variables[0].eval()

            print("#####################")
            print("Itr", i, "validation accuracy")

            y_vadpredmean = bval + np.dot(X_vad[:, :n_causes], wval) + np.dot(X_vad[:, n_causes:], vval)
            evalacc(Y_vad, y_vadpredmean, outtype)

            print("#####################")
            print("Itr", i, "training accuracy")

            y_trainpredmean = bval + np.dot(X_train[:, :n_causes], wval) + np.dot(X_train[:, n_causes:], vval)
            evalacc(Y_train, y_trainpredmean, outtype)

    wthetaval = qwtheta_variables[0].eval()
    wbetaval = qwbeta_variables[0].eval()
    wval = qw_variables[0].eval()
    vval = qv_variables[0].eval()
    bval = qb_variables[0].eval()

    loss = np.array(loss)

    print("#####################")
    print("final ", "validation accuracy")

    y_vadpredmean = bval + np.dot(X_vad[:, :n_causes], wval) + np.dot(X_vad[:, n_causes:], vval)
    vadacc = evalacc(Y_vad, y_vadpredmean, outtype)

    print("#####################")
    print("final ", "training accuracy")

    y_trainpredmean = bval + np.dot(X_train[:, :n_causes], wval) + np.dot(X_train[:, n_causes:], vval)
    trainacc = evalacc(Y_train, y_trainpredmean, outtype)

    return wval, bval, vadacc, trainacc