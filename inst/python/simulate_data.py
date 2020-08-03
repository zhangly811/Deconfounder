from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
    
def simulate_multicause_data(N, K, D, Nsim):
    # Simulate causes and confounders
    C = np.random.normal(0, 1, size = [N, K])
    lambd = np.random.normal(0, 0.5, size = [K, D])
    bernoulli_p = sigmoid(np.dot(C, lambd))                
    A = np.random.binomial(1, bernoulli_p, size = bernoulli_p.shape)
    
    # Simulate sets of coefficients and outcomes
    betas = np.zeros((Nsim, D))
    gammas = np.zeros((Nsim, K))
    Ys = np.zeros((Nsim, N))
    for sim in range(Nsim):
        beta = np.random.normal(0, 0.25, size = D)
        zero_coeff_idx = np.random.choice(np.arange(len(beta)), size = int(D*0.8), replace = False)
        beta[zero_coeff_idx] = 0
    
        gamma = np.random.normal(0, 0.25, size = K)
        
        noise = np.random.normal(0, 1, size = N).reshape(-1,1)
        Y = np.dot(A, beta.reshape(D,1)) + np.dot(C, gamma.reshape(10,1)) + noise
        
        betas[sim,:] = beta
        gammas[sim,:] = gamma
        Ys[sim, np.newaxis] = Y.T

    return A, C, Ys, betas

