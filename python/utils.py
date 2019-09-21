import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class MVNCholPrecisionTriL(tfd.TransformedDistribution):
    """MVN from loc and (Cholesky) precision matrix."""
    def __init__(self, loc, chol_precision_tril, name=None):
        super(MVNCholPrecisionTriL, self).__init__(
            distribution=tfd.Independent(tfd.Normal(tf.zeros_like(loc),
                                                    scale=tf.ones_like(loc)),
                                         reinterpreted_batch_ndims=1),
            bijector=tfb.Chain([
                tfb.Affine(shift=loc),
                tfb.Invert(tfb.Affine(scale_tril=chol_precision_tril,
                                      adjoint=True))]), name=name)

def load_data(path, data_filename):
    path = os.path.expanduser(path)
    filepath = os.path.join(path, data_filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError

    x_data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    x_data = x_data[:, 1:]
    y_data = np.loadtxt(filepath)
    meds = [str(med) for med in np.arange(x_data.shape[1])]
    return x_data, y_data, meds

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def simulate_data(num_subject, num_med, num_conf, num_output):
    '''
    Args
    :num_subject: number of observations
    :num_causes: number of medications
    :num_conf: number of confounders
    :num_output: number of correlated outcomes
    '''
    C = np.random.normal(0., 1., size=[num_subject, num_conf])
    lambd = np.random.normal(0., 10., size=[num_conf, num_med])
    bernoulli_p = sigmoid(np.dot(C, lambd))
    A = np.random.binomial(1, bernoulli_p, size=bernoulli_p.shape)
    beta_base = np.random.normal(0, 1., size=num_med).reshape(-1,1)
    true_beta = np.repeat(beta_base, num_output, axis=1) + \
                   np.random.normal(loc=0, scale=0.1, size=[num_med, num_output])
    # zero_coeff_idx = np.random.choice(np.arange(len(beta)), size=int(num_med*0.7), replace=False)
    true_beta[int(num_med/2):, :int(num_output/2)] = 0.
    true_beta[:int(num_med/2), int(num_output/2):] = 0.

    # effect size of confounders
    gamma1_base = np.random.normal(0, 5., size=num_conf).reshape(-1,1)
    gamma2_base = np.random.normal(0, 5., size=num_conf).reshape(-1,1)
    true_gamma1 = np.repeat(gamma1_base, int(num_output/2), axis=1) + \
                   np.random.normal(loc=0, scale=1., size=[num_conf, int(num_output/2)])
    true_gamma2 = np.repeat(gamma2_base, int(num_output/2), axis=1) + \
                   np.random.normal(loc=0, scale=1., size=[num_conf, int(num_output/2)])
    true_gamma = np.concatenate((true_gamma1, true_gamma2), axis=1)

    true_weights = np.concatenate((true_beta, true_gamma), axis=0)

    design_matrix = np.concatenate((A, C), axis=1)
    true_bias = np.random.randn(num_output)
    true_mean = np.dot(design_matrix, true_weights) + true_bias
    # Generate correlation
    true_corr = np.eye(num_output)
    # And we'll give the 2 coordinates different variances
    true_var = np.eye(num_output)
    # Combine the variances and correlations into a covariance matrix
    # true_cov = np.expand_dims(np.sqrt(true_var), axis=1).dot(
    #     np.expand_dims(np.sqrt(true_var), axis=1).T) * true_corr
    true_cov = np.sqrt(true_var).dot(np.sqrt(true_var).T) * true_corr
    # scale_chol = tf.linalg.cholesky(true_cov)
    # outcomes = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=scale_chol).sample()
    # true_precision = np.linalg.inv(true_cov)
    # true_chol_precision_tril = np.linalg.cholesky(true_precision)
    # outcomes = MVNCholPrecisionTriL(loc=true_mean, chol_precision_tril=true_chol_precision_tril).sample()
    outcomes = np.zeros((num_subject, num_output))
    for i in range(num_subject):
        outcomes[i] = np.random.multivariate_normal(true_mean[i], true_cov)
    # outcomes = tf.cast(outcomes, dtype=tf.float32)

    # true_chol_prec = np.linalg.cholesky(true_precision)
    # print('True weights:\n', W)
    # print('True bias:', b)
    # print('Error covariance:\n', true_cov)
    # print('Error chol precision:\n', true_chol_prec)
    # with tf.Session() as sess:
    #     print('Sample covariance:\n ', np.cov(sess.run(y)[:, 0], sess.run(y)[:, 1]))
    #     print('Sample precision:\n ', np.linalg.inv(np.cov(sess.run(y)[:, 0], sess.run(y)[:, 1])))
    #     print('Sample chol precision:\n ', np.linalg.cholesky(np.linalg.inv(np.cov(sess.run(y)[:, 0], sess.run(y)[:, 1]))))
    #     # Do a scatter plot of the observations to make sure they look like what we
    #     # expect (higher variance on the x-axis, y values strongly correlated with x)
    #     plt.scatter(sess.run(y)[:, 0], sess.run(y)[:, 1], alpha=0.75)
    #     plt.show()

    return true_weights, true_bias, true_cov, design_matrix, outcomes