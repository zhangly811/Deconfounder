from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
tfd = tfp.distributions
tfb = tfp.bijectors

# MODEL
class Multivariate_bayesian_regression():
    def __init__(self, learning_rate, max_steps):
        self.learning_rate = learning_rate
        self.max_steps = max_steps

    def bayesian_multivariate_regression(self, X, num_subject, num_feat, num_output):
        W = ed.Normal(loc=tf.zeros([num_feat, num_output]),
                      scale=tf.ones([num_feat, num_output]),
                      name="W")
        b = ed.Normal(loc=tf.zeros([num_output]),
                      scale=tf.ones([num_output]),
                      name="b")
        pred = ed.Normal(loc=tf.matmul(X, W) + b, scale=tf.ones([num_subject, num_output]), name='pred')
        return pred

    def trainable_normal(self, shape, min_scale=1e-5, name=None):
        """Learnable Normal via loc and scale parameterization."""
        with tf.compat.v1.variable_scope(None, default_name="trainable_normal"):
            loc = tf.compat.v1.get_variable(
                "loc",
                shape,
                initializer=tf.compat.v1.initializers.random_normal(
                    mean=0., stddev=1.))
            unconstrained_scale = tf.compat.v1.get_variable(
                "unconstrained_scale",
                shape,
                initializer=tf.compat.v1.initializers.random_normal(stddev=0.1))
            scale = tf.maximum(tf.nn.softplus(unconstrained_scale), min_scale)
            rv = ed.Normal(loc=loc, scale=scale, name=name)
            return rv

    def bayesian_multivariate_regression_variational(self, num_feat, num_output):
        """Posterior approx. for bayesian regression p(W, b| data)."""
        qW = self.trainable_normal([num_feat, num_output], name="qW")
        qb = self.trainable_normal([num_output], name="qb")
        return qW, qb

    def train(self, sess, x_data, y_data, model_dir):
        num_subject, num_feat = x_data.shape
        num_output = y_data.shape[1]

        qW, qb = self.bayesian_multivariate_regression_variational(num_feat, num_output)

        with ed.tape() as model_tape:
            with ed.interception(ed.make_value_setter(W=qW, b=qb)):
                posterior_predictive = self.bayesian_multivariate_regression(x_data, num_subject, num_feat, num_output)

        log_likelihood = posterior_predictive.distribution.log_prob(y_data)
        log_likelihood = tf.reduce_sum(input_tensor=log_likelihood)
        tf.compat.v1.summary.scalar("log_likelihood", log_likelihood)

        # Compute analytic KL-divergence between variational and prior distributions.
        kl = 0.
        for rv_name, variational_rv in [("W", qW), ("b", qb)]:
            kl += tf.reduce_sum(
                input_tensor=variational_rv.distribution.kl_divergence(
                    model_tape[rv_name].distribution))

        # tf.compat.v1.summary.scalar("kl", kl)
        elbo = log_likelihood - kl
        tf.compat.v1.summary.scalar("elbo_outcome", elbo)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(-elbo)

        summary = tf.compat.v1.summary.merge_all()
        summary_writer = tf.compat.v1.summary.FileWriter(model_dir, sess.graph)

        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        start_time = time.time()
        elbo_list = []
        for step in range(self.max_steps):
            _, elbo_value = sess.run([train_op, elbo])
            elbo_list.append(elbo_value)
            if step % 500 == 0:
                duration = time.time() - start_time
                print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(
                    step, elbo_value, duration))
                summary_str = sess.run(summary)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                # Compute perplexity of the full data set. The model's negative
                # log-likelihood of data is upper bounded by the variational objective.
                negative_log_likelihood = -elbo_value
                # perplexity = np.exp(negative_log_likelihood / total_count)
                print("Negative log-likelihood <= {:0.3f}".format(
                    negative_log_likelihood))
                # print("Perplexity <= {:0.3f}".format(perplexity))

        plt.figure(figsize=(5, 3))
        plt.plot(np.arange(self.max_steps), elbo_list)
        plt.savefig(model_dir + 'elbo.pdf', format='pdf')
        return qW.distribution.parameters['loc'].eval(session=sess), qW.distribution.parameters['scale'].eval(session=sess)

    def plot_mean_and_CI(self, w_post_loc, w_post_scale, w_true, model_dir, file_name):
        w_post_loc = w_post_loc.flatten('F')
        w_post_scale = w_post_scale.flatten('F')
        w_true = w_true.flatten('F')

        lower_CI = stats.norm.ppf(q=0.975, loc=w_post_loc, scale=w_post_scale)
        upper_CI = stats.norm.ppf(q=0.025, loc=w_post_loc, scale=w_post_scale)

        credible_intervals = np.array((w_post_loc - lower_CI, upper_CI-w_post_loc))

        plt.figure(figsize=(10, 3))

        plt.errorbar(x=np.arange(w_post_loc.shape[0]),
                     y=w_post_loc,
                     yerr=credible_intervals,
                     color='dodgerblue',
                     fmt='o')
        plt.scatter(x=np.arange(w_true.shape[0]),
                    y=w_true,
                    color='violet')
        plt.savefig("{}{}.pdf".format(model_dir, file_name),
                    format="pdf")
