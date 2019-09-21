
"""Trains a sparse Gamma deep exponential family on patients' medication records
to estimate a substitute confounder. This constitutes the first step of the medical deconfounder.
We apply a sparse Gamma deep exponential family [2] as a topic model on patients' medication records.
Note that [2] applies score function gradients with advanced variance reduction techniques; instead we apply
implicit reparameterization gradients [1]. Preliminary experiments for this
model and task suggest that implicit reparameterization exhibits lower gradient
variance and trains faster.
###$ This code is modified from
https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/deep_exponential_family.py
#### References
[1]: Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization
     Gradients, 2018.
     https://arxiv.org/abs/1805.08498.
[2]: Rajesh Ranganath, Linpeng Tang, Laurent Charlin, David M. Blei. Deep
     Exponential Families. In _Artificial Intelligence and Statistics_, 2015.
     https://arxiv.org/abs/1411.2581
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import time
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import edward2 as ed
from scipy import sparse, stats
tfd = tfp.distributions

def holdout_row_col_mask(x_data, holdout_portion):
    # randomly holdout some entries of x_data
    num_datapoints, data_dim = x_data.shape

    n_holdout = int(holdout_portion * num_datapoints * data_dim)

    holdout_row = np.random.randint(num_datapoints, size=n_holdout)
    holdout_col = np.random.randint(data_dim, size=n_holdout)
    holdout_mask = (sparse.coo_matrix((np.ones(n_holdout),
                                       (holdout_row, holdout_col)),
                                      shape=x_data.shape)).toarray()
    holdout_mask = np.minimum(holdout_mask, np.ones(x_data.shape))
    holdout_mask = np.float32(holdout_mask)
    return holdout_row, holdout_col, holdout_mask

class Deep_exponential_family():
    def __init__(self, learning_rate, max_steps, layer_sizes, shape, holdout_portion):
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.layer_sizes = [int(layer_size) for layer_size in layer_sizes]
        self.shape = shape
        self.holdout_portion = holdout_portion

    def deep_exponential_family(self, data_size, feature_size):
        """A multi-layered topic model over a documents-by-terms matrix."""
        # w2 = ed.Gamma(0.1, 0.3, sample_shape=[self.layer_sizes[2], self.layer_sizes[1]], name="w2")
        w1 = ed.Gamma(0.1, 0.3, sample_shape=[self.layer_sizes[1], self.layer_sizes[0]], name="w1")
        w0 = ed.Gamma(0.1, 0.3, sample_shape=[self.layer_sizes[0], feature_size], name="w0")

        # z2 = ed.Gamma(0.1, 0.1, sample_shape=[data_size, self.layer_sizes[2]], name="z2")
        z1 = ed.Gamma(0.1, 0.1, sample_shape=[data_size, self.layer_sizes[1]], name="z1")
        z0 = ed.Gamma(self.shape, self.shape / tf.matmul(z1, w1), name="z0")
        x = ed.Poisson(tf.matmul(z0, w0), name="x")
        return x

    def trainable_positive_deterministic(self, shape, min_loc=1e-3, name=None):
        """Learnable Deterministic distribution over positive reals."""
        with tf.compat.v1.variable_scope(
                None, default_name="trainable_positive_deterministic"):
            unconstrained_loc = tf.compat.v1.get_variable("unconstrained_loc", shape)
            loc = tf.maximum(tf.nn.softplus(unconstrained_loc), min_loc)
            rv = ed.Deterministic(loc=loc, name=name)
        return rv

    def trainable_gamma(self, shape, min_concentration=1e-3, min_scale=0.003, name=None):
        """Learnable Gamma via concentration and scale parameterization."""
        with tf.compat.v1.variable_scope(None, default_name="trainable_gamma"):
            unconstrained_concentration = tf.compat.v1.get_variable(
                "unconstrained_concentration",
                shape,
                initializer=tf.compat.v1.initializers.random_normal(
                    mean=0.5, stddev=0.1))
            unconstrained_scale = tf.compat.v1.get_variable(
                "unconstrained_scale",
                shape,
                initializer=tf.compat.v1.initializers.random_normal(stddev=0.1))
            concentration = tf.maximum(tf.nn.softplus(unconstrained_concentration),
                                       min_concentration)
            rate = tf.maximum(1. / tf.nn.softplus(unconstrained_scale), 1. / min_scale)
            rv = ed.Gamma(concentration=concentration, rate=rate, name=name)
        return rv

    def deep_exponential_family_variational(self, data_size, feature_size):
        """Posterior approx. for deep exponential family p(w{0,1,2}, z{1,2,3} | x)."""
        # qw2 = trainable_positive_deterministic([self.layer_sizes[2], self.layer_sizes[1]], name="qw2")
        qw1 = self.trainable_positive_deterministic([self.layer_sizes[1], self.layer_sizes[0]], name="qw1")
        qw0 = self.trainable_positive_deterministic([self.layer_sizes[0], feature_size], name="qw0")
        # qz2 = trainable_gamma([data_size, self.layer_sizes[2]], name="qz2")
        qz1 = self.trainable_gamma([data_size, self.layer_sizes[1]], name="qz1")
        qz0 = self.trainable_gamma([data_size, self.layer_sizes[0]], name="qz0")
        return qw1, qw0, qz1, qz0

    def predictive_check(self, sess, x_vad, holdout_mask, holdout_row, x_post, z0_post, qw0):
        # Generate 100 replicated datasets at the heldout entries
        n_rep = 100  # number of replicated datasets we generate
        holdout_gen = np.zeros((n_rep, *(x_post.shape)))
        for i in range(n_rep):
            x_generated = x_post.distribution.sample().eval(session=sess)
            # look only at the heldout entries
            holdout_gen[i] = np.multiply(holdout_mask, x_generated)

        n_eval = 10  # we draw samples from the inferred Z and W
        obs_ll = []
        rep_ll = []

        pbar = tf.compat.v1.keras.utils.Progbar(n_eval)
        for j in range(1, n_eval + 1):
            z0_sample = z0_post.distribution.sample().eval(session=sess)
            w0_sample = qw0.distribution.sample().eval(session=sess)

            holdoutmean_sample = np.multiply(z0_sample.dot(w0_sample), holdout_mask)
            obs_ll.append(\
                np.mean(np.ma.masked_invalid(stats.poisson.logpmf(np.array(x_vad, dtype=int), mu=holdoutmean_sample)),\
                        axis=1))
            rep_ll.append(
                np.mean(np.ma.masked_invalid(stats.poisson.logpmf(holdout_gen, mu=holdoutmean_sample)),
                        axis=2))
            pbar.update(j)

        obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0) # average across each subject

        pvals = np.array([np.mean(rep_ll_per_zi[:, i] < obs_ll_per_zi[i]) for i in range(len(obs_ll_per_zi))])
        holdout_subjects = np.unique(holdout_row)
        predictive_score = np.mean(pvals[holdout_subjects])
        return predictive_score

    def extract_substitute_confounders(self, sess, z0_post):
        z0_post_np = z0_post.distribution.mean().eval(session=sess)
        return z0_post_np

    def extract_reconstructed_causes(self, sess, x_post):
        x_post_np = x_post.distribution.mean().eval(session=sess)
        return x_post_np

    def train(self, sess, x_data, meds, model_dir):
        if len(self.layer_sizes) != 2:
            raise NotImplementedError("Specifying fewer or more than 2 layers is not "
                                      "currently available.")
        if tf.io.gfile.exists(model_dir):
            tf.compat.v1.logging.warning(
                "Warning: deleting old log directory at {}".format(model_dir))
            tf.io.gfile.rmtree(model_dir)
        tf.io.gfile.makedirs(model_dir)

        total_count = np.sum(x_data)
        x_data = tf.cast(x_data, dtype=tf.float32)
        data_size, feature_size = x_data.shape

        # Compute expected log-likelihood. First, sample from the variational
        # distribution; second, compute the log-likelihood given the sample.
        qw1, qw0, qz1, qz0 = self.deep_exponential_family_variational(
            data_size,
            feature_size)

        with ed.tape() as model_tape:
            with ed.interception(ed.make_value_setter(w1=qw1, w0=qw0,
                                                      z1=qz1, z0=qz0)):
                posterior_predictive = self.deep_exponential_family(data_size, feature_size)

        log_likelihood = posterior_predictive.distribution.log_prob(x_data)
        log_likelihood = tf.reduce_sum(input_tensor=log_likelihood)
        tf.compat.v1.summary.scalar("log_likelihood", log_likelihood)

        # Compute analytic KL-divergence between variational and prior distributions.
        kl = 0.
        for rv_name, variational_rv in [("z0", qz0), ("z1", qz1),
                                      ("w0", qw0), ("w1", qw1)]:
            kl += tf.reduce_sum(
                input_tensor=variational_rv.distribution.kl_divergence(
                    model_tape[rv_name].distribution))


        elbo = log_likelihood - kl
        tf.compat.v1.summary.scalar("elbo_factor", elbo)
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
                perplexity = np.exp(negative_log_likelihood / total_count)
                print("Negative log-likelihood <= {:0.3f}".format(
                    negative_log_likelihood))
                print("Perplexity <= {:0.3f}".format(perplexity))

                # # Print top 10 meds for first 10 topics.
                # qw0_values = sess.run(qw0)
                # for k in range(min(10, self.layer_sizes[-1])):
                #     top_words_idx = qw0_values[k, :].argsort()[-10:][::-1]
                #     top_words = " ".join([meds[i] for i in top_words_idx])
                #     print("Topic {}: {}".format(k, top_words))

        z0_post = ed.Gamma(self.shape, self.shape / tf.matmul(qz1, qw1))
        x_post = ed.Poisson(tf.matmul(z0_post, qw0))

        plt.figure(figsize=(5, 3))
        plt.plot(np.arange(self.max_steps), elbo_list)
        plt.savefig(model_dir + 'elbo_def.pdf', format='pdf')
        plt.close()
        return z0_post, x_post, qw0

