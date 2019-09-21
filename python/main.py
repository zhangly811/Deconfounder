from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import time

# Dependency imports
from absl import flags
from absl import app
import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow_probability import edward2 as ed

# medical deconfounder specific dependencies
# from deep_exponential_family import *
# from ed_bayesian_regression import *
# from utils import *

def main(learning_rate, max_steps, layer_sizes, shape, holdout_portion, 
data_dir, data_filename, factor_model_dir, outcome_model_dir, fake_data):
    # del args # unused args
    # load data
    if fake_data:
        num_subject = 300
        num_med = 10
        num_conf = 5
        num_output = 6
        W, b, true_cov, x_data_complete, y_data = simulate_data(num_subject, num_med, num_conf, num_output)
        x_data = x_data_complete[:, :num_med]
        meds = [str(med) for med in np.arange(num_med)]
    else:
        x_data, y_data, meds = load_data(data_dir, data_filename)
        num_subject, num_med = x_data.shape
        num_output = y_data.shape[1]

    # hold out entries from the X matrix
    holdout_row, holdout_col, holdout_mask = holdout_row_col_mask(x_data, holdout_portion)
    x_train = np.multiply(1 - holdout_mask, x_data)
    x_vad = np.multiply(holdout_mask, x_data)

    with tf.compat.v1.Session() as sess1:
        # FACTOR MODEL
        print('Step 1: Train a deep exponential family')
        factor_mod = Deep_exponential_family(learning_rate, max_steps, layer_sizes, shape, holdout_portion)
        z0_post, x_post, qw0 = factor_mod.train(sess1, x_train, meds, factor_model_dir)
        predictive_score = factor_mod.predictive_check(sess1, x_vad, holdout_mask, holdout_row, x_post, z0_post, qw0)
        print("Predictive check score", predictive_score)
        # extract output
        # z0_post_np = factor_mod.extract_substitute_confounders(sess1, z0_post)
        x_post_np = factor_mod.extract_reconstructed_causes(sess1, x_post)
        # save output
        # np.savetxt(z0_post_np, factor_model_dir + 'z0_post_np_def')
        # np.savetxt(x_post_np, factor_model_dir + 'x_post_np_def')
        sess1.close()

        
    # OUTCOME MODEL: bayesian multivariate regression
    out_mod = Multivariate_bayesian_regression(learning_rate, max_steps)

    print ('Train an unadjusted outcome model')
    with tf.compat.v1.Session() as sess2:
        x_data = x_data.astype(np.float32)
        w_post_loc_unadj, w_post_scale_unadj = out_mod.train(sess2, x_data, y_data, outcome_model_dir)

        out_mod.plot_mean_and_CI(w_post_loc_unadj[:num_med], w_post_scale_unadj[:num_med],
                                 W[:num_med], outcome_model_dir, 'w_post_unadj')
        sess2.close()
    # OUTCOME MODEL: adjust for reconstructed causes
    print('Train an outcome adjusting for the reconstructed causes')
    x_data_def = np.concatenate((x_data, x_post_np), axis=1)
    x_data_def = x_data_def.astype(np.float32)
    with tf.compat.v1.Session() as sess3:
        w_post_loc_def, w_post_scale_def = out_mod.train(sess3, x_data_def, y_data, outcome_model_dir)

        out_mod.plot_mean_and_CI(w_post_loc_def[:num_med], w_post_scale_def[:num_med],
                                 W[:num_med], outcome_model_dir, 'w_post_def')

        sess3.close()
    print('RMSE: ', np.sqrt(np.mean((w_post_loc_unadj[:num_med, :] - W[:num_med, :])**2)))
    print('RMSE: ', np.sqrt(np.mean((w_post_loc_def[:num_med, :] - W[:num_med, :])**2)))


# if __name__ == "__main__":
    # DEFINE_float("learning_rate",
    #                    default=1e-4,
    #                    help="Initial learning rate.")
    # DEFINE_integer("max_steps",
    #                      default=5000,
    #                      help="Number of training steps to run.")
    # DEFINE_list("layer_sizes",
    #                   default=["50", "10"],
    #                   help="Comma-separated list denoting number of latent "
    #                        "variables (stochastic units) per layer.")
    # DEFINE_float("shape",
    #                    default=1.0,
    #                    help="Shape hyperparameter for Gamma priors on latents.")
    # DEFINE_float("holdout_portion",
    #                    default=0.5,
    #                    help="Number of entries to holdout for evaluation")
    # DEFINE_string("data_dir",
    #                     default='/Users/linyingzhang/LargeFiles/Blei/multivariate_medical_deconfounder/dat/',
    #                     help="Directory where data is stored (if using real data).")
    # DEFINE_string("data_filename",
    #                     default='sparse_matrix_X.csv',
    #                     help="Name of the medication file (if using real data).")
    # DEFINE_string("factor_model_dir",
    #                     default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
    #                                          "factor_model/"),
    #                     help="Directory to put the factor model's fit.")
    # DEFINE_string("outcome_model_dir",
    #                     default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
    #                                          "outcome_model/"),
    #                     help="Directory to put the outcome model's fit.")
    # DEFINE_bool("fake_data",
    #                   default=True,
    #                   help="If true, uses fake data. Defaults to real data.")
    # 
    # FLAGS = FLAGS
    
    # app.run(main)


