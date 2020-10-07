import os
import shutil
import time
import torch

from absl import app
from absl import flags
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse, stats
from torch.utils.tensorboard import SummaryWriter
## For outcome model
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, train_test_split
# from models import *

def fit_deconfounder(learning_rate=0.001,
                      max_steps=5000,
                      latent_dim=1,
                      batch_size=1024,
                      num_samples=1,
                      holdout_portion=0.2,
                      print_steps=50,
                      tolerance=3,
                      num_confounder_sampels=30,
                      cv=5,
                      outcome_type='linear'):
    __file__ = sys.argv[0]
    project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                               os.pardir))

    data_dir = os.path.join(project_dir, "dat")
    save_dir = os.path.join(project_dir, "res")
    param_save_dir = os.path.join(save_dir, "params/")

    if os.path.exists(save_dir):
        print("Deleting old log directory at {}".format(save_dir))
        shutil.rmtree(save_dir)
    if not os.path.exists(param_save_dir):
        os.makedirs(param_save_dir)

    kwargs = ({'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available()
              else {})

    dataset = CustomDataset("ppca_data.txt", data_dir, holdout_portion)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
    iterator = data_loader.__iter__()

    num_datapoints, data_dim = dataset.counts.shape
    stddv_datapoints = 0.5

    summary_writer = SummaryWriter(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPCA(device, num_datapoints, data_dim, latent_dim,
                 stddv_datapoints, num_samples,
                 print_steps, summary_writer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    prev_loss = 1e8
    tol = 0 #tolerance counter
    for step in range(max_steps):
        try:
            datapoints_indices, x_train, holdout_mask = iterator.next()
        except StopIteration:
            iterator = data_loader.__iter__()
            datapoints_indices, x_train, holdout_mask = iterator.next()

        datapoints_indices = datapoints_indices.to(device)
        x_train = x_train.to(device)
        optimizer.zero_grad()
        elbo = model(datapoints_indices, x_train, holdout_mask, step)
        loss = -elbo
        loss.backward()
        optimizer.step()

        if step == 0 or step % print_steps == print_steps-1:
            duration = (time.time() - start_time) / (step + 1)
            print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec)".format(
                step+1, -loss, duration))
            summary_writer.add_scalar("loss", loss, step)

            if loss < prev_loss:
                tol = 0
                prev_loss = loss
            else:
                tol += 1
                prev_loss = loss

            if step == max_steps - 1 or tol >= tolerance:
                w_samples = model.qw_distribution.sample(1000).mean(axis=0)
                z_samples = model.qz_distribution.sample(1000).mean(axis=0)
                loc = torch.transpose(torch.matmul(w_samples, z_samples), 0, 1)
                data_posterior = torch.distributions.Normal(loc=loc, scale=stddv_datapoints)
                x_generated = data_posterior.sample(torch.Size([1]))
                x_generated = torch.squeeze(x_generated)


                model.predictive_check(dataset.vad, dataset.holdout_mask, dataset.holdout_subjects, 100, 100)

                # plt.scatter(dataset.counts[:, 0], dataset.counts[:, 1], color='blue', alpha=0.1, label='Actual data')
                # plt.scatter(x_generated[:, 0], x_generated[:, 1], color='red', alpha=0.1, label='Simulated data')
                # plt.legend()
                # plt.axis([-20, 20, -20, 20])
                # plt.show()
                # plt.close()

                np.save(os.path.join(param_save_dir, "qw_loc"),
                        model.qw_distribution.location.cpu().detach())
                np.save(os.path.join(param_save_dir, "qw_scale"),
                        model.qw_distribution.scale().cpu().detach())
                np.save(os.path.join(param_save_dir, "qz_loc"),
                        model.qz_distribution.location.cpu().detach())
                np.save(os.path.join(param_save_dir, "qz_scale"),
                        model.qz_distribution.scale().cpu().detach())

                if tol >= tolerance:
                    print("Loss goes up for {} consecutive prints. Stop training.".format(tol))
                    break

                if step == max_steps - 1:
                    print("Maximum step reached. Stop training.")

    # fit outcome model
    y = np.loadtxt(os.path.join(data_dir, "ppca_outcome.txt"))
    treatment_effect = np.zeros((num_confounder_samples, data_dim))
    for sample in range(num_confounder_samples):
        substitute_confounder = np.squeeze(model.qz_distribution.sample(1).detach().numpy())
        X = np.column_stack([dataset.counts, substitute_confounder])

        outcome_model = fit_outcome_model(X, y, data_dim, outcome_type, true_betas, CV=CV, verbose=False)
        if outcome_type == 'linear':
            treatment_effect[sample, :] = outcome_model.coef_[:data_dim]
        if outcome_type == 'binary':
            treatment_effect[sample, :] = outcome_model.coef_[0][:data_dim]

    np.save(os.path.join(param_save_dir, "treatment_effect"),
            treatment_effect)


