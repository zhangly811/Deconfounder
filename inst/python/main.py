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

class CustomDataset(torch.utils.data.Dataset):
    """A customed torch dataset."""

    def __init__(self, txt_file, data_dir, holdout_portion, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            data_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.counts = np.loadtxt(os.path.join(data_dir, txt_file))
        self.transform = transform
        num_datapoints, data_dim = self.counts.shape
        self.num_datapoints = num_datapoints
        self.data_dim = data_dim
        self.__train_valid_split__(holdout_portion)

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, idx):
        return idx, self.train[idx], self.holdout_mask[idx]

    def __train_valid_split__(self, holdout_portion):
        # randomly holdout some entries
        n_holdout = int(holdout_portion * self.num_datapoints * self.data_dim)

        holdout_row = np.random.randint(self.num_datapoints, size=n_holdout)
        holdout_col = np.random.randint(self.data_dim, size=n_holdout)
        holdout_mask_initial = (sparse.coo_matrix((np.ones(n_holdout), \
                                                   (holdout_row, holdout_col)), \
                                                  shape=self.counts.shape)).toarray()

        self.holdout_subjects = np.unique(holdout_row)
        self.holdout_mask = np.minimum(1, holdout_mask_initial)

        self.train = np.multiply(1 - self.holdout_mask, self.counts)
        self.vad = np.multiply(self.holdout_mask, self.counts)

class VariationalFamily(torch.nn.Module):
    """Object to store variational parameters and get sample statistics."""

    def __init__(self, device, family, shape, initial_loc=None):
        """Initialize variational family.

        Args:
          device: Device where operations take place.
          family: A string representing the variational family, either "normal" or
            "lognormal".
          shape: A list representing the shape of the variational family.
          initial_loc: An optional tensor with shape `shape`, denoting the initial
            location of the variational family.
        """
        super(VariationalFamily, self).__init__()
        if initial_loc is None:
            if len(shape) > 1:
                self.location = torch.nn.init.xavier_uniform_(
                    torch.nn.Parameter(torch.ones(shape)))
            else:
                self.location = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.ones(shape)),
                    std=0.1)
        else:
            self.location = torch.nn.Parameter(
                torch.FloatTensor(np.log(initial_loc)))
        self.log_scale = torch.nn.Parameter(torch.zeros(shape))
        self.family = family
        if self.family == 'normal':
            self.prior = torch.distributions.Normal(loc=0., scale=1.)
        elif self.family == 'lognormal':
            self.prior = torch.distributions.Gamma(concentration=0.3, rate=0.3)
        else:
            raise ValueError("Unrecognized prior distribution.")
        self.device = device

    def scale(self):
        """Constrain scale to be positive using softplus."""
        return torch.nn.functional.softplus(self.log_scale)

    def distribution(self):
        """Create variational distribution."""
        if self.family == 'normal':
            distribution = torch.distributions.Normal(
                loc=self.location,
                scale=self.scale())
        elif self.family == 'lognormal':
            distribution = torch.distributions.LogNormal(
                loc=self.location,
                scale=self.scale())
        return distribution

    def get_log_prior(self, samples):
        """Compute log prior of samples."""
        # Sum all but first axis.
        log_prior = torch.sum(self.prior.log_prob(samples).to(self.device),
                              dim=tuple(range(1, len(samples.shape))))
        return log_prior

    def get_entropy(self, samples):
        """Compute entropy of samples from variational distribution."""
        # Sum all but first axis.
        entropy = -torch.sum(self.distribution().log_prob(samples).to(self.device),
                             dim=tuple(range(1, len(samples.shape))))
        return entropy

    def sample(self, num_samples):
        """Sample from variational family using reparameterization."""
        return self.distribution().rsample([num_samples])

class PPCA(torch.nn.Module):
    """Object to hold model parameters and approximate ELBO."""
    def __init__(self, device, num_datapoints, data_dim, latent_dim,
                 stddv_datapoints, num_samples,
                 print_steps, summary_writer):
        super(PPCA, self).__init__()
        self.num_datapoints = num_datapoints
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.stddv_datapoints = stddv_datapoints
        self.num_samples = num_samples
        self.print_steps = print_steps
        self.summary_writer = summary_writer
        self.qw_distribution = VariationalFamily(
            device,
            'normal',
            [data_dim, latent_dim]
        )
        self.qz_distribution = VariationalFamily(
            device,
            'normal',
            [latent_dim, num_datapoints]
        )

    def get_samples(self):
        """Return samples from variational distributions."""
        w_samples = self.qw_distribution.sample(self.num_samples)
        z_samples = self.qz_distribution.sample(self.num_samples)

        samples = [w_samples, z_samples]
        return samples

    def get_log_prior(self, samples):
        """Calculate log prior of variational samples.

        Args:
          samples: A list of samples. The length of the list is the number of variables being sampled.

        Returns:
          log_prior: A Monte-Carlo approximation of the log prior, summed across
            latent dimensions and averaged over the number of samples.
        """
        (w_samples, z_samples) = samples
        w_log_prior = self.qw_distribution.get_log_prior(
            w_samples)
        z_topic_log_prior = self.qz_distribution.get_log_prior(
            z_samples)
        log_prior = (w_log_prior +
                     z_topic_log_prior)
        return torch.mean(log_prior)

    def get_entropy(self, samples):
        """Calculate entropy of variational samples.

        Args:
            samples: A list of samples. The length of the list is the number of variables being sampled.

        Returns:
          log_prior: A Monte-Carlo approximation of the variational entropy,
            summed across latent dimensions and averaged over the number of
            samples.
        """
        (w_samples, z_samples) = samples
        w_entropy = self.qw_distribution.get_entropy(
            w_samples)
        z_entropy = self.qz_distribution.get_entropy(
            z_samples)
        entropy = (w_entropy +
                   z_entropy)
        return torch.mean(entropy)

    def get_data_log_likelihood(self,
                                samples,
                                counts,
                                datapoints_indices,
                                holdout_mask
                                ):
        """Approximate log-likelihood term of ELBO using Monte Carlo samples.

        Args:
          samples: A list of samples. The length of the list is the number of variables being sampled.
          counts: A float-tensor with shape [batch_size, num_words].

          holdout_mask: A binary tensor with shape [batch_size, data_dim]. 1=valid, 0=train


        Returns:
          data_log_likelihood: A Monte-Carlo approximation of the count
            log-likelihood, summed across latent dimensions and averaged over the
            number of samples.

        """
        (w_samples, z_samples) = samples
        selected_z_samples = z_samples[:, :, datapoints_indices]
        loc_datapoints = torch.transpose(torch.matmul(w_samples, selected_z_samples),
                                         1,2)

        loc_datapoints = torch.mul(loc_datapoints, 1-holdout_mask)
        data_distribution = torch.distributions.Normal(loc=loc_datapoints, scale=self.stddv_datapoints)
        data_log_likelihood = data_distribution.log_prob(counts)
        data_log_likelihood = torch.sum(data_log_likelihood, dim=1)
        # Adjust for the fact that we're only using a minibatch.
        batch_size = len(counts)
        data_log_likelihood = data_log_likelihood * (
                self.num_datapoints / batch_size)
        return torch.mean(data_log_likelihood)

    def forward(self, datapoints_indices, data, holdout_mask, step):
        """Approximate variational Lognormal ELBO using reparameterization.

        Args:
            datapoints_indices: An int-vector with shape [batch_size].
            data: A matrix with shape `[batch_size, num_words]`.
            step: The training step, used to log summaries to Tensorboard.

        Returns:
            elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value
              is averaged across samples and summed across batches.
        """
        samples = self.get_samples()
        log_prior = self.get_log_prior(samples)
        data_log_likelihood = self.get_data_log_likelihood(
            samples,
            data,
            datapoints_indices,
            holdout_mask
        )
        entropy = self.get_entropy(samples)
        elbo = data_log_likelihood + log_prior + entropy
        if step % self.print_steps == 0:
            self.summary_writer.add_scalar("elbo/entropy", entropy, step)
            self.summary_writer.add_scalar("elbo/log_prior", log_prior, step)
            self.summary_writer.add_scalar("elbo/data_log_likelihood",
                                           data_log_likelihood,
                                           step)
            self.summary_writer.add_scalar('elbo/elbo', elbo, step)
        return elbo

    def predictive_check(self, x_vad, holdout_mask, holdout_subjects, n_rep, n_eval):
        """Predictive model checking.

        Args:
            x_vad: The validation data matrix with shape `[num_datapoints, data_dim]`.
            holdout_mask: A binary tensor to mask the train/validation data with shape [num_datapoints, data_dim]. 1=valid, 0=train
            n_rep: The number of replicated datasets we generate.
            n_eval: The number of samples drawn from the variational posterior
        Returns:
            elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value
              is averaged across samples and summed across batches.
        """
        holdout_gen = np.zeros((n_rep, self.num_datapoints, self.data_dim))

        for i in range(n_rep):
            w_samples = self.qw_distribution.sample(1)
            z_samples = self.qz_distribution.sample(1)
            loc = torch.squeeze(torch.transpose(torch.matmul(w_samples, z_samples), 1, 2))
            data_posterior = torch.distributions.Normal(loc=loc, scale=self.stddv_datapoints)
            x_generated = data_posterior.sample(torch.Size([1]))
            x_generated = torch.squeeze(x_generated)
            # look only at the heldout entries
            holdout_gen[i] = np.multiply(x_generated, holdout_mask)

        obs_ll = []
        rep_ll = []
        for j in range(n_eval):
            w_samples = self.qw_distribution.sample(1).detach().numpy()
            z_samples = self.qz_distribution.sample(1).detach().numpy()

            holdoutmean_sample = np.transpose(np.squeeze(w_samples.dot(z_samples)))
            holdoutmean_sample = np.multiply(holdoutmean_sample, holdout_mask)
            obs_ll.append(np.mean(stats.norm(holdoutmean_sample, self.stddv_datapoints).logpdf(x_vad), axis=1))

            rep_ll.append(np.mean(stats.norm(holdoutmean_sample, self.stddv_datapoints).logpdf(holdout_gen), axis=2))

        obs_ll_per_zi, rep_ll_per_zi = np.mean(np.array(obs_ll), axis=0), np.mean(np.array(rep_ll), axis=0)

        pvals = np.array([np.mean(rep_ll_per_zi[:,i] < obs_ll_per_zi[i]) for i in range(self.num_datapoints)])
        overall_pval = np.mean(pvals[holdout_subjects])
        print("Predictive check p-values", overall_pval)

def regression_CV(X, y, outtype="linear", verbose=False):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=123)

    if outtype == "linear":
        reg = linear_model.Ridge()
    elif outtype == "logistic":
        reg = linear_model.RidgeClassifier()

    parameters = dict(alpha=np.logspace(-5, 5, 10))
    clf = GridSearchCV(reg, parameters)
    clf.fit(X_train, y_train)

    bestalpha = clf.best_estimator_.alpha
    print("best alpha", bestalpha)

    if outtype == "linear":
        reg = linear_model.Ridge(alpha=bestalpha)
    elif outtype == "logistic":
        reg = linear_model.RidgeClassifier(alpha=bestalpha)

    reg.fit(X_train, y_train)

    if verbose:
        print("training score", reg.score(X_train, y_train))
        print("predictive score", reg.score(X_test, y_test))

    return reg

def regression_noCV(X, y, outtype="linear", verbose=False):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=123)

    if outtype == "linear":
        reg = linear_model.Ridge(alpha=0)
    elif outtype == "logistic":
        reg = linear_model.RidgeClassifier(alpha=0)

    reg.fit(X_train, y_train)

    if verbose:
        print("training score", reg.score(X_train, y_train))
        print("predictive score", reg.score(X_test, y_test))

    return reg

def fit_outcome_linear(X, y, n_causes, true_betas=None, CV=False, verbose=False):
    if CV == True:
        linear_reg = regression_CV(X, y, outtype="linear", verbose=verbose)
    else:
        linear_reg = regression_noCV(X, y, outtype="linear", verbose=verbose)
    if true_betas is not None:
        linear_rmse = np.sqrt(((true_betas - linear_reg.coef_[:n_causes]) ** 2).mean())
        trivial_rmse = np.sqrt(((true_betas - 0) ** 2).mean())
        if verbose:
            print("linear outcome rmse", linear_rmse, "\nlinear - trivial", linear_rmse - trivial_rmse)
    return linear_reg

def fit_outcome_logistic(X, y_bin, n_causes, true_betas=None, CV=False, verbose=False):
    if CV == True:
        logistic_reg = regression_CV(X, y_bin, outtype="logistic", verbose=verbose)
    else:
        logistic_reg = regression_noCV(X, y_bin, outtype="logistic", verbose=verbose)
    if true_betas is not None:
        logistic_rmse = np.sqrt(((true_betas - logistic_reg.coef_[0][:n_causes]) ** 2).mean())
        trivial_rmse = np.sqrt(((true_betas - 0) ** 2).mean())
        if verbose:
            print("logistic outcome rmse: ", logistic_rmse, \
                  "\nlogistic - trivial: ", logistic_rmse - trivial_rmse)
    return logistic_reg

def fit_outcome_model(X, y, n_causes, outcome_type, true_betas=None, CV=False, verbose=False):
    if outcome_type == 'linear':
        return fit_outcome_linear(X, y, n_causes, true_betas, CV, verbose)
    if outcome_type == 'binary':
        return fit_outcome_logistic(X, y, n_causes, true_betas, CV, verbose)

def fit_deconfounder(learning_rate,
                   max_steps,
                   latent_dim,
                   batch_size,
                   num_samples,
                   holdout_portion,
                   print_steps,
                   tolerance,
                   num_confounder_samples,
                   CV,
                   outcome_type,
                   project_dir):
    # project_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),
    #                                            os.pardir))

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

        outcome_model = fit_outcome_model(X, y, data_dim, outcome_type, CV=CV, verbose=False)
        if outcome_type == 'linear':
            treatment_effect[sample, :] = outcome_model.coef_[:data_dim]
        if outcome_type == 'binary':
            treatment_effect[sample, :] = outcome_model.coef_[0][:data_dim]

    np.save(os.path.join(param_save_dir, "treatment_effect"),
            treatment_effect)

