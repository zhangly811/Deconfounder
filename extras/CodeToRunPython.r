# specify which python to use
reticulate::use_condaenv("dcf", conda = "/opt/anaconda2/bin/conda", required=TRUE)
e <- environment()
reticulate::source_python(system.file(package='MvDeconfounder','python','simulate_data.py'), envir = e)
reticulate::source_python(system.file(package='MvDeconfounder','python','dcf.py'), envir = e)
reticulate::source_python(system.file(package='MvDeconfounder','python','utils.py'), envir = e)

# data <- reticulate::r_to_py(x$data)

N=5000
K=10
D=50
Nsim = 500

X, C, Ys, betas <- simulate_multicause_data(N, K, D, Nsim)

# specify parameters

x_train, x_vad, holdout_mask <- holdout_data(X)
x_post, U_post, V_post, pmf_x_post_np, pmf_z_post_np <- fit_pmf(x_train, gamma_prior=0.1, M=100, K=10, n_iter=20000, optimizer=tf.train.RMSPropOptimizer(1e-4))
overall_pval <- pmf_predictive_check(x_train, x_vad, holdout_mask, x_post, V_post, U_post,n_rep=10, n_eval=10)
