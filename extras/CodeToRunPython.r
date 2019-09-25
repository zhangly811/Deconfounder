reticulate::use_python(Sys.which("python3"))
reticulate::source_python("python/deep_exponential_family.py")
reticulate::source_python("python/ed_bayesian_regression.py")
reticulate::source_python("python/utils.py")
reticulate::source_python("python/main.py")

e <- environment()
reticulate::source_python(system.file(package='MvDeconfounder','python','deep_exponential_family.py'), envir = e)
reticulate::source_python(system.file(package='MvDeconfounder','python','ed_bayesian_regression.py'), envir = e)
reticulate::source_python(system.file(package='MvDeconfounder','python','utils.py'), envir = e)
reticulate::source_python(system.file(package='MvDeconfounder','python','main.py'), envir = e)

data <- reticulate::r_to_py(x$data)

# specify parameters
learning_rate = 1e-4
max_steps = as.integer(1000)
layer_sizes = c("50", "10")
shape = 1.0
holdout_portion = 0.5
# data directory and file name
data_dir = '/Users/linyingzhang/LargeFiles/Blei/multivariate_medical_deconfounder/dat/'
data_filename = 'sparse_matrix_X.csv'
# directory for outputs
factor_model_dir = "/tmp/factor_model/"
outcome_model_dir = "/tmp/outcome_model/"
# simulation
fake_data = TRUE

main(learning_rate, max_steps, layer_sizes, shape, holdout_portion,
     data_dir, data_filename, factor_model_dir, outcome_model_dir, fake_data)
