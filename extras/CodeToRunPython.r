# specify which python to use
reticulate::use_condaenv("deconfounder_py3",required=TRUE)
fitDeconfounder <- function(learning_rate,
                            max_steps,
                            latent_dim,
                            batch_size,
                            num_samples,
                            holdout_portion,
                            print_steps,
                            tolerance,
                            num_confounder_samples,
                            cv,
                            outcome_type,
                            project_dir){
  e <- environment()
  reticulate::source_python('inst/python/main.py', envir=e)
  fit_deconfounder(learning_rate,
                   max_steps,
                   latent_dim,
                   batch_size,
                   num_samples,
                   holdout_portion,
                   print_steps,
                   tolerance,
                   num_confounder_samples,
                   cv,
                   outcome_type,
                   project_dir)
}

learning_rate=0.01
max_steps=as.integer(5000)
latent_dim=as.integer(1)
batch_size=as.integer(1024)
num_samples=as.integer(1)
holdout_portion=0.2
print_steps=as.integer(50)
tolerance=as.integer(3)
num_confounder_samples=as.integer(100)
CV=as.integer(5)
outcome_type='linear'
project_dir="C:/Users/lz2629/git/zhangly811/MvDeconfounder"

fitDeconfounder(learning_rate,
                max_steps,
                latent_dim,
                batch_size,
                num_samples,
                holdout_portion,
                print_steps,
                tolerance,
                num_confounder_samples,
                cv,
                outcome_type,
                project_dir)

