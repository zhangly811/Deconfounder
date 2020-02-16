# install.packages("jtools")
# install.packages('rstanarm')
# install.packages('rstan')
# install.packages('glmnet')

library(rstanarm)
library(rstan)
library(glmnet)
#plotting
library(jtools)
library(ggplot2)
library(gtable)
library(gridExtra)
library(grid)
# load data
WORK_DIR <- "/Users/linyingzhang/LargeFiles/Hripcsak/deconfounder/"
DATA_PATH <- paste0(WORK_DIR, "data/20190316_cohort505_t2dm/")
x_df <- read.csv(paste0(DATA_PATH, "sparse_matrix_X.csv"), row.names = 1, header = TRUE)
T_hat <- read.table(paste0(DATA_PATH, "x_post_np_DEF_30_6.txt"))
x_t_df <- as.data.frame(cbind(x_df, T_hat))
lab2 <- read.csv(paste0(DATA_PATH, 'deconfounder_hba1c_lab_test_after_closest.csv'))
lab1<- read.csv(paste0(DATA_PATH, 'deconfounder_hba1c_lab_test_before_closest.csv'))
y <- lab2$value_as_number - lab1$value_as_number

n_causes <- dim(x_df)[2]
##########################################################################################################################
#fit ridge models
fitridge_no_control = stan_glm(y~., data = x_df, family = gaussian(), prior = normal(), 
                    algorithm = "meanfield", adapt_delta = NULL, QR = FALSE,
                    sparse = TRUE)

fitridge_def = stan_glm(y~., data = x_t_df, family = gaussian(), prior = normal(), 
                               algorithm = "meanfield", adapt_delta = NULL, QR = FALSE,
                               sparse = TRUE)
# CI
ci95_no_control <- posterior_interval(fitridge_no_control, prob = 0.95)
ci95_def <- posterior_interval(fitridge_def, prob = 0.95)
# store coefficients mean and 95%CI in a dataframe
res <- round(cbind(fitridge_no_control$coefficients[2:n_causes], ci95_no_control[2:n_causes,], 
                   fitridge_def$coefficients[2:n_causes], ci95_def[2:n_causes,]), 2)
res <- as.data.frame(res)
colnames(res) <- c('mean_no_control', '2.5%_nc', '97.5%_nc', 'mean_def', '2.5%_def', '97.5%_def')
res <- res[order(res$mean_def),]
# plot
g1<-ggplotGrob(stan_plot(fitridge_no_control, point_est = 'mean', ci_level = 0.80, pars = rownames(res), 
                         est_color = 'dodgerblue2', fill_color = 'dodgerblue4', outline_color = 'dodgerblue2')+
                 ggtitle("Unadjusted model") + scale_x_continuous(name="Estimated coefficients", breaks=seq(-0.8, 0.8, 0.2))
)

g2<-ggplotGrob(stan_plot(fitridge_def, point_est = 'mean', ci_level = 0.80, pars = rownames(res), 
                         est_color = 'dodgerblue2', fill_color = 'dodgerblue4', outline_color = 'dodgerblue2')+
                 ggtitle("Deconfounder") + scale_x_continuous(name="Estimated coefficients", breaks=seq(-0.8, 0.8, 0.2))
)
g <- cbind(g1,g2, size = 'first')
grid.newpage()
grid.draw(g)

# save
write.csv(res, file = paste0(DATA_PATH, "coeffs_ridge_in_R.csv"))
saveRDS(fitridge_no_control, file = paste0(DATA_PATH, 'fitridge_no_control.rds'))
saveRDS(fitridge_def, file = paste0(DATA_PATH, 'fitridge_def.rds'))
# load
res <- read.csv(paste0(DATA_PATH, "coeffs_ridge_in_R.csv"), row.names = 1)
fitridge_no_control <- readRDS(paste0(DATA_PATH, 'fitridge_no_control.rds'))
fitridge_def <- readRDS(paste0(DATA_PATH, 'fitridge_def.rds'))

#########################################################################################################################
# # fit horseshoe models
# fiths_no_control = stan_glm(y~., data = x_df, family = gaussian(), prior = hs(),
#                  algorithm = "meanfield", adapt_delta = NULL, QR = FALSE,
#                  sparse = TRUE)
# fiths_def = stan_glm(y~., data = x_t_df, family = gaussian(), prior = hs(),
#                             algorithm = "meanfield", adapt_delta = NULL, QR = FALSE,
#                             sparse = TRUE)
#
# # CI
# ci95_no_control <- posterior_interval(fiths_no_control, prob = 0.95)
# ci95_def <- posterior_interval(fiths_def, prob = 0.95)
# # store coefficients mean and 95%CI in a dataframe
# res <- round(cbind(fiths_no_control$coefficients[2:n_causes], ci95_no_control[2:n_causes,],
#                    fiths_def$coefficients[2:n_causes], ci95_def[2:n_causes,]), 2)
# res <- as.data.frame(res)
# colnames(res) <- c('mean_no_control', '2.5%_nc', '97.5%_nc', 'mean_def', '2.5%_def', '97.5%_def')
# res <- res[order(res$mean_def),]
# # plotting
# g1<-ggplotGrob(stan_plot(fiths_no_control, point_est = 'mean', ci_level = 0.80, pars = rownames(res),
#           show_outer_line = TRUE, est_color = 'dodgerblue2', fill_color = 'dodgerblue4', outline_color = 'dodgerblue2')+
#             ggtitle("Unadjusted model") + scale_x_continuous(name="Coefficients", limits=c(-0.4, 0.3)))
#
# g2<-ggplotGrob(stan_plot(fiths_def, point_est = 'mean', ci_level = 0.80, pars = rownames(res),
#           show_outer_line = TRUE, est_color = 'dodgerblue2', fill_color = 'dodgerblue4', outline_color = 'dodgerblue2')+
#             ggtitle("Deconfounder")+coord_cartesian(xlim = c(-0.4, 0.3)))
# g <- cbind(g1,g2, size = 'first')
# grid.newpage()
# grid.draw(g)
# plot(fiths_def)
# # save
# write.csv(res, file = paste0(DATA_PATH, "coeffs_hs_in_R.csv"))
# saveRDS(fiths_no_control, file = paste0(DATA_PATH, 'fiths_no_control.rds'))
# saveRDS(fiths_def, file = paste0(DATA_PATH, 'fiths_def.rds'))
# # load
# res <- read.csv(paste0(DATA_PATH, "coeffs_hs_in_R_v1.csv"), row.names = 1)
# fiths_no_control <- readRDS(paste0(DATA_PATH, 'fiths_no_control_v1.rds'))
# fiths_def <- readRDS(paste0(DATA_PATH, 'fiths_def_v1.rds'))
#
