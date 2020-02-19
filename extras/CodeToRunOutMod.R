install.packages("jtools")
install.packages('rstanarm')
install.packages('rstan')
install.packages('glmnet')

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
inputFolder <- "dat/1FullCohort"
factorModResFolder <- "res/factorMod"
outputFolder <- "res/outcomeMod"
if (!dir.exists(outputFolder)){dir.create(outputFolder)}
drug <- Matrix::readMM(file=file.path(inputFolder, "drugSparseMat.txt"))
drug <- as.data.frame(as.matrix(drug))
drug <- drug*1
meas <- Matrix::readMM(file=file.path(inputFolder, "measChangeSparseMat.txt"))
meas <- as.data.frame(as.matrix(meas))
measIdx <- Matrix::readMM(file=file.path(inputFolder, "measChangeIndexMat.txt"))
drugName <- as.character(read.csv(file=file.path(inputFolder, "drugName.csv"), row.names = 1)[,1])
measName <- as.character(read.csv(file=file.path(inputFolder, "measName.csv"), row.names = 1)[,1])

# x_df <- read.csv(paste0(DATA_PATH, "drugSparseMat.txt"), row.names = 1, header = TRUE)
conf <- read.table(file.path(factorModResFolder, "pmf_z_post_np.txt"))
drugConf <- as.data.frame(cbind(drug, conf))

numUnits <- dim(drug)[1]
numCauses <- dim(drug)[2]
numOutcomes <- dim(meas)[2]
numConfs <- dim(conf)[2]
##########################################################################################################################
#fit ridge models
coefMat <- matrix(data = NA, nrow = numCauses, ncol = numOutcomes)
rownames(coefMat) <- drugName
colnames(coefMat) <- measName

for (o in seq(numOutcomes)){
  print(paste0("Running outcome ", o))
  rowIdx <- which(measIdx[,o]!=0)

  y <- meas[rowIdx, o]
  x <- drug[rowIdx,]
  xc <- drugConf[rowIdx,]


  if (length(y)>=numCauses){
    ridgeNoCtrl = rstanarm::stan_glm(y~., data = x, family = gaussian(), prior = normal(),
                        algorithm = "meanfield", adapt_delta = NULL, QR = FALSE,
                        sparse = TRUE)

    ridgeDcf = stan_glm(y~., data = xc, family = gaussian(), prior = normal(),
                                   algorithm = "meanfield", adapt_delta = NULL, QR = FALSE,
                                   sparse = TRUE)
  }
}
# CI
ci95_no_control <- posterior_interval(ridgeNoCtrl, prob = 0.95)
ci95_def <- posterior_interval(ridgeDcf, prob = 0.95)
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
write.csv(res, file = file.path(outputFolder, "coeffs_ridge_in_R.csv"))
saveRDS(fitridge_no_control, file = file.path(outputFolder, 'fitridge_no_control.rds'))
saveRDS(fitridge_def, file = file.path(outputFolder, 'fitridge_def.rds'))
# # load
# res <- read.csv(paste0(DATA_PATH, "coeffs_ridge_in_R.csv"), row.names = 1)
# fitridge_no_control <- readRDS(paste0(DATA_PATH, 'fitridge_no_control.rds'))
# fitridge_def <- readRDS(paste0(DATA_PATH, 'fitridge_def.rds'))

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
