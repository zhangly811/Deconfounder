# Copyright 2020 Observational Health Data Sciences and Informatics
#
# This file is part of Deconfounder
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install devtools from CRAN
# install.packages("devtools")
# devtools::install_github("ohdsi/SqlRender")
# devtools::install_github("ohdsi/DatabaseConnector")
# devtools::install_github("ohdsi/FeatureExtraction")
# devtools::install_github("ohdsi/PatientLevelPrediction")


connectionDetails = DatabaseConnector::createConnectionDetails(dbms = "sql server",
                                                               server = "omop.dbmi.columbia.edu")
connection = DatabaseConnector::connect(connectionDetails)
cdmDatabaseSchema = "ohdsi_cumc_deid_2020q2r2.dbo"
cohortDatabaseSchema = "ohdsi_cumc_deid_2020q2r2.results"
targetCohortTable = "Deconfounder_COHORT"
targetCohortId = 2
drugExposureTable = "SAMPLE_COHORT_DRUG_EXPOSURE"
measurementTable = "SAMPLE_COHORT_MEASUREMENT"
conditionConceptIds <- c(434610,437833) # Hypo and hyperkalemia
measurementConceptId <- c(3023103) # serum potassium

observationWindowBefore <- 7
observationWindowAfter <- 30
drugWindow <- 0


measFilename <- "meas.csv"
drugFilename <- "drug.csv"
dataFolder <- "C:/Users/lz2629/dat/DeconfounderSampleDat"
Deconfounder::generateData(connection,
             cdmDatabaseSchema,
             oracleTempSchema = NULL,
             vocabularyDatabaseSchema = cdmDatabaseSchema,
             cohortDatabaseSchema,
             targetCohortTable,
             drugExposureTable,
             measurementTable,
             conditionConceptIds,
             measurementConceptId,
             observationWindowBefore,
             observationWindowAfter,
             drugWindow,
             createTargetCohortTable = T,
             createTargetCohort = T,
             extractFeature = T,
             targetCohortId=targetCohortId,
             dataFolder,
             drugFilename,
             measFilename)


reticulate::use_condaenv("deconfounder_py3", required = TRUE)
Deconfounder::preprocessingData(dataFolder, measFilename, drugFilename, drugWindow)


factorModel <- 'DEF'
outputFolder <- "C:/Users/lz2629/dat/DeconfounderSampleRes"

Deconfounder::fitDeconfounder(data_dir=dataFolder,
                save_dir=outputFolder,
                factor_model=factorModel,
                learning_rate=0.0001,
                max_steps=as.integer(100000),
                latent_dim=as.integer(1), # number of latent var for PMF
                layer_dim=c(as.integer(20), as.integer(4)), # number of latent var in each layer of DEF
                batch_size=as.integer(1024),
                num_samples=as.integer(1), # number of samples from variational distribution
                holdout_portion=0.5,
                print_steps=as.integer(50),
                tolerance=as.integer(100), # termination criteria for the factor model: 3 consecutive increase of the ELBO
                num_confounder_samples=as.integer(30), # number of samples of substitute confounder from the posterior
                CV=as.integer(5), # fold of cross-val in the outcome model
                outcome_type='linear'
)


library(ggplot2)
stats <- read.csv(file = file.path(outputFolder,
                                           "DEF_lr0.0001_maxsteps100000_latentdim1_layerdim[20, 4]_batchsize1024_numsamples1_holdoutp0.5_tolerance100_numconfsamples30_CV5_outTypelinear",
                                           "treatment_effects_stats.csv"))

stats$drug_name <- factor(stats$drug_name, levels = stats$drug_name[order(-stats$mean)])
p2 <- ggplot(stats, aes(drug_name, mean)) + theme_gray(base_size=10)
p2 + geom_point(size=1) +
  geom_errorbar(aes(x = drug_name, ymin = ci95_lower, ymax = ci95_upper), width=0.2) +
  xlab("") +
  ylab("Estimated effect") +
  coord_flip()


