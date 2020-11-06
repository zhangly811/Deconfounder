# Copyright 2020 Observational Health Data Sciences and Informatics
#
# This file is part of MvDeconfounder
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
devtools::install_github("ohdsi/SqlRender")
# devtools::install_github("ohdsi/DatabaseConnector")
# devtools::install_github("ohdsi/FeatureExtraction")
# devtools::install_github("ohdsi/PatientLevelPrediction")

# install.packages("qlcMatrix")
# install.packages("d3heatmap")

Sys.setenv(TZ = "EST")
options(fftempdir = "C:/tmp")
memory.limit(size = 1024 * 12)

connectionDetails <- DatabaseConnector::createConnectionDetails(dbms = 'sql server',
                                                                server = 'omop.dbmi.columbia.edu')
connection <- DatabaseConnector::connect(connectionDetails)

cdmDatabaseSchema <- 'ohdsi_cumc_deid_2020q2r2.dbo'
cohortDatabaseSchema <-'ohdsi_cumc_deid_2020q2r2.results'
## Medical deconfounder (single outcome)
conditionConceptIds <- c(434610,437833) # Hypo and hyperkalemia
measurementConceptId <- c(3023103) # serum potassium

observationWindowBefore <- 7
observationWindowAfter <- 30
targetCohortTable <- 'DECONFOUNDER_COHORT'
targetCohortId <- 1
# createCohorts(connection = connection,
#               cdmDatabaseSchema = cdmDatabaseSchema,
#               oracleTempSchema = NULL,
#               vocabularyDatabaseSchema = cdmDatabaseSchema,
#               cohortDatabaseSchema = cohortDatabaseSchema,
#               targetCohortTable = targetCohortTable,
#               createTargetCohortTable = TRUE,
#               conditionConceptIds = conditionConceptIds,
#               measurementConceptId = measurementConceptId,
#               observationWindowBefore = observationWindowBefore,
#               observationWindowAfter = observationWindowAfter,
#               targetCohortId)



# Create drug exposure table and measurement table:
drugExposureTable <- "potassium_cohort_drug_exposure"
measurementTable <- "POTASSIUM_COHORT_MEASUREMENT"
sql <- SqlRender::loadRenderTranslateSql(sqlFilename = "extractDrugAndMeas.sql",
                                         packageName = "MvDeconfounder",
                                         dbms = attr(connection, "dbms"),
                                         oracleTempSchema = NULL,
                                         drug_exposure_table = drugExposureTable,
                                         measurement_table = measurementTable,
                                         target_cohort_id = targetCohortId,
                                         measurement_concept_id = measurementConceptId,
                                         observation_window_before = observationWindowBefore,
                                         observation_window_after = observationWindowAfter,
                                         cdm_database_schema = cdmDatabaseSchema,
                                         target_cohort_id = targetCohortId,
                                         target_cohort_table = targetCohortTable,
                                         target_database_schema = cohortDatabaseSchema)

DatabaseConnector::executeSql(connection, sql, progressBar = TRUE, reportOverallTime = FALSE)

# load data into R
sql<-SqlRender::render("SELECT * FROM @target_database_schema.@measurement_table",
                          target_database_schema = cohortDatabaseSchema,
                          measurement_table = measurementTable)
meas <- DatabaseConnector::querySql(connection, sql)
sql<-SqlRender::render("SELECT * FROM @target_database_schema.@drug_exposure_table",
                       target_database_schema = cohortDatabaseSchema,
                       drug_exposure_table = drugExposureTable)
drug <- DatabaseConnector::querySql(connection, sql)
#
#
# ## Multivariate deconfounder
# targetCohortTable <- 'MVDECONFOUNDER_COHORT'
# targetCohortId <- 1
# DataFolder <- 'dat/PackageTest'
# ResFolder <- 'res/PackageTest'
# ingredientList <- readRDS(file = file.path(DataFolder, 'ingredientList.rds'))
# ingredientConceptIds <-ingredientList[[1]]$conceptId
# measurementList <- readRDS(file = file.path(DataFolder,'measurementList.rds'))
# measurementConceptIds <- measurementList[[1]]$conceptId
# mVdData <- MvDeconfounder::generateMvdData(connection = connection,
#                                            cdmDatabaseSchema = cdmDatabaseSchema,
#                                            oracleTempSchema = NULL,
#                                            vocabularyDatabaseSchema = cdmDatabaseSchema,
#                                            cohortDatabaseSchema = cohortDatabaseSchema,
#                                            targetCohortTable = targetCohortTable,
#                                            minimumProportion = 0.001,
#                                            targetDrugTable = 'DRUG_ERA',
#                                            ingredientConceptIds = ingredientConceptIds,
#                                            measurementConceptIds = measurementConceptIds,
#                                            createTargetCohort = F,
#                                            extractDrugFeature = T,
#                                            extractMeasFeature = T,
#                                            labWindow = 35,
#                                            targetCohortId = targetCohortId,
#                                            temporalStartDays = c(-35, 1),
#                                            temporalEndDays =c(-1, 35),
#                                            sampleSize = NULL,
#                                            DataFolder)
#
# result <- MvDeconfounder::unadjIndeptRidgeReg(DataFolder,
#                                               ResFolder,
#                                               cvFold = 5,
#                                               lambda = 10^seq(2, -2, by = -0.1))
# plpData.meas <-PatientLevelPrediction::loadPlpData(file = 'dat/20191116Complete/plpData.meas')
# plpData.drug <-PatientLevelPrediction::loadPlpData(file = 'dat/20191116Complete/plpData.drug')
#
# # specify the python to use, i.e. from a conda env
# reticulate::use_condaenv("deconfounder_py3", required = TRUE)
#
# learning_rate <- 0.01
# max_steps <- as.integer(5000)
# latent_dim <- as.integer(1)
# batch_size <- as.integer(1024)
# num_samples <- as.integer(1)
# holdout_portion <- 0.2
# print_steps <- as.integer(50)
# tolerance <- as.integer(3)
# num_confounder_samples <- as.integer(100)
# CV <- as.integer(5)
# outcome_type <- "linear"
# project_dir <- "C:/Users/lz2629/git/zhangly811/MvDeconfounder"
#
# MvDeconfounder::fitDeconfounder(learning_rate,
#                                 max_steps,
#                                 latent_dim,
#                                 batch_size,
#                                 num_samples,
#                                 holdout_portion,
#                                 print_steps,
#                                 tolerance,
#                                 num_confounder_samples,
#                                 CV,
#                                 outcome_type,
#                                 project_dir)
