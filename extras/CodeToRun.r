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
# devtools::install_github("ohdsi/SqlRender")
# devtools::install_github("ohdsi/DatabaseConnector")
# devtools::install_github("ohdsi/FeatureExtraction")
# devtools::install_github("ohdsi/PatientLevelPrediction")


connectionDetails = DatabaseConnector::createConnectionDetails(dbms = "sql server",
                                                               server = "omop.dbmi.columbia.edu")
connection = DatabaseConnector::connect(connectionDetails)
cdmDatabaseSchema = "ohdsi_cumc_deid_2020q2r2.dbo"
cohortDatabaseSchema = "ohdsi_cumc_deid_2020q2r2.results"
targetCohortTable = "MVDECONFOUNDER_COHORT"
targetCohortId = 1
drugExposureTable = "SAMPLE_COHORT_DRUG_EXPOSURE"
measurementTable = "SAMPLE_COHORT_MEASUREMENT"
conditionConceptIds <- c(434610,437833) # Hypo and hyperkalemia
measurementConceptId <- c(3023103) # serum potassium

observationWindowBefore <- 7
observationWindowAfter <- 30
drugWindow <- 7


measFilename <- "meas.csv"
drugFilename <- "drug.csv"
generateData(connection,
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
reticulate::source_python("inst/python/preprocessing.py")
preprocessing(dataFolder, measFilename, drugFilename)
factorModel <- 'PMF'
fitDeconfounder(data_dir=dataFolder,
                save_dir=outputFolder,
                factor_model=factorModel,
                learning_rate=0.001,
                max_steps=5000,
                latent_dim=1,
                batch_size=1024,
                num_samples=1, # number of samples from variational distribution
                holdout_portion=0.5,
                print_steps=50,
                tolerance=3,
                num_confounder_samples=30, # number of samples of substitute confounder from the posterior
                CV=5,
                outcome_type='linear'
                )
