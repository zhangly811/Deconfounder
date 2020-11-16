# Copyright 2020 Observational Health Data Sciences and Informatics
#
# This file is part of MvConfounder
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


#' Generate data
#'
#' @details
#' This function generates cohort and extracts data of the cohort.
#'
#' @param connection Connection to the database. An object of class connect as created by the connect function in the DatabaseConnector package.
#' @param cdmDatabaseSchema A schema where OMOP CDM data are stored.
#' @param oracleTempSchema            A schema that can be used to create temp tables in when using Oracle.
#' @param vocabularyDatabaseSchema A schema where vocabulary is stored
#' @param cohortDatabaseSchema A schema where the cohort is stored
#' @param targetCohortTable A string corresponds to the name of the cohort table
#' @param drugExposureTable A string corresponds to the name of the drug exposure table
#' @param measurementTable A string corresponds to the name of the measurement table
#' @param createTargetCohortTable A boolean that indicates whether the targetCohortTable will be created. Default TRUE.
#' @param conditionConceptIds A list of condition concept IDs that correspond to the disease of interest.
#' @param measurementConceptId A numeric of measurement concept ID
#' @param observationWindowBefore An integer indicates the number of pre-treatment days used to look for pre-treatment measurement.
#' @param observationWindowAfter An integer indicates the number of post-treatment days used to look for post-treatment measurement.
#' @param drugWindow An integer indicates the number of post-treatment days during which drug exposure are also considered. Default is 0
#' @param targetCohortId An integer of the cohort ID.
#' @param createTargetCohort A boolean that indicates whether to create the cohort. Default TRUE.
#' @param extractFeature A boolean that indicates whether to extract features for the cohort. Default TRUE.
#' @param dataFolder A string indicates where output will be stored.
#' @param drugFilename A string for the name of the table to store drug exposure.
#' @param measFilename A string for the name of the table to store measurement.
#' @return
#' Returns a string containing the rendered SQL.
#'
#' @export
generateData<-function(connection,
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
                       targetCohortId=NULL,
                       dataFolder,
                       drugFilename,
                       measFilename){
  ParallelLogger::addDefaultFileLogger(file.path(dataFolder, "log.txt"))



  if(createTargetCohort){

    if(is.null(targetCohortId)){
      ParallelLogger::logWarn("Warning: target Cohort Id was set as 9999 automatically")
      targetCohortId <- 9999
    }
    ParallelLogger::logInfo("The cohorts are being generated")

    Deconfounder::createCohorts(connection = connection,
                                  cdmDatabaseSchema = cdmDatabaseSchema,
                                  oracleTempSchema = oracleTempSchema,
                                  vocabularyDatabaseSchema = cdmDatabaseSchema,
                                  cohortDatabaseSchema = cohortDatabaseSchema,
                                  targetCohortTable = targetCohortTable,
                                  createTargetCohortTable = createTargetCohortTable,
                                  conditionConceptIds = conditionConceptIds,
                                  measurementConceptId = measurementConceptId,
                                  observationWindowBefore = observationWindowBefore,
                                  observationWindowAfter = observationWindowAfter,
                                  drugWindow = drugWindow,
                                  targetCohortId)


    ParallelLogger::logInfo("Cohort was generated")

  } else {
    if(is.null(targetCohortId)) stop ("You should specify targetCohortId if you don't create the cohort")
  }

  # get features
  if (extractFeature){

    # Create drug exposure table and measurement table:
    sql <- SqlRender::loadRenderTranslateSql(sqlFilename = "extractDrugAndMeas.sql",
                                             packageName = "Deconfounder",
                                             dbms = attr(connection, "dbms"),
                                             oracleTempSchema = oracleTempSchema,
                                             drug_exposure_table = drugExposureTable,
                                             measurement_table = measurementTable,
                                             target_cohort_id = targetCohortId,
                                             measurement_concept_id = measurementConceptId,
                                             observation_window_before = observationWindowBefore,
                                             observation_window_after = observationWindowAfter,
                                             drug_window = drugWindow,
                                             cdm_database_schema = cdmDatabaseSchema,
                                             target_cohort_id = targetCohortId,
                                             target_cohort_table = targetCohortTable,
                                             target_database_schema = cohortDatabaseSchema)

    DatabaseConnector::executeSql(connection, sql, progressBar = TRUE, reportOverallTime = TRUE)

    # load data into R
    sql<-SqlRender::render("SELECT * FROM @target_database_schema.@measurement_table",
                           target_database_schema = cohortDatabaseSchema,
                           measurement_table = measurementTable)
    meas <- DatabaseConnector::querySql(connection, sql)
    sql<-SqlRender::render("SELECT * FROM @target_database_schema.@drug_exposure_table",
                           target_database_schema = cohortDatabaseSchema,
                           drug_exposure_table = drugExposureTable)
    drug <- DatabaseConnector::querySql(connection, sql)
    write.csv(meas, file.path(dataFolder, measFilename))
    write.csv(drug, file.path(dataFolder, drugFilename))
    ParallelLogger::logInfo("Features were generated and saved at data folder")
  } else {
    ParallelLogger::logInfo("Features were not generated.")
  }
}



#' Preprocess data
#' @param dataFolder A string indicates where output will be stored.
#' @param drugFilename A string for the name of the table to store drug exposure.
#' @param measFilename A string for the name of the table to store measurement.
#' @param drugWindow An integer indicates the number of post-treatment days during which drug exposure are also considered. Default is 0
#'
#' @export
preprocessingData <- function(dataFolder, measFilename, drugFilename, drugWindow){
  reticulate::source_python("inst/python/preprocessing.py")
  preprocessing(dataFolder, measFilename, drugFilename, drugWindow)
}


#' fit the deconfounder to estimate average treatment effect
#' @param data_dir String: the directory where cohort data are stored
#' @param save_dir String: the directory where results will be stored
#' @param factor_model String: the type of probabilistic factor model to fit. Choices are: PMF or DEF.
#' @param learning_rate Float: The learning rate for the probabilistic factor model.
#' @param max_steps Integer: the maximum steps to run the probabilistic factor model.
#' @param latent_dim Integer: the number of latent dimensions in PMF.
#' @param layer_dim List: a list of length 2. The number of latent dimensions in each layer of the 2-layer DEF.
#' @param batch_size Integer: the number of datapoints to use in each training step of the probabilistic model.
#' @param num_samples Integer: number of samples from variational distribution used in updating variational parameters.
#' @param holdout_portion Float: A value between 0 and 1. The proportion of data heldout for predictive model checking in checking the probabilistic model.
#' @param print_steps Integer: Print the results during training.
#' @param tolerance Integer: The termination criteria for training the probabilistic model.
#' @param num_confounder_samples Integer: number of samples of substitute confounder from the posterior, input for the outcome model for estimating ATE.
#' @param CV Integer: Fold of cross validation in the outcome model.
#' @param outcome_type String: The type of outcome. Choices are: linear
#'
#' @export
fitDeconfounder <- function(data_dir,
                            save_dir,
                            factor_model,
                            learning_rate=0.0001,
                            max_steps=100000,
                            latent_dim=1,
                            layer_dim=c(30, 10),
                            batch_size=1024,
                            num_samples=1,
                            holdout_portion=0.5,
                            print_steps=50,
                            tolerance=3,
                            num_confounder_samples=30,
                            CV=5,
                            outcome_type='linear'){
  e <- environment()
  reticulate::source_python(system.file(package='Deconfounder','python','main.py'), envir=e)
  fit_deconfounder(data_dir,
                   save_dir,
                   factor_model,
                   learning_rate,
                   max_steps,
                   latent_dim,
                   layer_dim,
                   batch_size,
                   num_samples,
                   holdout_portion,
                   print_steps,
                   tolerance,
                   num_confounder_samples,
                   CV,
                   outcome_type)
}
