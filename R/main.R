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


#' Listing function
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
                       dataFolder){
  ParallelLogger::addDefaultFileLogger(file.path(dataFolder, "log.txt"))



  if(createTargetCohort){

    if(is.null(targetCohortId)){
      ParallelLogger::logWarn("Warning: target Cohort Id was set as 9999 automatically")
      targetCohortId <- 9999
    }
    ParallelLogger::logInfo("The cohorts are being generated")

    MvDeconfounder::createCohorts(connection = connection,
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
                                  targetCohortId)


    ParallelLogger::logInfo("Cohort was generated")

  } else {
    if(is.null(targetCohortId)) stop ("You should specify targetCohortId if you don't create the cohort")
  }

  # get features
  if (extractFeature){

    # Create drug exposure table and measurement table:
    sql <- SqlRender::loadRenderTranslateSql(sqlFilename = "extractDrugAndMeas.sql",
                                             packageName = "MvDeconfounder",
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
    write.csv(meas, file.path(dataFolder, "meas.csv"))
    write.csv(drug, file.path(dataFolder, "drug.csv"))
    ParallelLogger::logInfo("Features were generated and saved at data folder")
  } else {
    ParallelLogger::logInfo("Features were not generated.")
  }
}


#' Listing function
#' @export
generateMvdData<-function(connection,
                          cdmDatabaseSchema,
                          oracleTempSchema = NULL,
                          vocabularyDatabaseSchema = cdmDatabaseSchema,
                          cohortDatabaseSchema,
                          targetCohortTable,
                          minimumProportion = 0.001,
                          targetDrugTable = 'DRUG_ERA',
                          ingredientConceptIds=NULL,
                          measurementConceptIds=NULL,
                          createTargetCohort = T,
                          extractDrugFeature = T,
                          extractMeasFeature = T,
                          labWindow = 35,
                          targetCohortId=NULL,
                          temporalStartDays = c(-35,1),
                          temporalEndDays   = c(-1,35),
                          sampleSize = NULL,
                          outputFolder){
  ParallelLogger::addDefaultFileLogger(file.path(outputFolder, "log.txt"))

  if(is.null(ingredientConceptIds)){
    ParallelLogger::logInfo("Ingrediet concept Ids are generated")

    ingredientList<-listingIngredients(connection=connection,
                                       cdmDatabaseSchema=cdmDatabaseSchema,
                                       vocabularyDatabaseSchema = vocabularyDatabaseSchema,
                                       minimumProportion = minimumProportion,
                                       targetDrugTable = targetDrugTable)
    saveRDs(ingredientList, file = file.path(outputFolder,"ingredientList.rds"))

    ParallelLogger::logInfo("Ingrediet concept Id List was save at ", file.path(outputFolder,"ingredientList.rds"))

    ingredientConceptIds <- ingredientList[[1]]$conceptId
  }

  if(is.null(measurementConceptIds)){
    ParallelLogger::logInfo("Ingrediet concept Ids are generated")
    measurementList<-listingMeasurements(connection=connection,
                                         cdmDatabaseSchema=cdmDatabaseSchema,
                                         vocabularyDatabaseSchema = vocabularyDatabaseSchema,
                                         minimumProportion = minimumProportion)
    saveRDs(measurementList, file = file.path(outfolder,"measurementList.rds"))

    ParallelLogger::logInfo("Measurement concept Id List was save at ", file.path(outputFolder,"measurementList.rds"))

    measurementConceptIds <- measurementList[[1]]$conceptId
  }

  if(createTargetCohort){

    if(is.null(targetCohortId)){
      ParallelLogger::logWarn("Warning: target Cohort Id was set as 9999 automatically")
      targetCohortId <- 9999
    }
    ParallelLogger::logInfo("The cohorts are being generated")

    MvDeconfounder::createMvdCohorts(connection=connection,
                                     cdmDatabaseSchema=cdmDatabaseSchema,
                                     oracleTempSchema = oracleTempSchema,
                                     vocabularyDatabaseSchema = vocabularyDatabaseSchema,
                                     cohortDatabaseSchema=cohortDatabaseSchema,
                                     targetCohortTable=targetCohortTable,
                                     ingredientConceptIds=ingredientConceptIds,
                                     measurementConceptIds=measurementConceptIds,
                                     labWindow = labWindow,
                                     targetCohortId = targetCohortId)

    ParallelLogger::logInfo("Cohort was generated")

  } else {
    if(is.null(targetCohortId)) stop ("You should specify targetCohortId if you don't create the cohort")
  }

  # get features
  if (extractMeasFeature){
    measCovariateSettings <- FeatureExtraction::createTemporalCovariateSettings(useMeasurementValue = TRUE,
                                                                                temporalStartDays = temporalStartDays,
                                                                                temporalEndDays   = temporalEndDays,
                                                                                includedCovariateConceptIds = measurementConceptIds
    )

    ParallelLogger::logInfo("Start generating measurement data ...")
    plpData.meas <- PatientLevelPrediction::getPlpData(connectionDetails = connectionDetails,
                                                       cdmDatabaseSchema = cdmDatabaseSchema,
                                                       cohortDatabaseSchema = cohortDatabaseSchema,
                                                       cohortTable = targetCohortTable,
                                                       cohortId = targetCohortId,
                                                       covariateSettings = measCovariateSettings,
                                                       outcomeDatabaseSchema = cohortDatabaseSchema,
                                                       outcomeTable = targetCohortTable,
                                                       outcomeIds = targetCohortId,
                                                       sampleSize = sampleSize
    )
    PatientLevelPrediction::savePlpData(plpData.meas, file=file.path(outputFolder, 'plpData.meas'), overwrite=TRUE)
    ParallelLogger::logInfo("Measurement data was saved at ",file.path(outputFolder,'plpData.meas'))

    #create sparse measurement matrices
    ##seperate two timeId
    measMappedCov<-MapCovariates (covariates=plpData.meas$covariates,
                                  covariateRef=plpData.meas$covariateRef,
                                  population=plpData.meas$cohorts,
                                  map=NULL)

    preMeasSparseList <- toSparseM(plpData.meas, map=measMappedCov$map, timeId=1)
    postMeasSparseList <- toSparseM(plpData.meas, map=measMappedCov$map, timeId=2)
    measChangeSparseMat <- postMeasSparseList$data - preMeasSparseList$data
    measChangeIndexMat <- preMeasSparseList$index + postMeasSparseList$index

    measChangeIndexMat[measChangeIndexMat!=2]<-0
    measChangeIndexMat[measChangeIndexMat==2]<-1

    Matrix::writeMM(measChangeSparseMat, file=file.path(outputFolder,"measChangeSparseMat.txt"))
    Matrix::writeMM(measChangeIndexMat, file=file.path(outputFolder,"measChangeIndexMat.txt"))
    #save measurement name to a csv file
    measName <- as.matrix(ff::as.ram(plpData.meas$covariateRef$covariateName))
    measName <- gsub(".*: ", "", measName)
    # measName <- gsub(" in .*", "", measName)
    # measName <- gsub("\\s*\\([^\\)]+\\)","", measName)
    # measName <- gsub("\\s*\\[[^\\)]+\\]","", measName)
    write.csv(measName, file=file.path(outputFolder, "measName.csv"))
  } else {
    ParallelLogger::logInfo("Measurement data were not generated.")
  }

  if (extractDrugFeature){
    ParallelLogger::logInfo("Start generating drug data ...")
    drugCovariateSettings <- FeatureExtraction::createCovariateSettings(useDrugEraShortTerm =TRUE,
                                                                        shortTermStartDays = 0,
                                                                        endDays = 0,
                                                                        includedCovariateConceptIds = ingredientConceptIds
    )

    plpData.drug <- PatientLevelPrediction::getPlpData(connectionDetails = connectionDetails,
                                                       cdmDatabaseSchema = cdmDatabaseSchema,
                                                       cohortDatabaseSchema = cohortDatabaseSchema,
                                                       cohortTable = targetCohortTable,
                                                       cohortId = targetCohortId,
                                                       covariateSettings = drugCovariateSettings,
                                                       outcomeDatabaseSchema = cohortDatabaseSchema,
                                                       outcomeTable = targetCohortTable,
                                                       outcomeIds = targetCohortId,
                                                       sampleSize = sampleSize
    )
    PatientLevelPrediction::savePlpData(plpData.drug, file=file.path(outputFolder,'plpData.drug'), overwrite=TRUE)
    ParallelLogger::logInfo("Drug data were saved at ",file.path(outputFolder,'plpData.drug'))


    #create sparse drug matrix
    drugMappedCov<-MapCovariates (covariates=plpData.drug$covariates,
                                  covariateRef=plpData.drug$covariateRef,
                                  population=plpData.drug$cohorts,
                                  map=NULL)
    drugSparseList <- toSparseM(plpData.drug, map=drugMappedCov$map)
    drugName <- as.matrix(ff::as.ram(drugSparseList$covariateRef$covariateName))
    drugName <- gsub(".*: ", "", drugName)
    # find and remove highly correlated drug pairs
    corPairs <- findCollinearFeatures(drugSparseList, threshold=0.8)
    drugToRemove<-unique(corPairs[,2])
    corDrug <- data.frame(drugName[corPairs[,1]], drugName[corPairs[,2]])
    colnames(corDrug)<- c("DrugName1", "DrugName2")
    # remove highly correlated drugs
    drugSparseList$data <- drugSparseList$data[, -c(drugToRemove)]
    drugName <- drugName[-c(drugToRemove)]
    #save drug mat and drug names
    Matrix::writeMM(drugSparseList$data, file=file.path(outputFolder, "drugSparseMat.txt"))
    write.csv(drugName, file=file.path(outputFolder, "drugName.csv"))
  } else {
    ParallelLogger::logInfo("Drug data were not generated.")
  }
  # return(
  #   list(
  #     #plpDataMeas=plpData.meas,
  #     #plpDataDrug = plpData.drug,
  #     measChangeMatrix = measChangeSparseMat,
  #     measChangeIndexMatrix = measChangeIndexMat,
  #     drugMatrix = drugSparseMat)
  # )

}



#' Listing function
#' @export
fitDeconfounder <- function(data_dir,
                            save_dir,
                            factor_model,
                            learning_rate=0.0001,
                            max_steps=100000,
                            latent_dim=1,
                            layer_dim=c(30, 10),
                            batch_size=1024,
                            num_samples=1, # number of samples from variational distribution
                            holdout_portion=0.5,
                            print_steps=50,
                            tolerance=3,
                            num_confounder_samples=30, # number of samples of substitute confounder from the posterior
                            CV=5,
                            outcome_type='linear'){
  e <- environment()
  reticulate::source_python(system.file(package='MvDeconfounder','python','main.py'), envir=e)
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
