# Copyright 2019 Observational Health Data Sciences and Informatics
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

    ParallelLogger::logInfo("The cohorts was generated")

  } else {
    if(is.null(targetCohortId)) stop ("You should specify targetCohortId if you don't create the cohorts")
  }

  # sql = "select * from @cohort_database_schema.@target_cohort_table where cohort_definition_id = @target_cohort_id;"
  # sql<-SqlRender::render(sql,
  #                        cohort_database_schema = cohortDatabaseSchema,
  #                        target_cohort_table = targetCohortTable,
  #                        target_cohort_id = targetCohortId
  # )
  # sql<-SqlRender::translate(sql, targetDialect = connectionDetails$dbms)
  # cohort<-DatabaseConnector::querySql(connection, sql)
  # colnames(cohort)<-SqlRender::snakeCaseToCamelCase(colnames(cohort))

  # Create unique id: rowId
  #cohort$rowId <- seq(nrow(cohort))
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

    preMeasSparseMat <- toSparseM(plpData.meas, map=measMappedCov$map, timeId=1)
    postMeasSparseMat <- toSparseM(plpData.meas, map=measMappedCov$map, timeId=2)
    measChangeSparseMat <- postMeasSparseMat$data - preMeasSparseMat$data
    measChangeIndexMat <- preMeasSparseMat$index + postMeasSparseMat$index

    measChangeIndexMat[measChangeIndexMat!=2]<-0
    measChangeIndexMat[measChangeIndexMat==2]<-1

    # measChangeSparseMat[indexMat==0]<-NA
    Matrix::writeMM(measChangeSparseMat, file=file.path(outputFolder,"measChangeSparseMat.txt"))
    Matrix::writeMM(measChangeIndexMat, file=file.path(outputFolder,"measChangeIndexMat.txt"))
    #save measurement name to a csv file
    measName <- as.matrix(ff::as.ram(plpData.meas$covariateRef$covariateName))
    measName <- gsub(".*: ", "", measName)
    measName<-noquote(measName)
    write.csv(measName, file=file.path(outputFolder, "measName.csv"))

  } else {
    ParallelLogger::logInfo("Measurement data were previously generated.")
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
    drugSparseMat <- toSparseM(plpData.drug, map=drugMappedCov$map)
    Matrix::writeMM(drugSparseMat$data, file=file.path(outputFolder, "drugSparseMat.txt"))
    #save drug names to a csv file
    drugName <- as.matrix(ff::as.ram(drugSparseMat$covariateRef$covariateName))
    drugName <- gsub(".*: ", "", drugName)
    drugName<-noquote(drugName)
    write.csv(drugName, file=file.path(outputFolder, "drugName.csv"))
  } else {
    ParallelLogger::logInfo("Drug data were previously generated.")
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
