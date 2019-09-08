# devtools::install_github("ohdsi/SqlRender")
# devtools::install_github("ohdsi/DatabaseConnector")
# devtools::install_github("ohdsi/FeatureExtraction")
# devtools::install_github("ohdsi/PatientLevelPrediction")

connectionDetails = DatabaseConnector::createConnectionDetails(dbms = "sql server",
                                             server = "omop.dbmi.columbia.edu")
connection = DatabaseConnector::connect(connectionDetails)

cdmDatabaseSchema = "ohdsi_cumc_deid_pending.dbo"
cohortDatabaseSchema = "ohdsi_cumc_deid_pending.results"
targetCohortTable = "MVDECONFOUNDER_COHORT"
targetCohortId = 1

ingredientList<-MvDeconfounder::listingIngredients(connection,
                           cdmDatabaseSchema,
                           vocabularyDatabaseSchema = cdmDatabaseSchema,
                           minimumProportion = 0.01,
                           targetDrugTable = 'DRUG_ERA')

ingredientConceptIds = ingredientList[[1]]$conceptId
# ingredientConceptIds = as.numeric(c(967823, 1124957, 1125315, 1177480))
measurementConceptIds = as.numeric(c(3013682, 3016723, 3023103, 3015632, 3014576, 3019550, 3006923, 3013721, 3035995, 3006906, 3024128, 3024561, 33000483))


MvDeconfounder::createCohorts(connection,
                              cdmDatabaseSchema,
                              oracleTempSchema = NULL,
                              vocabularyDatabaseSchema = cdmDatabaseSchema,
                              cohortDatabaseSchema,
                              targetCohortTable,
                              ingredientConceptIds,
                              measurementConceptIds,
                              drugWindow = 35,
                              labWindow = 35,
                              targetCohortId = targetCohortId)

sql = "select * from @cohort_database_schema.@target_cohort_table where cohort_definition_id = @target_cohort_id;"
sql<-SqlRender::render(sql,
                       cohort_database_schema = cohortDatabaseSchema,
                       target_cohort_table = targetCohortTable,
                       target_cohort_id = targetCohortId
)
sql<-SqlRender::translate(sql, targetDialect = connectionDetails$dbms)
cohort<-DatabaseConnector::querySql(connection, sql)
# Create unique id: rowId
colnames(cohort)<-SqlRender::snakeCaseToCamelCase(colnames(cohort))
cohort$rowId <- seq(nrow(cohort))

# get features
measCovariateSettings <- FeatureExtraction::createTemporalCovariateSettings(useMeasurementValue = TRUE,
                                                                            temporalStartDays = c(-35,1),
                                                                            temporalEndDays   = c(-1,35),
                                                                            includedCovariateConceptIds = measurementConceptIds
                                                                            )

drugCovariateSettings <- FeatureExtraction::createCovariateSettings(useDrugEraShortTerm =TRUE,
                                                                    shortTermStartDays = 0,
                                                                    endDays = 0,
                                                                    includedCovariateConceptIds = ingredientConceptIds
)

set.seed(1)
plpData.meas <- PatientLevelPrediction::getPlpData(connectionDetails = connectionDetails,
                                                      cdmDatabaseSchema = cdmDatabaseSchema,
                                                      cohortDatabaseSchema = cohortDatabaseSchema,
                                                      cohortTable = targetCohortTable,
                                                      cohortId = targetCohortId,
                                                      covariateSettings = measCovariateSettings,
                                                      outcomeDatabaseSchema = cohortDatabaseSchema,
                                                      outcomeTable = targetCohortTable,
                                                      outcomeIds = targetCohortId,
                                                      sampleSize = NULL
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
                                                   sampleSize = NULL
)

# install.packages('magrittr')
library(dplyr)
library(magrittr)
# compute the change of measurement
measCovariates <- ff::as.ram(plpData.meas$covariates)
measCovariates <- measCovariates[order(measCovariates$rowId, measCovariates$covariateId, measCovariates$timeId),] %>%
  dplyr::group_by(rowId, covariateId) %>%
  dplyr::mutate(change = c(NA, covariateValue[timeId==2] - covariateValue[timeId==1]))

measCovariatesDiff <- measCovariates[complete.cases(measCovariates$change), c("rowId", "covariateId", "change")]


toSparseM <- function(plpData,
                      map=plpMap$map){
  # # check logger
  # if(length(ParallelLogger::getLoggers())==0){
  #   logger <- ParallelLogger::createLogger(name = "SIMPLE",
  #                                          threshold = "INFO",
  #                                          appenders = list(ParallelLogger::createConsoleAppender(layout = ParallelLogger::layoutTimestamp)))
  #   ParallelLogger::registerLogger(logger)
  # }
  cohorts = plpData$cohorts

  ParallelLogger::logDebug(paste0('covariates nrow: ', nrow(plpData$covariates)))
  cov <- plpData$covariates #ff::clone(plpData$covariates)
  ParallelLogger::logDebug(paste0('covariateRef nrow: ', nrow(plpData$covariateRef)))
  covref <- plpData$covariateRef#ff::clone(plpData$covariateRef)

  # plpData.mapped <- MapCovariates(covariates=cov, covariateRef=ff::clone(covref),
  #                                 population, map)

  cov<-ff::as.ram(cov)
  covMapped<-merge(cov2,map,by.x="covariateId", by.y = "oldIds", all =FALSE)

  data <- Matrix::sparseMatrix(i=ff::as.ram(covMapped$rowId),
                               j=ff::as.ram(covMapped$newIds),
                               x=ff::as.ram(covMapped$covariateValue),
                               dims=c(max(covMapped$rowId), max(covMapped$newIds))) # edit this to max(map$newIds)


  result <- list(data=data,
                 covariateRef=covref,
                 map=map)
  return(result)
}


drugMat <- toSparseM(plpData.drug, map=plpMap$map)


#
# drugCovariates <- FeatureExtraction::getDbCovariateData(connectionDetails = connectionDetails,
#                                                         cdmDatabaseSchema = cdmDatabaseSchema,
#                                                         cohortDatabaseSchema = cohortDatabaseSchema,
#                                                         cohortTable = targetCohortTable,
#                                                         cohortId = targetCohortId,
#                                                         covariateSettings = preCovariateSettings,
#                                                         rowIdField = "subject_id")
#
# cov<-ff::as.ram(drugCovariates$covariates)
# length(unique(cov$rowId))
# length(unique(cohort$subjectId))
# cohort[cohort$subjectId==2889209,]
#
#
#
# measCovariates <- FeatureExtraction::getDbCovariateData(connectionDetails = connectionDetails,
#                                                         cdmDatabaseSchema = cdmDatabaseSchema,
#                                                         cohortDatabaseSchema = cohortDatabaseSchema,
#                                                         cohortTable = targetCohortTable,
#                                                         cohortId = targetCohortId,
#                                                         covariateSettings = measCovariateSettings)

# for(drugConceptId in ingredientConceptIds){
#   for (measurementConceptId in measurementConceptIds){
#     MvDeconfounder::createCohorts(connection,
#                                   cdmDatabaseSchema,
#                                   vocabularyDatabaseSchema = cdmDatabaseSchema,
#                                   cohortDatabaseSchema,
#                                   targetCohortTable,
#                                   # oracleTempSchema,
#                                   ingredientConceptIds = ingredientConceptIds,
#                                   measurementConceptIds = measurementConceptIds,
#                                   drugWindow = 35,
#                                   labWindow = 35,
#                                   targetCohortId = drugConceptId)
#     }
# }
