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

# devtools::install_github("ohdsi/SqlRender")
# devtools::install_github("ohdsi/DatabaseConnector")
# devtools::install_github("ohdsi/FeatureExtraction")
# devtools::install_github("ohdsi/PatientLevelPrediction")''

options(fftempdir = tempdir())
memory.limit(size=1024*12)

connectionDetails = DatabaseConnector::createConnectionDetails(dbms = "sql server",
                                                               server = "omop.dbmi.columbia.edu")
connection = DatabaseConnector::connect(connectionDetails)

cdmDatabaseSchema = "ohdsi_cumc_deid_pending.dbo"
cohortDatabaseSchema = "ohdsi_cumc_deid_pending.results"
targetCohortTable = "MVDECONFOUNDER_COHORT"
targetCohortId = 1

outputFolder <- "C:/data/MvConfounderV1T1"



mVdData<-generateMvdData(connection=connection,
                          cdmDatabaseSchema=cdmDatabaseSchema,
                          oracleTempSchema = NULL,
                          vocabularyDatabaseSchema = cdmDatabaseSchema,
                          cohortDatabaseSchema=cohortDatabaseSchema,
                          targetCohortTable=targetCohortTable,
                          minimumProportion = 0.001,
                          targetDrugTable = 'DRUG_ERA',
                          ingredientConceptIds=c(967823, 1124957, 1125315, 1177480),
                          measurementConceptIds=as.numeric(c(3013682, 3016723, 3023103, 3015632, 3014576, 3019550, 3006923, 3013721, 3035995, 3006906, 3024128, 3024561, 33000483))
,
                          createTargetCohort = F,
                          labWindow = 35,
                          targetCohortId=targetCohortId,
                          temporalStartDays = c(-35,1),
                          temporalEndDays   = c(-1,35),
                          sampleSize = 100,
                          outputFolder)



# ingredientList<-MvDeconfounder::listingIngredients(connection,
#                                                    cdmDatabaseSchema,
#                                                    vocabularyDatabaseSchema = cdmDatabaseSchema,
#                                                    minimumProportion = 0.001,
#                                                    targetDrugTable = 'DRUG_ERA')
# saveRDS(ingredientList, file="ingredientList.rds")
ingredientList <- readRDS(file="dat/ingredientList.rds")
ingredientConceptIds = ingredientList[[1]]$conceptId
# ingredientConceptIds = as.numeric(c(967823, 1124957, 1125315, 1177480))

# measurementList<-MvDeconfounder::listingMeasurements(connection,
#                                                     cdmDatabaseSchema,
#                                                     vocabularyDatabaseSchema = cdmDatabaseSchema,
#                                                     minimumProportion = 0.001)
# saveRDS(measurementList, file="measurementList.rds")
measurementList <- readRDS(file="dat/measurementList.rds")
measurementConceptIds = measurementList[[1]]$conceptId
# measurementConceptIds = as.numeric(c(3013682, 3016723, 3023103, 3015632, 3014576, 3019550, 3006923, 3013721, 3035995, 3006906, 3024128, 3024561, 33000483))


# MvDeconfounder::createCohorts(connection,
#                               cdmDatabaseSchema,
#                               oracleTempSchema = NULL,
#                               vocabularyDatabaseSchema = cdmDatabaseSchema,
#                               cohortDatabaseSchema,
#                               targetCohortTable,
#                               ingredientConceptIds,
#                               measurementConceptIds,
#                               # drugWindow = 35,
#                               labWindow = 35,
#                               targetCohortId = targetCohortId)

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

options(fftempdir = tempdir())
memory.limit(size=1024*12)
memory.size(max=NA)
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
                                                   sampleSize = 1e+5#NULL
)
PatientLevelPrediction::savePlpData(plpData.meas,file='dat/plpData_meas_sampleSize1e5', overwrite=TRUE)
plpData.meas <- PatientLevelPrediction::loadPlpData(file='dat/plpData_meas_sampleSize1e5')


plpData.drug <- PatientLevelPrediction::getPlpData(connectionDetails = connectionDetails,
                                                   cdmDatabaseSchema = cdmDatabaseSchema,
                                                   cohortDatabaseSchema = cohortDatabaseSchema,
                                                   cohortTable = targetCohortTable,
                                                   cohortId = targetCohortId,
                                                   covariateSettings = drugCovariateSettings,
                                                   outcomeDatabaseSchema = cohortDatabaseSchema,
                                                   outcomeTable = targetCohortTable,
                                                   outcomeIds = targetCohortId,
                                                   sampleSize = 1e+5#NULL
)
PatientLevelPrediction::savePlpData(plpData.drug,file='dat/plpData_drug_sampleSize1e5', overwrite=TRUE)
plpData.drug <- PatientLevelPrediction::loadPlpData(file='dat/plpData_drug_sampleSize1e5')

length(unique(plpData.meas$cohorts$subjectId))
length(unique(plpData.drug$cohorts$subjectId))

#create sparse measurement matrices
##seperate two timeId
measMappedCov<-MapCovariates (covariates=plpData.meas$covariates,
                              covariateRef=plpData.meas$covariateRef,
                              population=plpData.meas$cohorts,
                              map=NULL)

preMeasSparseMat <- toSparseM(plpData.meas, map=measMappedCov$map, timeId=1)
postMeasSparseMat <- toSparseM(plpData.meas, map=measMappedCov$map, timeId=2)
measChangeSparseMat <- postMeasSparseMat$data - preMeasSparseMat$data
measChangeIndexMat <- preMeasSparseMat$index + preMeasSparseMat$index
measChangeIndexMat[measChangeIndexMat!=2]<-0
measChangeIndexMat[measChangeIndexMat==2]<-1
# measChangeSparseMat[indexMat==0]<-NA
Matrix::writeMM(measChangeSparseMat, file="dat/measChangeSparseMat.txt")
Matrix::writeMM(measChangeIndexMat, file="dat/measChangeIndexMat.txt")

#create sparse drug matrix
drugMappedCov<-MapCovariates (covariates=plpData.drug$covariates,
                              covariateRef=plpData.drug$covariateRef,
                              population=plpData.drug$cohorts,
                              map=NULL)
drugSparseMat <- toSparseM(plpData.drug, map=drugMappedCov$map)
Matrix::writeMM(drugSparseMat$data, file="dat/drugSparseMat.txt")

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
