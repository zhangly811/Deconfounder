# devtools::install_github("ohdsi/SqlRender")
# install.packages('SqlRender')
#
devtools::install_github("ohdsi/DatabaseConnector")
devtools::install_github("ohdsi/FeatureExtraction")
devtools::install_github("ohdsi/PatientLevelPrediction")

connectionDetails = DatabaseConnector::createConnectionDetails(dbms = "sql server",
                                             server = "omop.dbmi.columbia.edu")
connection = DatabaseConnector::connect(connectionDetails)

cdmDatabaseSchema = "ohdsi_cumc_deid_pending.dbo"
cohortDatabaseSchema = "ohdsi_cumc_deid_pending.results"
targetCohortTable = "MVDECONFOUNDER_COHORT"

ingredientList<-MvDeconfounder::listingIngredients(connection,
                           cdmDatabaseSchema,
                           vocabularyDatabaseSchema = cdmDatabaseSchema,
                           minimumProportion = 0.07,
                           targetDrugTable = 'DRUG_ERA')
ingredientConceptIds = c(ingredientList$drugConceptIds['conceptId'])
# ingredientConceptIds = c(967823, 1125315)
ingredientConceptIds = ingredientList[[1]]$conceptId
measurementConceptIds = c(3016723)

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
                              targetCohortId = 1)

sql = "select * from @cohort_database_schema.@target_cohort_table where cohort_definition_id = @target_cohort_id;"
sql<-SqlRender::render(sql,
                       cohort_database_schema = cohortDatabaseSchema,
                       target_cohort_table = targetCohortTable,
                       target_cohort_id = targetCohortId
)
sql<-SqlRender::translate(sql, targetDialect = connectionDetails$dbms)
table<-DatabaseConnector::querySql(connection, sql)
nrow(table)
length(unique(table$SUBJECT_ID))

##get features
preCovariateSettings <- FeatureExtraction::createCovariateSettings(useDrugEraShortTerm =TRUE,
                                                                useMeasurementValueShortTerm = TRUE,
                                                                shortTermStartDays = -35,
                                                                endDays = 0,
                                                                includedCovariateConceptIds = c(ingredientConceptIds,measurementConceptIds),
                                                                addDescendantsToInclude = TRUE)
# FeatureExtraction::createTemporalCovariateSettings
postCovariateSettings <- FeatureExtraction::createCovariateSettings(useMeasurementValueShortTerm = TRUE,
                                                                shortTermStartDays = 35,
                                                                endDays = 1,
                                                                includedCovariateConceptIds = measurementConceptIds,
                                                                addDescendantsToInclude = TRUE)




preCovariates <- FeatureExtraction::getDbCovariateData(connectionDetails = connectionDetails,
                                 cdmDatabaseSchema = cdmDatabaseSchema,
                                 cohortDatabaseSchema = cohortDatabaseSchema,
                                 cohortTable = targetCohortTable,
                                 cohortId = 1,
                                 covariateSettings = preCovariateSettings)

postCovariates <- FeatureExtraction::getDbCovariateData(connectionDetails = connectionDetails,
                                                         cdmDatabaseSchema = cdmDatabaseSchema,
                                                         cohortDatabaseSchema = cohortDatabaseSchema,
                                                         cohortTable = targetCohortTable,
                                                         cohortId = 1,
                                                         covariateSettings = postCovariateSettings)

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
