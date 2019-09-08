getOldestMeasData <- function(connection,
                                  oracleTempSchema = NULL,
                                  cdmDatabaseSchema,
                                  cohortTable = "#cohort_person",
                                  cohortId = -1,
                                  cdmVersion = "5",
                                  rowIdField = "subject_id",
                                  covariateSettings,
                                  aggregated = FALSE) {
  writeLines("Constructing the oldest measurement covariates")
  if (covariateSettings$getOldestMeasData == FALSE) {
    return(NULL)
  }
  if (aggregated)
    stop("Aggregation not supported")
  # Some SQL to construct the covariate:
  sql <- "SELECT @row_id_field as row_id, 1 AS covariate_id, meas.NUMBER_AS_VALUE as covariate_value
FROM @cohort_table c
INNER JOIN @cdm_database_schema.measurement meas
ON meas.person_id = c.subject_id,
WHERE cohor_start_date >=
{@cohort_id != -1} ? {AND cohort_definition_id = @cohort_id}

  "
  sql <- paste("SELECT @row_id_field AS row_id, 1 AS covariate_id,",
               "DATEDIFF(DAY, observation_period_start_date, cohort_start_date)",
               "AS covariate_value",
               "FROM @cohort_table c",
               "INNER JOIN @cdm_database_schema.observation_period op",
               "ON op.person_id = c.subject_id",
               "WHERE cohort_start_date >= observation_period_start_date",
               "AND cohort_start_date <= observation_period_end_date",
               "{@cohort_id != -1} ? {AND cohort_definition_id = @cohort_id}")
  sql <- SqlRender::render(sql,
                           cohort_table = cohortTable,
                           cohort_id = cohortId,
                           row_id_field = rowIdField,
                           cdm_database_schema = cdmDatabaseSchema)
  sql <- SqlRender::translate(sql, targetDialect = attr(connection, "dbms"))
  # Retrieve the covariate:
  covariates <- DatabaseConnector::querySql.ffdf(connection, sql)
  # Convert colum names to camelCase:
  colnames(covariates) <- SqlRender::snakeCaseToCamelCase(colnames(covariates))
  # Construct covariate reference:
  covariateRef <- data.frame(covariateId = 1,
                             covariateName = "Length of observation",
                             analysisId = 1,
                             conceptId = 0)
  covariateRef <- ff::as.ffdf(covariateRef)
  # Construct analysis reference:
  analysisRef <- data.frame(analysisId = 1,
                            analysisName = "Length of observation",
                            domainId = "Demographics",
                            startDay = 0,
                            endDay = 0,
                            isBinary = "N",
                            missingMeansZero = "Y")
  analysisRef <- ff::as.ffdf(analysisRef)
  # Construct analysis reference:
  metaData <- list(sql = sql, call = match.call())
  result <- list(covariates = covariates,
                 covariateRef = covariateRef,
                 analysisRef = analysisRef,
                 metaData = metaData)
  class(result) <- "covariateData"
  return(result)
}

