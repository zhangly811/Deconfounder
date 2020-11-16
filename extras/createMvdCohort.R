
createMvdCohorts <- function(connection,
                             cdmDatabaseSchema,
                             oracleTempSchema = NULL,
                             vocabularyDatabaseSchema = cdmDatabaseSchema,
                             cohortDatabaseSchema,
                             targetCohortTable,
                             ingredientConceptIds,
                             measurementConceptIds,
                             # drugWindow = 35,
                             labWindow = 35,
                             targetCohortId) {
  ingredientConceptIds<-paste(ingredientConceptIds,collapse=",")
  measurementConceptIds<-paste(measurementConceptIds,collapse=",")

  # Create study cohort table structure:
  sql <- SqlRender::loadRenderTranslateSql(sqlFilename = "MvdCohort.sql",
                                           packageName = "Deconfounder",
                                           dbms = attr(connection, "dbms"),
                                           oracleTempSchema = oracleTempSchema,
                                           cdm_database_schema = cdmDatabaseSchema,
                                           ingredient_concept_ids = ingredientConceptIds,
                                           measurement_concept_ids= measurementConceptIds,
                                           #drug_window = drugWindow,
                                           lab_window = labWindow,
                                           target_cohort_id = targetCohortId,
                                           target_cohort_table = targetCohortTable,
                                           target_database_schema = cohortDatabaseSchema,
                                           vocabulary_database_schema=vocabularyDatabaseSchema)
  DatabaseConnector::executeSql(connection, sql, progressBar = TRUE, reportOverallTime = FALSE)
}
