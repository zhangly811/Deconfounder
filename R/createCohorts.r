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

#' Generate cohort
#'
#' @details
#' This function generates cohort for the medical deconfounder (single outcome)
#'
#' @param connection Connection to the database. An object of class connect as created by the connect function in the DatabaseConnector package.
#' @param cdmDatabaseSchema A schema where OMOP CDM data are stored.
#' @param oracleTempSchema            A schema that can be used to create temp tables in when using Oracle.
#' @param vocabularyDatabaseSchema A schema where vocabulary is stored
#' @param cohortDatabaseSchema A schema where the cohort is stored
#' @param targetCohortTable A string corresponds to the name of cohort table
#' @param createTargetCohortTable A boolean that indicates whether the targetCohortTable will be created. Default TRUE.
#' @param conditionConceptIds A list of condition concept IDs that correspond to the disease of interest.
#' @param measurementConceptId A numeric of measurement concept ID
#' @param observationWindowBefore An integer indicates the number of pre-treatment days used to look for pre-treatment measurement.
#' @param observationWindowAfter An integer indicates the number of post-treatment days used to look for post-treatment measurement.
#' @param drugWindow An integer indicates the number of post-treatment days during which drug exposure are also considered. Default is 0
#' @param targetCohortId An integer of the cohort ID.
#' @return
#' Returns a string containing the rendered SQL.
#'
#' @export
createCohorts <- function(
  connection,
  cdmDatabaseSchema,
  oracleTempSchema = NULL,
  vocabularyDatabaseSchema = cdmDatabaseSchema,
  cohortDatabaseSchema,
  targetCohortTable,
  createTargetCohortTable=TRUE,
  conditionConceptIds,
  measurementConceptId,
  observationWindowBefore,
  observationWindowAfter,
  drugWindow,
  targetCohortId
) {

  # Create study cohort table structure:
  if (createTargetCohortTable){
    sql <- SqlRender::loadRenderTranslateSql(sqlFilename = "CreateCohortTable.sql",
                                             packageName = "MvDeconfounder",
                                             dbms = attr(connection, "dbms"),
                                             cohort_table = targetCohortTable,
                                             cohort_database_schema = cohortDatabaseSchema)
    DatabaseConnector::executeSql(connection, sql, progressBar = FALSE, reportOverallTime = FALSE)
  }

  sql <- SqlRender::loadRenderTranslateSql(sqlFilename = "cohort.sql",
                                           packageName = "MvDeconfounder",
                                           dbms = attr(connection, "dbms"),
                                           oracleTempSchema = oracleTempSchema,
                                           cdm_database_schema = cdmDatabaseSchema,
                                           condition_concept_ids = conditionConceptIds,
                                           measurement_concept_id = measurementConceptId,
                                           observation_window_before = observationWindowBefore,
                                           observation_window_after = observationWindowAfter,
                                           drug_window = drugWindow,
                                           target_cohort_id = targetCohortId,
                                           target_cohort_table = targetCohortTable,
                                           target_database_schema = cohortDatabaseSchema,
                                           vocabulary_database_schema = vocabularyDatabaseSchema
  )
  DatabaseConnector::executeSql(connection, sql, progressBar = TRUE, reportOverallTime = TRUE)
}




#' Generate cohort
#'
#' @details
#' This function generates cohort for the medical deconfounder with multiple outcomes.
#'
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
                                           packageName = "MvDeconfounder",
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
