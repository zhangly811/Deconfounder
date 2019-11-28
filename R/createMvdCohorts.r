# Copyright 2019 Observational Health Data Sciences and Informatics
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
#' This function generates cohort for the multivariate deconfounder
#'
#' @param connection         Name of local folder where the results were generated; make sure to use forward slashes
#'                             (/). Do not use a folder on a network drive since this greatly impacts
#'                             performance.
#' @param cdmDatabaseSchema     How many parallel cores should be used? If more cores are made
#'                              available this can speed up the analyses.
#'
#' @export
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
  sql <- SqlRender::loadRenderTranslateSql(sqlFilename = "cohort.sql",
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
