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


#' Generate list of concept Ids of interest
#'
#' @details
#' This function
#'
#' @param connection         Name of local folder where the results were generated; make sure to use forward slashes
#'                             (/). Do not use a folder on a network drive since this greatly impacts
#'                             performance.
#' @param cdmDatabaseSchema     How many parallel cores should be used? If more cores are made
#'                              available this can speed up the analyses.
#'
#' @export
listingMeasurements <- function(connection,
                               cdmDatabaseSchema,
                               vocabularyDatabaseSchema = cdmDatabaseSchema,
                               minimumProportion = 0.01){

  #extract whole number of population from the database
  sql<-"SELECT COUNT(PERSON_ID) as total_person_count FROM @cdm_database_schema.PERSON;"
  sql<-SqlRender::render(sql,
                         cdm_database_schema = cdmDatabaseSchema)
  sql<-SqlRender::translate(sql, targetDialect = connectionDetails$dbms)

  totalPersonCount<-DatabaseConnector::querySql(connection, sql)
  colnames(totalPersonCount)<-SqlRender::snakeCaseToCamelCase(colnames(totalPersonCount))

  limitNumber = round(as.numeric(totalPersonCount) * minimumProportion,0)


  #extract list of measurement concept Ids from measurement table
  sql <- "select concept.concept_id, concept.concept_name, COUNT(distinct person_id) AS person_count
  from
  @cdm_database_schema.MEASUREMENT meas
  JOIN @vocabulary_database_schema.concept concept
  on meas.measurement_concept_id = concept.concept_id AND concept.invalid_reason is NULL AND concept.standard_concept = 'S'
  GROUP BY concept.concept_id, concept.concept_name
  HAVING COUNT(distinct person_id) > @limit_number
  "
  sql<-SqlRender::render(sql,
                         cdm_database_schema = cdmDatabaseSchema,
                         vocabulary_database_schema = vocabularyDatabaseSchema,
                         limit_number = limitNumber
  )
  sql<-SqlRender::translate(sql, targetDialect = connectionDetails$dbms)
  measConceptIds<-DatabaseConnector::querySql(connection, sql)
  colnames(measConceptIds)<-SqlRender::snakeCaseToCamelCase(colnames(measConceptIds))

  return(list(measConceptIds=measConceptIds,limitNumber=limitNumber))
}
