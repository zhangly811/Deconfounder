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

listingIngredients <- function(connection,
                               cdmDatabaseSchema,
                               vocabularyDatabaseSchema = cdmDatabaseSchema,
                               minimumProportion = 0.05,
                               outputFolder){
  sql<-"SELECT COUNT(PERSON_ID) as total_person_count FROM @cdmDatabaseSchema.PERSON"
  
  
  totalPersonCount 
  
  limitNumber = totalPersonCount * minimumProportion
  
  sql <- "select drug.drug_concept_id, COUNT(distinct person_id)
  from
          @cdmDatabaseSchema.drug_era drug
          JOIN @vocabularyDatabaseSchema.concept concept
          on drug.drug_concept_id = concept.concept_id AND concept.invalid_reason is NULL AND concept.standard_concept = 'S'
  GROUP BY drug.drug_concept_id
  HAVING COUNT(distinct person_id) > @limit_number
  "
  
  sql <- SqlRender::loadRenderTranslateSql(sqlFilename = "cohort.sql",
                                           packageName = "MvConfounder",
                                           dbms = attr(connection, "dbms"),
                                           oracleTempSchema = oracleTempSchema,
                                           cdm_database_schema = cdmDatabaseSchema,
                                           ingredient_concept_id = ingredientConceptId,
                                           measurement_concept_ids= measurementConceptIds,
                                           drug_window = drugWindow,
                                           lab_window = labWindow,
                                           target_cohort_id = targetCohortId,
                                           cohort_table = cohortTable,
                                           target_database_schmea = cohortDatabaseSchema,
                                           vocabulary_database_schema=vocabularyDatabaseSchema)
}