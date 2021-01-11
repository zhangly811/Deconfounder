IF OBJECT_ID('@target_database_schema.@confounder_table', 'U') IS NOT NULL
DROP TABLE @target_database_schema.@confounder_table;

-- age and sex extraction
CREATE TABLE  @target_database_schema.@confounder_table(
    person_id VARCHAR(255),
    cohort_start_date DATE,
    year_of_birth INTEGER,
    gender_concept_id VARCHAR(255)
);

INSERT INTO @target_database_schema.@confounder_table(person_id, cohort_start_date, year_of_birth, gender_concept_id)
SELECT DISTINCT person_id, cohort_start_date, year_of_birth, gender_concept_id
  FROM @target_database_schema.@target_cohort_table c
  JOIN @cdm_database_schema.PERSON p ON c.subject_id = p.person_id
  WHERE c.cohort_definition_id = @target_cohort_id;
