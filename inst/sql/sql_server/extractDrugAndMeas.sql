IF OBJECT_ID('@target_database_schema.@drug_exposure_table', 'U') IS NOT NULL
DROP TABLE @target_database_schema.@drug_exposure_table;
IF OBJECT_ID('@target_database_schema.@measurement_table', 'U') IS NOT NULL
DROP TABLE @target_database_schema.@measurement_table;

CREATE TABLE @target_database_schema.@drug_exposure_table(
    subject_id VARCHAR(255),
    cohort_start_date DATE,
    drug_exposure_start_date DATE,
    drug_concept_id VARCHAR(255),
    ancestor_concept_id VARCHAR(255),
    concept_name VARCHAR(255)
);

INSERT INTO @target_database_schema.@drug_exposure_table(subject_id, cohort_start_date, drug_exposure_start_date, drug_concept_id, ancestor_concept_id, concept_name)

SELECT DISTINCT SUBJECT_ID, cohort_start_date, drug_exposure_start_date, drug_concept_id, ancestor_concept_id, concept_name FROM @target_database_schema.@target_cohort_table
JOIN @cdm_database_schema.drug_exposure ON @target_cohort_table.SUBJECT_ID = @cdm_database_schema.drug_exposure.person_id
JOIN @cdm_database_schema.concept_ancestor ON drug_exposure.drug_concept_id = @cdm_database_schema.concept_ancestor.descendant_concept_id
JOIN @cdm_database_schema.concept ON @cdm_database_schema.concept_ancestor.ancestor_concept_id=@cdm_database_schema.concept.concept_id
WHERE COHORT_DEFINITION_ID = @target_cohort_id
and @cdm_database_schema.drug_exposure.drug_exposure_start_date <= DATEADD(day, @drug_window, CAST(@target_database_schema.@target_cohort_table.cohort_start_date AS DATE))
and @cdm_database_schema.drug_exposure.drug_exposure_start_date >= @target_database_schema.@target_cohort_table.cohort_start_date
and drug_concept_id != 0
and concept_class_id = 'Ingredient';


-- measurement extraction
CREATE TABLE  @target_database_schema.@measurement_table(
    person_id VARCHAR(255),
    cohort_start_date DATE,
    measurement_concept_id VARCHAR(255),
    measurement_date DATE,
    value_as_number FLOAT,
    concept_name VARCHAR(255)
);

INSERT INTO @target_database_schema.@measurement_table(person_id, cohort_start_date, measurement_concept_id, measurement_date, value_as_number, concept_name)
SELECT DISTINCT person_id, cohort_start_date, measurement_concept_id, measurement_date, value_as_number, concept_name FROM @target_database_schema.@target_cohort_table
JOIN @cdm_database_schema.measurement ON @target_cohort_table.SUBJECT_ID = @cdm_database_schema.measurement.person_id
JOIN @cdm_database_schema.concept ON @cdm_database_schema.measurement.measurement_concept_id = @cdm_database_schema.concept.concept_id
WHERE COHORT_DEFINITION_ID = @target_cohort_id
and measurement_concept_id = @measurement_concept_id
and @cdm_database_schema.measurement.measurement_date > DATEADD(day, -@observation_window_before, CAST(@target_database_schema.@target_cohort_table.cohort_start_date AS DATE))
and @cdm_database_schema.measurement.measurement_date <= DATEADD(day, @observation_window_after, CAST(@target_database_schema.@target_cohort_table.cohort_start_date AS DATE))
and domain_id = 'Measurement'
and standard_concept = 'S';


-- age and sex extraction
CREATE TABLE  @target_database_schema.@confounder_table(
    person_id VARCHAR(255),
    cohort_start_date DATE,
    year_of_birth DATE,
    gender_concept_id VARCHAR(255)
);
INSERT INTO @target_database_schema.@confounder_table(person_id, cohort_start_date, year_of_birth, gender_concept_id)
SELECT DISTINCT p.person_id, c.cohort_start_date, p.year_of_birth, p.gender_concept_id
  FROM @target_database_schema.@target_cohort_table c, @cdm_database_schema.PERSON p
  WHERE c.cohort_definition_id = @target_cohort_id
  AND c.subject_id = p.person_id
  AND standard_concept = 'S';
