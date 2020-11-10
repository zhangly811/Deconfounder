
from __future__ import print_function
import pandas as pd
import numpy as np
import os
import re



def read_data(DATA_PATH, measFilename, drugFilename):
    lab = pd.read_csv(os.path.join(DATA_PATH, measFilename), index_col=0)
    lab.columns = ['subject_id', 'cohort_start_date', 'measurement_concept_id', 'measurement_date', 'value_as_number',
                   'concept_name']
    drug = pd.read_csv(os.path.join(DATA_PATH, drugFilename), index_col=0)
    drug.columns = ['subject_id', 'cohort_start_date', 'drug_exposure_start_date', 'drug_concept_id',
                    'ancestor_concept_id', 'concept_name']
    return lab, drug

def remove_vaccines(drug):
    # only drugs prescribed on cohort start date
    #     drug = drug[drug['cohort_start_date'] == drug['drug_exposure_start_date']]

    # remove vaccines
    drug_to_remove = ['vaccine', 'Vaccine', 'antigen', 'virus']
    pattern = '|'.join(drug_to_remove)
    mask = drug['concept_name'].str.contains(pattern, re.IGNORECASE)
    drug = drug[~mask]

    return drug

def extract_closest_before_and_after_lab(lab):
    # if multiple values exist within a day, take the mean
    lab = lab.groupby(['subject_id', 'cohort_start_date', \
                       'measurement_concept_id', 'measurement_date', 'concept_name'])[
        'value_as_number'].mean().reset_index()
    lab = lab.sort_values(['subject_id', 'measurement_date'], ascending=[True, True])

    lab_before = lab[lab['measurement_date'] <= lab['cohort_start_date']].sort_values(
        ['subject_id', 'measurement_date'], ascending=[True, True])
    lab_before = lab_before.groupby('subject_id').tail(1)

    lab_after = lab[lab['measurement_date'] > lab['cohort_start_date']].sort_values(['subject_id', 'measurement_date'],
                                                                                    ascending=[True, True])
    lab_after = lab_after.groupby('subject_id').head(1)

    # find common patients in the two dfs
    common = \
        set.intersection(set(lab_before.subject_id), set(lab_after.subject_id))

    lab_before = lab_before[lab_before.subject_id.isin(common)][['subject_id', 'cohort_start_date', 'value_as_number']]
    lab_after = lab_after[lab_after.subject_id.isin(common)][['subject_id', 'cohort_start_date', 'value_as_number']]
    return lab_before, lab_after


def preprocessing(DATA_PATH, measFilename, drugFilename):
    # order drugs by time within patient
    lab, drug = read_data(DATA_PATH, measFilename, drugFilename)

    drug = drug.sort_values(['subject_id', 'drug_exposure_start_date'], ascending=[True, True])
    drug['cohort_start_date'] = pd.to_datetime(drug['cohort_start_date'], format="%Y-%m-%d")
    drug['drug_exposure_start_date'] = pd.to_datetime(drug['drug_exposure_start_date'], format="%Y-%m-%d")

    lab = lab.sort_values(['subject_id', 'measurement_date'], ascending=[True, True])
    lab['cohort_start_date'] = pd.to_datetime(lab['cohort_start_date'], format="%Y-%m-%d")
    lab['measurement_date'] = pd.to_datetime(lab['measurement_date'], format="%Y-%m-%d")
    # remove NA in lab values
    lab.dropna(subset=['value_as_number'], inplace=True)

    drug = remove_vaccines(drug)
    lab_before, lab_after = extract_closest_before_and_after_lab(lab)


    # remove rare drugs (<5%)
    drug_occurrence_counts = drug.groupby('ancestor_concept_id')['subject_id'].nunique()
    mask = drug_occurrence_counts.values > drug['subject_id'].nunique() * 0.05
    drug_occurrence_counts_reduced = drug_occurrence_counts[mask]
    drug = drug[drug['ancestor_concept_id'].isin(drug_occurrence_counts_reduced.index)]

    print("Number of unique drugs", drug['ancestor_concept_id'].nunique())

    # find common patients in the two dfs: drug and lab
    common = \
        set.intersection(set(lab_before.subject_id), set(drug.subject_id))
    drug = drug[drug.subject_id.isin(common)]
    lab_before = lab_before[lab_before.subject_id.isin(common)]
    lab_after = lab_after[lab_after.subject_id.isin(common)]

    print("Number of patients in drug: {}; in lab_before: {}; in lab_after: {}".format(drug['subject_id'].nunique(), \
                                                                                       lab_before['subject_id'].nunique(), \
                                                                                       lab_after['subject_id'].nunique()))

    # Create sparse matrix
    drug_with_counts = drug.groupby(["subject_id", "concept_name"]).size().reset_index(name="Time")
    X_df = pd.pivot_table(drug_with_counts, values='Time', index='subject_id', columns='concept_name').fillna(0).astype(int)
    X_df[X_df.values != 0] = 1

    # Create correlation matrix
    corr_matrix = X_df.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.6
    to_drop = [str(column) for column in upper.columns if any(upper[column] > 0.6)]
    print("Highly correlated drugs to drop:", to_drop)
    # Drop features
    X_df.drop(X_df.columns[X_df.columns.isin(to_drop)], axis=1, inplace=True)

    # sparsity of the matrix
    sparsity = np.count_nonzero(X_df) / (X_df.shape[0] * X_df.shape[1]) * 100
    print("Sparsity of the matrix {} %".format(100 - sparsity))

    X_df.to_csv(os.path.join(DATA_PATH, 'drug_exposure_sparse_matrix.csv'))
    lab_before.to_csv(os.path.join(DATA_PATH, "pre_treatment_lab.csv"), index=False)
    lab_after.to_csv(os.path.join(DATA_PATH, "post_treatment_lab.csv"), index=False)

