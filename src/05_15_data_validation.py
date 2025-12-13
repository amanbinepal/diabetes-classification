# 05_15_data_validation.py
# Author: Vy Phan
# Date: 2025-12-11

# This script goes through the following data validation checks:
# Correct data file format
# Correct column names
# No empty observations
# Missingness not beyond expected threshold
# Correct data types in each column
# No duplicate observations
# No outlier or anomalous values
# Correct category levels (i.e., no string mismatches or single values)
# Target/response variable follows expected distribution
# No anomalous correlations between target/response variable and features/explanatory variables
# No anomalous correlations between features/explanatory variables


import pandas as pd
import click
import pointblank as pb
from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import ClassImbalance, FeatureLabelCorrelation, FeatureFeatureCorrelation

TRAIN_CSV = "data/processed/train_df.csv"

def load_data():
    return pd.read_csv(TRAIN_CSV)


def check_column_count(df):
    validation = pb.Validate(data=df).col_count_match(len(df.columns)).interrogate()
    save_result(validation, "data_validation1_1")


def check_row_count(df, original_csv="data/raw/diabetes_binary_health_indicators_BRFSS2015.csv"):
    original_data = pd.read_csv(original_csv)
    rows, _ = original_data.shape
    expected_rows = int(rows * 0.8)
    validation = pb.Validate(data=df).row_count_match(expected_rows).interrogate()
    save_result(validation, "data_validation1_2")


def check_required_columns(df):
    expected_columns = ['ID', 'Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'BMI',
                        'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
                        'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                        'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
                        'Education', 'Income']
    validation = pb.Validate(data=df).col_exists(columns=expected_columns).interrogate()
    save_result(validation, "data_validation2")


def check_no_missing_rows(df):
    validation = pb.Validate(data=df).rows_complete().interrogate()
    save_result(validation, "data_validation3")


def check_no_missing_columns(df):
    validator = pb.Validate(data=df)
    for col in df.columns:
        validator = validator.col_vals_not_null(columns=str(col), thresholds=1)
    validation = validator.interrogate()
    save_result(validation, "data_validation4")


def check_column_data_types(df):
    schema_columns = [(col, "int64") for col in df.columns]
    schema = pb.Schema(columns=schema_columns)
    validation = pb.Validate(data=df).col_schema_match(schema=schema).interrogate()
    save_result(validation, "data_validation5")


def check_no_duplicates(df):
    validation = pb.Validate(data=df).rows_distinct(columns_subset=["ID"]).interrogate()
    save_result(validation, "data_validation6")


def check_bmi_range(df):
    validation = pb.Validate(data=df).col_vals_between(columns="BMI", left=10, right=100).interrogate()
    save_result(validation, "data_validation7")


def check_ordinal_ranges(df):
    ordinal_features = ["GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    for f in ordinal_features:
        temp_col = df[f]
        print(f"Feature: {f}, dtype: {temp_col.dtype}, value range: {temp_col.sort_values().unique()}")


def check_binary_and_categorical_levels(df):
    binary_features = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'Smoker',
                       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
                       'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                       'DiffWalk', 'Sex']
    validation = (
        pb.Validate(data=df)
        .col_vals_in_set(columns=binary_features, set=[0, 1])
        .col_vals_in_set(columns="GenHlth", set=list(range(1, 6)))
        .col_vals_between(columns=["MentHlth", "PhysHlth"], left=0, right=30)
        .col_vals_in_set(columns="Age", set=list(range(1, 14)))
        .col_vals_in_set(columns="Education", set=list(range(1, 7)))
        .col_vals_in_set(columns="Income", set=list(range(1, 9)))
        .interrogate()
    )
    save_result(validation, "data_validation9")


def deepchecks_validation(df):
    binary_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 
                       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits',
                       'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
                       'DiffWalk', 'Sex']

    deep_train = Dataset(df.drop(columns=['ID']), label="Diabetes_binary", cat_features=binary_features)
    suite = Suite(
        "Validation",
        ClassImbalance(),
        FeatureLabelCorrelation(correlation_threshold=0.5),
        FeatureFeatureCorrelation(correlation_threshold=0.7),
    )

    result = suite.run(deep_train)
    save_result(result, "data_validation10", deepchecks=True)


def save_result(validation, name, deepchecks=False):
    if deepchecks:
        passed = validation.passed()
    else:
        passed = validation.all_passed()
    file_name = f"src/objects/pass_{name}.txt" if passed else f"src/objects/failed_{name}.txt"
    pd.DataFrame({"result": ["PASS" if passed else "FAILED"]}).to_csv(file_name, index=False)
    if not passed:
        print(f"Validation {name} failed!")

if __name__ == "__main__":
    df = load_data()
    check_column_count(df)
    check_row_count(df)
    check_required_columns(df)
    check_no_missing_rows(df)
    check_no_missing_columns(df)
    check_column_data_types(df)
    check_no_duplicates(df)
    check_bmi_range(df)
    check_ordinal_ranges(df)  # prints value ranges; no pass/fail
    check_binary_and_categorical_levels(df)
    deepchecks_validation(df)
    click.echo(
        "-------------Data Validation Complete-------------\nSaved:\nValidation .txt scripts under objects folder\n"
    )
