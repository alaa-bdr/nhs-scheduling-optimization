# Repository Guide

This guide explains where the main project evidence is stored.

## Cleaning and preprocessing

`src/nbt_pipeline/preprocessing/selection.py` defines the selected analysis columns and excluded columns.

`src/nbt_pipeline/preprocessing/features.py` defines operation length, duration error, tolerance, meaningful overrun, meaningful underrun and duration status.

`notebooks/nbt_smallset_data_cleaning.ipynb` documents duplicate removal, column exclusion and missingness after cleaning.

## Statistical analysis

`notebooks/nbt_smallset_statistical_analysis.ipynb` contains the main exploratory and statistical evidence for bottlenecks, including duration status, procedure group patterns, theatre room pressure and missingness analysis.

## Modelling

`src/nbt_pipeline/modeling/experiments.py` defines the reusable modelling pipeline, approved predictors, leakage columns, missing value strategies, model comparisons and final selected model specifications.

`src/nbt_pipeline/main.py` is the packaged end to end pipeline. Running `uv run python -m nbt_pipeline.main` rebuilds the cleaned analysis file in `data/processed/nbt_smallset_analysis_room_only.xlsx` and saves the three final model pipelines under `result/modeling/final_models/`.

The modelling notebooks are:

1. `notebooks/nbt_meaningful_overrun_classification.ipynb`
2. `notebooks/nbt_duration_error_regression.ipynb`
3. `notebooks/nbt_operation_length_regression.ipynb`
4. `notebooks/nbt_modeling_summary.ipynb`

## Generated result files

Generated result files are written under `result/`. This folder is ignored by Git because outputs can be large and can contain generated data. The key local result files used by the report are:

1. `result/modeling/cross_target_winner_summary.csv`
2. `result/modeling/operation_length_mins/final_test_metrics.csv`
3. `result/modeling/operation_length_mins/r2_experiments/operation_length_r2_experiments.csv`
4. `result/modeling/operation_length_mins/r2_experiments/full_procedure_code_sensitivity_experiments.csv`
5. `result/modeling/operation_length_mins/permutation_importance.csv`
6. `result/modeling/meaningful_overrun_flag/final_test_metrics.csv`
7. `result/modeling/duration_error_mins/final_test_metrics.csv`

## Report pack

The final report example is stored in `docs/`.

1. `docs/final_project_organised_report_pack.md`
2. `docs/final_project_organised_report_pack.html`
3. `docs/final_project_organised_report_pack.pdf`

The report pack should be treated as an example and evidence guide. The submitted group report should be checked by the group and written in the group voice.
