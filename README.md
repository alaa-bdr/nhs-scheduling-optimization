# Data Driven Identification and Optimisation of Workflow Bottlenecks in Operating Theatre Scheduling

This repository contains the code, notebooks and report pack for the NHS theatre scheduling group project. The project uses routine operating theatre scheduling data to identify bottlenecks and test whether machine learning can improve operation duration prediction.

## Project aim

The aim is to support theatre planning by answering three practical questions.

1. Which clinical and operational factors are linked with meaningful theatre overruns?
2. Can operation length be predicted better than simple scheduling benchmarks?
3. Which modelling pipeline is safest and most useful for future operational review?

## Data note

The original theatre dataset is internal project data and is not intended for public sharing. The repository keeps reusable code and documentation under version control. Local generated outputs are stored under `result/`, which is ignored by Git because it can contain generated data and large artifacts.

## Main project structure

```text
notebooks/
  nbt_smallset_data_cleaning.ipynb
  nbt_smallset_statistical_analysis.ipynb
  nbt_meaningful_overrun_classification.ipynb
  nbt_duration_error_regression.ipynb
  nbt_operation_length_regression.ipynb
  nbt_modeling_summary.ipynb

src/nbt_pipeline/
  preprocessing/
    clean.py
    codes.py
    features.py
    pipeline.py
    selection.py
  modeling/
    experiments.py
  outputs/
    export.py

tools/
  build_cleaning_notebook.py
  build_statistical_notebook.py
  build_modeling_notebooks.py
  execute_modeling_notebooks.py
  experiment_operation_length_r2.py

docs/
  final_project_organised_report_pack.md
  final_project_organised_report_pack.html
  final_project_organised_report_pack.pdf
```

## Recommended notebook order

1. `nbt_smallset_data_cleaning.ipynb`
2. `nbt_smallset_statistical_analysis.ipynb`
3. `nbt_meaningful_overrun_classification.ipynb`
4. `nbt_duration_error_regression.ipynb`
5. `nbt_operation_length_regression.ipynb`
6. `nbt_modeling_summary.ipynb`

## Key modelling results

| Target | Recommended model | Main result |
|---|---|---|
| Meaningful overrun | XGBoost | Accuracy 81.07 percent and balanced accuracy 80.17 percent |
| Operation length | XGBoost | Selected clean model R squared 0.732349 and MAE 29.37 minutes |
| Duration error | XGBoost | R squared about 0.424 and MAE about 31.11 minutes |

The highest raw sensitivity R squared for operation length was 0.732565 after adding the full procedure code. That setup was not selected as the final recommendation because the gain was extremely small, the MAE was slightly worse at about 29.45 minutes, and the cleaner model is easier to justify.

## Important modelling choices

The selected modelling approach uses:

1. duplicate removal before column selection;
2. leakage control, especially excluding operation end time and outcome derived columns from predictors;
3. missing aware preprocessing;
4. grouped cross validation;
5. benchmark comparison across multiple models;
6. sensitivity testing for start hour, planned duration, procedure detail, consultant fields and flagged records.

The overrun rule is a project working definition. It is not an official NBT policy threshold.

```text
duration error minutes = operation length minutes minus ExpectedDurationMins
tolerance = max(10 minutes, 10 percent of ExpectedDurationMins)
meaningful overrun = duration error minutes greater than tolerance
meaningful underrun = duration error minutes less than negative tolerance
within tolerance = all other cases
```

## Reproducing the project locally

Install dependencies with uv.

```powershell
uv sync
```

Run tests.

```powershell
uv run pytest
```

Run the packaged project pipeline.

```powershell
uv run python -m nbt_pipeline.main
```

This creates the cleaned analysis file in `data/processed/nbt_smallset_analysis_room_only.xlsx` and saves the three final model pipelines under `result/modeling/final_models/`.

Rebuild notebooks from the reusable pipeline scripts.

```powershell
uv run python tools/build_cleaning_notebook.py
uv run python tools/build_statistical_notebook.py
uv run python tools/build_modeling_notebooks.py
```

Execute the modelling notebooks.

```powershell
uv run python tools/execute_modeling_notebooks.py
```

Run the supplementary operation length improvement experiments.

```powershell
uv run python tools/experiment_operation_length_r2.py
```

## Final report pack

The organised report pack is available in three formats.

1. `docs/final_project_organised_report_pack.md`
2. `docs/final_project_organised_report_pack.html`
3. `docs/final_project_organised_report_pack.pdf`

The PDF version includes the key figures, citations, references and the GitHub repository link at the bottom.

## References used in the report

The report cites operating theatre duration prediction literature, XGBoost, CatBoost, scikit learn, pandas and NumPy. It also cites internal project notebooks and result files for project defined measures such as the overrun tolerance rule and within minutes accuracy.

## GitHub repository

https://github.com/alaa-bdr/nhs-scheduling-optimization

