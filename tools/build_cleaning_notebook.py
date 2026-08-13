"""Build the reproducible NBT data-cleaning notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "nbt_smallset_data_cleaning.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md("""
# NBT theatre data cleaning

This notebook creates the cleaned analysis dataset used after preprocessing and exploratory analysis. The original Excel source is never overwritten. Cleaning and column selection are performed through reusable functions in `src/nbt_pipeline/preprocessing`, so the same rules can be applied consistently whenever the source data are refreshed.

The main purpose of this stage is to remove fields that are unsuitable for the planned analysis because they are unstructured, identifying, redundant, unavailable before an operation, or derived from an unconfirmed time-reconstruction rule. Removal takes place only after feature engineering, because retained fields such as `session_specialty` and `operation_start_hour` are derived from source columns that are later excluded.
"""),
    md("""
## 1. Cleaning decisions

The cleaned analysis dataset follows these principles:

- `SessionIDdesc` and `theatre_notes` are dropped completely from the exported analysis dataset. The original source remains available separately.
- Staff identifiers and the unvalidated consultant text are excluded for governance, interpretability and overfitting reasons.
- High-cardinality procedure text is replaced by the more stable `procedure_code_group` feature.
- `theatre_area` and `TheatreRoom` are retained, while redundant room-prefix, room-number and IR indicators are removed.
- Raw event times and provisional reconstructed stage durations are excluded.
- `operation_start_hour` is retained only as a provisional sensitivity feature. It must not be treated as a confirmed operational timestamp.
- Outcome columns remain available for analysis and target construction, but must not be used as predictors.

No missing clinical category is guessed or automatically replaced with a more common category.
"""),
    code("""
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nbt_pipeline.outputs import save_dataframe
from nbt_pipeline.preprocessing import (
    ANALYSIS_DROP_COLUMNS,
    build_analysis_dataset,
    build_preprocessed_dataset,
    dropped_column_summary,
    missing_summary,
)

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 120)
"""),
    md("""
## 2. Build the feature-engineered working dataset

This intermediate dataframe is created in memory from the source dataset. It is used to calculate retained structured features before the unsuitable source and derived columns are removed.
"""),
    code("""
preprocessed_df = build_preprocessed_dataset()

pd.DataFrame({
    "measure": ["rows", "columns", "exact_duplicate_rows"],
    "value": [len(preprocessed_df), preprocessed_df.shape[1], preprocessed_df.duplicated().sum()],
})
"""),
    md("""
**Interpretation.** Row count is preserved at this point. Exact duplicates are reported for audit rather than silently removed in the column-selection step. Duplicate handling should be implemented as a separate, documented cleaning decision after confirming whether repeated rows represent duplicate records or valid repeated cases.
"""),
    md("""
## 3. Column-exclusion audit

The exclusion list is maintained in one Python module rather than repeated manually in the notebook. This makes the decision reproducible and reduces the risk that future runs drop a different set of columns.
"""),
    code("""
exclusion_groups = {
    "Unstructured source text": ["SessionIDdesc", "theatre_notes"],
    "Procedure detail replaced by group": [
        "actual_proc_1_procedure_code", "ProcedureDescription", "procedure_code_chapter"
    ],
    "Staff or consultant identifiers": [
        "listing_cons_code", "theat_surg_1_national_code",
        "theat_anae_1_national_code", "session_consultant"
    ],
    "Unvalidated or redundant session fields": [
        "session_theatre_code", "session_code_prefix", "session_list_type", "session_time_band"
    ],
    "Redundant theatre fields": [
        "theatre_room_prefix", "theatre_room_number", "theatre_is_ir"
    ],
    "Raw event times": [
        "into_theatre", "anaesthetic_start_time", "incision", "closure",
        "out_of_theatre", "operation_end_time", "recovery_time"
    ],
    "Provisional reconstructed timing": [
        "into_theatre_inferred", "operation_end_time_inferred",
        "anaesthetic_start_time_inferred", "incision_inferred", "closure_inferred",
        "operation_start_hour_band", "post_operation_theatre_time_mins",
        "theatre_occupancy_mins", "theatre_to_anaesthetic_start_mins",
        "anaesthetic_to_incision_mins", "incision_to_closure_mins",
        "closure_to_operation_end_mins"
    ],
    "Provisional timing-validation fields": [
        "operation_length_rule_valid", "time_sequence_valid", "time_reconstruction_status"
    ],
}

exclusion_audit = pd.DataFrame(
    [(group, column) for group, columns in exclusion_groups.items() for column in columns],
    columns=["reason_group", "column"],
).merge(dropped_column_summary(preprocessed_df), on="column", how="left")

exclusion_audit
"""),
    md("""
**Interpretation.** `present_before_drop=True` confirms that a named field existed and was available for removal. A false value would not stop the pipeline; it would indicate that the source schema or feature pipeline had changed and should be reviewed. `operation_start_hour` is deliberately absent from this table because it is retained as a provisional sensitivity variable.
"""),
    md("""
## 4. Create the cleaned analysis dataset

The reusable pipeline now applies the documented exclusions after all retained features have been created.
"""),
    code("""
analysis_df = build_analysis_dataset()

selection_summary = pd.DataFrame({
    "measure": [
        "rows_before", "rows_after", "columns_before", "columns_after",
        "columns_removed", "excluded_columns_still_present"
    ],
    "value": [
        len(preprocessed_df), len(analysis_df), preprocessed_df.shape[1], analysis_df.shape[1],
        preprocessed_df.shape[1] - analysis_df.shape[1],
        len(set(ANALYSIS_DROP_COLUMNS).intersection(analysis_df.columns)),
    ],
})
selection_summary
"""),
    code("""
assert len(analysis_df) == len(preprocessed_df), "Column selection unexpectedly changed the row count."
assert not set(ANALYSIS_DROP_COLUMNS).intersection(analysis_df.columns)
assert "SessionIDdesc" not in analysis_df.columns
assert "theatre_notes" not in analysis_df.columns
assert "operation_start_hour" in analysis_df.columns

analysis_df.columns.to_frame(index=False, name="retained_column")
"""),
    md("""
**Interpretation.** Column selection changes the analytical schema without deleting cases. The two free-text fields are absent, all documented exclusions have been removed, and `operation_start_hour` remains available for explicitly labelled sensitivity analysis. The resulting dataframe is an analysis dataset, not yet a leakage-safe predictor matrix.
"""),
    md("""
## 5. Missingness after column selection

Missing values are retained rather than replaced with guessed clinical categories. This table supports later decisions about explicit missing categories, missingness indicators and train-only numerical imputation.
"""),
    code("""
missing_summary(analysis_df)
"""),
    md("""
**Interpretation.** Missingness alone is not a sufficient reason to remove a clinically relevant field. In later modelling, categorical missingness can be represented explicitly, while any numerical imputation must be fitted on training data only to prevent information leakage. Priority and other incomplete clinical fields should therefore be evaluated through sensitivity analysis rather than filled by assumption.
"""),
    md("""
## 6. Outcome and predictor separation

The cleaned analysis file intentionally retains outcome columns for descriptive analysis and target construction. The following fields must be excluded from the predictor matrix because they contain actual or outcome-derived information:

`operation_length_mins`, `calculated_operation_length_mins`, `duration_error_mins`, `is_overrun`, `overrun_minutes`, `underrun_minutes`, `duration_tolerance_mins`, `meaningful_overrun_flag`, `meaningful_underrun_flag`, `duration_status`, and duration-review flags.

The final modelling pipeline should create predictors and targets separately. `operation_start_hour` should be tested only in a sensitivity model because its reconstruction has not been validated by NBT.
"""),
    code("""
outcome_or_leakage_columns = [
    "operation_length_mins", "calculated_operation_length_mins", "duration_error_mins",
    "is_overrun", "overrun_minutes", "underrun_minutes", "duration_tolerance_mins",
    "meaningful_overrun_flag", "meaningful_underrun_flag", "duration_status",
    "duration_timing_review_flag", "duration_timing_review_reason",
]

pd.DataFrame({
    "column": outcome_or_leakage_columns,
    "present_for_analysis_or_target_construction": [
        column in analysis_df.columns for column in outcome_or_leakage_columns
    ],
    "allowed_as_model_predictor": False,
})
"""),
    md("""
## 7. Export

The cleaned analysis dataset is written to the project `result` directory. The original source file remains unchanged.
"""),
    code("""
OUTPUT_PATH = PROJECT_ROOT / "result" / "nbt_smallset_analysis.xlsx"
save_dataframe(analysis_df, OUTPUT_PATH)

pd.DataFrame({
    "output_path": [str(OUTPUT_PATH)],
    "rows": [len(analysis_df)],
    "columns": [analysis_df.shape[1]],
})
"""),
    md("""
## 8. Next cleaning tasks

Before modelling:

1. Confirm and remove exact duplicate records only after checking their record-level meaning.
2. Apply the established review flag to the 12 questionable duration records; retain them in the audit dataset and compare models with and without them.
3. Define explicit predictor and target datasets to prevent outcome leakage.
4. Encode categorical variables and perform numerical imputation within cross-validation or training folds only.
5. Compare a primary model without `operation_start_hour` against a sensitivity model that includes it.

These steps should be implemented in the reusable pipeline and accompanied by a cleaning report recording every row and column decision.
"""),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook.metadata["language_info"] = {"name": "python", "version": "3"}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Built {OUTPUT}")
