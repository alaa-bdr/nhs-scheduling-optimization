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
- Raw admission, intended-management and priority codes are removed because their readable label columns are retained.
- `TheatreRoom` is retained as the only location field. `theatre_area`, room prefix, room number and the IR indicator are removed to avoid representing the same location at several overlapping levels.
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
    build_preprocessed_dataset,
    drop_analysis_columns,
    dropped_column_summary,
    missing_summary,
    remove_exact_source_duplicates,
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
**Interpretation.** Seven rows are exact duplicates across all 66 preprocessed source and derived columns. The agreed row-level cleaning decision is to remove these exact repeated records before any columns are discarded. This avoids counting the same complete record twice while preserving rows that only become analytically identical after excluded source details are removed.
"""),
    md("""
### 2.1 Row-level cleaning audit

Exact source-level duplicates are removed reproducibly in the pipeline. The 12 records carrying `duration_timing_review_flag` are **retained**, because the flag identifies values requiring sensitivity analysis rather than proving that those observations are incorrect. They will be compared with and without the flagged records during modelling.
"""),
    code("""
deduplicated_df = remove_exact_source_duplicates(preprocessed_df)

row_cleaning_audit = pd.DataFrame({
    "measure": [
        "rows_before_duplicate_removal",
        "exact_duplicate_rows_removed",
        "rows_after_duplicate_removal",
        "duration_review_records_retained",
    ],
    "value": [
        len(preprocessed_df),
        len(preprocessed_df) - len(deduplicated_df),
        len(deduplicated_df),
        int(deduplicated_df["duration_timing_review_flag"].fillna(False).astype(bool).sum()),
    ],
})
row_cleaning_audit
"""),
    md("""
## 3. Column-exclusion audit

The exclusion list is maintained in one Python module rather than repeated manually in the notebook. This makes the decision reproducible and reduces the risk that future runs drop a different set of columns.

### 3.1 Rationale for removal

Column removal was based on analytical purpose and data provenance rather than missingness alone. The exclusions have the following justification:

| Exclusion group | Reason for removal | Information retained instead |
|---|---|---|
| Unstructured source text | SessionIDdesc and theatre_notes are high-cardinality free text and may contain sensitive or identifying information. Their direct inclusion would require a separately governed and validated text-analysis method. | Structured features created before removal, particularly session_specialty; the source file remains unchanged. |
| Detailed procedure fields | The complete OPCS-like code and procedure description contain more than one thousand categories. Direct use would create sparse predictors and increase overfitting risk. procedure_code_chapter is very broad and overlaps with the more informative derived hierarchy. | procedure_code_group and procedure_code_category provide two documented levels of procedure detail for later comparison. |
| Raw coded values | Raw admission, intended-management and priority codes duplicate their decoded labels. Keeping both versions would represent the same information twice and make model interpretation less clear. | admission_type_label, intended_management_label and priority_level_label. Missing labels remain explicit rather than being guessed. |
| Staff identifiers | Consultant, surgeon and anaesthetist identifiers have unconfirmed local meanings and raise governance, fairness and overfitting concerns. Unadjusted staff comparisons could reflect case mix rather than individual performance. | No person-level predictor is used in the primary analysis dataset. Specialty and procedure information retain relevant case-mix context. |
| Session-derived fields | Session codes, list type and time band were extracted from locally formatted text and were not externally validated. Several also have substantial missingness. | session_specialty is retained because it is interpretable and useful for case-mix analysis, but its derived origin remains a limitation. |
| Theatre representations | Area, prefix, room number and IR indicator overlap with the complete room label. Including several nested location fields can create redundant predictors and obscure which level drives model performance. A room number alone is also ambiguous across areas. | TheatreRoom is retained as the single, most specific location variable. Rare-room handling will occur inside the modelling pipeline. |
| Raw event times | The source event fields do not consistently provide complete clock times and most occur during or after the operation. They are therefore unsuitable as preoperative predictors and could introduce leakage. | Planned duration and other information available before surgery remain. |
| Reconstructed timing and stage durations | These fields depend on the provisional under-one-hour reconstruction logic. NBT could not confirm the source semantics, so they are not sufficiently reliable for the primary dataset. | operation_start_hour alone is retained for a clearly labelled sensitivity model, not the primary model. |
| Timing-validation fields | These flags describe whether the provisional reconstruction succeeded. They are quality-control results rather than preoperative patient or scheduling characteristics. | The reconstruction remains documented in the preprocessing/EDA work but is excluded from primary modelling. |
| Duplicate or superseded outcomes | calculated_operation_length_mins exactly repeats the recorded operation length. The raw is_overrun flag classifies every positive difference, including trivial differences, and is superseded by the tolerance-based outcome definition. | operation_length_mins, duration_error_mins, duration_status, and meaningful overrun/underrun flags are retained for outcome construction and analysis. |

These decisions reduce duplication and leakage without changing the number of cases. They do not imply that every removed field lacks operational meaning; rather, the fields are unsuitable for the primary analysis under the available metadata and governance constraints.
"""),
    code("""
exclusion_groups = {
    "Unstructured source text": ["SessionIDdesc", "theatre_notes"],
    "Procedure detail replaced by group": [
        "actual_proc_1_procedure_code", "ProcedureDescription", "procedure_code_chapter"
    ],
    "Raw codes replaced by readable labels": [
        "admission_type", "intended_management", "PriorityLevelCode"
    ],
    "Staff or consultant identifiers": [
        "listing_cons_code", "theat_surg_1_national_code",
        "theat_anae_1_national_code", "session_consultant"
    ],
    "Unvalidated or redundant session fields": [
        "session_theatre_code", "session_code_prefix", "session_list_type", "session_time_band"
    ],
    "Redundant theatre fields": [
        "theatre_area", "theatre_room_prefix", "theatre_room_number", "theatre_is_ir"
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
    "Duplicate or superseded outcome fields": [
        "calculated_operation_length_mins", "is_overrun"
    ],
}

exclusion_audit = pd.DataFrame(
    [(group, column) for group, columns in exclusion_groups.items() for column in columns],
    columns=["reason_group", "column"],
).merge(dropped_column_summary(preprocessed_df), on="column", how="left")

exclusion_audit
"""),
    md("""
### 3.2 Audit interpretation

`present_before_drop=True` confirms that a named field existed and was removed by the reusable selection function. A false value would not stop the pipeline, because source schemas can change, but it would trigger a review before the dataset is released.

The audit also shows the distinction between **removing a column** and **discarding its underlying information**. For example, raw admission and priority codes are removed only after their readable labels have been created, while detailed room components are removed because the complete `TheatreRoom` value is retained. By contrast, staff identifiers and unvalidated timing stages are deliberately not replaced because their safe interpretation cannot be established from the available dataset.

`operation_start_hour` is the only provisional reconstructed timing variable retained. It will be excluded from the primary model and introduced only in a sensitivity model. Any improvement observed after adding it must therefore be reported as provisional rather than evidence of a confirmed time-of-day effect.
"""),
    md("""
## 4. Create the cleaned analysis dataset

The reusable pipeline now applies the documented exclusions after all retained features have been created.
"""),
    code("""
analysis_df = drop_analysis_columns(deduplicated_df)

selection_summary = pd.DataFrame({
    "measure": [
        "rows_before", "rows_after", "columns_before", "columns_after",
        "columns_removed", "excluded_columns_still_present"
    ],
    "value": [
        len(deduplicated_df), len(analysis_df), deduplicated_df.shape[1], analysis_df.shape[1],
        preprocessed_df.shape[1] - analysis_df.shape[1],
        len(set(ANALYSIS_DROP_COLUMNS).intersection(analysis_df.columns)),
    ],
})
selection_summary
"""),
    code("""
assert len(preprocessed_df) - len(deduplicated_df) == 7
assert len(analysis_df) == len(deduplicated_df), "Column selection unexpectedly changed the row count."
assert not set(ANALYSIS_DROP_COLUMNS).intersection(analysis_df.columns)
assert "SessionIDdesc" not in analysis_df.columns
assert "theatre_notes" not in analysis_df.columns
assert "operation_start_hour" in analysis_df.columns

analysis_df.columns.to_frame(index=False, name="retained_column")
"""),
    md("""
**Interpretation.** Seven exact source-level duplicate records were removed before column selection. Column selection itself changes only the schema and deletes no further cases. Some rows can have identical values across the final 22 analysis columns because excluded source fields previously distinguished them; these are not silently removed. The two free-text fields are absent, all documented exclusions have been removed, and `operation_start_hour` remains available for explicitly labelled sensitivity analysis.
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

`operation_length_mins`, `duration_error_mins`, `overrun_minutes`, `underrun_minutes`, `duration_tolerance_mins`, `meaningful_overrun_flag`, `meaningful_underrun_flag`, `duration_status`, and duration-review flags.

The final modelling pipeline should create predictors and targets separately. `operation_start_hour` should be tested only in a sensitivity model because its reconstruction has not been validated by NBT.
"""),
    code("""
outcome_or_leakage_columns = [
    "operation_length_mins", "duration_error_mins", "overrun_minutes", "underrun_minutes",
    "duration_tolerance_mins",
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
OUTPUT_PATH = PROJECT_ROOT / "result" / "nbt_smallset_analysis_room_only.xlsx"
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

1. Compare models with and without the 12 retained duration-review records.
2. Define explicit predictor and target datasets to prevent outcome leakage.
3. Use a documented complete-case modelling population rather than replacing unknown clinical values with assumed values.
4. Encode observed categorical values within cross-validation or training folds only.
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
