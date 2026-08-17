import pandas as pd


# Columns excluded from the cleaned analysis dataset. The original source file is
# never modified; source fields needed for feature engineering are dropped only
# after all retained features have been created.
ANALYSIS_DROP_COLUMNS = (
    # Unstructured or identifying source text.
    "SessionIDdesc",
    "theatre_notes",
    # Raw or high-cardinality procedure fields replaced by procedure_code_group.
    "actual_proc_1_procedure_code",
    "ProcedureDescription",
    "procedure_code_chapter",
    # Raw coded values replaced by their readable label columns.
    "admission_type",
    "intended_management",
    "PriorityLevelCode",
    # Staff identifiers and unvalidated consultant text.
    "listing_cons_code",
    "theat_surg_1_national_code",
    "theat_anae_1_national_code",
    "session_consultant",
    # Unvalidated or redundant session features.
    "session_theatre_code",
    "session_code_prefix",
    "session_list_type",
    "session_time_band",
    # Redundant theatre representations. Keep only the specific TheatreRoom.
    "theatre_area",
    "theatre_room_prefix",
    "theatre_room_number",
    "theatre_is_ir",
    # Raw event times. Keep operation_start_hour as a provisional sensitivity feature.
    "into_theatre",
    "anaesthetic_start_time",
    "incision",
    "closure",
    "out_of_theatre",
    "operation_end_time",
    "recovery_time",
    # Provisional reconstructed timestamps and stage-duration features.
    "into_theatre_inferred",
    "operation_end_time_inferred",
    "anaesthetic_start_time_inferred",
    "incision_inferred",
    "closure_inferred",
    "operation_start_hour_band",
    "post_operation_theatre_time_mins",
    "theatre_occupancy_mins",
    "theatre_to_anaesthetic_start_mins",
    "anaesthetic_to_incision_mins",
    "incision_to_closure_mins",
    "closure_to_operation_end_mins",
    # Validation fields tied to the provisional reconstruction.
    "operation_length_rule_valid",
    "time_sequence_valid",
    "time_reconstruction_status",
    # Exact or less informative outcome duplicates retained through better fields.
    "calculated_operation_length_mins",
    "is_overrun",
)


def drop_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned analysis view after documented column exclusions."""
    present_columns = [column for column in ANALYSIS_DROP_COLUMNS if column in df.columns]
    return df.drop(columns=present_columns).copy()


def remove_exact_source_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact records before analytical columns are discarded."""
    return df.drop_duplicates().reset_index(drop=True)


def dropped_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Report whether each planned exclusion was present in the input dataset."""
    return pd.DataFrame(
        {
            "column": ANALYSIS_DROP_COLUMNS,
            "present_before_drop": [column in df.columns for column in ANALYSIS_DROP_COLUMNS],
        }
    )
