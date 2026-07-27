import pandas as pd


ADMISSION_TYPE_LABELS = {
    "11": "elective_waiting_list",
    "12": "elective_booked",
    "13": "elective_planned",
    "21": "emergency_ae",
    "22": "emergency_gp",
    "23": "emergency_bed_bureau",
    "24": "emergency_outpatient_clinic",
    "25": "emergency_mental_health_crisis",
    "2A": "emergency_other_provider_ae",
    "2B": "emergency_transfer",
    "2C": "emergency_baby_born_at_home",
    "2D": "emergency_other",
    "31": "maternity_antenatal",
    "32": "maternity_postnatal",
    "81": "transfer_from_other_provider",
    "82": "baby_born_in_provider",
    "83": "baby_born_outside_provider",
    "98": "not_applicable",
}

INTENDED_MANAGEMENT_LABELS = {
    "1": "inpatient",
    "2": "day_case",
    "3": "regular_day_attender",
    "4": "regular_night_attender",
    "5": "mother_and_baby",
    "8": "not_applicable",
    "9": "not_known",
}

PRIORITY_LABELS = {
    "P1": "highest_priority",
    "P2": "high_priority",
    "P3": "medium_priority",
    "P4": "routine_or_lower_priority",
}

VALID_CODE_VALUES = {
    "admission_type": set(ADMISSION_TYPE_LABELS),
    "intended_management": set(INTENDED_MANAGEMENT_LABELS),
    "sex_national_code": {"1", "2"},
    "ASAScore": {"1", "2", "3", "4", "5", "6"},
    "PriorityLevelCode": set(PRIORITY_LABELS),
}


def _normalise_code(value) -> str | None:
    if pd.isna(value):
        return None
    value_str = str(value).strip()
    if value_str.endswith(".0"):
        value_str = value_str[:-2]
    return value_str.upper()


def add_code_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable labels for key coded columns."""
    df = df.copy()

    if "admission_type" in df:
        admission_codes = df["admission_type"].map(_normalise_code)
        df["admission_type_label"] = admission_codes.map(ADMISSION_TYPE_LABELS)

    if "intended_management" in df:
        intended_codes = df["intended_management"].map(_normalise_code)
        df["intended_management_label"] = intended_codes.map(INTENDED_MANAGEMENT_LABELS)

    if "PriorityLevelCode" in df:
        priority_codes = df["PriorityLevelCode"].map(_normalise_code)
        df["priority_level_label"] = priority_codes.map(PRIORITY_LABELS)

    if "actual_proc_1_procedure_code" in df:
        code = df["actual_proc_1_procedure_code"].astype("string").str.strip().str.upper()
        df["procedure_code_chapter"] = code.str[0]

    return df


def coded_value_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing and unexpected values in key coded columns."""
    rows = []
    for column, valid_values in VALID_CODE_VALUES.items():
        if column not in df:
            continue

        codes = df[column].map(_normalise_code)
        invalid_mask = codes.notna() & ~codes.isin(valid_values)
        invalid_values = sorted(codes[invalid_mask].dropna().unique().tolist())

        rows.append(
            {
                "column": column,
                "missing_count": codes.isna().sum(),
                "invalid_count": invalid_mask.sum(),
                "invalid_values": ", ".join(invalid_values) if invalid_values else "",
                "valid_values": ", ".join(sorted(valid_values)),
            }
        )

    return pd.DataFrame(rows)
