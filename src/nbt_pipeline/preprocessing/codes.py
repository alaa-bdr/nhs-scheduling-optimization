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


def _normalise_text(value) -> str | None:
    if pd.isna(value):
        return None
    value_str = str(value).strip()
    if not value_str:
        return None
    return " ".join(value_str.upper().split())


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
        df["procedure_code_chapter"] = code.str.extract(r"^([A-Z])", expand=False)
        df["procedure_code_group"] = code.str.extract(r"^([A-Z]\d)", expand=False)
        df["procedure_code_format_valid"] = code.str.fullmatch(r"[A-Z]\d{3}")

    df = add_procedure_description_quality_flags(df)

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


def add_procedure_description_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag procedure code-description mappings that are not one-to-one."""
    df = df.copy()
    code_column = "actual_proc_1_procedure_code"
    description_column = "ProcedureDescription"

    if code_column not in df or description_column not in df:
        return df

    code = df[code_column].map(_normalise_code)
    description = df[description_column].map(_normalise_text)

    mapped = pd.DataFrame(
        {
            "procedure_code": code,
            "procedure_description": description,
        }
    ).dropna()

    if mapped.empty:
        df["procedure_code_description_count"] = pd.NA
        df["procedure_description_code_count"] = pd.NA
        df["procedure_code_has_multiple_descriptions"] = False
        df["procedure_description_has_multiple_codes"] = False
        df["procedure_mapping_needs_review"] = False
        return df

    descriptions_per_code = mapped.groupby("procedure_code")["procedure_description"].nunique()
    codes_per_description = mapped.groupby("procedure_description")["procedure_code"].nunique()

    df["procedure_code_description_count"] = code.map(descriptions_per_code)
    df["procedure_description_code_count"] = description.map(codes_per_description)
    df["procedure_code_has_multiple_descriptions"] = df["procedure_code_description_count"].fillna(0) > 1
    df["procedure_description_has_multiple_codes"] = df["procedure_description_code_count"].fillna(0) > 1
    df["procedure_mapping_needs_review"] = (
        df["procedure_code_has_multiple_descriptions"]
        | df["procedure_description_has_multiple_codes"]
    )

    return df


def procedure_description_mapping_quality(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarise procedure code-description mapping quality in both directions."""
    code_column = "actual_proc_1_procedure_code"
    description_column = "ProcedureDescription"

    if code_column not in df or description_column not in df:
        empty = pd.DataFrame()
        return {
            "codes_with_multiple_descriptions": empty,
            "descriptions_with_multiple_codes": empty,
        }

    mapped = pd.DataFrame(
        {
            "procedure_code": df[code_column].map(_normalise_code),
            "procedure_description": df[description_column].map(_normalise_text),
        }
    ).dropna()

    codes_with_multiple_descriptions = (
        mapped
        .groupby("procedure_code")
        .agg(
            unique_descriptions=("procedure_description", "nunique"),
            row_count=("procedure_description", "size"),
            descriptions=("procedure_description", lambda values: "; ".join(sorted(values.unique())[:5])),
        )
        .query("unique_descriptions > 1")
        .sort_values(["unique_descriptions", "row_count"], ascending=False)
        .reset_index()
    )

    descriptions_with_multiple_codes = (
        mapped
        .groupby("procedure_description")
        .agg(
            unique_codes=("procedure_code", "nunique"),
            row_count=("procedure_code", "size"),
            codes=("procedure_code", lambda values: ", ".join(sorted(values.unique())[:10])),
        )
        .query("unique_codes > 1")
        .sort_values(["unique_codes", "row_count"], ascending=False)
        .reset_index()
    )

    return {
        "codes_with_multiple_descriptions": codes_with_multiple_descriptions,
        "descriptions_with_multiple_codes": descriptions_with_multiple_codes,
    }
