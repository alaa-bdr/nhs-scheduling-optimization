"""Improved feature preparation for the v2 modelling round.

Adds three things beyond the v1 feature set:
1. Out of fold target encoding for high cardinality columns such as
   procedure code and surgeon, computed without leaking the target.
2. A stated duration parsed from the free text theatre notes, which
   clinicians often record explicitly as "90 mins" or "Duration = 240".
3. Simple text volume features from the notes.

All features remain preoperative. Nothing here uses intraoperative timing.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from nbt_pipeline.preprocessing import build_preprocessed_dataset

TARGET = "operation_length_mins"
RANDOM_SEED = 42

BASE_FEATURES = [
    "ExpectedDurationMins",
    "age_at_operation",
    "ASAScore",
    "sex_national_code",
    "admission_type",
    "intended_management",
    "PriorityLevelCode",
    "anaesthetic_desc",
    "procedure_code_chapter",
    "procedure_code_group",
    "theatre_area",
    "TheatreRoom",
    "session_specialty",
    "session_time_band",
]

TARGET_ENCODE_COLUMNS = [
    "actual_proc_1_procedure_code",
    "theat_surg_1_national_code",
    "listing_cons_code",
    "ProcedureDescription",
]


def parse_stated_duration(text: str) -> float:
    """Pull an explicitly stated duration out of a theatre note."""
    s = str(text).lower()
    m = re.search(r"duration\s*=\s*(\d{2,3})", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d{1,3})\s*m(?:in|ins|nins)\b", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d(?:\.\d)?)\s*(?:hrs|hours|hr|h)\b", s)
    if m:
        return float(m.group(1)) * 60
    return np.nan


def add_out_of_fold_target_encoding(df, columns, target, n_splits=5, min_count=5):
    """Encode categories by their mean target, computed out of fold.

    Computing the mean on the same rows the model trains on would leak the
    target, so each row receives an encoding derived only from other folds.
    """
    out = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    global_mean = out[target].mean()

    for col in columns:
        if col not in out.columns:
            continue
        encoded = pd.Series(np.nan, index=out.index)
        for train_idx, holdout_idx in kf.split(out):
            stats = out.iloc[train_idx].groupby(col)[target].agg(["mean", "count"])
            stats = stats[stats["count"] >= min_count]
            encoded.iloc[holdout_idx] = (
                out.iloc[holdout_idx][col].map(stats["mean"]).values
            )
        out[f"te_{col}"] = encoded.fillna(global_mean)

    return out


def build_v2_dataset(duration_cap=None):
    """Return the modelling frame, the feature list, and the target name."""
    df = build_preprocessed_dataset()
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    if duration_cap is not None:
        df = df[df[TARGET] <= duration_cap].reset_index(drop=True)

    available_te = [c for c in TARGET_ENCODE_COLUMNS if c in df.columns]
    df = add_out_of_fold_target_encoding(df, available_te, TARGET)

    notes = df["theatre_notes"].fillna("").astype(str) if "theatre_notes" in df.columns else pd.Series([""] * len(df))
    df["stated_duration_mins"] = notes.map(parse_stated_duration)
    df["notes_character_count"] = notes.str.len()
    df["notes_word_count"] = notes.str.split().str.len()

    features = [c for c in BASE_FEATURES if c in df.columns]
    features += [f"te_{c}" for c in available_te]
    features += ["stated_duration_mins", "notes_character_count", "notes_word_count"]

    return df, features, TARGET


if __name__ == "__main__":
    frame, feats, target = build_v2_dataset()
    print(f"Rows: {len(frame)}")
    print(f"Features: {len(feats)}")
    for f in feats:
        print(f"  {f}")
    stated = frame["stated_duration_mins"]
    print(f"\nNotes with a stated duration: {stated.notna().sum()} ({stated.notna().mean():.1%})")
