"""Leakage-safe utilities for operation-duration-error regression."""

from __future__ import annotations

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


RANDOM_STATE = 42
TARGET_COLUMN = "duration_error_mins"
SENSITIVITY_COLUMN = "operation_start_hour"

# These fields contain the recorded outcome or are calculated from it. They may
# be used for reporting or target construction, but never as model predictors.
OUTCOME_OR_LEAKAGE_COLUMNS = (
    "operation_length_mins",
    "duration_error_mins",
    "overrun_minutes",
    "underrun_minutes",
    "duration_tolerance_mins",
    "meaningful_overrun_flag",
    "meaningful_underrun_flag",
    "duration_status",
    "duration_timing_review_flag",
)


def make_predictor_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return predictors and finite targets for rows eligible for regression."""
    if TARGET_COLUMN not in df:
        raise ValueError(f"Required target column is missing: {TARGET_COLUMN}")

    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    eligible = target.notna() & np.isfinite(target)
    target = target.loc[eligible].copy()
    modeling_df = df.loc[eligible].copy()
    sensitivity = modeling_df.drop(
        columns=[column for column in OUTCOME_OR_LEAKAGE_COLUMNS if column in df.columns]
    ).copy()
    for column in sensitivity.columns:
        if is_numeric_dtype(sensitivity[column]):
            sensitivity[column] = pd.to_numeric(sensitivity[column], errors="coerce").astype(float)
        else:
            values = sensitivity[column].astype(object)
            sensitivity[column] = values.where(pd.notna(values), np.nan)
    primary = sensitivity.drop(columns=[SENSITIVITY_COLUMN], errors="ignore").copy()

    if target.empty:
        raise ValueError("No rows have a finite regression target.")
    if set(OUTCOME_OR_LEAKAGE_COLUMNS).intersection(primary.columns):
        raise AssertionError("Outcome leakage columns remain in the predictor matrix.")
    return primary, sensitivity, target


def predictor_row_groups(predictors: pd.DataFrame) -> pd.Series:
    """Hash predictor rows so identical cases remain in the same data split."""
    normalized = predictors.astype("string").fillna("<MISSING>")
    rows = pd.Series(
        list(normalized.itertuples(index=False, name=None)),
        index=predictors.index,
        dtype="object",
    )
    codes, _ = pd.factorize(rows, sort=False)
    return pd.Series(codes, index=predictors.index, dtype="int64")


def build_preprocessor() -> ColumnTransformer:
    """Create fold-fitted numeric and categorical preprocessing."""
    numeric = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ]
    )


def build_model_pipeline(model: RegressorMixin) -> Pipeline:
    """Keep preprocessing and estimation together to prevent fold leakage."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def get_regression_models() -> dict[str, RegressorMixin]:
    """Return the benchmark and five prespecified regression models."""
    return {
        "Dummy median benchmark": DummyRegressor(strategy="median"),
        "Linear regression": LinearRegression(),
        "Decision tree": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=10, random_state=RANDOM_STATE
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=3,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "SVR": SVR(C=10.0, epsilon=0.1, kernel="rbf"),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
    }
