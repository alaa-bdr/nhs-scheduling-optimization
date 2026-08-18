"""Shared, leakage-safe utilities for the three supervised modelling targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


RANDOM_STATE = 42
START_HOUR_COLUMN = "operation_start_hour"
REVIEW_FLAG_COLUMN = "duration_timing_review_flag"
CLASSIFICATION_TARGET = "meaningful_overrun_flag"
REGRESSION_TARGETS = ("operation_length_mins", "duration_error_mins")

OUTCOME_OR_LEAKAGE_COLUMNS = (
    "operation_length_mins",
    "duration_error_mins",
    "overrun_minutes",
    "underrun_minutes",
    "duration_tolerance_mins",
    "meaningful_overrun_flag",
    "meaningful_underrun_flag",
    "duration_status",
    REVIEW_FLAG_COLUMN,
)

APPROVED_PREDICTORS = (
    "ExpectedDurationMins",
    "sex_national_code",
    "age_at_operation",
    "ASAScore",
    "anaesthetic_desc",
    "admission_type_label",
    "intended_management_label",
    "priority_level_label",
    "procedure_code_category",
    "procedure_code_group",
    "session_specialty",
    "TheatreRoom",
)

CLINICAL_PREDICTORS = (
    "ExpectedDurationMins",
    "sex_national_code",
    "age_at_operation",
    "ASAScore",
    "anaesthetic_desc",
    "admission_type_label",
    "intended_management_label",
    "priority_level_label",
)


@dataclass(frozen=True)
class TargetData:
    predictors: pd.DataFrame
    target: pd.Series
    review_flag: pd.Series
    excluded_target_rows: int


def _normalize_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if is_numeric_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
        else:
            values = result[column].astype(object)
            result[column] = values.where(pd.notna(values), np.nan)
    return result


def prepare_target_data(df: pd.DataFrame, target: str) -> TargetData:
    """Create an eligible target population with only approved predictors."""
    if target not in {CLASSIFICATION_TARGET, *REGRESSION_TARGETS}:
        raise ValueError(f"Unsupported target: {target}")
    if target not in df.columns:
        raise ValueError(f"Target column is missing: {target}")

    y = pd.to_numeric(df[target], errors="coerce")
    eligible = y.notna() & np.isfinite(y)
    if target == CLASSIFICATION_TARGET:
        eligible &= y.isin([0, 1])
    available = [column for column in (*APPROVED_PREDICTORS, START_HOUR_COLUMN) if column in df]
    X = _normalize_predictors(df.loc[eligible, available])
    y = y.loc[eligible].astype(int if target == CLASSIFICATION_TARGET else float)
    review = df.loc[eligible, REVIEW_FLAG_COLUMN].fillna(False).astype(bool)

    forbidden = set(OUTCOME_OR_LEAKAGE_COLUMNS).intersection(X.columns)
    if forbidden:
        raise AssertionError(f"Leakage columns entered predictors: {sorted(forbidden)}")
    return TargetData(X, y, review, int((~eligible).sum()))


def predictor_row_groups(predictors: pd.DataFrame) -> pd.Series:
    """Hash rows so identical predictor profiles cannot cross split boundaries."""
    normalized = predictors.astype("string").fillna("<MISSING>")
    rows = pd.Series(
        list(normalized.itertuples(index=False, name=None)),
        index=predictors.index,
        dtype="object",
    )
    codes, _ = pd.factorize(rows, sort=False)
    return pd.Series(codes, index=predictors.index, dtype="int64")


def feature_configurations(include_priority: bool = True) -> dict[str, list[str]]:
    """Return prespecified in-memory predictor configurations."""
    clinical = list(CLINICAL_PREDICTORS)
    if not include_priority:
        clinical.remove("priority_level_label")
    procedure_group = clinical + ["procedure_code_group"]
    procedure_category = clinical + ["procedure_code_category"]
    procedure_both = clinical + ["procedure_code_group", "procedure_code_category"]
    full = procedure_both + ["session_specialty", "TheatreRoom"]
    return {
        "Expected duration only": ["ExpectedDurationMins"],
        "Clinical and planning": clinical,
        "Procedure group": procedure_group,
        "Procedure category": procedure_category,
        "Both procedure levels": procedure_both,
        "Full approved": full,
        "Full without location": [column for column in full if column != "TheatreRoom"],
    }


def build_preprocessor(
    predictors: pd.DataFrame,
    *,
    missing_strategy: str,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build fold-fitted preprocessing for missing-aware or complete-case data."""
    if missing_strategy not in {"missing_aware", "complete_case"}:
        raise ValueError(f"Unknown missing strategy: {missing_strategy}")
    numeric_columns = [column for column in predictors if is_numeric_dtype(predictors[column])]
    categorical_columns = [column for column in predictors if column not in numeric_columns]

    numeric_steps = []
    if missing_strategy == "missing_aware":
        numeric_steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True)))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = []
    if missing_strategy == "missing_aware":
        categorical_steps.append(
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing/not recorded"))
        )
    categorical_steps.append(
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", Pipeline(categorical_steps), categorical_columns),
        ],
        remainder="drop",
    )


def build_supervised_pipeline(
    model,
    predictors: pd.DataFrame,
    *,
    missing_strategy: str = "missing_aware",
) -> Pipeline:
    """Bind preprocessing and estimation so transformations remain fold-local."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(predictors, missing_strategy=missing_strategy)),
            ("model", model),
        ]
    )


def regression_models() -> dict[str, object]:
    return {
        "Dummy median benchmark": DummyRegressor(strategy="median"),
        "Linear regression": LinearRegression(),
        "Decision tree": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=10, random_state=RANDOM_STATE
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=120,
            max_depth=16,
            min_samples_leaf=3,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "SVR": SVR(C=10.0, epsilon=1.0, kernel="rbf"),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "Neural network": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            random_state=RANDOM_STATE,
        ),
    }


def classification_models(y: pd.Series) -> dict[str, object]:
    positives = max(int(y.eq(1).sum()), 1)
    scale_pos_weight = int(y.eq(0).sum()) / positives
    return {
        "Dummy prevalence benchmark": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "SVC": SVC(C=2.0, kernel="rbf", class_weight="balanced", probability=True),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "Neural network": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=128,
            learning_rate_init=0.001,
            max_iter=250,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            random_state=RANDOM_STATE,
        ),
    }


def regression_parameter_grids() -> dict[str, dict[str, list[object]]]:
    return {
        "Linear regression": {"model__fit_intercept": [True, False]},
        "Decision tree": {"model__max_depth": [6, 12], "model__min_samples_leaf": [5, 20]},
        "Random forest": {"model__max_depth": [12, None], "model__min_samples_leaf": [2, 5]},
        "SVR": {"model__C": [1.0, 10.0], "model__epsilon": [1.0, 5.0]},
        "XGBoost": {"model__max_depth": [3, 6], "model__learning_rate": [0.03, 0.08]},
        "Neural network": {
            "model__hidden_layer_sizes": [(128, 64), (128, 64, 32)],
            "model__alpha": [0.0001, 0.001],
        },
    }


def classification_parameter_grids() -> dict[str, dict[str, list[object]]]:
    return {
        "Logistic regression": {"model__C": [0.1, 1.0, 10.0]},
        "Decision tree": {"model__max_depth": [5, 10], "model__min_samples_leaf": [10, 30]},
        "Random forest": {"model__max_depth": [10, None], "model__min_samples_leaf": [2, 5]},
        "SVC": {"model__C": [0.5, 2.0], "model__gamma": ["scale", "auto"]},
        "XGBoost": {"model__max_depth": [3, 6], "model__learning_rate": [0.03, 0.08]},
        "Neural network": {
            "model__hidden_layer_sizes": [(128, 64), (128, 64, 32)],
            "model__alpha": [0.0001, 0.001],
        },
    }


def aggregate_tree_importance(pipeline: Pipeline, original_columns: list[str]) -> pd.DataFrame:
    """Aggregate one-hot tree importance back to original predictor columns."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError("The fitted model does not expose tree feature importance.")
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(names) != len(importances):
        raise AssertionError("Transformed feature names and importances have different lengths.")

    ordered_columns = sorted(original_columns, key=len, reverse=True)
    rows = []
    for transformed_name, importance in zip(names, importances, strict=True):
        suffix = transformed_name.split("__", 1)[-1]
        original = next(
            (
                column
                for column in ordered_columns
                if suffix == column
                or suffix.startswith(f"{column}_")
                or suffix == f"missingindicator_{column}"
            ),
            "unmapped",
        )
        rows.append({"feature": original, "importance": importance})
    return (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False, ignore_index=True)
    )
