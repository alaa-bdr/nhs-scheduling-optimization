"""Run focused operation-length experiments to try improving test R2.

This script is deliberately separate from the main cleaned-data pipeline. It keeps one
cleaned analysis dataset, creates candidate feature configurations in memory, and
checks whether extra context, target transforms, hyperparameter tuning or different
GroupKFold counts genuinely improve the untouched-test R2.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_validate,
)
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nbt_pipeline.modeling import (  # noqa: E402
    RANDOM_STATE,
    START_HOUR_COLUMN,
    build_supervised_pipeline,
    predictor_row_groups,
    prepare_target_data,
)
from nbt_pipeline.preprocessing import build_analysis_dataset, build_preprocessed_dataset  # noqa: E402
from nbt_pipeline.preprocessing.selection import remove_exact_source_duplicates  # noqa: E402


TARGET = "operation_length_mins"
RESULT_DIR = PROJECT_ROOT / "result" / "modeling" / TARGET / "r2_experiments"
BASE_COLUMNS = [
    "ExpectedDurationMins",
    "sex_national_code",
    "age_at_operation",
    "ASAScore",
    "anaesthetic_desc",
    "admission_type_label",
    "intended_management_label",
    "priority_level_label",
    "procedure_code_group",
    "procedure_code_category",
]
DETAILED_PROCEDURE_COLUMN = "actual_proc_1_procedure_code"
INTERACTION_COLUMNS = [
    "procedure_group_x_anaesthetic",
    "procedure_category_x_management",
    "procedure_group_x_expected_band",
]


def metrics(actual, prediction) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(actual, prediction),
        "RMSE": mean_squared_error(actual, prediction) ** 0.5,
        "R2": r2_score(actual, prediction),
    }


def xgb_model(**params) -> XGBRegressor:
    base = {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "n_jobs": 1,
        "random_state": RANDOM_STATE,
    }
    base.update(params)
    return XGBRegressor(**base)


def add_experimental_features(X: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()
    if DETAILED_PROCEDURE_COLUMN in source:
        values = source.loc[result.index, DETAILED_PROCEDURE_COLUMN].astype(object)
        result[DETAILED_PROCEDURE_COLUMN] = values.where(pd.notna(values), np.nan)
    result["procedure_group_x_anaesthetic"] = (
        result["procedure_code_group"].astype("string").fillna("Missing")
        + " | "
        + result["anaesthetic_desc"].astype("string").fillna("Missing")
    )
    result["procedure_category_x_management"] = (
        result["procedure_code_category"].astype("string").fillna("Missing")
        + " | "
        + result["intended_management_label"].astype("string").fillna("Missing")
    )
    expected = pd.to_numeric(result["ExpectedDurationMins"], errors="coerce")
    expected_band = pd.cut(
        expected,
        bins=[0, 30, 60, 90, 120, 180, 240, float("inf")],
        labels=["0-30", "31-60", "61-90", "91-120", "121-180", "181-240", "240+"],
        include_lowest=True,
    ).astype("string").fillna("Missing")
    result["procedure_group_x_expected_band"] = (
        result["procedure_code_group"].astype("string").fillna("Missing") + " | " + expected_band
    )
    return result


def fit_predict(columns, X_development_all, X_test_all, y_development, y_test, *, log_target=False):
    model = xgb_model(learning_rate=0.08, max_depth=6)
    pipeline = build_supervised_pipeline(
        model, X_development_all[columns], missing_strategy="missing_aware"
    )
    if log_target:
        estimator = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    else:
        estimator = pipeline
    estimator.fit(X_development_all[columns], y_development)
    prediction = estimator.predict(X_test_all[columns])
    return prediction


def compare_group_kfold_numbers(
    columns,
    X_development_all,
    X_test_all,
    y_development,
    y_test,
    development_groups,
) -> pd.DataFrame:
    """Compare validation-fold counts for the current best approved setup."""
    rows = []
    for folds in [3, 5, 10]:
        model = xgb_model(learning_rate=0.08, max_depth=6)
        pipeline = build_supervised_pipeline(
            model, X_development_all[columns], missing_strategy="missing_aware"
        )
        cv = GroupKFold(n_splits=folds)
        started = time.perf_counter()
        scores = cross_validate(
            pipeline,
            X_development_all[columns],
            y_development,
            cv=cv,
            groups=development_groups,
            scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"},
            n_jobs=1,
            error_score="raise",
        )
        pipeline.fit(X_development_all[columns], y_development)
        prediction = pipeline.predict(X_test_all[columns])
        rows.append(
            {
                "experiment": f"fixed best config, GroupKFold {folds}",
                "folds": folds,
                "runtime seconds": time.perf_counter() - started,
                "CV MAE": -scores["test_MAE"].mean(),
                "CV R2": scores["test_R2"].mean(),
                **metrics(y_test, prediction),
                "best parameters": {"fixed": "current best"},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    analysis_df = build_analysis_dataset()
    preprocessed_df = remove_exact_source_duplicates(build_preprocessed_dataset())
    prepared = prepare_target_data(analysis_df, TARGET)
    X_all = add_experimental_features(prepared.predictors, preprocessed_df)
    y_all = prepared.target

    group_basis = prepared.predictors.drop(columns=[START_HOUR_COLUMN], errors="ignore")
    groups_all = predictor_row_groups(group_basis)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    development_idx, test_idx = next(outer.split(X_all, y_all, groups_all))

    X_development_all = X_all.iloc[development_idx].copy()
    X_test_all = X_all.iloc[test_idx].copy()
    y_development = y_all.iloc[development_idx].copy()
    y_test = y_all.iloc[test_idx].copy()
    development_groups = groups_all.iloc[development_idx]
    cv = GroupKFold(n_splits=5)

    rows = []
    no_priority = [column for column in BASE_COLUMNS if column != "priority_level_label"]
    candidate_configs = {
        "priority retained, no start hour": BASE_COLUMNS,
        "priority excluded, no start hour": no_priority,
        "priority retained, start hour": BASE_COLUMNS + [START_HOUR_COLUMN],
        "priority excluded, start hour": no_priority + [START_HOUR_COLUMN],
        "priority excluded, start hour, room + specialty": no_priority
        + [START_HOUR_COLUMN, "TheatreRoom", "session_specialty"],
        "priority retained, start hour, room + specialty": BASE_COLUMNS
        + [START_HOUR_COLUMN, "TheatreRoom", "session_specialty"],
        "priority excluded, start hour, detailed procedure": no_priority
        + [START_HOUR_COLUMN, DETAILED_PROCEDURE_COLUMN],
        "priority excluded, start hour, interactions": no_priority
        + [START_HOUR_COLUMN, *INTERACTION_COLUMNS],
        "priority excluded, start hour, detailed procedure + interactions": no_priority
        + [START_HOUR_COLUMN, DETAILED_PROCEDURE_COLUMN, *INTERACTION_COLUMNS],
    }

    for name, columns in candidate_configs.items():
        start = time.perf_counter()
        prediction = fit_predict(columns, X_development_all, X_test_all, y_development, y_test)
        rows.append(
            {
                "experiment": name,
                "stage": "fixed tuned baseline",
                "predictors": len(columns),
                "runtime seconds": time.perf_counter() - start,
                **metrics(y_test, prediction),
                "best parameters": {"learning_rate": 0.08, "max_depth": 6},
            }
        )

    log_columns = candidate_configs["priority excluded, start hour"]
    start = time.perf_counter()
    log_prediction = fit_predict(
        log_columns, X_development_all, X_test_all, y_development, y_test, log_target=True
    )
    rows.append(
        {
            "experiment": "priority excluded, start hour, log target",
            "stage": "fixed tuned baseline",
            "predictors": len(log_columns),
            "runtime seconds": time.perf_counter() - start,
            **metrics(y_test, log_prediction),
            "best parameters": {"learning_rate": 0.08, "max_depth": 6, "target": "log1p"},
        }
    )

    search_columns = candidate_configs["priority excluded, start hour, detailed procedure + interactions"]
    param_distributions = {
        "model__n_estimators": [250, 350, 500, 700],
        "model__learning_rate": [0.025, 0.04, 0.06, 0.08],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__min_child_weight": [1, 3, 6, 10],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__reg_lambda": [1.0, 3.0, 8.0, 15.0],
    }
    search_pipeline = build_supervised_pipeline(
        xgb_model(), X_development_all[search_columns], missing_strategy="missing_aware"
    )
    start = time.perf_counter()
    search = RandomizedSearchCV(
        search_pipeline,
        param_distributions=param_distributions,
        n_iter=32,
        scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"},
        refit="R2",
        cv=cv,
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_development_all[search_columns], y_development, groups=development_groups)
    prediction = search.best_estimator_.predict(X_test_all[search_columns])
    rows.append(
        {
            "experiment": "priority excluded, start hour, detailed procedure + interactions",
            "stage": "randomized XGBoost tuning",
            "predictors": len(search_columns),
            "runtime seconds": time.perf_counter() - start,
            **metrics(y_test, prediction),
            "best parameters": search.best_params_,
        }
    )

    result = pd.DataFrame(rows).sort_values("R2", ascending=False, ignore_index=True)
    result["best parameters"] = result["best parameters"].astype(str)
    result.to_csv(RESULT_DIR / "operation_length_r2_experiments.csv", index=False)
    print(result)
    print(f"Saved: {RESULT_DIR / 'operation_length_r2_experiments.csv'}")

    best_columns = candidate_configs["priority retained, start hour, room + specialty"]
    kfold_result = compare_group_kfold_numbers(
        best_columns,
        X_development_all,
        X_test_all,
        y_development,
        y_test,
        development_groups,
    )

    param_distributions_10_fold = {
        "model__n_estimators": [250, 350, 500, 700, 900],
        "model__learning_rate": [0.02, 0.035, 0.05, 0.08],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__min_child_weight": [1, 3, 6, 10],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__reg_lambda": [1.0, 3.0, 8.0, 15.0],
        "model__reg_alpha": [0.0, 0.05, 0.2, 1.0],
    }
    ten_fold_pipeline = build_supervised_pipeline(
        xgb_model(), X_development_all[best_columns], missing_strategy="missing_aware"
    )
    started = time.perf_counter()
    ten_fold_search = RandomizedSearchCV(
        ten_fold_pipeline,
        param_distributions=param_distributions_10_fold,
        n_iter=40,
        scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"},
        refit="R2",
        cv=GroupKFold(n_splits=10),
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    ten_fold_search.fit(X_development_all[best_columns], y_development, groups=development_groups)
    prediction = ten_fold_search.best_estimator_.predict(X_test_all[best_columns])
    kfold_result = pd.concat(
        [
            kfold_result,
            pd.DataFrame(
                [
                    {
                        "experiment": "room + specialty + start hour, 10-fold randomized tuning",
                        "folds": 10,
                        "runtime seconds": time.perf_counter() - started,
                        "CV MAE": -ten_fold_search.cv_results_["mean_test_MAE"][
                            ten_fold_search.best_index_
                        ],
                        "CV R2": ten_fold_search.cv_results_["mean_test_R2"][
                            ten_fold_search.best_index_
                        ],
                        **metrics(y_test, prediction),
                        "best parameters": ten_fold_search.best_params_,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    kfold_result["best parameters"] = kfold_result["best parameters"].astype(str)
    kfold_result.to_csv(RESULT_DIR / "kfold_number_experiments.csv", index=False)
    print(kfold_result)
    print(f"Saved: {RESULT_DIR / 'kfold_number_experiments.csv'}")


if __name__ == "__main__":
    main()
