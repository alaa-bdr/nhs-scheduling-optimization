"""Test operation-length prediction with and without ExpectedDurationMins.

The goal is to answer the supervisor question: can dropping the planned duration
improve operation-length prediction? The script uses a grouped train/validation/test
setup and XGBoost early stopping. It exports result tables and plots.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nbt_pipeline.modeling import (  # noqa: E402
    RANDOM_STATE,
    START_HOUR_COLUMN,
    predictor_row_groups,
    prepare_target_data,
)
from nbt_pipeline.modeling.experiments import build_preprocessor  # noqa: E402
from nbt_pipeline.preprocessing import build_analysis_dataset, build_preprocessed_dataset  # noqa: E402
from nbt_pipeline.preprocessing.selection import remove_exact_source_duplicates  # noqa: E402

TARGET = "operation_length_mins"
FULL_CODE = "actual_proc_1_procedure_code"
RESULT_DIR = PROJECT_ROOT / "result" / "modeling" / TARGET / "expected_duration_sensitivity"

BASE_COLUMNS = [
    "ExpectedDurationMins",
    "sex_national_code",
    "age_at_operation",
    "ASAScore",
    "anaesthetic_desc",
    "admission_type_label",
    "intended_management_label",
    "procedure_code_group",
    "procedure_code_category",
    START_HOUR_COLUMN,
    "TheatreRoom",
    "session_specialty",
    "actual_proc_1_procedure_code_min20",
]


def add_rare_grouped_full_code(X: pd.DataFrame, preprocessed_df: pd.DataFrame) -> pd.DataFrame:
    result = X.copy()
    raw = preprocessed_df.loc[result.index, FULL_CODE]
    result[FULL_CODE] = raw.astype("string").mask(raw.isna()).astype(object)
    result.loc[raw.isna(), FULL_CODE] = np.nan
    counts = result[FULL_CODE].value_counts(dropna=True)
    common_codes = set(counts[counts >= 20].index)
    result["actual_proc_1_procedure_code_min20"] = result[FULL_CODE].where(
        result[FULL_CODE].isin(common_codes), "Rare procedure code"
    )
    result.loc[result[FULL_CODE].isna(), "actual_proc_1_procedure_code_min20"] = np.nan
    return result


def rmse(actual, predicted) -> float:
    return mean_squared_error(actual, predicted) ** 0.5


def evaluate_config(
    name: str,
    columns: list[str],
    X_train_all: pd.DataFrame,
    X_validation_all: pd.DataFrame,
    X_test_all: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, object], dict[str, list[float]]]:
    preprocessor = build_preprocessor(X_train_all[columns], missing_strategy="missing_aware")
    X_train = preprocessor.fit_transform(X_train_all[columns])
    X_validation = preprocessor.transform(X_validation_all[columns])
    X_test = preprocessor.transform(X_test_all[columns])

    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=8.0,
        reg_alpha=0.05,
        gamma=0.1,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    started = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_validation, y_validation)],
        verbose=False,
    )
    validation_prediction = model.predict(X_validation)
    test_prediction = model.predict(X_test)
    evals = model.evals_result()
    best_iteration = getattr(model, "best_iteration", None)
    row = {
        "experiment": name,
        "predictors": len(columns),
        "columns": ", ".join(columns),
        "runtime seconds": time.perf_counter() - started,
        "best iteration": best_iteration,
        "validation MAE": mean_absolute_error(y_validation, validation_prediction),
        "validation RMSE": rmse(y_validation, validation_prediction),
        "validation R2": r2_score(y_validation, validation_prediction),
        "test MAE": mean_absolute_error(y_test, test_prediction),
        "test RMSE": rmse(y_test, test_prediction),
        "test R2": r2_score(y_test, test_prediction),
    }
    history = {
        "train_rmse": evals["validation_0"]["rmse"],
        "validation_rmse": evals["validation_1"]["rmse"],
    }
    return row, history


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    analysis_df = build_analysis_dataset()
    preprocessed_df = remove_exact_source_duplicates(build_preprocessed_dataset())
    prepared = prepare_target_data(analysis_df, TARGET)
    X_all = add_rare_grouped_full_code(prepared.predictors, preprocessed_df)
    y_all = prepared.target

    # First hold out an untouched grouped test set.
    group_basis = prepared.predictors.drop(columns=[START_HOUR_COLUMN], errors="ignore")
    groups_all = predictor_row_groups(group_basis)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    development_idx, test_idx = next(outer.split(X_all, y_all, groups_all))

    X_development_all = X_all.iloc[development_idx].copy()
    X_test_all = X_all.iloc[test_idx].copy()
    y_development = y_all.iloc[development_idx].copy()
    y_test = y_all.iloc[test_idx].copy()
    groups_development = groups_all.iloc[development_idx].copy()

    # Then split development into train and validation for early stopping.
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_local_idx, validation_local_idx = next(
        inner.split(X_development_all, y_development, groups_development)
    )
    X_train_all = X_development_all.iloc[train_local_idx].copy()
    X_validation_all = X_development_all.iloc[validation_local_idx].copy()
    y_train = y_development.iloc[train_local_idx].copy()
    y_validation = y_development.iloc[validation_local_idx].copy()

    configs = {
        "With ExpectedDurationMins": BASE_COLUMNS,
        "Without ExpectedDurationMins": [c for c in BASE_COLUMNS if c != "ExpectedDurationMins"],
    }

    rows = []
    histories = {}
    for name, columns in configs.items():
        row, history = evaluate_config(
            name,
            columns,
            X_train_all,
            X_validation_all,
            X_test_all,
            y_train,
            y_validation,
            y_test,
        )
        rows.append(row)
        histories[name] = history

    results = pd.DataFrame(rows).sort_values("test R2", ascending=False, ignore_index=True)
    results.to_csv(RESULT_DIR / "expected_duration_sensitivity_results.csv", index=False)

    # Plot validation and test R2/MAE side by side.
    plot_df = results.melt(
        id_vars="experiment",
        value_vars=["validation R2", "test R2", "validation MAE", "test MAE"],
        var_name="metric",
        value_name="value",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    r2_df = plot_df[plot_df["metric"].str.contains("R2")]
    mae_df = plot_df[plot_df["metric"].str.contains("MAE")]
    for metric, frame in r2_df.groupby("metric"):
        axes[0].plot(frame["experiment"], frame["value"], marker="o", label=metric)
    axes[0].set_title("Validation vs test R² for operation_length_mins")
    axes[0].set_ylabel("R² score, higher is better")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend()
    for metric, frame in mae_df.groupby("metric"):
        axes[1].plot(frame["experiment"], frame["value"], marker="o", label=metric)
    axes[1].set_title("Validation vs test MAE for operation_length_mins")
    axes[1].set_ylabel("MAE in minutes, lower is better")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(RESULT_DIR / "validation_test_comparison.png", dpi=160)
    plt.close(fig)

    # Plot early-stopping learning curves.
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in histories.items():
        ax.plot(history["validation_rmse"], label=f"{name}: validation RMSE")
    ax.set_title("Early stopping validation curve")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Validation RMSE, lower is better")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULT_DIR / "early_stopping_validation_curve.png", dpi=160)
    plt.close(fig)

    history_rows = []
    for name, history in histories.items():
        for iteration, (train_rmse, validation_rmse) in enumerate(
            zip(history["train_rmse"], history["validation_rmse"], strict=True)
        ):
            history_rows.append(
                {
                    "experiment": name,
                    "iteration": iteration,
                    "train RMSE": train_rmse,
                    "validation RMSE": validation_rmse,
                }
            )
    pd.DataFrame(history_rows).to_csv(RESULT_DIR / "early_stopping_history.csv", index=False)

    print(results[["experiment", "predictors", "best iteration", "validation MAE", "validation R2", "test MAE", "test R2"]].to_string(index=False))
    print("Saved:", RESULT_DIR / "expected_duration_sensitivity_results.csv")
    print("Saved:", RESULT_DIR / "validation_test_comparison.png")
    print("Saved:", RESULT_DIR / "early_stopping_validation_curve.png")


if __name__ == "__main__":
    main()


