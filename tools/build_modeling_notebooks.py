"""Build the three reproducible supervised-modelling notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "duration_error_mins": {
        "filename": "nbt_duration_error_regression.ipynb",
        "title": "NBT operation duration-error modelling",
        "task": "regression",
        "question": "How many minutes longer or shorter than planned will the operation be?",
    },
    "operation_length_mins": {
        "filename": "nbt_operation_length_regression.ipynb",
        "title": "NBT operation-length modelling",
        "task": "regression",
        "question": "How many minutes will the operation take?",
    },
    "meaningful_overrun_flag": {
        "filename": "nbt_meaningful_overrun_classification.ipynb",
        "title": "NBT meaningful-overrun classification",
        "task": "classification",
        "question": "Will the operation exceed its planned duration by more than the working tolerance?",
    },
}


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(target: str) -> Path:
    spec = SPECS[target]
    task = spec["task"]
    cells = [
        md(f"""
# {spec['title']}

**Question:** {spec['question']}

This notebook uses one cleaned analysis dataset and creates all experimental configurations in memory. It selects the missing-data approach, feature representation and algorithm using development cross-validation. The held-out test set is evaluated only after those choices are frozen.

Seven exact source-level duplicates are removed by the cleaning pipeline. `TheatreRoom` is the only location representation; `theatre_area` is not reintroduced. The 12 duration-review records are retained in the primary analysis and removed only in sensitivity analysis. `operation_start_hour` remains provisional and sensitivity-only.
"""),
        md("""
## 1. Prespecified experiment hierarchy

1. Build the target-eligible population and prohibit outcome leakage.
2. Freeze grouped development/test partitions and development cross-validation folds.
3. Use an interpretable baseline model to compare missing-data configurations.
4. Freeze the missing strategy and compare feature configurations.
5. Freeze the data configuration and compare six algorithms, including a neural network, plus a dummy benchmark.
6. Tune the two leading algorithms using development folds only.
7. Freeze the model and evaluate the untouched test set once.
8. Run start-hour, flagged-record and complete-case sensitivity analyses.

Missing-aware preprocessing does not claim that an unknown clinical value has a particular value. Categorical missingness is represented explicitly as `Missing/not recorded`; numerical imputation is a fold-fitted median placeholder accompanied by a missingness indicator. Complete cases are reported separately because they represent a smaller, selected population.
"""),
        code(f"""
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, r2_score, roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, GroupShuffleSplit, StratifiedGroupKFold,
    cross_val_predict, cross_validate,
)

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nbt_pipeline.modeling import (
    START_HOUR_COLUMN, aggregate_tree_importance, build_supervised_pipeline, classification_models,
    classification_parameter_grids, feature_configurations, prepare_target_data,
    predictor_row_groups, regression_models, regression_parameter_grids,
)
from nbt_pipeline.preprocessing import build_analysis_dataset

TARGET = {target!r}
TASK = {task!r}
RANDOM_STATE = 42
N_SPLITS = 5
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 100)
"""),
        md("""
## 2. Population, target and leakage audit

Rows without an observed target cannot train or evaluate a supervised model and are excluded. Predictor missingness is retained at this stage so missing-data strategies can be compared inside training folds. The duration-review flag is preserved for audit but prohibited from the predictor matrix.
"""),
        code("""
analysis_df = build_analysis_dataset()
prepared = prepare_target_data(analysis_df, TARGET)
X_all = prepared.predictors
y_all = prepared.target
review_all = prepared.review_flag

population_audit = pd.DataFrame({
    "measure": [
        "cleaned rows", "cleaned columns", "target-ineligible rows",
        "target-eligible rows", "identical predictor profiles",
        "duration-review records", "target mean/prevalence",
    ],
    "value": [
        len(analysis_df), analysis_df.shape[1], prepared.excluded_target_rows,
        len(y_all), int(X_all.duplicated().sum()), int(review_all.sum()), y_all.mean(),
    ],
})
missingness_audit = pd.DataFrame({
    "column": X_all.columns,
    "missing_n": [int(X_all[c].isna().sum()) for c in X_all],
    "missing_percent": [100 * X_all[c].isna().mean() for c in X_all],
    "unique_observed": [int(X_all[c].nunique(dropna=True)) for c in X_all],
}).sort_values("missing_percent", ascending=False, ignore_index=True)
display(population_audit)
missingness_audit
"""),
        code("""
assert analysis_df.shape == (14911, 22)
assert TARGET not in X_all.columns
assert not {
    "operation_length_mins", "duration_error_mins", "overrun_minutes",
    "underrun_minutes", "duration_tolerance_mins", "meaningful_overrun_flag",
    "meaningful_underrun_flag", "duration_status", "duration_timing_review_flag",
}.intersection(X_all.columns)
assert "theatre_area" not in X_all.columns
assert "TheatreRoom" in X_all.columns
assert y_all.notna().all()
"""),
        md("""
## 3. Fixed grouped development and untouched test partitions

Identical predictor profiles receive the same group hash. Classification uses grouped stratification to preserve class balance; regression uses a grouped random holdout. The test indices are then frozen for all later configurations.
"""),
        code("""
group_basis = X_all.drop(columns=[START_HOUR_COLUMN], errors="ignore")
groups_all = predictor_row_groups(group_basis)

if TASK == "classification":
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    development_idx, test_idx = next(outer.split(X_all, y_all, groups_all))
else:
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    development_idx, test_idx = next(outer.split(X_all, y_all, groups_all))

X_development_all = X_all.iloc[development_idx].copy()
X_test_all = X_all.iloc[test_idx].copy()
y_development = y_all.iloc[development_idx].copy()
y_test = y_all.iloc[test_idx].copy()
groups_development = groups_all.iloc[development_idx].copy()
groups_test = groups_all.iloc[test_idx].copy()
review_development = review_all.iloc[development_idx].copy()
review_test = review_all.iloc[test_idx].copy()
assert set(groups_development).isdisjoint(set(groups_test))

if TASK == "classification":
    inner_splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
else:
    inner_splitter = GroupKFold(n_splits=N_SPLITS)
cv_splits = list(inner_splitter.split(X_development_all, y_development, groups_development))

split_audit = pd.DataFrame({
    "partition": ["development", "untouched test"],
    "rows": [len(development_idx), len(test_idx)],
    "groups": [groups_development.nunique(), groups_test.nunique()],
    "target mean/prevalence": [y_development.mean(), y_test.mean()],
})
split_audit
"""),
        md("""
## 4. Scoring definitions and reusable comparison function

Classification selects configurations primarily by PR-AUC because it emphasizes performance on the overrun class. Regression selects by MAE because it is interpretable in minutes. Supporting metrics prevent selection from relying on one number alone.
"""),
        code("""
if TASK == "classification":
    SCORING = {
        "PR_AUC": "average_precision", "ROC_AUC": "roc_auc",
        "Balanced_accuracy": "balanced_accuracy", "F1": "f1",
        "Recall": "recall", "Precision": "precision", "Brier": "neg_brier_score",
    }
    PRIMARY_METRIC = "PR_AUC"
    HIGHER_IS_BETTER = True
    baseline_model = classification_models(y_development)["Logistic regression"]
else:
    SCORING = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
    PRIMARY_METRIC = "MAE"
    HIGHER_IS_BETTER = False
    baseline_model = regression_models()["Linear regression"]

def summarize_cv(scores):
    row = {}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        if metric in {"MAE", "RMSE", "Brier"}:
            values = -values
        row[f"CV {metric} mean"] = values.mean()
        row[f"CV {metric} SD"] = values.std(ddof=1)
    return row

def evaluate_cv(model, X, y, splits, missing_strategy="missing_aware"):
    return cross_validate(
        build_supervised_pipeline(model, X, missing_strategy=missing_strategy),
        X, y, cv=splits, scoring=SCORING, n_jobs=1, error_score="raise",
    )
"""),
        md("""
## 5. Missing-data strategy experiment

The interpretable baseline compares missing-aware preprocessing with and without the highly incomplete priority field on the same development rows. Complete cases are shown as a sensitivity population and do not compete directly because their case mix and sample size differ.
"""),
        code("""
base_columns = feature_configurations(include_priority=True)["Full approved"]
missing_rows = []
for label, columns in {
    "Missing-aware, priority retained": base_columns,
    "Missing-aware, priority excluded": [c for c in base_columns if c != "priority_level_label"],
}.items():
    X = X_development_all[columns]
    scores = evaluate_cv(baseline_model, X, y_development, cv_splits)
    missing_rows.append({"configuration": label, "rows": len(X), "predictors": len(columns), **summarize_cv(scores)})

complete_mask = X_development_all[base_columns].notna().all(axis=1)
X_complete = X_development_all.loc[complete_mask, base_columns]
y_complete = y_development.loc[complete_mask]
groups_complete = groups_development.loc[complete_mask]
if TASK == "classification":
    complete_splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
else:
    complete_splitter = GroupKFold(n_splits=N_SPLITS)
complete_cv = list(complete_splitter.split(X_complete, y_complete, groups_complete))
complete_scores = evaluate_cv(baseline_model, X_complete, y_complete, complete_cv, "complete_case")
missing_rows.append({
    "configuration": "Complete cases (sensitivity)", "rows": len(X_complete),
    "predictors": len(base_columns), **summarize_cv(complete_scores),
})
missing_strategy_results = pd.DataFrame(missing_rows)

candidate_missing = missing_strategy_results[missing_strategy_results["configuration"].str.startswith("Missing-aware")]
sort_column = f"CV {PRIMARY_METRIC} mean"
selected_missing_row = candidate_missing.sort_values(sort_column, ascending=not HIGHER_IS_BETTER).iloc[0]
include_priority = selected_missing_row["configuration"].endswith("retained")
selected_missing_strategy = selected_missing_row["configuration"]
missing_strategy_results.sort_values(sort_column, ascending=not HIGHER_IS_BETTER)
"""),
        md("""
## 6. Feature-configuration experiment

With the missing strategy frozen, the same baseline model and folds compare planning-only, clinical, procedure and operational representations. `TheatreRoom` is compared with no location; `theatre_area` remains excluded. Start hour is deferred to sensitivity analysis.
"""),
        code("""
feature_rows = []
configs = feature_configurations(include_priority=include_priority)
for config_name, columns in configs.items():
    X = X_development_all[columns]
    started = time.perf_counter()
    scores = evaluate_cv(baseline_model, X, y_development, cv_splits)
    feature_rows.append({
        "configuration": config_name, "rows": len(X), "predictors": len(columns),
        "runtime seconds": time.perf_counter() - started, **summarize_cv(scores),
    })
feature_results = pd.DataFrame(feature_rows).sort_values(sort_column, ascending=not HIGHER_IS_BETTER, ignore_index=True)
selected_feature_name = feature_results.iloc[0]["configuration"]
selected_columns = configs[selected_feature_name]
feature_results
"""),
        md("""
## 7. Six-model comparison

The missing strategy, predictor columns and folds are now frozen. All six algorithms—including a regularized multilayer neural network—and the dummy benchmark receive identical development information. The neural network uses early stopping within each training fit.
"""),
        code("""
X_development = X_development_all[selected_columns].copy()
X_test = X_test_all[selected_columns].copy()
models = classification_models(y_development) if TASK == "classification" else regression_models()
algorithm_rows = []
for model_name, model in models.items():
    started = time.perf_counter()
    scores = evaluate_cv(model, X_development, y_development, cv_splits)
    algorithm_rows.append({
        "model": model_name, "runtime seconds": time.perf_counter() - started,
        **summarize_cv(scores),
    })
algorithm_results = pd.DataFrame(algorithm_rows).sort_values(sort_column, ascending=not HIGHER_IS_BETTER, ignore_index=True)
algorithm_results
"""),
        code("""
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=algorithm_results, x=sort_column, y="model", color="#4C78A8", ax=ax)
ax.set(title=f"Grouped cross-validation: {PRIMARY_METRIC}", ylabel="")
plt.tight_layout()
"""),
        md("""
## 8. Compact tuning of the two leading algorithms

The two strongest non-dummy algorithms receive prespecified compact grids. Tuning uses development folds only and retains the same primary selection metric.
"""),
        code("""
is_benchmark = algorithm_results["model"].str.contains("Dummy")
candidate_names = algorithm_results.loc[~is_benchmark, "model"].head(2).tolist()
parameter_grids = classification_parameter_grids() if TASK == "classification" else regression_parameter_grids()
tuning_rows = []
tuned_searches = {}
for model_name in candidate_names:
    search = GridSearchCV(
        build_supervised_pipeline(models[model_name], X_development, missing_strategy="missing_aware"),
        parameter_grids[model_name], scoring=SCORING, refit=PRIMARY_METRIC,
        cv=cv_splits, n_jobs=1, error_score="raise", return_train_score=False,
    )
    started = time.perf_counter()
    search.fit(X_development, y_development)
    tuned_searches[model_name] = search
    idx = search.best_index_
    row = {"model": model_name, "best parameters": search.best_params_, "runtime seconds": time.perf_counter() - started}
    for metric in SCORING:
        value = search.cv_results_[f"mean_test_{metric}"][idx]
        row[f"tuned CV {metric}"] = -value if metric in {"MAE", "RMSE", "Brier"} else value
    tuning_rows.append(row)
tuning_results = pd.DataFrame(tuning_rows).sort_values(
    f"tuned CV {PRIMARY_METRIC}", ascending=not HIGHER_IS_BETTER, ignore_index=True
)
best_model_name = tuning_results.iloc[0]["model"]
best_model = tuned_searches[best_model_name].best_estimator_
tuning_results
"""),
        md("""
## 9. Frozen final test evaluation

After missingness, features, algorithm and hyperparameters are selected, the winning model is evaluated on the untouched test set. Classification selects its provisional decision threshold from development out-of-fold probabilities, never from test outcomes.
"""),
        code("""
if TASK == "classification":
    development_probability = cross_val_predict(
        clone(best_model), X_development, y_development, cv=cv_splits,
        method="predict_proba", n_jobs=1,
    )[:, 1]
    threshold_rows = []
    for threshold in np.linspace(0.10, 0.90, 81):
        predicted = (development_probability >= threshold).astype(int)
        threshold_rows.append({
            "threshold": threshold,
            "precision": precision_score(y_development, predicted, zero_division=0),
            "recall": recall_score(y_development, predicted, zero_division=0),
            "F1": f1_score(y_development, predicted, zero_division=0),
        })
    threshold_results = pd.DataFrame(threshold_rows)
    selected_threshold = threshold_results.sort_values(["F1", "recall"], ascending=False).iloc[0]["threshold"]
else:
    threshold_results = pd.DataFrame()
    selected_threshold = np.nan

best_model.fit(X_development, y_development)
if TASK == "classification":
    test_probability = best_model.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= selected_threshold).astype(int)
    final_test_results = pd.DataFrame([{
        "model": best_model_name, "threshold": selected_threshold,
        "accuracy": accuracy_score(y_test, test_prediction),
        "balanced accuracy": balanced_accuracy_score(y_test, test_prediction),
        "ROC-AUC": roc_auc_score(y_test, test_probability),
        "PR-AUC": average_precision_score(y_test, test_probability),
        "precision": precision_score(y_test, test_prediction, zero_division=0),
        "recall": recall_score(y_test, test_prediction, zero_division=0),
        "F1": f1_score(y_test, test_prediction, zero_division=0),
        "Brier": brier_score_loss(y_test, test_probability),
    }])
else:
    test_prediction = best_model.predict(X_test)
    residual = y_test.to_numpy() - test_prediction
    final_test_results = pd.DataFrame([{
        "model": best_model_name,
        "MAE": mean_absolute_error(y_test, test_prediction),
        "RMSE": mean_squared_error(y_test, test_prediction) ** 0.5,
        "R2": r2_score(y_test, test_prediction),
        "within 10 minutes": np.mean(np.abs(residual) <= 10),
        "within 20 minutes": np.mean(np.abs(residual) <= 20),
        "within 30 minutes": np.mean(np.abs(residual) <= 30),
        "within 60 minutes": np.mean(np.abs(residual) <= 60),
    }])
final_test_results
"""),
        code("""
if TASK == "classification":
    matrix = confusion_matrix(y_test, test_prediction)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set(xlabel="Predicted", ylabel="Observed", title="Untouched-test confusion matrix")
    calibration = pd.DataFrame({"observed": y_test.to_numpy(), "probability": test_probability})
    calibration["bin"] = pd.qcut(calibration["probability"], q=10, duplicates="drop")
    calibration_plot = calibration.groupby("bin", observed=True).agg(
        predicted_probability=("probability", "mean"), observed_rate=("observed", "mean")
    ).reset_index()
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].plot(calibration_plot["predicted_probability"], calibration_plot["observed_rate"], marker="o")
    axes[1].set(xlabel="Predicted probability", ylabel="Observed proportion", title="Calibration")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    residual = y_test.to_numpy() - test_prediction
    axes[0].scatter(y_test, test_prediction, alpha=0.25, s=16)
    bounds = [min(y_test.min(), test_prediction.min()), max(y_test.max(), test_prediction.max())]
    axes[0].plot(bounds, bounds, "--", color="grey")
    axes[0].set(xlabel="Observed", ylabel="Predicted", title="Observed versus predicted")
    axes[1].scatter(test_prediction, residual, alpha=0.25, s=16)
    axes[1].axhline(0, linestyle="--", color="grey")
    axes[1].set(xlabel="Predicted", ylabel="Observed - predicted", title="Residuals")
plt.tight_layout()
"""),
        md("""
## 10. Operational benchmarks

Regression includes the dummy benchmark. Direct operation-length modelling additionally compares the hospital's `ExpectedDurationMins`. Duration-error modelling converts its predicted correction back to an operation-length prediction where the hospital plan is observed.
"""),
        code("""
benchmark_results = pd.DataFrame()
if TASK == "regression":
    dummy = build_supervised_pipeline(regression_models()["Dummy median benchmark"], X_development, missing_strategy="missing_aware")
    dummy.fit(X_development, y_development)
    dummy_prediction = dummy.predict(X_test)
    rows = [{
        "benchmark": "Dummy median", "MAE": mean_absolute_error(y_test, dummy_prediction),
        "RMSE": mean_squared_error(y_test, dummy_prediction) ** 0.5,
        "R2": r2_score(y_test, dummy_prediction),
    }]
    expected = X_test_all.loc[y_test.index, "ExpectedDurationMins"]
    available = expected.notna()
    if TARGET == "operation_length_mins":
        rows.append({
            "benchmark": "Hospital ExpectedDurationMins",
            "MAE": mean_absolute_error(y_test.loc[available], expected.loc[available]),
            "RMSE": mean_squared_error(y_test.loc[available], expected.loc[available]) ** 0.5,
            "R2": r2_score(y_test.loc[available], expected.loc[available]),
        })
    elif TARGET == "duration_error_mins":
        actual_duration = y_test.loc[available] + expected.loc[available]
        corrected_duration = test_prediction[available.to_numpy()] + expected.loc[available].to_numpy()
        rows.append({
            "benchmark": "Error-corrected operation length",
            "MAE": mean_absolute_error(actual_duration, corrected_duration),
            "RMSE": mean_squared_error(actual_duration, corrected_duration) ** 0.5,
            "R2": r2_score(actual_duration, corrected_duration),
        })
    benchmark_results = pd.DataFrame(rows)
benchmark_results
"""),
        md("""
## 11. Prespecified sensitivity analyses

The final specification is refitted with provisional start hour, without flagged records, and on complete cases. These results assess stability; they do not replace the primary test result or reopen model selection.
"""),
        code("""
def compact_metrics(actual, prediction, probability=None):
    if TASK == "classification":
        return {
            "PR-AUC": average_precision_score(actual, probability),
            "ROC-AUC": roc_auc_score(actual, probability),
            "F1": f1_score(actual, prediction, zero_division=0),
            "recall": recall_score(actual, prediction, zero_division=0),
        }
    return {
        "MAE": mean_absolute_error(actual, prediction),
        "RMSE": mean_squared_error(actual, prediction) ** 0.5,
        "R2": r2_score(actual, prediction),
    }

sensitivity_rows = [{"analysis": "Primary", "development rows": len(y_development), "test rows": len(y_test), **compact_metrics(y_test, test_prediction, test_probability if TASK == "classification" else None)}]

# Provisional start hour.
start_columns = selected_columns + [START_HOUR_COLUMN]
start_model = build_supervised_pipeline(clone(best_model.named_steps["model"]), X_development_all[start_columns], missing_strategy="missing_aware")
start_model.fit(X_development_all[start_columns], y_development)
if TASK == "classification":
    start_probability = start_model.predict_proba(X_test_all[start_columns])[:, 1]
    start_prediction = (start_probability >= selected_threshold).astype(int)
else:
    start_probability = None
    start_prediction = start_model.predict(X_test_all[start_columns])
sensitivity_rows.append({"analysis": "Start hour included", "development rows": len(y_development), "test rows": len(y_test), **compact_metrics(y_test, start_prediction, start_probability)})

# Duration-review records excluded.
keep_development = ~review_development.to_numpy()
keep_test = ~review_test.to_numpy()
flag_model = clone(best_model)
flag_model.fit(X_development.iloc[keep_development], y_development.iloc[keep_development])
if TASK == "classification":
    flag_probability = flag_model.predict_proba(X_test.iloc[keep_test])[:, 1]
    flag_prediction = (flag_probability >= selected_threshold).astype(int)
else:
    flag_probability = None
    flag_prediction = flag_model.predict(X_test.iloc[keep_test])
sensitivity_rows.append({
    "analysis": "Flagged records excluded", "development rows": int(keep_development.sum()),
    "test rows": int(keep_test.sum()),
    **compact_metrics(y_test.iloc[keep_test], flag_prediction, flag_probability),
})

# Complete cases.
complete_development = X_development.notna().all(axis=1)
complete_test = X_test.notna().all(axis=1)
complete_model = build_supervised_pipeline(
    clone(best_model.named_steps["model"]), X_development.loc[complete_development], missing_strategy="complete_case"
)
complete_model.fit(X_development.loc[complete_development], y_development.loc[complete_development])
if TASK == "classification":
    complete_probability = complete_model.predict_proba(X_test.loc[complete_test])[:, 1]
    complete_prediction = (complete_probability >= selected_threshold).astype(int)
else:
    complete_probability = None
    complete_prediction = complete_model.predict(X_test.loc[complete_test])
sensitivity_rows.append({
    "analysis": "Complete cases", "development rows": int(complete_development.sum()),
    "test rows": int(complete_test.sum()),
    **compact_metrics(y_test.loc[complete_test], complete_prediction, complete_probability),
})
sensitivity_results = pd.DataFrame(sensitivity_rows)
sensitivity_results
"""),
        md("""
## 12. Model interpretation

Permutation importance measures the reduction in untouched-test performance when one original column is shuffled. It explains any winning model, including a neural network, without relying on its internal structure. Supporting Decision Tree, Random Forest and XGBoost importances are aggregated from one-hot levels back to original columns.
"""),
        code("""
importance_scoring = "average_precision" if TASK == "classification" else "neg_mean_absolute_error"
permutation = permutation_importance(
    best_model, X_test, y_test, scoring=importance_scoring,
    n_repeats=10, random_state=RANDOM_STATE, n_jobs=1,
)
permutation_importance_results = pd.DataFrame({
    "feature": selected_columns,
    "importance mean": permutation.importances_mean,
    "importance SD": permutation.importances_std,
}).sort_values("importance mean", ascending=False, ignore_index=True)
permutation_importance_results["rank"] = np.arange(1, len(permutation_importance_results) + 1)

tree_importance_frames = []
for tree_name in ["Decision tree", "Random forest", "XGBoost"]:
    if tree_name == best_model_name:
        fitted_tree = best_model
    else:
        fitted_tree = build_supervised_pipeline(
            clone(models[tree_name]), X_development, missing_strategy="missing_aware"
        ).fit(X_development, y_development)
    aggregated = aggregate_tree_importance(fitted_tree, selected_columns)
    aggregated.insert(0, "model", tree_name)
    tree_importance_frames.append(aggregated)
tree_importance_results = pd.concat(tree_importance_frames, ignore_index=True)

fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(permutation_importance_results))))
sns.barplot(
    data=permutation_importance_results,
    x="importance mean", y="feature", color="#59A14F", ax=ax,
)
ax.set(title=f"{best_model_name}: original-column permutation importance", xlabel="Performance reduction after shuffling", ylabel="")
plt.tight_layout()
permutation_importance_results
"""),
        md("""
Permutation and tree importance are predictive, not causal. Correlated columns can divide importance between themselves, and missingness patterns may contribute to predictions. These results explain the frozen model and are not used to reopen feature or model selection.
"""),
        md("""
## 13. Export reproducible evidence

Only result tables and index-linked predictions are exported. No duplicate cleaned datasets are created.
"""),
        code("""
RESULT_DIR = PROJECT_ROOT / "result" / "modeling" / TARGET
RESULT_DIR.mkdir(parents=True, exist_ok=True)
tables = {
    "population_audit": population_audit,
    "missingness_audit": missingness_audit,
    "split_audit": split_audit,
    "missing_strategy_comparison": missing_strategy_results,
    "feature_configuration_comparison": feature_results,
    "algorithm_comparison": algorithm_results,
    "tuning_results": tuning_results.assign(**{"best parameters": tuning_results["best parameters"].astype(str)}),
    "final_test_metrics": final_test_results,
    "sensitivity_results": sensitivity_results,
    "permutation_importance": permutation_importance_results,
    "tree_importance": tree_importance_results,
}
if not benchmark_results.empty:
    tables["operational_benchmarks"] = benchmark_results
if not threshold_results.empty:
    tables["threshold_comparison"] = threshold_results
for name, table in tables.items():
    table.to_csv(RESULT_DIR / f"{name}.csv", index=False)

predictions = pd.DataFrame({"source_index": y_test.index, "observed": y_test.to_numpy(), "predicted": test_prediction})
if TASK == "classification":
    predictions["predicted_probability"] = test_probability
predictions.to_csv(RESULT_DIR / "test_predictions.csv", index=False)

selection_summary = pd.DataFrame({
    "selection": ["target", "missing strategy", "feature configuration", "model", "threshold"],
    "value": [TARGET, selected_missing_strategy, selected_feature_name, best_model_name, selected_threshold],
})
selection_summary.to_csv(RESULT_DIR / "selection_summary.csv", index=False)
selection_summary
"""),
        md("""
## 14. Interpretation boundaries

- Configuration and algorithm selection are internal-validation results, not evidence of causal effects.
- The test result estimates performance only for this dataset and requires temporal or external validation.
- Missing-aware preprocessing preserves cases but may learn from recording patterns; complete-case results describe a smaller population.
- Start hour is provisional and cannot become a primary predictor until its reconstruction is validated.
- Flagged-record sensitivity may be imprecise because only 12 records are flagged.
- Operational thresholds require stakeholder input about the relative cost of missed overruns and false warnings.
"""),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    path = ROOT / "notebooks" / spec["filename"]
    nbf.write(notebook, path)
    return path


def main() -> None:
    for target in SPECS:
        print(f"Built {build_notebook(target)}")


if __name__ == "__main__":
    main()
