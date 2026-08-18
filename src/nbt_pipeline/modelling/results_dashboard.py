"""Generate a visual results dashboard comparing all models."""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

DATA_DIR = Path("data/modelling")
TARGET = "operation_length_mins"
OUTPUT_DIR = Path("data/modelling/plots")


def encode_categoricals(train, validation):
    encoders = {}
    train_enc = train.copy()
    val_enc = validation.copy()
    for col in train.select_dtypes(include=["object", "string", "category"]).columns:
        le = LabelEncoder()
        train_enc[col] = train_enc[col].fillna("missing")
        val_enc[col] = val_enc[col].fillna("missing")
        le.fit(train_enc[col])
        val_enc[col] = val_enc[col].map(
            lambda x, le=le: x if x in le.classes_ else "missing"
        )
        le_classes = list(le.classes_)
        if "missing" not in le_classes:
            le_classes.append("missing")
            le = LabelEncoder()
            le.fit(le_classes)
        train_enc[col] = le.transform(train_enc[col])
        val_enc[col] = le.transform(val_enc[col])
        encoders[col] = le
    return train_enc, val_enc, encoders


def run():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "validation.parquet")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_train = train[TARGET]
    y_val = val[TARGET]
    x_train = train.drop(columns=[TARGET])
    x_val = val.drop(columns=[TARGET])

    x_train_enc, x_val_enc, _ = encode_categoricals(x_train, x_val)
    x_train_enc = x_train_enc.fillna(-1)
    x_val_enc = x_val_enc.fillna(-1)

    num_feats = ["ExpectedDurationMins", "age_at_operation", "ASAScore"]
    medians = train[num_feats].median()

    predictions = {}

    # Hospital baseline
    hosp_mask = val["ExpectedDurationMins"].notna()
    predictions["Hospital estimate"] = (
        val.loc[hosp_mask, "ExpectedDurationMins"].values,
        y_val[hosp_mask].values,
    )

    # Linear regression
    lr = LinearRegression().fit(
        train[num_feats].fillna(medians), y_train
    )
    predictions["Linear regression"] = (
        lr.predict(val[num_feats].fillna(medians)),
        y_val.values,
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=5,
        random_state=42, n_jobs=-1,
    )
    rf.fit(x_train_enc, y_train)
    predictions["Random Forest"] = (
        rf.predict(x_val_enc), y_val.values,
    )

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb.fit(x_train_enc, y_train)
    predictions["XGBoost"] = (
        xgb.predict(x_val_enc), y_val.values,
    )

    # Collect metrics
    rows = []
    for name, (y_pred, y_true) in predictions.items():
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "R2": r2_score(y_true, y_pred),
        })
    results = pd.DataFrame(rows)
    print(results.to_string(index=False))

    # ---- CHART 1: Model comparison bar chart ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colours = ["#B0B0B0", "#7BAFD4", "#4A90D9", "#1B4F72"]

    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        bars = ax.barh(
            results["Model"], results[metric],
            color=colours, edgecolor="white", height=0.6,
        )
        for bar, val in zip(bars, results[metric]):
            fmt = f"{val:.1f}" if metric != "R2" else f"{val:.3f}"
            ax.text(
                bar.get_width() + (0.3 if metric != "R2" else 0.005),
                bar.get_y() + bar.get_height() / 2,
                fmt, va="center", fontsize=10,
            )
        label = "Minutes" if metric != "R2" else "Score (1.0 = perfect)"
        ax.set_xlabel(label)
        ax.set_title(metric, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Model Performance: Predicting Operation Duration",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR / 'model_comparison.png'}")

    # ---- CHART 2: Predicted vs Actual scatter for best model ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, colour) in zip(axes2, [
        ("Hospital estimate", "#B0B0B0"),
        ("XGBoost", "#1B4F72"),
    ]):
        y_pred, y_true = predictions[name]
        ax.scatter(y_true, y_pred, alpha=0.15, s=8, color=colour)
        lims = [0, max(y_true.max(), y_pred.max()) * 1.05]
        ax.plot(lims, lims, "--", color="#D95F02", linewidth=1.5, label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual duration (mins)")
        ax.set_ylabel("Predicted duration (mins)")
        ax.set_title(name, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig2.suptitle(
        "Predicted vs Actual Operation Duration",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'predicted_vs_actual.png'}")

    # ---- CHART 3: Feature importances ----
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model, name in zip(axes3, [rf, xgb], ["Random Forest", "XGBoost"]):
        imp = pd.Series(
            model.feature_importances_, index=x_train_enc.columns
        ).sort_values()
        imp.plot.barh(ax=ax, color="#4A90D9", edgecolor="white")
        ax.set_title(f"{name} Feature Importances", fontweight="bold")
        ax.set_xlabel("Importance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "feature_importances.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'feature_importances.png'}")


if __name__ == "__main__":
    run()
