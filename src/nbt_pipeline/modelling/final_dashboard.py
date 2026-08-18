"""Final results dashboard comparing all five models plus the hospital baseline.

Trains every model on the training set, evaluates on validation, and
saves comparison charts. This is the headline figure set for the report.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

DATA_DIR = Path("data/modelling")
TARGET = "operation_length_mins"
OUTPUT_DIR = Path("data/modelling/plots")


def label_encode(train, val):
    tr, vl = train.copy(), val.copy()
    for col in train.select_dtypes(include=["object", "string", "category"]).columns:
        le = LabelEncoder()
        tr[col] = tr[col].fillna("missing")
        vl[col] = vl[col].fillna("missing")
        le.fit(tr[col])
        vl[col] = vl[col].map(lambda x, le=le: x if x in le.classes_ else "missing")
        classes = list(le.classes_)
        if "missing" not in classes:
            classes.append("missing")
            le = LabelEncoder().fit(classes)
        tr[col] = le.transform(tr[col])
        vl[col] = le.transform(vl[col])
    return tr.fillna(-1), vl.fillna(-1)


def onehot_prep(x):
    num = x.select_dtypes(include=[np.number]).columns.tolist()
    cat = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def clean(x_train, x_val):
    xt, xv = x_train.copy(), x_val.copy()
    for col in xt.select_dtypes(include=[np.number]).columns:
        m = xt[col].median()
        xt[col] = xt[col].fillna(m)
        xv[col] = xv[col].fillna(m)
    for col in xt.select_dtypes(include=["object", "string", "category"]).columns:
        xt[col] = xt[col].fillna("missing").astype(str)
        xv[col] = xv[col].fillna("missing").astype(str)
    return xt, xv


def run():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "validation.parquet")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_train = train[TARGET]
    y_val = val[TARGET]
    x_train = train.drop(columns=[TARGET])
    x_val = val.drop(columns=[TARGET])

    predictions = {}

    # Hospital baseline
    mask = val["ExpectedDurationMins"].notna()
    predictions["Hospital estimate"] = (
        val.loc[mask, "ExpectedDurationMins"].values, y_val[mask].values,
    )

    # Ridge regression
    xt_c, xv_c = clean(x_train, x_val)
    ridge = Pipeline([("prep", onehot_prep(xt_c)), ("m", Ridge(alpha=10.0, random_state=42))])
    ridge.fit(xt_c, y_train)
    predictions["Ridge regression"] = (ridge.predict(xv_c), y_val.values)

    # Random Forest
    xt_le, xv_le = label_encode(x_train, x_val)
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(xt_le, y_train)
    predictions["Random Forest"] = (rf.predict(xv_le), y_val.values)

    # XGBoost
    xgb = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8, random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(xt_le, y_train)
    predictions["XGBoost"] = (xgb.predict(xv_le), y_val.values)

    # SVR
    svr = Pipeline([("prep", onehot_prep(xt_c)), ("m", SVR(kernel="rbf", C=100.0, epsilon=5.0, gamma="scale"))])
    svr.fit(xt_c, y_train)
    predictions["SVR"] = (svr.predict(xv_c), y_val.values)

    rows = []
    for name, (pred, true) in predictions.items():
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(true, pred),
            "RMSE": mean_squared_error(true, pred) ** 0.5,
            "R2": r2_score(true, pred),
        })
    results = pd.DataFrame(rows).sort_values("MAE", ascending=False).reset_index(drop=True)
    print(results.to_string(index=False))
    results.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    # Chart 1: comparison bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colours = ["#B0B0B0", "#9BB7D4", "#6699CC", "#3366A6", "#1B3A5C"]
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        bars = ax.barh(results["Model"], results[metric], color=colours, edgecolor="white", height=0.65)
        for bar, v in zip(bars, results[metric]):
            fmt = f"{v:.1f}" if metric != "R2" else f"{v:.3f}"
            ax.text(bar.get_width() + (0.3 if metric != "R2" else 0.004),
                    bar.get_y() + bar.get_height() / 2, fmt, va="center", fontsize=10)
        ax.set_xlabel("Minutes" if metric != "R2" else "Score (1.0 = perfect)")
        ax.set_title(metric, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model Performance: Predicting Operation Duration", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_model_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'final_model_comparison.png'}")

    # Chart 2: best model vs hospital scatter
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name, colour in zip(axes2, ["Hospital estimate", "SVR"], ["#B0B0B0", "#1B3A5C"]):
        pred, true = predictions[name]
        ax.scatter(true, pred, alpha=0.15, s=8, color=colour)
        lims = [0, max(true.max(), pred.max()) * 1.05]
        ax.plot(lims, lims, "--", color="#D95F02", linewidth=1.5, label="Perfect prediction")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual duration (mins)")
        ax.set_ylabel("Predicted duration (mins)")
        ax.set_title(name, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig2.suptitle("Best Model vs Hospital Estimate", fontsize=14, fontweight="bold", y=1.02)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "final_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'final_predicted_vs_actual.png'}")


if __name__ == "__main__":
    run()
