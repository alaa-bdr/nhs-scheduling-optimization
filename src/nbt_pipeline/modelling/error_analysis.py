"""Error analysis for the best model (SVR) on the test set.

Shows where predictions are strong and where they break down, broken
out by operation length, specialty, admission type, and anaesthetic.
This identifies the model's limitations for the discussion section.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data/modelling")
TARGET = "operation_length_mins"
OUTPUT_DIR = Path("data/modelling/plots")


def onehot_prep(x):
    num = x.select_dtypes(include=[np.number]).columns.tolist()
    cat = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def clean(a, b):
    aa, bb = a.copy(), b.copy()
    for col in aa.select_dtypes(include=[np.number]).columns:
        m = aa[col].median()
        aa[col] = aa[col].fillna(m)
        bb[col] = bb[col].fillna(m)
    for col in aa.select_dtypes(include=["object", "string", "category"]).columns:
        aa[col] = aa[col].fillna("missing").astype(str)
        bb[col] = bb[col].fillna("missing").astype(str)
    return aa, bb


def grouped_error(df, group_col, min_n=30):
    out = df.groupby(group_col).agg(
        n=("abs_error", "size"),
        mean_abs_error=("abs_error", "mean"),
        mean_actual=("actual", "mean"),
    )
    out = out[out["n"] >= min_n].sort_values("mean_abs_error", ascending=False)
    return out.round(1)


def run():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_train = train[TARGET]
    y_test = test[TARGET]
    x_train = train.drop(columns=[TARGET])
    x_test = test.drop(columns=[TARGET])

    xt_c, xte_c = clean(x_train, x_test)
    svr = Pipeline([("prep", onehot_prep(xt_c)), ("m", SVR(kernel="rbf", C=100.0, epsilon=5.0, gamma="scale"))])
    svr.fit(xt_c, y_train)
    pred = svr.predict(xte_c)

    an = test.copy()
    an["predicted"] = pred
    an["actual"] = y_test.values
    an["error"] = an["predicted"] - an["actual"]
    an["abs_error"] = an["error"].abs()

    print("=== Overall test MAE:", round(an['abs_error'].mean(), 1), "mins ===")

    an["length_band"] = pd.cut(
        an["actual"], [0, 30, 60, 120, 240, 10000],
        labels=["0-30", "30-60", "60-120", "120-240", "240+"],
    )
    print("\n--- Error by actual operation length ---")
    print(grouped_error(an, "length_band", min_n=1).to_string())

    if "session_specialty" in an.columns:
        print("\n--- Worst specialties by error ---")
        print(grouped_error(an, "session_specialty").head(8).to_string())

    if "admission_type" in an.columns:
        print("\n--- Error by admission type ---")
        print(grouped_error(an, "admission_type").to_string())

    if "anaesthetic_desc" in an.columns:
        print("\n--- Error by anaesthetic ---")
        print(grouped_error(an, "anaesthetic_desc").head(8).to_string())

    # Chart: error by operation length band
    band = grouped_error(an, "length_band", min_n=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(band.index.astype(str), band["mean_abs_error"], color="#3366A6", edgecolor="white")
    for i, v in enumerate(band["mean_abs_error"]):
        ax.text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=10)
    ax.set_xlabel("Actual operation length (mins)")
    ax.set_ylabel("Mean absolute error (mins)")
    ax.set_title("Where the Model Struggles: Error by Operation Length", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_by_length.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_DIR / 'error_by_length.png'}")

    # Residual distribution
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.hist(an["error"], bins=60, color="#6699CC", edgecolor="white")
    ax2.axvline(0, color="#D95F02", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Prediction error (predicted minus actual, mins)")
    ax2.set_ylabel("Number of operations")
    ax2.set_title("Prediction Error Distribution (Test Set)", fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "error_distribution.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'error_distribution.png'}")


if __name__ == "__main__":
    run()
