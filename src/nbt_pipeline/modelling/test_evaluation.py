"""Final test set evaluation for all five models.

Every model is trained on the training set, its hyperparameters were
selected using the validation set earlier, and here it is evaluated
once on the held-out test set. These test scores are the final,
unbiased results reported in the article.
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


def label_encode(train, other):
    tr, ot = train.copy(), other.copy()
    for col in train.select_dtypes(include=["object", "string", "category"]).columns:
        le = LabelEncoder()
        tr[col] = tr[col].fillna("missing")
        ot[col] = ot[col].fillna("missing")
        le.fit(tr[col])
        ot[col] = ot[col].map(lambda x, le=le: x if x in le.classes_ else "missing")
        classes = list(le.classes_)
        if "missing" not in classes:
            classes.append("missing")
            le = LabelEncoder().fit(classes)
        tr[col] = le.transform(tr[col])
        ot[col] = le.transform(ot[col])
    return tr.fillna(-1), ot.fillna(-1)


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


def metrics(true, pred):
    return {
        "MAE": mean_absolute_error(true, pred),
        "RMSE": mean_squared_error(true, pred) ** 0.5,
        "R2": r2_score(true, pred),
    }


def run():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_train = train[TARGET]
    y_test = test[TARGET]
    x_train = train.drop(columns=[TARGET])
    x_test = test.drop(columns=[TARGET])

    predictions = {}

    # Hospital baseline (no training, uses the planned time directly)
    mask = test["ExpectedDurationMins"].notna()
    predictions["Hospital estimate"] = (
        test.loc[mask, "ExpectedDurationMins"].values, y_test[mask].values,
    )

    xt_c, xte_c = clean(x_train, x_test)
    xt_le, xte_le = label_encode(x_train, x_test)

    ridge = Pipeline([("prep", onehot_prep(xt_c)), ("m", Ridge(alpha=10.0, random_state=42))])
    ridge.fit(xt_c, y_train)
    predictions["Ridge regression"] = (ridge.predict(xte_c), y_test.values)

    rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(xt_le, y_train)
    predictions["Random Forest"] = (rf.predict(xte_le), y_test.values)

    xgb = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8, random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(xt_le, y_train)
    predictions["XGBoost"] = (xgb.predict(xte_le), y_test.values)

    svr = Pipeline([("prep", onehot_prep(xt_c)), ("m", SVR(kernel="rbf", C=100.0, epsilon=5.0, gamma="scale"))])
    svr.fit(xt_c, y_train)
    predictions["SVR"] = (svr.predict(xte_c), y_test.values)

    rows = []
    for name, (pred, true) in predictions.items():
        m = metrics(true, pred)
        m["Model"] = name
        rows.append(m)
    results = pd.DataFrame(rows)[["Model", "MAE", "RMSE", "R2"]]
    results = results.sort_values("MAE", ascending=False).reset_index(drop=True)

    hosp_mae = results.loc[results["Model"] == "Hospital estimate", "MAE"].values[0]
    results["Mins better than hospital"] = (hosp_mae - results["MAE"]).round(1)

    print("=== FINAL TEST SET RESULTS ===")
    print(results.to_string(index=False))
    results.to_csv(OUTPUT_DIR / "test_set_results.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'test_set_results.csv'}")

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
    fig.suptitle("Final Test Set Performance (Held-Out Data)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_set_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_DIR / 'test_set_comparison.png'}")


if __name__ == "__main__":
    run()
