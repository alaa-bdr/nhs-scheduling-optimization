"""Statistical significance testing between the leading models.

Point estimates alone cannot show whether one model genuinely beats
another. This compares per case absolute errors on the same test rows
using paired tests, so differences are assessed on identical cases.
"""

from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from nbt_pipeline.modelling_v2.features_v2 import build_v2_dataset

OUT = Path("data/modelling_v2/plots")
SEED = 42
CAP = 480


def encode(frame, features):
    X = frame[features].copy()
    for c in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].fillna("missing").astype(str))
    return X.fillna(-1)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, features, target = build_v2_dataset(duration_cap=CAP)
    X = encode(frame, features)
    y = frame[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    scaler = StandardScaler().fit(Xtr)

    errors = {}

    errors["Hospital estimate"] = np.abs(yte.values - Xte["ExpectedDurationMins"].values)

    ridge = Ridge(alpha=10.0).fit(scaler.transform(Xtr), ytr)
    errors["Ridge regression"] = np.abs(yte.values - ridge.predict(scaler.transform(Xte)))

    rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=5,
                               random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    errors["Random Forest"] = np.abs(yte.values - rf.predict(Xte))

    xgb = XGBRegressor(n_estimators=863, max_depth=10, learning_rate=0.0144,
                       subsample=0.6557, colsample_bytree=0.5539, min_child_weight=1,
                       reg_alpha=1.1266, reg_lambda=3.9776,
                       random_state=SEED, n_jobs=-1, verbosity=0).fit(Xtr, ytr)
    errors["XGBoost"] = np.abs(yte.values - xgb.predict(Xte))

    svr = SVR(kernel="rbf", C=100.0, epsilon=5.0, gamma="scale").fit(scaler.transform(Xtr), ytr)
    errors["SVR"] = np.abs(yte.values - svr.predict(scaler.transform(Xte)))

    print(f"Test cases: {len(yte)}\n")
    print("Mean absolute error per model")
    for name, err in sorted(errors.items(), key=lambda kv: kv[1].mean()):
        print(f"  {name:20} {err.mean():.2f} mins")

    print("\nPaired comparisons (Wilcoxon signed rank on absolute errors)")
    rows = []
    for a, b in combinations(errors, 2):
        diff = errors[a] - errors[b]
        stat, p = stats.wilcoxon(errors[a], errors[b])
        better = a if errors[a].mean() < errors[b].mean() else b
        sig = "yes" if p < 0.05 else "no"
        rows.append({"model_a": a, "model_b": b,
                     "mae_a": errors[a].mean(), "mae_b": errors[b].mean(),
                     "mean_difference": diff.mean(), "p_value": p,
                     "significant_at_0.05": sig, "better_model": better})
        print(f"  {a:20} vs {b:20} p={p:.2e}  {sig:3}  better: {better}")

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "significance_tests.csv", index=False)

    names = list(errors)
    n = len(names)
    matrix = np.ones((n, n))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                matrix[i, j] = stats.wilcoxon(errors[a], errors[b])[1]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    masked = np.where(matrix < 1e-300, 1e-300, matrix)
    im = ax.imshow(np.log10(masked), cmap="RdYlGn_r", vmin=-50, vmax=0)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=10)
            else:
                mark = "*" if matrix[i, j] < 0.05 else ""
                ax.text(j, i, f"{matrix[i, j]:.1e}{mark}", ha="center",
                        va="center", fontsize=7.5)
    ax.set_title("Pairwise Significance (p values, * = significant at 0.05)",
                 fontweight="bold")
    fig.colorbar(im, ax=ax, label="log10 p value")
    fig.tight_layout()
    fig.savefig(OUT / "significance_tests.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT / 'significance_tests.png'}")


if __name__ == "__main__":
    run()
