"""It was best for us to look at the higher accuracy folds and understand what generalises."""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from nbt_pipeline.modelling_v2.features_v2 import build_v2_dataset

OUT = Path("data/modelling_v3/plots")
SEED = 42
CAP = 480
FOLDS = 5
TARGET = "operation_length_mins"

PARAMS = dict(n_estimators=600, max_depth=8, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.7, random_state=SEED,
              n_jobs=-1, verbosity=0)


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

    kf = KFold(FOLDS, shuffle=True, random_state=SEED)
    rows = []
    residual_store = np.zeros(len(X))

    for i, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        model = XGBRegressor(**PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        actual = y.iloc[test_idx].values
        residual_store[test_idx] = pred - actual

        fold_frame = frame.iloc[test_idx]
        rows.append({
            "fold": i,
            "R2": r2_score(actual, pred),
            "MAE": mean_absolute_error(actual, pred),
            "n": len(test_idx),
            "mean_duration": actual.mean(),
            "sd_duration": actual.std(),
            "pct_over_180min": float(np.mean(actual > 180)),
            "pct_emergency": float(fold_frame["admission_type"].astype(str).isin(["21", "22", "23", "24", "2A", "2B", "2D"]).mean()),
            "pct_general_anaesthetic": float(fold_frame["anaesthetic_desc"].astype(str).str.contains("GA", na=False).mean()),
            "mean_planned": float(fold_frame["ExpectedDurationMins"].mean()),
        })

    results = pd.DataFrame(rows)
    print("Per fold performance and composition\n")
    print(results.round(3).to_string(index=False))
    results.to_csv(OUT / "fold_diagnostic.csv", index=False)

    best = results.loc[results["R2"].idxmax()]
    worst = results.loc[results["R2"].idxmin()]
    print(f"\nBest fold {int(best['fold'])}: R2={best['R2']:.4f}")
    print(f"Worst fold {int(worst['fold'])}: R2={worst['R2']:.4f}")
    print(f"Spread: {best['R2'] - worst['R2']:.4f}\n")

    print("What differs between the best and worst folds")
    for col in ["sd_duration", "pct_over_180min", "pct_emergency",
                "pct_general_anaesthetic", "mean_duration"]:
        print(f"  {col:26} best={best[col]:.3f}  worst={worst[col]:.3f}")

    print("\nCorrelation of fold composition with fold R2")
    for col in ["sd_duration", "pct_over_180min", "pct_emergency",
                "pct_general_anaesthetic", "mean_duration"]:
        r = results["R2"].corr(results[col])
        print(f"  {col:26} r={r:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = axes[0].bar(results["fold"].astype(str), results["R2"],
                       color="#3366A6", edgecolor="white")
    for b, v in zip(bars, results["R2"]):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.004,
                     f"{v:.3f}", ha="center", fontsize=10)
    axes[0].axhline(results["R2"].mean(), linestyle="--", color="#D95F02",
                    label=f"mean {results['R2'].mean():.3f}")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("R2")
    axes[0].set_ylim(0, 0.85)
    axes[0].set_title("Performance Varies Between Folds", fontweight="bold")
    axes[0].legend(fontsize=9)

    axes[1].scatter(results["sd_duration"], results["R2"], s=110,
                    color="#1B3A5C", zorder=3)
    for _, r in results.iterrows():
        axes[1].annotate(f"fold {int(r['fold'])}",
                         (r["sd_duration"], r["R2"]),
                         textcoords="offset points", xytext=(7, 5), fontsize=9)
    axes[1].set_xlabel("Spread of durations within the fold (sd, mins)")
    axes[1].set_ylabel("Fold R2")
    axes[1].set_title("Folds With More Variable Cases Are Harder", fontweight="bold")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fold_diagnostic.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT / 'fold_diagnostic.png'}")


if __name__ == "__main__":
    run()
