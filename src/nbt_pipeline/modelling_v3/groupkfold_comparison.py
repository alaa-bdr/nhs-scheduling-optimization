"""Compares random folds against grouped folds, since grouping changes how optimistic the score is."""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from nbt_pipeline.modelling_v2.features_v2 import build_v2_dataset

OUT = Path("data/modelling_v3/plots")
SEED = 42
CAP = 480
FOLDS = 5

GROUPINGS = {
    "surgeon": "theat_surg_1_national_code",
    "procedure": "actual_proc_1_procedure_code",
    "theatre room": "TheatreRoom",
    "consultant": "listing_cons_code",
}

PARAMS = dict(n_estimators=600, max_depth=8, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.7, random_state=SEED,
              n_jobs=-1, verbosity=0)


def encode(frame, features):
    X = frame[features].copy()
    for c in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].fillna("missing").astype(str))
    return X.fillna(-1)


def evaluate(X, y, cv, groups, label, rows):
    scores = cross_validate(XGBRegressor(**PARAMS), X, y, cv=cv, groups=groups,
                            scoring=["r2", "neg_mean_absolute_error"], n_jobs=-1)
    r2 = scores["test_r2"]
    mae = -scores["test_neg_mean_absolute_error"]
    rows.append({"scheme": label, "mean_R2": r2.mean(), "sd_R2": r2.std(),
                 "min_R2": r2.min(), "max_R2": r2.max(), "mean_MAE": mae.mean()})
    print(f"{label:28} R2={r2.mean():.4f} sd={r2.std():.4f} "
          f"range=[{r2.min():.3f}, {r2.max():.3f}]  MAE={mae.mean():.2f}")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, features, target = build_v2_dataset(duration_cap=CAP)
    X = encode(frame, features)
    y = frame[target]
    print(f"Rows {len(X)}, features {X.shape[1]}, folds {FOLDS}\n")

    rows = []
    evaluate(X, y, KFold(FOLDS, shuffle=True, random_state=SEED), None,
             "random KFold", rows)

    for name, col in GROUPINGS.items():
        if col not in frame.columns:
            continue
        groups = frame[col].fillna("missing").astype(str)
        print(f"  ({name}: {groups.nunique()} groups)")
        evaluate(X, y, GroupKFold(FOLDS), groups, f"GroupKFold by {name}", rows)

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "groupkfold_comparison.csv", index=False)

    random_r2 = results.iloc[0]["mean_R2"]
    print(f"\nRandom KFold gives {random_r2:.4f}.")
    for _, r in results.iloc[1:].iterrows():
        gap = random_r2 - r["mean_R2"]
        print(f"  {r['scheme']:28} is {gap:+.4f} lower")

    colours = ["#1B3A5C"] + ["#6699CC"] * (len(results) - 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(results["scheme"], results["mean_R2"],
                   xerr=results["sd_R2"], color=colours, edgecolor="white",
                   height=0.6, capsize=4)
    for b, v in zip(bars, results["mean_R2"]):
        ax.text(b.get_width() + 0.012, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Mean cross validation R2 (bars show spread across folds)")
    ax.set_xlim(0, 0.85)
    ax.set_title("Random KFold vs GroupKFold: Effect of Grouping on Reported Accuracy",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "groupkfold_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT / 'groupkfold_comparison.png'}")


if __name__ == "__main__":
    run()
