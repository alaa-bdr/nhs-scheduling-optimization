"""Run all five models on the v2 feature set and produce every chart."""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def score(name, true, pred, rows):
    r = {"Model": name,
         "MAE": mean_absolute_error(true, pred),
         "RMSE": mean_squared_error(true, pred) ** 0.5,
         "R2": r2_score(true, pred)}
    rows.append(r)
    print(f"{name:24} MAE={r['MAE']:6.2f}  RMSE={r['RMSE']:6.2f}  R2={r['R2']:.4f}")
    return r


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, features, target = build_v2_dataset(duration_cap=CAP)
    X = encode(frame, features)
    y = frame[target]
    idx = np.arange(len(frame))

    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, idx, test_size=0.2, random_state=SEED)
    print(f"Train {len(Xtr)}, test {len(Xte)}, features {X.shape[1]}\n")

    rows, preds = [], {}

    planned = Xte["ExpectedDurationMins"].values
    ok = planned > 0
    score("Hospital estimate", yte.values[ok], planned[ok], rows)
    preds["Hospital estimate"] = (yte.values[ok], planned[ok])

    sc = StandardScaler().fit(Xtr)
    ridge = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr)
    p = ridge.predict(sc.transform(Xte))
    score("Ridge regression", yte.values, p, rows)
    preds["Ridge regression"] = (yte.values, p)

    rf = RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=5,
                               random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    p = rf.predict(Xte)
    score("Random Forest", yte.values, p, rows)
    preds["Random Forest"] = (yte.values, p)

    xgb = XGBRegressor(n_estimators=863, max_depth=10, learning_rate=0.0144,
                       subsample=0.6557, colsample_bytree=0.5539, min_child_weight=1,
                       reg_alpha=1.1266, reg_lambda=3.9776,
                       random_state=SEED, n_jobs=-1, verbosity=0).fit(Xtr, ytr)
    p = xgb.predict(Xte)
    score("XGBoost", yte.values, p, rows)
    preds["XGBoost"] = (yte.values, p)

    svr = SVR(kernel="rbf", C=100.0, epsilon=5.0, gamma="scale").fit(sc.transform(Xtr), ytr)
    p = svr.predict(sc.transform(Xte))
    score("SVR", yte.values, p, rows)
    preds["SVR"] = (yte.values, p)

    notes = frame["theatre_notes"].fillna("").astype(str)
    tf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=5, stop_words="english")
    Ttr = tf.fit_transform(notes.iloc[itr])
    Tte = tf.transform(notes.iloc[ite])
    nlp = Ridge(alpha=1.0).fit(Ttr, ytr)
    p = nlp.predict(Tte)
    score("NLP (text only)", yte.values, p, rows)

    results = pd.DataFrame(rows).sort_values("MAE", ascending=False).reset_index(drop=True)
    results.to_csv(OUT / "model_results_v2.csv", index=False)

    colours = ["#B0B0B0", "#C49A6C", "#9BB7D4", "#6699CC", "#3366A6", "#1B3A5C"][:len(results)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, met in zip(axes, ["MAE", "RMSE", "R2"]):
        bars = ax.barh(results["Model"], results[met], color=colours,
                       edgecolor="white", height=0.65)
        for b, v in zip(bars, results[met]):
            ax.text(b.get_width() + (0.4 if met != "R2" else 0.005),
                    b.get_y() + b.get_height() / 2,
                    f"{v:.1f}" if met != "R2" else f"{v:.3f}", va="center", fontsize=10)
        ax.set_xlabel("Minutes" if met != "R2" else "Score (1.0 = perfect)")
        ax.set_title(met, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model Performance v2: All Five Models vs Hospital Estimate",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison_v2.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: model_comparison_v2.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name, col in zip(axes2, ["Hospital estimate", "XGBoost"], ["#B0B0B0", "#1B3A5C"]):
        t, p = preds[name]
        ax.scatter(t, p, alpha=0.15, s=8, color=col)
        lim = [0, max(t.max(), p.max()) * 1.05]
        ax.plot(lim, lim, "--", color="#D95F02", linewidth=1.5, label="Perfect prediction")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual duration (mins)")
        ax.set_ylabel("Predicted duration (mins)")
        ax.set_title(name, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig2.suptitle("Predicted vs Actual, v2 Features", fontsize=14, fontweight="bold", y=1.02)
    fig2.tight_layout()
    fig2.savefig(OUT / "predicted_vs_actual_v2.png", dpi=150, bbox_inches="tight")
    print("Saved: predicted_vs_actual_v2.png")

    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 6))
    for ax, model, name in zip(axes3, [rf, xgb], ["Random Forest", "XGBoost"]):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(15)
        imp.plot.barh(ax=ax, color="#3366A6", edgecolor="white")
        ax.set_title(f"{name} Feature Importances", fontweight="bold")
        ax.set_xlabel("Importance")
        ax.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()
    fig3.savefig(OUT / "feature_importances_v2.png", dpi=150, bbox_inches="tight")
    print("Saved: feature_importances_v2.png")

    t, p = preds["XGBoost"]
    err = p - t
    band = pd.cut(t, [0, 30, 60, 120, 240, 10000],
                  labels=["0-30", "30-60", "60-120", "120-240", "240+"])
    grouped = pd.DataFrame({"band": band, "abs_err": np.abs(err)}).groupby("band", observed=True)["abs_err"].mean()

    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))
    axes4[0].bar(grouped.index.astype(str), grouped.values, color="#3366A6", edgecolor="white")
    for i, v in enumerate(grouped.values):
        axes4[0].text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=10)
    axes4[0].set_xlabel("Actual operation length (mins)")
    axes4[0].set_ylabel("Mean absolute error (mins)")
    axes4[0].set_title("Error by Operation Length", fontweight="bold")

    axes4[1].hist(err, bins=60, color="#6699CC", edgecolor="white")
    axes4[1].axvline(0, color="#D95F02", linestyle="--", linewidth=1.5)
    axes4[1].set_xlabel("Prediction error (predicted minus actual, mins)")
    axes4[1].set_ylabel("Number of operations")
    axes4[1].set_title("Prediction Error Distribution", fontweight="bold")

    for ax in axes4:
        ax.spines[["top", "right"]].set_visible(False)
    fig4.suptitle("Error Analysis v2 (XGBoost)", fontsize=14, fontweight="bold", y=1.02)
    fig4.tight_layout()
    fig4.savefig(OUT / "error_analysis_v2.png", dpi=150, bbox_inches="tight")
    print("Saved: error_analysis_v2.png")


if __name__ == "__main__":
    run()
