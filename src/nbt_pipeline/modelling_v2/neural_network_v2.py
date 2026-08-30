"""Two layer neural network for operation duration prediction."""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    print(f"Train {len(Xtr)}, test {len(Xte)}, features {X.shape[1]}")

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("net", MLPRegressor(max_iter=300, early_stopping=True,
                             n_iter_no_change=10, random_state=SEED)),
    ])
    grid = {
        "net__hidden_layer_sizes": [(64, 32), (128, 64), (256, 128)],
        "net__alpha": [0.0001, 0.01],
    }
    print("Two layer network, 6 combinations x 3 folds = 18 fits")

    search = GridSearchCV(pipeline, grid,
                          cv=KFold(3, shuffle=True, random_state=SEED),
                          scoring="r2", n_jobs=-1, verbose=1)
    search.fit(Xtr, ytr)

    print(f"Best architecture: {search.best_params_['net__hidden_layer_sizes']}")
    print(f"Best alpha: {search.best_params_['net__alpha']}")
    print(f"Best CV R2: {search.best_score_:.4f}")

    pred = search.best_estimator_.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    r2 = r2_score(yte, pred)
    print(f"Test MAE  = {mae:.2f}")
    print(f"Test RMSE = {rmse:.2f}")
    print(f"Test R2   = {r2:.4f}")

    pd.DataFrame([{"Model": "Neural network (two layer)", "MAE": mae,
                   "RMSE": rmse, "R2": r2,
                   "architecture": str(search.best_params_["net__hidden_layer_sizes"])}
                  ]).to_csv(OUT / "neural_network_v2.csv", index=False)

    scores = pd.DataFrame(search.cv_results_)
    scores["arch"] = scores["param_net__hidden_layer_sizes"].astype(str)
    best = scores.groupby("arch")["mean_test_score"].max().sort_values()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(best.index, best.values, color="#3366A6", edgecolor="white", height=0.6)
    for b, v in zip(bars, best.values):
        ax.text(b.get_width() + 0.004, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Best cross validation R2")
    ax.set_ylabel("Hidden layer sizes")
    ax.set_title("Two Layer Network: Architecture Comparison", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "neural_network_v2.png", dpi=150, bbox_inches="tight")
    print("Saved: neural_network_v2.png")


if __name__ == "__main__":
    run()
