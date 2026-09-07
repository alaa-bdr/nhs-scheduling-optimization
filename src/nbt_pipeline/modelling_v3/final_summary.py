"""Pulls every result into one table and one chart, so the final numbers sit in a single place."""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

V2 = Path("data/modelling_v2/plots")
V3 = Path("data/modelling_v3/plots")
OUT = V3


def load(path, label):
    if path.exists():
        return pd.read_csv(path)
    print(f"missing, run that stage first: {label}")
    return None


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    sections = []

    models = load(V2 / "model_results_v2.csv", "dashboard_v2")
    if models is not None:
        print("\nModel performance on the held out test set\n")
        print(models.round(3).to_string(index=False))
        sections.append(("Model performance", models))

    folds = load(V2 / "kfold_comparison.csv", "kfold_comparison")
    if folds is not None:
        print("\nFold count comparison\n")
        print(folds.round(4).to_string(index=False))
        sections.append(("Fold count", folds))

    groups = load(V3 / "groupkfold_comparison.csv", "groupkfold_comparison")
    if groups is not None:
        print("\nRandom folds against grouped folds\n")
        print(groups.round(4).to_string(index=False))
        sections.append(("Grouping scheme", groups))

    leak = load(V2 / "leakage_analysis.csv", "leakage_analysis")
    if leak is not None:
        print("\nEffect of the prediction horizon\n")
        print(leak.round(4).to_string(index=False))
        sections.append(("Prediction horizon", leak))

    outliers = load(V2 / "outlier_sensitivity.csv", "outlier_sensitivity")
    if outliers is not None:
        print("\nOutlier handling\n")
        print(outliers.round(4).to_string(index=False))
        sections.append(("Outlier handling", outliers))

    sig = load(V2 / "significance_tests.csv", "significance_tests")
    if sig is not None:
        print("\nPairwise significance\n")
        print(sig.round(5).to_string(index=False))
        sections.append(("Significance", sig))

    diag = load(V3 / "fold_diagnostic.csv", "fold_diagnostic")
    if diag is not None:
        print("\nPer fold breakdown\n")
        print(diag.round(3).to_string(index=False))
        sections.append(("Fold diagnostic", diag))

    with pd.ExcelWriter(OUT / "final_results.xlsx") as writer:
        for name, frame in sections:
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"\nSaved: {OUT / 'final_results.xlsx'} with {len(sections)} sheets")

    if models is not None:
        best = models.loc[models["R2"].idxmax()]
        hospital = models[models["Model"].str.contains("Hospital", case=False)]
        print("\nHeadline")
        print(f"  Best model: {best['Model']} at R2 {best['R2']:.3f}, "
              f"average error {best['MAE']:.1f} minutes")
        if len(hospital):
            h = hospital.iloc[0]
            print(f"  Hospital booking: R2 {h['R2']:.3f}, "
                  f"average error {h['MAE']:.1f} minutes")
            print(f"  Improvement: {h['MAE'] - best['MAE']:.1f} minutes per case")

        fig, ax = plt.subplots(figsize=(10, 5))
        plot = models.sort_values("MAE", ascending=False)
        colours = ["#B0B0B0" if "Hospital" in m else "#3366A6"
                   for m in plot["Model"]]
        colours[-1] = "#1B3A5C"
        bars = ax.barh(plot["Model"], plot["MAE"], color=colours,
                       edgecolor="white", height=0.6)
        for b, v in zip(bars, plot["MAE"]):
            ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2,
                    f"{v:.1f}", va="center", fontsize=10)
        ax.set_xlabel("Average error in minutes (lower is better)")
        ax.set_title("Final Results: Predicting Operation Duration",
                     fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "final_summary.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {OUT / 'final_summary.png'}")


if __name__ == "__main__":
    run()
