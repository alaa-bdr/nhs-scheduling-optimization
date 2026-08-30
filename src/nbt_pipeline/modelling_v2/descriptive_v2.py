"""Descriptive analysis of procedure frequency and operator variation."""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nbt_pipeline.preprocessing import build_preprocessed_dataset

OUT = Path("data/modelling_v2/plots")
TARGET = "operation_length_mins"
MIN_CASES = 40


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_preprocessed_dataset()
    df = df.dropna(subset=[TARGET])
    df["planning_error"] = df[TARGET] - df["ExpectedDurationMins"]

    print("=== MOST FREQUENT PROCEDURES ===")
    freq = df["ProcedureDescription"].value_counts().head(12)
    for name, n in freq.items():
        print(f"  {n:5d}  {str(name)[:70]}")
    freq.to_csv(OUT / "procedure_frequency.csv")

    print("\n=== PROCEDURES WITH LARGEST OVERRUN (min 40 cases) ===")
    proc = df.groupby("ProcedureDescription").agg(
        n=("planning_error", "size"),
        median_error=("planning_error", "median"),
        median_length=(TARGET, "median"),
    )
    proc = proc[proc["n"] >= MIN_CASES].sort_values("median_error", ascending=False)
    print(proc.head(8).round(1).to_string())
    proc.to_csv(OUT / "procedure_overrun.csv")

    surgeon_col = "theat_surg_1_national_code"
    if surgeon_col in df.columns:
        print(f"\n=== OPERATOR VARIATION (min {MIN_CASES} cases) ===")
        surg = df.groupby(surgeon_col).agg(
            n=("planning_error", "size"),
            median_error=("planning_error", "median"),
            median_length=(TARGET, "median"),
        )
        surg = surg[surg["n"] >= MIN_CASES].sort_values("median_error", ascending=False)
        print(f"Operators with at least {MIN_CASES} cases: {len(surg)}")
        print("\nLargest median overrun:")
        print(surg.head(5).round(1).to_string())
        print("\nLargest median underrun:")
        print(surg.tail(5).round(1).to_string())
        surg.to_csv(OUT / "operator_variation.csv")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        top = freq.head(10)[::-1]
        labels = [str(i)[:42] for i in top.index]
        axes[0].barh(labels, top.values, color="#3366A6", edgecolor="white")
        axes[0].set_xlabel("Number of operations")
        axes[0].set_title("Most Frequent Procedures", fontweight="bold")

        axes[1].hist(surg["median_error"], bins=25, color="#6699CC", edgecolor="white")
        axes[1].axvline(0, color="#D95F02", linestyle="--", linewidth=1.5)
        axes[1].set_xlabel("Median planning error (mins)")
        axes[1].set_ylabel("Number of operators")
        axes[1].set_title(
            f"Planning Error Varies Between Operators (n={len(surg)})",
            fontweight="bold")

        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "descriptive_v2.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved: {OUT / 'descriptive_v2.png'}")

        spread = surg["median_error"].max() - surg["median_error"].min()
        print(f"Spread in median planning error across operators: {spread:.0f} mins")


if __name__ == "__main__":
    run()
