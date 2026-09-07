"""Most operations finish early rather than late, so this looks at the size and direction of that planning error."""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nbt_pipeline.preprocessing import build_preprocessed_dataset

OUT = Path("data/modelling_v3/plots")
TARGET = "operation_length_mins"
PLANNED = "ExpectedDurationMins"
MIN_CASES = 40


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_preprocessed_dataset().dropna(subset=[TARGET, PLANNED])
    df = df[df[PLANNED] > 0]
    df["error"] = df[TARGET] - df[PLANNED]

    total = len(df)
    under = (df["error"] < 0).sum()
    over = (df["error"] > 0).sum()
    exact = (df["error"] == 0).sum()

    print("Direction of planning error\n")
    print(f"  Cases analysed          {total}")
    print(f"  Finished early          {under} ({under / total:.1%})")
    print(f"  Ran over                {over} ({over / total:.1%})")
    print(f"  Exactly on plan         {exact} ({exact / total:.1%})")
    print(f"\n  Median error            {df['error'].median():.0f} mins")
    print(f"  Mean error              {df['error'].mean():.1f} mins")

    early = df.loc[df["error"] < 0, "error"]
    late = df.loc[df["error"] > 0, "error"]
    print(f"\n  Median minutes early    {abs(early.median()):.0f}")
    print(f"  Median minutes late     {late.median():.0f}")

    wasted = abs(early.sum())
    print(f"\n  Total theatre minutes booked but unused: {wasted:,.0f}")
    print(f"  Equivalent full theatre days (8 hours):  {wasted / 480:,.0f}")

    print("\nPlanned durations are template values, not estimates")
    top = df[PLANNED].value_counts().head(6)
    for value, count in top.items():
        print(f"  {value:.0f} mins used {count} times ({count / total:.1%})")
    print(f"  Six values account for {top.sum() / total:.1%} of all bookings")

    print(f"\nSpecialties that most over book (min {MIN_CASES} cases)")
    spec = df.groupby("session_specialty").agg(
        n=("error", "size"),
        median_error=("error", "median"),
        pct_early=("error", lambda s: float((s < 0).mean())),
    )
    spec = spec[spec["n"] >= MIN_CASES].sort_values("median_error")
    print(spec.head(6).round(2).to_string())
    spec.to_csv(OUT / "underrun_by_specialty.csv")

    print("\nProcedures that most over book")
    proc = df.groupby("ProcedureDescription").agg(
        n=("error", "size"), median_error=("error", "median"))
    proc = proc[proc["n"] >= MIN_CASES].sort_values("median_error")
    print(proc.head(6).round(1).to_string())
    proc.to_csv(OUT / "underrun_by_procedure.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(df["error"], bins=70, color="#6699CC", edgecolor="white")
    ax.axvline(0, color="#D95F02", linestyle="--", linewidth=1.6, label="on plan")
    ax.axvline(df["error"].median(), color="#1B3A5C", linestyle="-",
               linewidth=1.6, label=f"median {df['error'].median():.0f} mins")
    ax.set_xlabel("Planning error (realised minus planned, mins)")
    ax.set_ylabel("Number of operations")
    ax.set_title("Most Operations Finish Early", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    labels = ["Finished early", "Ran over", "Exactly on plan"]
    values = [under, over, exact]
    ax.bar(labels, values, color=["#3366A6", "#C97B4E", "#B0B0B0"], edgecolor="white")
    for i, v in enumerate(values):
        ax.text(i, v + total * 0.01, f"{v}\n({v / total:.0%})",
                ha="center", fontsize=10)
    ax.set_ylabel("Number of operations")
    ax.set_title("Direction of Planning Error", fontweight="bold")

    ax = axes[1, 0]
    counts = df[PLANNED].value_counts().head(10).sort_values()
    ax.barh([f"{int(v)} mins" for v in counts.index], counts.values,
            color="#3366A6", edgecolor="white")
    ax.set_xlabel("Number of bookings")
    ax.set_title("Planned Durations Cluster on Round Numbers", fontweight="bold")

    ax = axes[1, 1]
    plot_spec = spec.head(8).sort_values("median_error", ascending=False)
    colours = ["#3366A6" if v < 0 else "#C97B4E" for v in plot_spec["median_error"]]
    ax.barh(plot_spec.index.astype(str), plot_spec["median_error"],
            color=colours, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Median planning error (mins, negative means finished early)")
    ax.set_title("Over Booking Varies by Specialty", fontweight="bold")

    for row in axes:
        for ax in row:
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Planning Error Analysis: Theatre Time Is Systematically Over Booked",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "underrun_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT / 'underrun_analysis.png'}")


if __name__ == "__main__":
    run()
