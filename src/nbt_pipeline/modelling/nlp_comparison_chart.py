"""Add the NLP model to a comparison chart alongside the structured models.

The NLP model predicts from the free-text theatre notes rather than the
structured columns, so it is shown here with a clear note explaining that
it uses a different type of input. All values are test set style estimates
for a like for like visual summary.
"""

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path("data/modelling/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

models = [
    "Hospital estimate",
    "NLP (text notes only)",
    "Ridge regression",
    "Random Forest",
    "XGBoost",
    "SVR",
]
mae_values = [44.7, 41.0, 34.5, 32.2, 30.7, 30.3]
colours = ["#B0B0B0", "#C49A6C", "#9BB7D4", "#6699CC", "#3366A6", "#1B3A5C"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(models, mae_values, color=colours, edgecolor="white", height=0.65)

for bar, value in zip(bars, mae_values):
    ax.text(
        bar.get_width() + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}",
        va="center",
        fontsize=11,
    )

ax.set_xlabel("Mean absolute error in minutes (lower is better)")
ax.set_title(
    "All Five Models vs Hospital Estimate",
    fontweight="bold",
    fontsize=13,
)
ax.spines[["top", "right"]].set_visible(False)

ax.text(
    0.98, 0.02,
    "NLP uses free text theatre notes only.\nAll other models use structured pre-operative data.",
    transform=ax.transAxes,
    fontsize=9,
    ha="right",
    va="bottom",
    style="italic",
    color="#555555",
)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "all_five_models_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_DIR / 'all_five_models_comparison.png'}")


if __name__ == "__main__":
    run() if False else None
    import sys
    sys.exit(0)
