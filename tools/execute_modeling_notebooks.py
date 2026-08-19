"""Execute and validate supervised-modelling notebooks."""

from argparse import ArgumentParser

from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "duration_error_mins": "nbt_duration_error_regression.ipynb",
    "operation_length_mins": "nbt_operation_length_regression.ipynb",
    "meaningful_overrun_flag": "nbt_meaningful_overrun_classification.ipynb",
}


parser = ArgumentParser()
parser.add_argument(
    "--target",
    choices=tuple(NOTEBOOKS),
    action="append",
    help="Target notebook to execute. Repeat to execute more than one. Defaults to all targets.",
)
args = parser.parse_args()
targets = tuple(args.target) if args.target else tuple(NOTEBOOKS)

for target in targets:
    filename = NOTEBOOKS[target]
    path = ROOT / "notebooks" / filename
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=3600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"Executed {path}")


summary_rows = []
importance_frames = []
for target in NOTEBOOKS:
    result_dir = ROOT / "result" / "modeling" / target
    selection = pd.read_csv(result_dir / "selection_summary.csv").set_index("selection")["value"]
    metrics = pd.read_csv(result_dir / "final_test_metrics.csv").iloc[0]
    if target == "meaningful_overrun_flag":
        main_metric, main_value = "PR-AUC", metrics["PR-AUC"]
        secondary_metric, secondary_value = "Recall", metrics["recall"]
    else:
        main_metric, main_value = "MAE", metrics["MAE"]
        secondary_metric, secondary_value = "R2", metrics["R2"]
    summary_rows.append(
        {
            "target": target,
            "winning missing strategy": selection["missing strategy"],
            "winning feature configuration": selection["feature configuration"],
            "winning model": selection["model"],
            "primary metric": main_metric,
            "primary value": main_value,
            "secondary metric": secondary_metric,
            "secondary value": secondary_value,
        }
    )
    importance = pd.read_csv(result_dir / "permutation_importance.csv")
    importance.insert(0, "target", target)
    importance_frames.append(importance)

summary_dir = ROOT / "result" / "modeling"
winner_summary = pd.DataFrame(summary_rows)
cross_target_importance = pd.concat(importance_frames, ignore_index=True)
winner_summary.to_csv(summary_dir / "cross_target_winner_summary.csv", index=False)
cross_target_importance.to_csv(summary_dir / "cross_target_feature_importance.csv", index=False)

summary_path = ROOT / "notebooks" / "nbt_modeling_summary.ipynb"
summary_notebook = nbformat.v4.new_notebook(
    cells=[
        nbformat.v4.new_markdown_cell(
            """# NBT modelling summary

This notebook compares the winning data treatment, feature configuration and algorithm across the three prespecified prediction targets. Model selection occurred in development cross-validation; the values below are from the frozen untouched test sets."""
        ),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
RESULT_DIR = PROJECT_ROOT / "result" / "modeling"
winner_summary = pd.read_csv(RESULT_DIR / "cross_target_winner_summary.csv")
winner_summary"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Winning type of data

The winner table states whether priority was retained through the selected missing strategy, which feature representation won, and whether the neural network or another algorithm performed best. Start hour, flagged-record exclusion and complete cases remain sensitivity analyses rather than primary data choices."""
        ),
        nbformat.v4.new_code_cell(
            """importance = pd.read_csv(RESULT_DIR / "cross_target_feature_importance.csv")
top_features = (
    importance.sort_values(["target", "rank"])
    .groupby("target", as_index=False, group_keys=False)
    .head(10)
)
top_features"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Interpretation

Permutation importance measures predictive contribution on the frozen test population and is not a causal effect. Correlated features can share importance. A low-ranked feature may still be clinically important, and external validation remains necessary before deployment."""
        ),
    ]
)
summary_notebook.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3"
}
nbformat.write(summary_notebook, summary_path)
summary_client = NotebookClient(
    summary_notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
)
summary_client.execute()
nbformat.write(summary_notebook, summary_path)
print(f"Executed {summary_path}")
