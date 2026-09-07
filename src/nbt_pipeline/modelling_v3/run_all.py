"""Runs the whole modelling pipeline in order, so the results can be reproduced from scratch."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

STAGES = [
    ("Prepare data splits", "nbt_pipeline.modelling.prepare", "quick"),
    ("Baseline and hospital benchmark", "nbt_pipeline.modelling.baseline", "quick"),
    ("Fold count comparison, 3 5 7 9", "nbt_pipeline.modelling_v2.kfold_comparison", "medium"),
    ("Outlier sensitivity", "nbt_pipeline.modelling_v2.outlier_sensitivity", "medium"),
    ("Leakage analysis", "nbt_pipeline.modelling_v2.leakage_analysis", "medium"),
    ("Random folds against grouped folds", "nbt_pipeline.modelling_v3.groupkfold_comparison", "medium"),
    ("Fold diagnostic", "nbt_pipeline.modelling_v3.fold_diagnostic", "medium"),
    ("Planning error analysis", "nbt_pipeline.modelling_v3.underrun_analysis", "quick"),
    ("Procedure and surgeon breakdown", "nbt_pipeline.modelling_v2.descriptive_v2", "quick"),
    ("All five models and charts", "nbt_pipeline.modelling_v2.dashboard_v2", "medium"),
    ("Significance testing", "nbt_pipeline.modelling_v2.significance_tests", "medium"),
    ("Two layer neural network", "nbt_pipeline.modelling_v2.neural_network_v2", "slow"),
    ("XGBoost hyperparameter search", "nbt_pipeline.modelling_v2.tune_xgboost_v2", "slow"),
]


def run_stage(name, module):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    start = time.time()
    result = subprocess.run([sys.executable, "-m", module])
    elapsed = time.time() - start
    ok = result.returncode == 0
    print(f"\n[{'ok' if ok else 'FAILED'}] {name} ({elapsed:.0f}s)")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-slow", action="store_true",
                        help="skip the long hyperparameter searches")
    parser.add_argument("--quick", action="store_true",
                        help="run only the fast stages")
    args = parser.parse_args()

    stages = STAGES
    if args.quick:
        stages = [s for s in stages if s[2] == "quick"]
    elif args.skip_slow:
        stages = [s for s in stages if s[2] != "slow"]

    print(f"Running {len(stages)} stages")
    failed = [name for name, module, _ in stages if not run_stage(name, module)]

    print(f"\n{'=' * 70}")
    print(f"Finished. {len(stages) - len(failed)} of {len(stages)} stages completed.")
    if failed:
        print("Failed stages:")
        for name in failed:
            print(f"  {name}")
        return 1
    print("Charts and result tables are in data/modelling_v2/plots and data/modelling_v3/plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
