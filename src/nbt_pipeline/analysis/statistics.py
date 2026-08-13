"""Statistical tests used by the NBT theatre analysis notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


def _complete_pairs(data: pd.DataFrame, *columns: str) -> pd.DataFrame:
    return data.loc[:, columns].dropna().copy()


def kruskal_comparison(
    data: pd.DataFrame,
    category: str,
    outcome: str,
    min_group_size: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Kruskal-Wallis and FDR-corrected pairwise Mann-Whitney tests."""
    frame = _complete_pairs(data, category, outcome)
    counts = frame[category].value_counts()
    keep = counts[counts >= min_group_size].index
    frame = frame[frame[category].isin(keep)]
    groups = {
        str(name): values[outcome].to_numpy()
        for name, values in frame.groupby(category, observed=True)
    }
    if len(groups) < 2:
        return pd.DataFrame(), pd.DataFrame()

    statistic, p_value = stats.kruskal(*groups.values())
    n = len(frame)
    k = len(groups)
    epsilon_squared = max(0.0, (statistic - k + 1) / (n - k)) if n > k else np.nan
    omnibus = pd.DataFrame(
        [{
            "category": category,
            "outcome": outcome,
            "groups": k,
            "n": n,
            "kruskal_h": statistic,
            "p_value": p_value,
            "epsilon_squared": epsilon_squared,
        }]
    )

    rows = []
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            u_stat, pair_p = stats.mannwhitneyu(
                groups[left], groups[right], alternative="two-sided"
            )
            rows.append({
                "group_1": left,
                "group_2": right,
                "n_1": len(groups[left]),
                "n_2": len(groups[right]),
                "median_1": np.median(groups[left]),
                "median_2": np.median(groups[right]),
                "median_difference": np.median(groups[left]) - np.median(groups[right]),
                "u_statistic": u_stat,
                "p_value": pair_p,
            })
    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        rejected, adjusted, _, _ = multipletests(pairwise["p_value"], method="fdr_bh")
        pairwise["p_fdr_bh"] = adjusted
        pairwise["reject_fdr_0_05"] = rejected
    return omnibus, pairwise


def categorical_association(
    data: pd.DataFrame,
    category: str,
    outcome: str = "duration_status",
    min_group_size: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run chi-square and report bias-corrected Cramer's V."""
    frame = _complete_pairs(data, category, outcome)
    counts = frame[category].value_counts()
    frame = frame[frame[category].isin(counts[counts >= min_group_size].index)]
    table = pd.crosstab(frame[category], frame[outcome])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return pd.DataFrame(), table
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, cols = table.shape
    phi2_corrected = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / (n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / (n - 1)
    cols_corrected = cols - ((cols - 1) ** 2) / (n - 1)
    denominator = min(cols_corrected - 1, rows_corrected - 1)
    cramers_v = np.sqrt(phi2_corrected / denominator) if denominator > 0 else np.nan
    summary = pd.DataFrame([{
        "category": category,
        "outcome": outcome,
        "n": n,
        "chi_square": chi2,
        "degrees_of_freedom": dof,
        "p_value": p_value,
        "cramers_v_corrected": cramers_v,
        "minimum_expected_count": expected.min(),
        "cells_expected_below_5_pct": (expected < 5).mean() * 100,
    }])
    return summary, table


def spearman_test(data: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    frame = _complete_pairs(data, left, right)
    rho, p_value = stats.spearmanr(frame[left], frame[right])
    return pd.DataFrame([{
        "variable_1": left,
        "variable_2": right,
        "n": len(frame),
        "spearman_rho": rho,
        "p_value": p_value,
    }])


def missingness_association(
    data: pd.DataFrame,
    column: str,
    outcome: str = "meaningful_overrun_flag",
) -> pd.DataFrame:
    frame = data[[column, outcome]].copy()
    frame["is_missing"] = frame[column].isna()
    frame = frame.dropna(subset=[outcome])
    table = pd.crosstab(frame["is_missing"], frame[outcome])
    if table.shape != (2, 2):
        return pd.DataFrame()
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    rates = frame.groupby("is_missing")[outcome].mean()
    return pd.DataFrame([{
        "column": column,
        "n": len(frame),
        "missing_n": int(frame["is_missing"].sum()),
        "overrun_rate_present": rates.get(False, np.nan),
        "overrun_rate_missing": rates.get(True, np.nan),
        "chi_square": chi2,
        "p_value": p_value,
    }])


def _collapse_rare(series: pd.Series, minimum: int = 100) -> pd.Series:
    values = series.astype("string").fillna("Missing")
    counts = values.value_counts()
    frequent = set(counts[counts >= minimum].index) - {"Missing"}
    return values.where(values.eq("Missing") | values.isin(frequent), "Other")


def logistic_overrun_model(data: pd.DataFrame):
    """Fit the prespecified adjusted overrun model without outcome leakage."""
    columns = [
        "meaningful_overrun_flag",
        "ExpectedDurationMins",
        "age_at_operation",
        "TheatreRoom",
        "procedure_code_category",
        "admission_type_label",
        "ASAScore",
        "PriorityLevelCode",
        "anaesthetic_desc",
        "session_specialty",
    ]
    frame = data[columns].copy()
    frame = frame[frame["meaningful_overrun_flag"].notna()]
    frame = frame[frame["ExpectedDurationMins"].gt(0)]
    frame["age_at_operation"] = frame["age_at_operation"].fillna(
        frame["age_at_operation"].median()
    )
    for column in columns[3:]:
        frame[column] = _collapse_rare(frame[column], minimum=100)
    formula = (
        "meaningful_overrun_flag ~ np.log1p(ExpectedDurationMins) + "
        "age_at_operation + C(TheatreRoom) + C(procedure_code_category) + "
        "C(admission_type_label) + C(ASAScore) + C(PriorityLevelCode) + "
        "C(anaesthetic_desc) + C(session_specialty)"
    )
    model = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit(
        cov_type="HC3"
    )
    confidence = model.conf_int()
    results = pd.DataFrame({
        "term": model.params.index,
        "odds_ratio": np.exp(model.params.values),
        "ci_95_lower": np.exp(confidence[0].values),
        "ci_95_upper": np.exp(confidence[1].values),
        "p_value": model.pvalues.values,
    })
    rejected, adjusted, _, _ = multipletests(results["p_value"], method="fdr_bh")
    results["p_fdr_bh"] = adjusted
    results["reject_fdr_0_05"] = rejected
    return model, results, frame
