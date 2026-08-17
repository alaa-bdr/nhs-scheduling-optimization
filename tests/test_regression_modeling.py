import pandas as pd

from nbt_pipeline.modeling import (
    OUTCOME_OR_LEAKAGE_COLUMNS,
    get_regression_models,
    make_predictor_sets,
    predictor_row_groups,
)


def test_make_predictor_sets_separates_target_leakage_and_start_hour() -> None:
    source = pd.DataFrame(
        {
            "TheatreRoom": ["A", "B"],
            "ExpectedDurationMins": [60, 90],
            "operation_start_hour": [8, 10],
            "operation_length_mins": [75, 80],
            "duration_error_mins": [15, -10],
            "meaningful_overrun_flag": [1, 0],
        }
    )

    primary, sensitivity, target = make_predictor_sets(source)

    assert "operation_start_hour" not in primary
    assert "operation_start_hour" in sensitivity
    assert not set(OUTCOME_OR_LEAKAGE_COLUMNS).intersection(primary.columns)
    assert not set(OUTCOME_OR_LEAKAGE_COLUMNS).intersection(sensitivity.columns)
    assert target.tolist() == [15, -10]


def test_predictor_row_groups_match_for_identical_rows() -> None:
    predictors = pd.DataFrame(
        {"room": ["A", "A", "B"], "age": [40, 40, 50]}, index=[2, 5, 9]
    )

    groups = predictor_row_groups(predictors)

    assert groups.iloc[0] == groups.iloc[1]
    assert groups.iloc[0] != groups.iloc[2]


def test_make_predictor_sets_excludes_rows_without_a_target() -> None:
    source = pd.DataFrame(
        {
            "TheatreRoom": ["A", "B", "C"],
            "duration_error_mins": [5.0, None, -2.0],
        }
    )

    primary, sensitivity, target = make_predictor_sets(source)

    assert primary.index.tolist() == [0, 2]
    assert sensitivity.index.tolist() == [0, 2]
    assert target.tolist() == [5.0, -2.0]


def test_model_registry_contains_benchmark_and_five_models() -> None:
    assert list(get_regression_models()) == [
        "Dummy median benchmark",
        "Linear regression",
        "Decision tree",
        "Random forest",
        "SVR",
        "XGBoost",
    ]
