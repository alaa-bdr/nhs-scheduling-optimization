import numpy as np
import pandas as pd

from nbt_pipeline.modeling import (
    APPROVED_PREDICTORS,
    aggregate_tree_importance,
    build_supervised_pipeline,
    classification_models,
    feature_configurations,
    prepare_target_data,
    regression_models,
)


def sample_analysis_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ExpectedDurationMins": [60, 90, 45],
            "age_at_operation": [40, np.nan, 70],
            "priority_level_label": ["P2", None, "P4"],
            "TheatreRoom": ["A", "B", "C"],
            "operation_start_hour": [8, 10, np.nan],
            "operation_length_mins": [75, 80, np.nan],
            "duration_error_mins": [15, -10, np.nan],
            "meaningful_overrun_flag": [1, 0, np.nan],
            "duration_timing_review_flag": [False, True, False],
        }
    )


def test_prepare_target_data_excludes_missing_target_and_all_leakage() -> None:
    prepared = prepare_target_data(sample_analysis_data(), "meaningful_overrun_flag")

    assert prepared.target.tolist() == [1, 0]
    assert prepared.excluded_target_rows == 1
    assert "operation_length_mins" not in prepared.predictors
    assert "duration_error_mins" not in prepared.predictors
    assert "meaningful_overrun_flag" not in prepared.predictors
    assert "operation_start_hour" in prepared.predictors
    assert prepared.review_flag.tolist() == [False, True]


def test_feature_configurations_use_only_approved_fields_and_no_theatre_area() -> None:
    configurations = feature_configurations()

    assert "TheatreRoom" in configurations["Full approved"]
    assert "TheatreRoom" not in configurations["Full without location"]
    assert all("theatre_area" not in columns for columns in configurations.values())
    assert all(set(columns).issubset(APPROVED_PREDICTORS) for columns in configurations.values())


def test_missing_aware_pipeline_fits_without_replacing_source_dataframe() -> None:
    prepared = prepare_target_data(sample_analysis_data(), "meaningful_overrun_flag")
    X = prepared.predictors[["ExpectedDurationMins", "age_at_operation", "priority_level_label"]]
    original = X.copy(deep=True)
    pipeline = build_supervised_pipeline(
        classification_models(prepared.target)["Logistic regression"],
        X,
        missing_strategy="missing_aware",
    )

    pipeline.fit(X, prepared.target)

    pd.testing.assert_frame_equal(X, original)
    assert pipeline.predict_proba(X).shape == (2, 2)


def test_registries_contain_five_models_plus_benchmark() -> None:
    assert len(regression_models()) == 7
    assert len(classification_models(pd.Series([0, 0, 1]))) == 7
    assert "Neural network" in regression_models()
    assert "Neural network" in classification_models(pd.Series([0, 0, 1]))


def test_predictor_groups_are_order_stable_integer_codes() -> None:
    from nbt_pipeline.modeling import predictor_row_groups

    frame = pd.DataFrame({"a": ["x", "y", "x"], "b": [1, 2, 1]})

    assert predictor_row_groups(frame).tolist() == [0, 1, 0]


def test_tree_importance_is_aggregated_to_original_columns() -> None:
    from sklearn.tree import DecisionTreeClassifier

    frame = pd.DataFrame({"numeric": [1, 2, 3, 4], "category": ["a", "a", "b", "b"]})
    target = pd.Series([0, 0, 1, 1])
    pipeline = build_supervised_pipeline(
        DecisionTreeClassifier(random_state=42), frame, missing_strategy="missing_aware"
    ).fit(frame, target)

    result = aggregate_tree_importance(pipeline, frame.columns.tolist())

    assert set(result["feature"]).issubset({"numeric", "category"})
    assert np.isclose(result["importance"].sum(), 1.0)
