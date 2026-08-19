import numpy as np
import pandas as pd

from nbt_pipeline.modeling import (
    APPROVED_PREDICTORS,
    REQUIRED_NON_IMPUTED_PREDICTORS,
    aggregate_tree_importance,
    build_supervised_pipeline,
    classification_models,
    feature_configurations,
    final_model_spec,
    prepare_target_data,
    regression_models,
    save_final_model_artifact,
    train_final_model,
)


def sample_analysis_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ExpectedDurationMins": [60, 90, 45],
            "sex_national_code": ["1", "2", "1"],
            "age_at_operation": [40, np.nan, 70],
            "ASAScore": ["2", "3", "1"],
            "anaesthetic_desc": ["GA", "LA", "Sedation"],
            "admission_type_label": ["Day case", "Inpatient", "Day case"],
            "intended_management_label": ["Elective", "Emergency", "Elective"],
            "priority_level_label": ["P2", None, "P4"],
            "procedure_code_group": ["A", "B", "A"],
            "procedure_code_category": ["A01", "B01", "A02"],
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
    assert prepared.excluded_required_predictor_rows == 0
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
    assert REQUIRED_NON_IMPUTED_PREDICTORS == ("ExpectedDurationMins",)


def test_prepare_target_data_excludes_missing_required_expected_duration() -> None:
    source = sample_analysis_data()
    extra = source.iloc[[0]].copy()
    extra["ExpectedDurationMins"] = np.nan
    extra["meaningful_overrun_flag"] = 1
    source = pd.concat([source, extra], ignore_index=True)

    prepared = prepare_target_data(source, "meaningful_overrun_flag")

    assert prepared.target.tolist() == [1, 0]
    assert prepared.excluded_target_rows == 1
    assert prepared.excluded_required_predictor_rows == 1
    assert prepared.predictors["ExpectedDurationMins"].notna().all()


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


def test_final_model_spec_uses_primary_winning_columns_without_start_hour() -> None:
    spec = final_model_spec("meaningful_overrun_flag")

    assert spec.model_name == "XGBoost"
    assert spec.feature_configuration == "Both procedure levels"
    assert spec.threshold == 0.54
    assert "ExpectedDurationMins" in spec.columns
    assert "operation_start_hour" not in spec.columns
    assert "TheatreRoom" not in spec.columns


def test_save_final_model_artifact_writes_pipeline_and_metadata(tmp_path) -> None:
    frame = sample_analysis_data().iloc[:2].copy()

    metadata = save_final_model_artifact(frame, "duration_error_mins", tmp_path)

    assert metadata["training_rows"] == 2
    assert metadata["columns"] == list(final_model_spec("duration_error_mins").columns)
    assert (tmp_path / "duration_error_mins" / "final_pipeline.joblib").exists()
    assert (tmp_path / "duration_error_mins" / "metadata.json").exists()


def test_train_final_model_returns_fitted_pipeline_for_classification() -> None:
    frame = sample_analysis_data().iloc[:2].copy()

    model, X, y, prepared = train_final_model(frame, "meaningful_overrun_flag")

    assert X.columns.tolist() == list(final_model_spec("meaningful_overrun_flag").columns)
    assert len(y) == 2
    assert prepared.excluded_target_rows == 0
    assert hasattr(model.named_steps["model"], "predict_proba")
