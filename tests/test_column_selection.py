import pandas as pd

from nbt_pipeline.preprocessing.selection import drop_analysis_columns


def test_drop_analysis_columns_removes_exclusions_and_keeps_selected_features() -> None:
    source = pd.DataFrame(
        {
            "SessionIDdesc": ["TH 1 AM"],
            "theatre_notes": ["note"],
            "recovery_time": ["10:00"],
            "session_time_band": ["AM"],
            "operation_end_time_inferred": ["09:30"],
            "operation_start_hour": [9],
            "ExpectedDurationMins": [60],
            "procedure_code_group": ["S0"],
            "theatre_area": ["BRUNEL"],
            "TheatreRoom": ["BRUNEL TH 01"],
            "duration_status": ["within_tolerance"],
        }
    )

    selected = drop_analysis_columns(source)

    assert "SessionIDdesc" not in selected
    assert "theatre_notes" not in selected
    assert "recovery_time" not in selected
    assert "session_time_band" not in selected
    assert "operation_end_time_inferred" not in selected
    assert "operation_start_hour" in selected
    assert "ExpectedDurationMins" in selected
    assert "procedure_code_group" in selected
    assert "theatre_area" in selected
    assert "TheatreRoom" in selected
    assert "duration_status" in selected


def test_drop_analysis_columns_ignores_columns_that_are_not_present() -> None:
    source = pd.DataFrame({"ExpectedDurationMins": [60]})

    selected = drop_analysis_columns(source)

    pd.testing.assert_frame_equal(selected, source)
    assert selected is not source
