"""Tests for the feature engineering used by the models."""

import numpy as np
import pandas as pd
import pytest

from nbt_pipeline.modelling_v2.features_v2 import (
    BASE_FEATURES,
    TARGET,
    add_out_of_fold_target_encoding,
    parse_stated_duration,
)


class TestParseStatedDuration:
    """Staff often write the expected time into the notes."""

    @pytest.mark.parametrize("text,expected", [
        ("60mins/day case/LA/ excision of lesion", 60.0),
        ("Duration = 240 RCS Priority = Level 2", 240.0),
        ("90 mins day case", 90.0),
        ("2 hours GA", 120.0),
        ("1.5 hrs", 90.0),
        ("60mnins/day case", 60.0),
    ])
    def test_extracts_duration(self, text, expected):
        assert parse_stated_duration(text) == expected

    @pytest.mark.parametrize("text", ["", ".", "no duration here", "P4 LEFT THR"])
    def test_returns_nan_when_absent(self, text):
        assert np.isnan(parse_stated_duration(text))

    def test_handles_non_string(self):
        assert np.isnan(parse_stated_duration(np.nan))

    def test_duration_keyword_takes_priority(self):
        assert parse_stated_duration("Duration = 180 and 30 mins prep") == 180.0


class TestTargetEncoding:
    """The encoding must not use the row's own answer."""

    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "category": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
            TARGET: np.concatenate([
                rng.normal(60, 5, 40),
                rng.normal(120, 5, 40),
                rng.normal(30, 5, 40),
            ]),
        })

    def test_creates_encoded_column(self, frame):
        out = add_out_of_fold_target_encoding(frame, ["category"], TARGET)
        assert "te_category" in out.columns

    def test_no_missing_values_produced(self, frame):
        out = add_out_of_fold_target_encoding(frame, ["category"], TARGET)
        assert out["te_category"].notna().all()

    def test_encoding_separates_categories(self, frame):
        out = add_out_of_fold_target_encoding(frame, ["category"], TARGET)
        means = out.groupby("category")["te_category"].mean()
        assert means["b"] > means["a"] > means["c"]

    def test_encoding_is_not_the_row_target(self, frame):
        """Matching the row's own value would mean the model is cheating."""
        out = add_out_of_fold_target_encoding(frame, ["category"], TARGET)
        assert not np.allclose(out["te_category"], out[TARGET])

    def test_original_frame_unchanged(self, frame):
        before = frame.copy()
        add_out_of_fold_target_encoding(frame, ["category"], TARGET)
        pd.testing.assert_frame_equal(frame, before)

    def test_missing_column_is_skipped(self, frame):
        out = add_out_of_fold_target_encoding(frame, ["not_a_column"], TARGET)
        assert "te_not_a_column" not in out.columns

    def test_rare_categories_fall_back_to_global_mean(self):
        frame = pd.DataFrame({
            "category": ["common"] * 50 + ["rare"],
            TARGET: [60.0] * 50 + [500.0],
        })
        out = add_out_of_fold_target_encoding(frame, ["category"], TARGET, min_count=5)
        assert out["te_category"].notna().all()


class TestFeatureContract:
    """Every feature has to be knowable before the operation starts."""

    FORBIDDEN = [
        "incision", "closure", "out_of_theatre", "operation_end",
        "occupancy", "duration_error", "overrun", "underrun",
        "inferred", "recovery_time", TARGET,
    ]

    def test_no_intraoperative_features(self):
        for feature in BASE_FEATURES:
            for banned in self.FORBIDDEN:
                assert banned not in feature.lower(), (
                    f"{feature} looks like it is not knowable before surgery"
                )

    def test_planned_duration_is_included(self):
        assert "ExpectedDurationMins" in BASE_FEATURES

    def test_features_are_unique(self):
        assert len(BASE_FEATURES) == len(set(BASE_FEATURES))
