import unittest

import pandas as pd

from nbt_pipeline.preprocessing.outliers import add_duration_timing_review_flag


class DurationTimingReviewFlagTests(unittest.TestCase):
    def test_flags_only_unusual_duration_with_failed_timing(self):
        normal_rows = 20
        source = pd.DataFrame(
            {
                "ExpectedDurationMins": [100] * normal_rows + [100, 100],
                "operation_length_mins": [100] * normal_rows + [1000, 1000],
                "operation_length_rule_valid": [True] * normal_rows + [False, True],
                "time_sequence_valid": [True] * normal_rows + [False, True],
            }
        )

        result = add_duration_timing_review_flag(source)

        self.assertTrue(result.loc[normal_rows, "duration_timing_review_flag"])
        self.assertFalse(result.loc[normal_rows + 1, "duration_timing_review_flag"])
        self.assertFalse(result.loc[: normal_rows - 1, "duration_timing_review_flag"].any())

    def test_timing_failure_without_unusual_duration_is_not_flagged(self):
        source = pd.DataFrame(
            {
                "ExpectedDurationMins": [100] * 10,
                "operation_length_mins": [100] * 10,
                "operation_length_rule_valid": [False] + [True] * 9,
                "time_sequence_valid": [False] + [True] * 9,
            }
        )

        result = add_duration_timing_review_flag(source)

        self.assertFalse(result["duration_timing_review_flag"].any())

    def test_preserves_source_dataframe(self):
        source = pd.DataFrame(
            {
                "ExpectedDurationMins": [60, 60],
                "operation_length_mins": [60, 600],
                "operation_length_rule_valid": [True, False],
                "time_sequence_valid": [True, False],
            }
        )
        original = source.copy(deep=True)

        add_duration_timing_review_flag(source)

        pd.testing.assert_frame_equal(source, original)


if __name__ == "__main__":
    unittest.main()
