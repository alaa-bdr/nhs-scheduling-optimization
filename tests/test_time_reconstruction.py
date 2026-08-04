import unittest
from datetime import time

import pandas as pd

from nbt_pipeline.preprocessing.time import add_theatre_flow_time_features


class ProvisionalTimeReconstructionTests(unittest.TestCase):
    def test_short_example_reconstructs_ordered_timeline(self):
        source = pd.DataFrame(
            {
                "into_theatre": [time(0, 20)],
                "anaesthetic_start_time": [time(0, 20)],
                "incision": [time(0, 28)],
                "closure": [time(0, 32)],
                "operation_end_time": [time(0, 39)],
                "out_of_theatre": [time(9, 43)],
                "operation_length_mins": [19],
            }
        )

        result = add_theatre_flow_time_features(source).iloc[0]

        self.assertEqual(result["into_theatre_inferred"], "09:20")
        self.assertEqual(result["incision_inferred"], "09:28")
        self.assertEqual(result["closure_inferred"], "09:32")
        self.assertEqual(result["operation_end_time_inferred"], "09:39")
        self.assertEqual(result["closure_to_operation_end_mins"], 7)
        self.assertEqual(result["time_reconstruction_status"], "valid_provisional_sequence")

    def test_long_example_reconstructs_closure_backwards_from_end(self):
        source = pd.DataFrame(
            {
                "into_theatre": [time(0, 48)],
                "anaesthetic_start_time": [time(0, 48)],
                "incision": [time(0, 20)],
                "closure": [time(0, 0)],
                "operation_end_time": [time(0, 45)],
                "out_of_theatre": [time(2, 8)],
                "operation_length_mins": [1017],
            }
        )

        result = add_theatre_flow_time_features(source).iloc[0]

        self.assertEqual(result["into_theatre_inferred"], "08:48")
        self.assertEqual(result["incision_inferred"], "09:20")
        self.assertEqual(result["closure_inferred"], "01:00")
        self.assertEqual(result["operation_end_time_inferred"], "01:45")
        self.assertEqual(result["closure_to_operation_end_mins"], 45)
        self.assertEqual(result["post_operation_theatre_time_mins"], 23)
        self.assertEqual(result["time_reconstruction_status"], "valid_provisional_sequence")

    def test_invalid_event_order_is_flagged_and_stage_values_are_missing(self):
        source = pd.DataFrame(
            {
                "into_theatre": [time(0, 20)],
                "anaesthetic_start_time": [time(0, 35)],
                "incision": [time(0, 30)],
                "closure": [time(0, 32)],
                "operation_end_time": [time(0, 39)],
                "out_of_theatre": [time(9, 43)],
                "operation_length_mins": [19],
            }
        )

        result = add_theatre_flow_time_features(source).iloc[0]

        self.assertFalse(result["time_sequence_valid"])
        self.assertEqual(result["time_reconstruction_status"], "provisional_rule_failed")
        self.assertTrue(pd.isna(result["incision_to_closure_mins"]))


if __name__ == "__main__":
    unittest.main()
