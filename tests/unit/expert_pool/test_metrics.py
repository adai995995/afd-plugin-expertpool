# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fixed-size timing aggregates for decode and multi-row Pool calls."""

import unittest

from afd_plugin.expert_pool.metrics import CallMetrics


class CallMetricsTests(unittest.TestCase):
    def test_empty_decode_and_multi_row_timings_remain_separate(self) -> None:
        metrics = CallMetrics()
        metrics.record(0, 0, {"roundtrip_ms": 0.5})
        metrics.record(1, 1, {"roundtrip_ms": 2.0, "compute_ms": 0.25})
        metrics.record(1, 1, {"roundtrip_ms": 4.0, "compute_ms": 0.75})
        metrics.record(2, 4, {"roundtrip_ms": 8.0, "compute_ms": 3.0})
        metrics.record(2, 8, {"roundtrip_ms": 12.0})

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["calls"], 5)
        self.assertEqual(snapshot["token_rows"], 14)
        self.assertEqual(snapshot["sum_ms"]["roundtrip_ms"], 26.5)
        self.assertEqual(snapshot["max_ms"]["roundtrip_ms"], 12.0)
        self.assertEqual(snapshot["batch_calls"], {"0": 1, "1": 2, "4": 1, "8": 1})
        self.assertEqual(
            snapshot["timings_by_rows"],
            {
                "zero": {
                    "calls": 1,
                    "sum_ms": {"roundtrip_ms": 0.5},
                    "max_ms": {"roundtrip_ms": 0.5},
                },
                "one": {
                    "calls": 2,
                    "sum_ms": {"roundtrip_ms": 6.0, "compute_ms": 1.0},
                    "max_ms": {"roundtrip_ms": 4.0, "compute_ms": 0.75},
                },
                "many": {
                    "calls": 2,
                    "sum_ms": {"roundtrip_ms": 20.0, "compute_ms": 3.0},
                    "max_ms": {"roundtrip_ms": 12.0, "compute_ms": 3.0},
                },
            },
        )

        snapshot["timings_by_rows"]["one"]["sum_ms"]["roundtrip_ms"] = 0
        self.assertEqual(
            metrics.snapshot()["timings_by_rows"]["one"]["sum_ms"]["roundtrip_ms"],
            6.0,
        )

    def test_invalid_value_does_not_change_row_bucket(self) -> None:
        metrics = CallMetrics()
        for value in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(ValueError):
                metrics.record(1, 1, {"time_ms": value})

        self.assertEqual(metrics.calls, 0)
        self.assertEqual(
            [bucket["calls"] for bucket in metrics.snapshot()["timings_by_rows"].values()],
            [0, 0, 0],
        )


if __name__ == "__main__":
    unittest.main()
