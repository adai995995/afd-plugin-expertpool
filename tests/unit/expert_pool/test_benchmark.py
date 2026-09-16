# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Regression checks for benchmark fairness, timing math and bounded aggregates."""

import unittest

from afd_plugin.expert_pool.metrics import CallMetrics
from tools.expert_pool.benchmark_engines import assignments, percentile, summarize


class BenchmarkTests(unittest.TestCase):
    def test_identical_domain_budget_and_request_coverage(self) -> None:
        for mode in ("native", "pool"):
            for concurrency in (1, 4, 8):
                for repeat in range(2):
                    with self.subTest(mode=mode, c=concurrency, repeat=repeat):
                        plans = assignments(mode, concurrency, 32, repeat)
                        for domain in range(2):
                            local = [p for p in plans if p["domain"] == domain]
                            self.assertEqual(
                                sum(p["concurrency"] for p in local), concurrency
                            )
                            indices = [i for p in local for i in p["indices"]]
                            self.assertEqual(sorted(indices), list(range(32)))
                            self.assertTrue(all(p["concurrency"] > 0 for p in local))

    def test_all_three_native_replicas_used_above_single_request_load(self) -> None:
        for repeat in range(2):
            plans = assignments("native", 4, 16, repeat)
            self.assertEqual({p["replica"] for p in plans}, {0, 1, 2})
            doubled_domain = 0 if repeat == 0 else 1
            self.assertEqual(sum(p["domain"] == doubled_domain for p in plans), 2)

    def test_throughput_uses_common_window_not_sum_of_replica_rates(self) -> None:
        records = [
            {
                "started_ns": 0,
                "finished_ns": end,
                "output_tokens": 32,
                "ttft_ms": 10,
                "tpot_ms": 20,
                "e2e_ms": end / 1e6,
            }
            for end in (1_000_000_000, 2_000_000_000)
        ]
        result = summarize(records)
        self.assertEqual(result["output_throughput_tps"], 32)
        self.assertEqual(result["request_throughput_rps"], 1)
        self.assertEqual(result["e2e_ms"]["p95"], 1950)

    def test_percentiles_and_invalid_inputs(self) -> None:
        self.assertEqual(percentile([4, 1, 3, 2], 0.5), 2.5)
        self.assertEqual(percentile([1], 0.95), 1)
        with self.assertRaises(ValueError):
            summarize([])
        with self.assertRaises(ValueError):
            assignments("native", 0, 4, 0)

    def test_metrics_aggregate_without_retaining_calls(self) -> None:
        metrics = CallMetrics()
        for _ in range(100):
            metrics.record(1, 8, {"receive_gpu_ms": 0.25, "client_roundtrip_ms": 2.0})
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["calls"], 100)
        self.assertEqual(snapshot["token_rows"], 800)
        self.assertEqual(snapshot["sum_ms"]["client_roundtrip_ms"], 200)
        self.assertEqual(snapshot["max_ms"]["receive_gpu_ms"], 0.25)
        self.assertEqual(snapshot["batch_calls"], {"8": 100})
        self.assertEqual(snapshot["layer_calls"], {"1": 100})
        snapshot["sum_ms"]["client_roundtrip_ms"] = 0
        self.assertEqual(metrics.snapshot()["sum_ms"]["client_roundtrip_ms"], 200)

    def test_bad_timings_do_not_change_aggregates(self) -> None:
        metrics = CallMetrics()
        for value in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(ValueError):
                metrics.record(1, 1, {"time_ms": value})
        self.assertEqual(metrics.calls, 0)


if __name__ == "__main__":
    unittest.main()
