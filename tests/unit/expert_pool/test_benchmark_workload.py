# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Offered arrivals must survive routing, queueing and failed completions."""

import unittest

from tools.expert_pool.benchmark_workload import (
    arrival_offsets,
    assignments,
    domain_replicas,
    mixed_lengths,
    open_loop_assignments,
    summarize,
    verify_cost_accounting,
)


class WorkloadTests(unittest.TestCase):
    def test_mixed_lengths_do_not_lock_long_inputs_to_one_replica(self):
        lengths = mixed_lengths(128, [128, 512], 41)
        self.assertEqual(lengths.count(128), 64)
        self.assertEqual(lengths.count(512), 64)
        self.assertEqual(lengths, mixed_lengths(128, [128, 512], 41))
        self.assertNotEqual(lengths, mixed_lengths(128, [128, 512], 42))
        for replica in range(2):
            self.assertEqual(set(lengths[replica::2]), {128, 512})

    def test_same_offered_trace_for_different_replica_counts(self):
        for count in (2, 8, 31):
            for replicas in (2, 3, 4, 6, 8):
                for repeat in (0, 1):
                    plans = open_loop_assignments(replicas, count, repeat)
                    offered = [(p["domain"], i) for p in plans for i in p["indices"]]
                    self.assertEqual(
                        sorted(offered),
                        [(d, i) for d in range(2) for i in range(count)],
                    )
                    self.assertTrue(all(p["concurrency"] == 0 for p in plans))

    def test_closed_loop_preserves_domain_budget_when_pool_grows(self):
        for replicas in (2, 4, 6):
            for concurrency in (1, 4, 8):
                plans = assignments("pool", concurrency, 16, 0, replicas)
                for domain in range(2):
                    local = [p for p in plans if p["domain"] == domain]
                    self.assertEqual(sum(p["concurrency"] for p in local), concurrency)
                    self.assertEqual(
                        sorted(i for p in local for i in p["indices"]), list(range(16))
                    )
        self.assertEqual(tuple(map(len, domain_replicas(5, 0))), (3, 2))
        self.assertEqual(tuple(map(len, domain_replicas(5, 1))), (2, 3))

    def test_arrivals_deterministic_and_independent_of_completions(self):
        self.assertEqual(
            arrival_offsets(4, 2, "steady", 0), [0, 500000000, 1000000000, 1500000000]
        )
        self.assertEqual(arrival_offsets(8, 4, "burst", 0), [0] * 4 + [1000000000] * 4)
        a = arrival_offsets(100, 2, "poisson", 1)
        self.assertEqual(a, arrival_offsets(100, 2, "poisson", 1))
        self.assertNotEqual(a, arrival_offsets(100, 2, "poisson", 2))
        self.assertEqual(a, sorted(a))
        for rate in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                arrival_offsets(2, rate, "steady", 0)

    def test_domain_goodput_shares_window_and_counts_timeouts(self):
        records = [
            {
                "domain": 0,
                "started_ns": 0,
                "finished_ns": 1000000000,
                "ttft_ms": 100,
                "tpot_ms": 20,
                "e2e_ms": 1000,
                "output_tokens": 4,
                "status": "completed",
            },
            {
                "domain": 1,
                "started_ns": 0,
                "finished_ns": 2000000000,
                "ttft_ms": 101,
                "tpot_ms": 20,
                "e2e_ms": 2000,
                "output_tokens": 4,
                "status": "completed",
            },
            {
                "domain": 0,
                "started_ns": 0,
                "finished_ns": 2000000000,
                "output_tokens": 0,
                "status": "timeout",
            },
        ]
        slos = {0: (100, 20), 1: (100, 20)}
        result = summarize(records, slos=slos)
        self.assertEqual(result["slo"]["good_requests"], 1)
        self.assertEqual(result["slo"]["attainment"], 1 / 3)
        self.assertEqual(result["slo"]["goodput_rps"], 0.5)
        self.assertEqual(result["request_throughput_rps"], 1)
        domains = [
            summarize(
                [r for r in records if r["domain"] == d],
                window=(0, 2000000000),
                slos=slos,
            )
            for d in range(2)
        ]
        self.assertEqual(sum(d["slo"]["goodput_rps"] for d in domains), 0.5)
        self.assertEqual(sum(d["request_throughput_rps"] for d in domains), 1)
        self.assertNotIn("slo", summarize(records))
        failed = summarize(records[2:], slos=slos)
        self.assertIsNone(failed["ttft_ms"])
        self.assertEqual(failed["slo"]["attainment"], 0)

    def test_arrival_clock_includes_load_generator_lag(self):
        record = {
            "domain": 0,
            "started_ns": 0,
            "submitted_ns": 400000000,
            "finished_ns": 1000000000,
            "output_tokens": 4,
            "ttft_ms": 500,
            "tpot_ms": 20,
            "e2e_ms": 1000,
            "submission_lag_ms": 400,
        }
        result = summarize([record], slos={0: (200, 20)})
        self.assertEqual(result["slo"]["good_requests"], 0)
        self.assertEqual(result["submission_lag_ms"]["p95"], 400)
        with self.assertRaises(ValueError):
            summarize([record], window=(400000000, 1000000000))

    def test_physical_cost_reconciliation_rejects_double_counting(self):
        totals = {
            "executions": 1,
            "child_calls": 2,
            "token_rows": 8,
            "sum_ms": {"compute_phase": 2, "input_merge": 0.1},
        }
        controller = {
            "active_cost_tickets": 0,
            "execution_cost_by_worker": {"e0": {"totals": totals}},
        }
        worker = {
            "worker_id": "e0",
            "pipeline": {
                "batching": {
                    "executions": 1,
                    "calls": 2,
                    "token_rows": 8,
                    "compute_phase_gpu_ms_total": 2,
                    "input_merge_phase_gpu_ms_total": 0.1,
                }
            },
        }
        self.assertEqual(
            verify_cost_accounting(controller, [worker])["e0"]["executions"], 1
        )
        totals["executions"] = 2
        with self.assertRaises(AssertionError):
            verify_cost_accounting(controller, [worker])


if __name__ == "__main__":
    unittest.main()
