# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Readiness must stay on one host clock and physical costs must not duplicate."""

import unittest

from tools.expert_pool.analyze_ready_trace import analyze


class ReadyTraceTests(unittest.TestCase):
    def audit(self):
        return {
            "dropped_batches": 0,
            "pending_recorded_members": 0,
            "records": [
                {
                    "layer_id": 1,
                    "members": [
                        {
                            "client_id": client,
                            "call_seq": 10,
                            "completed": True,
                            "input_ready_ns": ready,
                            "compute_submitted_ns": 100_000,
                            "metrics": {
                                "compute_phase_cost_gpu_ms": cost,
                                "input_merge_phase_cost_gpu_ms": cost / 10,
                            },
                        }
                        for client, ready, cost in (("a", 0, 2), ("b", 50_000, 0))
                    ],
                },
                {
                    "layer_id": 2,
                    "members": [
                        {
                            "client_id": "a",
                            "call_seq": 11,
                            "completed": True,
                            "input_ready_ns": 1_000_000,
                            "compute_submitted_ns": 1_000_000,
                            "metrics": {
                                "compute_phase_cost_gpu_ms": 1,
                                "input_merge_phase_cost_gpu_ms": 0,
                            },
                        }
                    ],
                },
            ],
        }

    def test_same_layer_peer_gaps_and_single_physical_cost(self):
        result = analyze(self.audit(), {"a": (10, 12), "b": (10, 12)})
        self.assertEqual(result["calls"], 3)
        self.assertEqual(result["executions"], 2)
        self.assertEqual(result["cross_a_executions"], 1)
        self.assertEqual(result["nearest_other_a_same_layer_ready"]["samples"], 2)
        self.assertEqual(result["calls_with_peer_gap_at_most_us"]["50"], 2)
        self.assertEqual(
            result["cuda_event_interval_costs_ms"]["compute_phase_cost_gpu_ms"], 3
        )

    def test_truncated_or_missing_timestamps_cannot_claim_opportunities(self):
        for key in ("dropped_batches", "pending_recorded_members"):
            audit = self.audit()
            audit[key] = 1
            with self.assertRaises(ValueError):
                analyze(audit, {"a": (10, 12), "b": (10, 12)})
        audit = self.audit()
        audit["records"][0]["members"][0].pop("input_ready_ns")
        with self.assertRaises(ValueError):
            analyze(audit, {"a": (10, 12), "b": (10, 12)})
        with self.assertRaises(ValueError):
            analyze(self.audit(), {"a": (20, 30), "b": (20, 30)})


if __name__ == "__main__":
    unittest.main()
