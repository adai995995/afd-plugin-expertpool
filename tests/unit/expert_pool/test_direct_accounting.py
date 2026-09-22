# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Reject lost results, stale slots and missing asynchronous completion feedback."""

import unittest

from tools.expert_pool.direct_accounting import verify_direct_pipeline


class DirectAccountingTests(unittest.TestCase):
    def setUp(self):
        counts = {"a0": 3, "a1": 2}
        self.clients = [
            {
                "client_id": client,
                "dispatch": {
                    "direct_dispatch": True,
                    "pending_feedback": 0,
                    "controller_roundtrips": 0,
                    "feedback_messages": count,
                    "workers": {"e0": {"calls": count}},
                },
            }
            for client, count in counts.items()
        ]
        self.pipeline = {
            "direct_dispatch": True,
            "enabled": True,
            "receive_slots": 2,
            "compute_lanes": 1,
            "active_slots": 0,
            "direct_active": 0,
            "direct_pending": 0,
            "direct_closed_clients": 2,
            "direct_slot_clients": ["a0", "a1"],
            "peak_occupied_slots": 2,
            "slot_generations": [3, 2],
            "slot_completed_calls": [3, 2],
            "direct_completed_by_client": dict(counts),
            "batching": {
                "calls": 5,
                "executions": 3,
                "calls_per_execution": {"1": 1, "2": 2},
            },
        }
        self.workers = [
            {
                "worker_id": "e0",
                "pipeline": self.pipeline,
                "completed_calls": 5,
                "client_calls": counts,
            }
        ]

    def test_results_gpu_generations_and_feedback_all_match(self):
        result = verify_direct_pipeline(self.workers, self.clients, 2)
        self.assertEqual(result["worker_child_calls"], 5)
        self.assertTrue(result["all_results_and_feedback_reconciled"])

    def test_missing_feedback_is_not_success(self):
        self.clients[0]["dispatch"]["feedback_messages"] = 2
        with self.assertRaises(AssertionError):
            verify_direct_pipeline(self.workers, self.clients, 2)

    def test_pending_generation_is_not_a_completed_result(self):
        self.pipeline["slot_generations"] = [4, 2]
        with self.assertRaises(AssertionError):
            verify_direct_pipeline(self.workers, self.clients, 2)

    def test_gpu_execution_cannot_drop_a_batch_member(self):
        self.pipeline["batching"]["calls_per_execution"] = {"1": 2, "2": 1}
        with self.assertRaises(AssertionError):
            verify_direct_pipeline(self.workers, self.clients, 2)


if __name__ == "__main__":
    unittest.main()
