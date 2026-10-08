# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Execution provenance uses admitted assignments, including replica slices."""

import unittest
from dataclasses import replace

from afd_plugin.expert_pool.batch_audit import BatchAudit
from afd_plugin.expert_pool.protocol import AssignmentSlice
from tests.unit.expert_pool.test_batching import group, ready_call


class BatchAuditTests(unittest.TestCase):
    def test_distinct_clients_preserve_counts_and_out_of_order_completion(self):
        first, second = ready_call(0).plan, ready_call(1, rows=5).plan
        audit = BatchAudit()
        audit.record_batch(group(first, second), (first, second))
        record = audit.snapshot()["records"][0]
        self.assertEqual(record["expert_assignments"], {"0": 8, "1": 8})
        self.assertEqual(record["token_rows"], 8)
        self.assertEqual([m["client_id"] for m in record["members"]], ["a0", "a1"])
        audit.record_done(second)
        self.assertEqual(audit.summary()["pending_recorded_members"], 1)
        audit.record_done(first)
        self.assertTrue(all(m["completed"] for m in record["members"]))

    def test_sliced_copy_records_only_its_own_assignments(self):
        plan = replace(
            ready_call(0, rows=5).plan,
            assignment_slices=(AssignmentSlice(0, 2, 3), AssignmentSlice(1, 0, 1)),
            num_assignments=4,
        )
        audit = BatchAudit()
        audit.record_batch(group(plan), (plan,))
        self.assertEqual(audit.records[0]["expert_assignments"], {"0": 3, "1": 1})
        member = audit.records[0]["members"][0]
        self.assertEqual(
            member["assignment_ranges"],
            {
                "0": {"start": 2, "count": 3},
                "1": {"start": 0, "count": 1},
            },
        )
        self.assertEqual(member["expert_demand"], {"0": 5, "1": 5})

    def test_overflow_is_explicit_and_does_not_grow_memory(self):
        audit = BatchAudit(1)
        first = ready_call(0).plan
        second = replace(ready_call(1).plan, plan_id=10)
        audit.record_batch(group(first), (first,))
        audit.record_batch(group(second, sequence=2), (second,))
        audit.record_done(first)
        audit.record_done(second)
        self.assertEqual(audit.summary()["dropped_batches"], 1)
        self.assertEqual(len(audit.records), 1)
        self.assertFalse(audit.pending)

    def test_unrelated_plan_cannot_be_recorded_as_a_batch_member(self):
        first, second = ready_call(0).plan, ready_call(1).plan
        audit = BatchAudit()
        with self.assertRaises(ValueError):
            audit.record_batch(group(first), (second,))
        self.assertEqual(audit.observed, 0)

    def test_ready_timeline_and_physical_cost_follow_the_right_member(self):
        first, second = ready_call(0).plan, ready_call(1).plan
        audit = BatchAudit()
        audit.record_batch(group(first, second), (first, second))
        audit.record_ready_timing(first, 10, 20, 30)
        audit.record_ready_timing(second, 11, 25, 30)
        with self.assertRaises(ValueError):
            audit.record_ready_timing(first, 10, 31, 30)
        metrics = {"compute_phase_cost_gpu_ms": 2.0}
        audit.record_done(second, {"compute_phase_cost_gpu_ms": 0.0})
        audit.record_done(first, metrics)
        metrics["compute_phase_cost_gpu_ms"] = 0.0
        members = audit.records[0]["members"]
        self.assertEqual(members[0]["input_ready_ns"], 20)
        self.assertEqual(members[1]["input_ready_ns"], 25)
        self.assertEqual(
            sum(m["metrics"]["compute_phase_cost_gpu_ms"] for m in members), 2.0
        )
        self.assertFalse(audit.pending)
