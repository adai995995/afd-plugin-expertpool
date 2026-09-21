# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Physical-batch accounting, bounded history and fail-closed feedback."""

import unittest
from dataclasses import replace

from afd_plugin.expert_pool.batching import BatchingOptions
from afd_plugin.expert_pool.controller import ControllerClientIdentity, directory_digest
from afd_plugin.expert_pool.cost_feedback import ExecutionCostBook, ExecutionShape
from afd_plugin.expert_pool.deployment import ExecutionOptions
from afd_plugin.expert_pool.fanout_controller import FanoutControllerLedger
from tests.unit.expert_pool.test_batching import group, ready_call
from tests.unit.expert_pool.test_compact_contract import deployment, directory
from tests.unit.expert_pool.test_pipeline_controller import grant, progress, submit


def prepared(batching=True):
    options = BatchingOptions(2, 128, 100) if batching else BatchingOptions()
    book = FanoutControllerLedger(
        directory(),
        tuple(ControllerClientIdentity(f"a{i}", 1, f"d{i}") for i in range(2)),
        demand_aware=True,
        compact_output=True,
        receive_slots=2,
        batching=options,
        collect_cost_feedback=True,
    )
    for worker, resident in book.directories.items():
        book.ready(
            worker,
            directory_digest(resident),
            receive_slots=2,
            batching=options,
            collect_cost_feedback=1,
        )
    submit(book, 0)
    submit(book, 1)
    first, second = grant(book), grant(book)
    for plan in (*first, *second):
        progress(book, plan, ("grant", "input_ready"))
    return book, first, second


def feedback(plans, anchor=True):
    return {
        "execution_cost_sample": int(anchor),
        "execution_batch_calls": len(plans),
        "execution_batch_tokens": sum(p.request.num_tokens for p in plans),
        "compute_phase_cost_gpu_ms": 2.0 if anchor else 0.0,
        "input_merge_phase_cost_gpu_ms": 0.1 if anchor else 0.0,
        "execution_batch_pack_gpu_ms": 0.2 if anchor else 0.0,
    }


class ExecutionCostTests(unittest.TestCase):
    def test_shape_aggregates_experts_before_bucketing(self):
        plans = tuple(ready_call(i, rows=n).plan for i, n in enumerate((3, 5)))
        shape = ExecutionShape.from_plans(plans)
        self.assertEqual(shape.assignments_per_expert, (8, 8))
        self.assertEqual(shape.bucket, (1, 2, 8, (8, 8)))
        # Same total load, very different GEMM shape must not share a bucket.
        self.assertNotEqual(shape.bucket, ExecutionShape(1, 2, 8, (16,)).bucket)
        with self.assertRaises(ValueError):
            ExecutionShape.from_plans((plans[0], plans[0]))
        with self.assertRaises(ValueError):
            ExecutionShape.from_plans((replace(plans[0], num_assignments=1),))

    def test_lru_eviction_keeps_lifetime_totals_and_snapshot_is_detached(self):
        book = ExecutionCostBook(2)
        for layer in (1, 2, 1, 3):
            book.record(ExecutionShape(layer, 2, 8, (8, 8)), 2, 0.1, 0.2)
        result = book.snapshot()
        self.assertEqual([b["layer"] for b in result["buckets"]], [1, 3])
        self.assertEqual(result["evictions"], 1)
        self.assertEqual(result["totals"]["executions"], 4)
        self.assertEqual(result["totals"]["child_calls"], 8)
        self.assertEqual(result["totals"]["expert_assignments"], 64)
        result["totals"]["sum_ms"]["compute_phase"] = 0
        self.assertEqual(book.snapshot()["totals"]["sum_ms"]["compute_phase"], 8)
        for value in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                book.record(ExecutionShape(4, 1, 1, (1,)), value, 0, 0)
        self.assertEqual(book.snapshot()["totals"]["executions"], 4)

    def test_out_of_order_batch_returns_count_once_and_drain(self):
        book, first, second = prepared()
        for i in range(2):
            plans = (first[i], second[i])
            book.start_batch(plans[0].worker_id, group(*plans))
            for p in plans:
                book.progress(p.worker_id, "output_ready", p)
            # Compute membership outlives the active batch, including when
            # the non-anchor client receives its output first.
            self.assertIsNone(book.active_batches[plans[0].worker_id])
            for p, anchor in ((plans[1], False), (plans[0], True)):
                book.progress(p.worker_id, "done", p, feedback(plans, anchor))
            totals = book.cost_books[plans[0].worker_id].snapshot()["totals"]
            self.assertEqual((totals["executions"], totals["child_calls"]), (1, 2))
        self.assertEqual(book.snapshot()["active_cost_tickets"], 0)
        self.assertEqual(book.reserved_count, 0)

    def test_wrong_membership_duplicate_and_nan_cannot_return_credit(self):
        book, first, second = prepared()
        plans = (first[0], second[0])
        book.start_batch("e0", group(*plans))
        for p in plans:
            book.progress("e0", "output_ready", p)
        for metrics in (
            None,
            {},
            {**feedback(plans), "execution_cost_sample": 0},
            {**feedback(plans), "execution_batch_calls": 1},
            {**feedback(plans), "compute_phase_cost_gpu_ms": float("nan")},
        ):
            with self.assertRaises(ValueError):
                book.progress("e0", "done", plans[0], metrics)
            self.assertEqual(book.workers["e0"].slots[0].phase, "returning")
            self.assertEqual(book.cost_books["e0"].totals["executions"], 0)
        with self.assertRaises(ValueError):
            book.progress("e0", "done", plans[1], feedback(plans))
        book.progress("e0", "done", plans[0], feedback(plans))
        with self.assertRaises(ValueError):
            book.progress("e0", "done", plans[0], feedback(plans))
        self.assertEqual(book.cost_books["e0"].totals["executions"], 1)

    def test_singleton_without_batching_and_mode_handshake(self):
        book, first, _ = prepared(batching=False)
        p = first[0]
        progress(book, p, ("executing", "output_ready"))
        book.progress("e0", "done", p, feedback((p,)))
        self.assertEqual(book.cost_books["e0"].totals["executions"], 1)
        book.workers["e1"].ready = False
        with self.assertRaises(ValueError):
            book.ready("e1", directory_digest(book.directories["e1"]), receive_slots=2)
        with self.assertRaises(ValueError):
            replace(
                deployment(), execution=ExecutionOptions(collect_cost_feedback=True)
            )


if __name__ == "__main__":
    unittest.main()
