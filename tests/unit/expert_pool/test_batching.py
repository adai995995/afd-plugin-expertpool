# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded collection, atomic compute groups and per-parent return credits."""

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from afd_plugin.expert_pool.batching import BatchingOptions, ReadyCall, select_batch
from afd_plugin.expert_pool.controller import ControllerClientIdentity, directory_digest
from afd_plugin.expert_pool.deployment import ExecutionOptions, PoolDeployment
from afd_plugin.expert_pool.fanout_controller import FanoutControllerLedger
from afd_plugin.expert_pool.protocol import (
    BatchExecution,
    BatchSlot,
    CallKey,
    ExecutionPlan,
    Message,
    decode_message,
    encode_message,
)
from tests.unit.expert_pool.test_compact_contract import deployment, directory, request
from tests.unit.expert_pool.test_pipeline_controller import grant, progress, submit


def ready_call(index, *, layer=1, rows=3, ready_ns=100):
    call = replace(
        request((rows, rows, 0, 0)),
        key=CallKey(f"a{index}", 1, 0),
        layer_id=layer,
    )
    return ReadyCall(
        ExecutionPlan(call, "e0", index + 1, index, 1, (0, 1), rows * 2), ready_ns
    )


def book_with_ready_calls(options=None):
    options = options or BatchingOptions(2, 128, 100)
    book = FanoutControllerLedger(
        directory(),
        tuple(ControllerClientIdentity(f"a{i}", 1, f"d{i}") for i in range(3)),
        demand_aware=True,
        compact_output=True,
        receive_slots=2,
        batching=options,
    )
    for worker, resident in book.directories.items():
        book.ready(
            worker, directory_digest(resident), receive_slots=2, batching=options
        )
    submit(book, 0)
    submit(book, 1)
    first, second = grant(book), grant(book)
    for plan in (*first, *second):
        progress(book, plan, ("grant", "input_ready"))
    return book, first, second


def group(*plans, sequence=1):
    return BatchExecution(
        plans[0].worker_id, sequence, tuple(BatchSlot.from_plan(p) for p in plans)
    )


class BatchSelectionTests(unittest.TestCase):
    def test_no_wait_if_all_slots_are_ready_but_incompatible(self):
        calls = (ready_call(0), ready_call(1, layer=2))
        decision = select_batch(
            calls, BatchingOptions(2, 64, 1000), 100, can_grow=False
        )
        self.assertEqual(decision.slot_ids, (0,))
        self.assertEqual(decision.wait_ns, 0)

    def test_deadline_releases_singleton_and_never_restarts(self):
        options = BatchingOptions(2, 64, 10)
        pending = (ready_call(0, ready_ns=100),)
        for now, remaining in ((100, 10000), (10000, 100), (10100, 0), (99999, 0)):
            decision = select_batch(pending, options, now)
            self.assertEqual(decision.slot_ids, (0,))
            self.assertEqual(decision.deadline_ns, 10100)
            self.assertEqual(decision.wait_ns, remaining)

    def test_full_batch_dispatches_early_and_zero_wait_merges_ready_calls(self):
        calls = (ready_call(0), ready_call(1, rows=5, ready_ns=101))
        for wait_us in (0, 1000):
            selected = select_batch(calls, BatchingOptions(2, 64, wait_us), 102)
            self.assertEqual(selected.slot_ids, (0, 1))
            self.assertEqual(selected.num_tokens, 8)
            self.assertEqual(selected.wait_ns, 0)

    def test_oldest_layer_is_not_bypassed_by_a_larger_hot_batch(self):
        calls = (ready_call(0, layer=2), ready_call(1, layer=1), ready_call(2, layer=1))
        selected = select_batch(calls, BatchingOptions(3, 64, 0), 100)
        self.assertEqual(selected.slot_ids, (0,))
        self.assertEqual(
            select_batch(calls[1:], BatchingOptions(3, 64, 0), 100).slot_ids, (1, 2)
        )

    def test_token_budget_skips_an_unfitting_call_without_losing_oldest(self):
        calls = (ready_call(0, rows=32), ready_call(1, rows=40), ready_call(2, rows=32))
        selected = select_batch(calls, BatchingOptions(3, 64, 100), 100)
        self.assertEqual(selected.slot_ids, (0, 2))
        self.assertEqual(selected.num_tokens, 64)
        self.assertEqual(selected.wait_ns, 0)
        with self.assertRaises(ValueError):
            select_batch((ready_call(0, rows=65),), BatchingOptions(2, 64), 100)

    def test_checkpoint_placement_and_shape_are_not_interchangeable(self):
        first, second = ready_call(0), ready_call(1)
        for change in (
            {"model_id": "other"},
            {"placement_version": 2},
            {"hidden_size": 64},
        ):
            changed = replace(
                second,
                plan=replace(
                    second.plan, request=replace(second.plan.request, **change)
                ),
            )
            self.assertEqual(
                select_batch((first, changed), BatchingOptions(2, 64), 100).slot_ids,
                (0,),
            )
        self.assertIsNone(select_batch((), BatchingOptions(), 100))
        self.assertEqual(
            select_batch((first, second), BatchingOptions(), 100).slot_ids, (0,)
        )

    def test_configuration_round_trip_and_bounds(self):
        options = BatchingOptions(2, 128, 100)
        configured = replace(
            deployment(),
            receive_slots=2,
            batching=options,
            execution=ExecutionOptions(
                validate_worker_values=False, defer_output_sync=True
            ),
        )
        with TemporaryDirectory() as root:
            path = Path(root) / "deployment.json"
            path.write_text(json.dumps(asdict(configured)))
            self.assertEqual(PoolDeployment.read(path), configured)
        for values in (
            (True, 64, 0),
            (2, 0, 0),
            (2, 64, -1),
            (2, 64, 1_000_001),
            (1, 64, 0),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BatchingOptions(*values)
        for options in (
            BatchingOptions(3, 128),
            BatchingOptions(2, 63),
            BatchingOptions(2, 129),
        ):
            with self.assertRaises(ValueError):
                replace(configured, batching=options)


class ComputeGroupTests(unittest.TestCase):
    def test_unreceived_input_and_oversized_batch_rejected_atomically(self):
        book, first, second = book_with_ready_calls(BatchingOptions(2, 64))
        book.workers["e0"].slots[1].phase = "receiving"
        with self.assertRaises(ValueError):
            book.start_batch("e0", group(first[0], second[0]))
        self.assertEqual(book.workers["e0"].slots[0].phase, "input_ready")
        # Both calls individually fit the directory, but not the merged cap.
        book, first, second = book_with_ready_calls(BatchingOptions(2, 64))
        plans = []
        for slot in book.workers["e0"].slots:
            call = replace(request((20, 20, 20, 20)), key=slot.active.request.key)
            slot.active = replace(
                slot.active, request=call, expert_ids=(0, 1), num_assignments=40
            )
            plans.append(slot.active)
        with self.assertRaises(ValueError):
            book.start_batch("e0", group(*plans))
        self.assertEqual(
            [s.phase for s in book.workers["e0"].slots], ["input_ready"] * 2
        )

    def test_wire_binds_all_slots_without_repeating_demand(self):
        batch = group(ready_call(0).plan, ready_call(1).plan)
        message = Message("batch_executing", batch=batch)
        self.assertEqual(decode_message(encode_message(message)), message)
        self.assertNotIn(b"counts", encode_message(message))
        with self.assertRaises(ValueError):
            replace(batch, members=(batch.members[0], batch.members[0]))
        with self.assertRaises(ValueError):
            Message("executing", plan=ready_call(0).plan, batch=batch)
        with self.assertRaises(ValueError):
            Message("batch_executing")
        raw = json.loads(encode_message(message))
        raw["batch"]["members"][0]["generation"] = True
        with self.assertRaises(ValueError):
            decode_message(json.dumps(raw).encode())

    def test_stale_member_rejects_whole_batch_without_partial_transition(self):
        book, first, second = book_with_ready_calls()
        batch = group(first[0], second[0])
        for bad in (
            replace(batch, sequence=2),
            replace(batch, worker_id="e1"),
            replace(
                batch,
                members=(batch.members[0], replace(batch.members[1], generation=99)),
            ),
            replace(
                batch, members=(batch.members[0], replace(batch.members[1], slot_id=7))
            ),
        ):
            with self.assertRaises(ValueError):
                book.start_batch("e0", bad)
            self.assertEqual(
                [s.phase for s in book.workers["e0"].slots], ["input_ready"] * 2
            )
            self.assertEqual(book.batch_sequences["e0"], 0)
        book.start_batch("e0", batch)
        with self.assertRaises(ValueError):
            book.start_batch("e0", batch)

    def test_single_compute_lane_and_independent_parent_release(self):
        book, first, second = book_with_ready_calls()
        for plans in zip(first, second, strict=True):
            with self.assertRaises(ValueError):
                book.progress(plans[0].worker_id, "executing", plans[0])
            book.start_batch(plans[0].worker_id, group(*plans))
        progress(book, first[0], ("output_ready", "done"))
        self.assertEqual(book.reserved_count, 4)
        self.assertIsNotNone(book.active_batches["e0"])
        progress(book, first[1], ("output_ready", "done"))
        self.assertEqual(book.reserved_count, 2)
        self.assertEqual(book.completed_by_client["a0"], 1)
        for p in second:
            progress(book, p, ("output_ready", "done"))
        self.assertEqual(book.reserved_count, 0)
        self.assertEqual(book.snapshot()["active_compute_batches"], 0)
        self.assertEqual(book.batch_calls, {"e0": 2, "e1": 2})
        self.assertEqual(book.batch_sequences, {"e0": 1, "e1": 1})
        with self.assertRaises(ValueError):
            book.start_batch("e0", group(first[0], second[0], sequence=2))

    def test_disabled_and_mismatched_workers_cannot_use_batching(self):
        book = FanoutControllerLedger(
            directory(),
            (ControllerClientIdentity("a0", 1, "d"),),
            demand_aware=True,
            compact_output=True,
            receive_slots=2,
        )
        with self.assertRaises(ValueError):
            book.ready(
                "e0",
                directory_digest(book.directories["e0"]),
                receive_slots=2,
                batching=BatchingOptions(2, 128),
            )
        self.assertFalse(book.workers["e0"].ready)
        with self.assertRaises(ValueError):
            book.start_batch("e0", group(ready_call(0).plan))


if __name__ == "__main__":
    unittest.main()
