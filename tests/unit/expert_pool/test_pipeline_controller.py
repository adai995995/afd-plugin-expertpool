# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Admission capacity, per-slot generations and one compute lane contracts."""

import unittest
from dataclasses import replace

from afd_plugin.expert_pool.controller import ControllerClientIdentity, directory_digest
from afd_plugin.expert_pool.demand import plan_expert_demand
from afd_plugin.expert_pool.deployment import ExecutionOptions
from afd_plugin.expert_pool.fanout_controller import FanoutControllerLedger
from afd_plugin.expert_pool.fanout_protocol import FanoutReplies
from afd_plugin.expert_pool.protocol import CallKey, Message
from tests.unit.expert_pool.test_compact_contract import deployment, directory, request
from tools.expert_pool.validate_engines import verify_pipeline


def ledger():
    book = FanoutControllerLedger(
        directory(),
        tuple(ControllerClientIdentity(f"a{i}", 1, f"domain-{i}") for i in range(3)),
        demand_aware=True,
        compact_output=True,
        receive_slots=2,
    )
    for worker, resident in book.directories.items():
        book.ready(worker, directory_digest(resident), receive_slots=2)
    return book


def submit(book, client, sequence=0):
    parent = replace(request(), key=CallKey(f"a{client}", 1, sequence))
    book.submit(parent.key.client_id, parent, 100)
    return parent


def grant(book):
    return [book.grant(200)[0] for _ in range(2)]


def progress(book, plan, kinds):
    for kind in kinds:
        book.progress(plan.worker_id, kind, plan)


class PipelineControllerTests(unittest.TestCase):
    def test_two_parents_reserve_distinct_slots_third_waits_without_partial_credit(
        self,
    ):
        book = ledger()
        for i in range(3):
            submit(book, i)
        first, second = grant(book), grant(book)
        self.assertEqual([p.slot_id for p in first], [0, 0])
        self.assertEqual([p.slot_id for p in second], [1, 1])
        self.assertEqual(book.reserved_count, 4)
        self.assertIsNone(book.grant(300))
        self.assertEqual(book.pending_count, 1)
        for p in first:
            progress(
                book, p, ("grant", "input_ready", "executing", "output_ready", "done")
            )
        third = grant(book)
        self.assertEqual([p.slot_id for p in third], [0, 0])
        self.assertEqual([p.generation for p in third], [2, 2])
        self.assertEqual(
            [book.workers[p.worker_id].slots[1].active for p in second], second
        )

    def test_receiving_and_returning_do_not_occupy_compute_lane(self):
        book = ledger()
        submit(book, 0)
        submit(book, 1)
        first, second = grant(book), grant(book)
        progress(book, first[0], ("grant", "input_ready", "executing"))
        progress(book, second[0], ("grant", "input_ready"))
        with self.assertRaisesRegex(ValueError, "compute lane"):
            book.progress("e0", "executing", second[0])
        self.assertEqual(book.workers["e0"].slots[1].phase, "input_ready")
        book.progress("e0", "output_ready", first[0])
        book.progress("e0", "executing", second[0])
        self.assertEqual(book.workers["e0"].slots[0].phase, "returning")
        self.assertEqual(book.workers["e0"].slots[1].phase, "executing")

    def test_parent_hold_and_release_are_scoped_to_its_slots(self):
        book = ledger()
        submit(book, 0)
        submit(book, 1)
        first, second = grant(book), grant(book)
        for p in (first[0], second[1]):
            progress(
                book, p, ("grant", "input_ready", "executing", "output_ready", "done")
            )
        self.assertEqual(book.reserved_count, 4)
        self.assertEqual(book.completed_by_client["a0"], 0)
        progress(
            book,
            first[1],
            ("grant", "input_ready", "executing", "output_ready", "done"),
        )
        self.assertEqual(book.reserved_count, 2)
        self.assertEqual(book.completed_by_client["a0"], 1)
        self.assertEqual(book.workers["e1"].slots[1].phase, "held")
        progress(
            book,
            second[0],
            ("grant", "input_ready", "executing", "output_ready", "done"),
        )
        self.assertEqual(book.reserved_count, 0)
        self.assertEqual(book.completed_by_client["a1"], 1)

    def test_stale_wrong_slot_and_duplicate_feedback_cannot_release_capacity(self):
        book = ledger()
        submit(book, 0)
        submit(book, 1)
        first, second = grant(book), grant(book)
        for bad in (
            replace(first[0], slot_id=1),
            replace(first[0], slot_id=2),
            replace(first[0], generation=99),
            replace(first[0], plan_id=99),
        ):
            with self.subTest(plan=bad), self.assertRaises(ValueError):
                book.progress("e0", "grant", bad)
            self.assertEqual(book.reserved_count, 4)
        book.progress("e0", "grant", first[0])
        with self.assertRaises(ValueError):
            book.progress("e0", "grant", first[0])
        with self.assertRaises(RuntimeError):
            book.progress("e0", "error", second[0])
        self.assertEqual(book.reserved_count, 4)

    def test_ready_capacity_must_match_before_any_admission(self):
        book = FanoutControllerLedger(
            directory(),
            (ControllerClientIdentity("a0", 1, "d"),),
            demand_aware=True,
            compact_output=True,
            receive_slots=2,
        )
        digest = directory_digest(book.directories["e0"])
        for capacity in (0, 1, 3, 2.0, True):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                book.ready("e0", digest, receive_slots=capacity)
            self.assertFalse(book.workers["e0"].ready)
        book.ready("e0", digest, receive_slots=2)
        self.assertEqual(book.workers["e0"].available_slots, 2)

    def test_reply_accepts_declared_slot_and_keeps_plan_generation_bound(self):
        book = ledger()
        submit(book, 0)
        parent = submit(book, 1)
        grant(book)
        plans = grant(book)
        replies = FanoutReplies(
            parent,
            ("e0", "e1"),
            dispatch_plan=plan_expert_demand(book.directory, parent),
            receive_slots=2,
        )
        replies.accept(Message("grant", plan=plans[0]))
        for bad in (replace(plans[0], generation=2), replace(plans[0], slot_id=0)):
            with self.assertRaises(RuntimeError):
                replies.accept(Message("output_ready", plan=bad))
        replies.accept(Message("output_ready", plan=plans[0]))
        legacy = FanoutReplies(
            parent,
            ("e0", "e1"),
            dispatch_plan=plan_expert_demand(book.directory, parent),
        )
        with self.assertRaises(RuntimeError):
            legacy.accept(Message("grant", plan=plans[1]))

    def test_configuration_requires_bounded_trusted_compact_execution(self):
        execution = ExecutionOptions(
            validate_worker_values=False, defer_output_sync=True
        )
        configured = replace(deployment(), receive_slots=2, execution=execution)
        self.assertEqual(configured.receive_slots, 2)
        for value in (0, -1, 9, True, 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(configured, receive_slots=value)
        for changes in (
            {"compact_output": False},
            {"execution": replace(execution, validate_worker_values=True)},
            {"execution": replace(execution, defer_output_sync=False)},
        ):
            with self.assertRaises(ValueError):
                replace(configured, **changes)
        self.assertEqual(deployment().receive_slots, 1)

    def test_pipeline_reports_must_match_each_slot_completion_not_just_total(self):
        book = ledger()
        submit(book, 0)
        submit(book, 1)
        first, second = grant(book), grant(book)
        for plans in (first, second):
            for p in plans:
                progress(
                    book,
                    p,
                    ("grant", "input_ready", "executing", "output_ready", "done"),
                )
        workers = [
            {
                "worker_id": f"e{i}",
                "completed_calls": 2,
                "pipeline": {
                    "enabled": True,
                    "receive_slots": 2,
                    "compute_lanes": 1,
                    "active_slots": 0,
                    "peak_occupied_slots": 2,
                    "slot_generations": [1, 1],
                    "slot_completed_calls": [1, 1],
                },
            }
            for i in range(2)
        ]
        self.assertTrue(
            verify_pipeline(workers, book.snapshot(), 2)["all_slots_drained"]
        )
        workers[0]["pipeline"]["slot_completed_calls"] = [2, 0]
        with self.assertRaises(AssertionError):
            verify_pipeline(workers, book.snapshot(), 2)


if __name__ == "__main__":
    unittest.main()
