# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Atomic parent reservations and bounded interleaved replies without CUDA."""

import multiprocessing
import threading
import unittest
from dataclasses import replace

from afd_plugin.expert_pool.controller import (
    ControllerClientIdentity,
    directory_digest,
)
from afd_plugin.expert_pool.controller_service import ControllerRuntime
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.fanout_controller import FanoutControllerLedger
from afd_plugin.expert_pool.fanout_protocol import FanoutReplies
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import (
    CallKey,
    CallRequest,
    ExecutionPlan,
    Message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory


def directory(mixed_layers=False):
    placements = (
        (ExpertPlacement(1, 4, (0, 1)), ExpertPlacement(2, 4, (0, 1))),
        (ExpertPlacement(1, 4, (2, 3)), ExpertPlacement(2, 4, (2, 3))),
    )
    if mixed_layers:
        placements = (
            (ExpertPlacement(1, 4, (0, 1)), ExpertPlacement(2, 4, (0, 1))),
            (ExpertPlacement(1, 4, (2, 3)),),
            (ExpertPlacement(2, 4, (2, 3)), ExpertPlacement(3, 4, (0, 1, 2, 3))),
        )
    return PoolDirectory(
        tuple(
            StaticDirectory(
                "checkpoint",
                1,
                f"e{i}",
                placement,
                32,
                2,
                64,
                allow_partial_experts=True,
            )
            for i, placement in enumerate(placements)
        ),
        expert_partitioned=True,
    )


def ledger(mixed_layers=False):
    return FanoutControllerLedger(
        directory(mixed_layers),
        tuple(ControllerClientIdentity(f"a{i}", 1, f"domain-{i}") for i in range(2)),
    )


def request(client="a0", sequence=0, layer=1, tokens=8):
    return CallRequest(
        CallKey(client, 1, sequence), "checkpoint", 1, layer, tokens, 32, 2
    )


def make_ready(book):
    for worker_id, resident in book.directories.items():
        book.ready(worker_id, directory_digest(resident))


def finish(book, plan):
    for kind in ("grant", "input_ready", "executing", "output_ready", "done"):
        book.progress(plan.worker_id, kind, plan)


def grant_group(book, owners=2):
    return tuple(book.grant(1000000)[0] for _ in range(owners))


class FanoutLedgerTests(unittest.TestCase):
    def test_requires_partitioned_directory_and_supported_policy(self):
        clients = (ControllerClientIdentity("a0", 1, "domain-0"),)
        with self.assertRaises(ValueError):
            FanoutControllerLedger(directory(), clients, "round_robin")
        complete = StaticDirectory(
            "checkpoint", 1, "e0", (ExpertPlacement(1, 4, (0, 1, 2, 3)),), 32, 2, 64
        )
        with self.assertRaises(ValueError):
            FanoutControllerLedger(PoolDirectory((complete,)), clients)

    def test_first_child_grant_atomically_reserves_all_owners(self):
        book = ledger()
        make_ready(book)
        book.submit("a0", request(), 0)
        first, queue_ms = book.grant(1000000)
        self.assertEqual(first.worker_id, "e0")
        self.assertEqual(queue_ms, 1.0)
        self.assertTrue(
            all(worker.active is not None for worker in book.workers.values())
        )
        self.assertEqual(book.workers["e0"].phase, "reserved")
        self.assertEqual(book.workers["e1"].phase, "pending_grant")
        self.assertEqual(book.snapshot()["pending_child_plans"], 1)
        self.assertEqual(book.snapshot()["active_parent_calls"], 1)
        second, _ = book.grant(2000000)
        self.assertEqual(second.request, first.request)
        self.assertNotEqual(second.plan_id, first.plan_id)
        self.assertEqual(second.worker_id, "e1")
        self.assertIsNone(book.grant(3000000))

    def test_two_clients_cannot_hold_partial_gangs(self):
        book = ledger()
        make_ready(book)
        for client in book.clients:
            book.submit(client, request(client), 0)
        first = grant_group(book)
        self.assertEqual({plan.request.key.client_id for plan in first}, {"a0"})
        self.assertIsNone(book.grant(2))
        finish(book, first[0])
        self.assertIsNone(book.grant(3))
        self.assertEqual(book.workers["e0"].phase, "held")
        self.assertEqual(book.workers["e0"].active, first[0])
        self.assertEqual(book.snapshot()["completed_parent_calls"], 0)
        self.assertEqual(book.snapshot()["completed_child_calls"], 1)
        with self.assertRaises(ValueError):
            book.close_client("a0")
        finish(book, first[1])
        second = grant_group(book)
        self.assertEqual({plan.request.key.client_id for plan in second}, {"a1"})
        self.assertEqual([plan.generation for plan in second], [2, 2])
        for plan in second:
            finish(book, plan)
        snapshot = book.snapshot()
        self.assertEqual(snapshot["completed_parent_calls"], 2)
        self.assertEqual(snapshot["completed_child_calls"], 4)
        self.assertEqual(snapshot["completed_by_client"], {"a0": 1, "a1": 1})
        self.assertEqual(snapshot["outstanding"], 0)
        self.assertEqual(snapshot["active_parent_calls"], 0)
        self.assertEqual(snapshot["pending_child_plans"], 0)
        for worker in snapshot["workers"].values():
            self.assertEqual(worker["available_slots"], 1)
            self.assertEqual(worker["client_calls"], {"a0": 1, "a1": 1})
            self.assertEqual(worker["layer_calls"], {"1": 2, "2": 0})

    def test_unready_owner_prevents_any_reservation(self):
        book = ledger()
        book.ready("e0", directory_digest(book.directories["e0"]))
        book.submit("a0", request(), 0)
        self.assertIsNone(book.grant(1))
        self.assertTrue(all(worker.active is None for worker in book.workers.values()))
        self.assertEqual(
            [worker.generation for worker in book.workers.values()], [0, 0]
        )
        book.ready("e1", directory_digest(book.directories["e1"]))
        self.assertEqual(len(grant_group(book)), 2)

    def test_overlapping_layers_do_not_reserve_free_partial_owner(self):
        book = ledger(mixed_layers=True)
        make_ready(book)
        book.submit("a0", request(layer=1), 0)
        first = grant_group(book)
        book.submit("a1", request("a1", layer=2), 0)
        self.assertIsNone(book.grant(1))
        self.assertIsNone(book.workers["e2"].active)
        finish(book, first[0])
        self.assertIsNone(book.grant(2))
        finish(book, first[1])
        second = grant_group(book)
        self.assertEqual([plan.worker_id for plan in second], ["e0", "e2"])
        self.assertEqual([plan.generation for plan in second], [2, 1])

    def test_disjoint_layer_owners_can_progress_independently(self):
        book = ledger(mixed_layers=True)
        make_ready(book)
        book.submit("a0", request(layer=1), 0)
        first = grant_group(book)
        book.submit("a1", request("a1", layer=3), 0)
        independent = book.grant(1)[0]
        self.assertEqual(independent.worker_id, "e2")
        self.assertEqual(book.snapshot()["active_parent_calls"], 2)
        finish(book, independent)
        self.assertEqual(set(book.outstanding), {"a0"})
        self.assertTrue(all(book.workers[p.worker_id].active == p for p in first))

    def test_bad_submission_does_not_consume_sequence(self):
        book = ledger()
        original = request()
        for bad in (
            replace(original, model_id="other"),
            replace(original, placement_version=9),
            replace(original, layer_id=99),
            replace(original, num_tokens=65),
            replace(original, key=CallKey("a1", 1, 0)),
            replace(original, key=CallKey("a0", 2, 0)),
        ):
            with self.assertRaises(ValueError):
                book.submit("a0", bad, 0)
        self.assertEqual(book.last_sequence["a0"], -1)
        self.assertFalse(book.outstanding)
        book.submit("a0", original, 0)
        with self.assertRaises(ValueError):
            book.submit("a0", request(sequence=1), 0)

    def test_stale_duplicate_and_out_of_order_feedback_cannot_release_gang(self):
        book = ledger()
        make_ready(book)
        book.submit("a0", request(), 0)
        first, second = grant_group(book)
        for source, kind, plan in (
            ("e0", "done", first),
            ("e0", "grant", replace(first, generation=99)),
            ("e1", "grant", first),
            ("unknown", "grant", first),
        ):
            with self.assertRaises(ValueError):
                book.progress(source, kind, plan)
            self.assertEqual(book.workers["e0"].active, first)
            self.assertEqual(book.workers["e1"].active, second)
        finish(book, first)
        with self.assertRaises(ValueError):
            book.progress("e0", "done", first)
        finish(book, second)
        book.submit("a0", request(sequence=1), 0)
        current = grant_group(book)
        with self.assertRaises(ValueError):
            book.progress("e0", "done", first)
        self.assertEqual(book.workers["e0"].active, current[0])

    def test_unpublished_plan_cannot_acknowledge_grant(self):
        book = ledger()
        make_ready(book)
        book.submit("a0", request(), 0)
        book.grant(1)
        unpublished = book.workers["e1"].active
        with self.assertRaises(ValueError):
            book.progress("e1", "grant", unpublished)
        self.assertEqual(book.workers["e1"].phase, "pending_grant")

    def test_child_error_keeps_every_reservation_until_supervisor_shutdown(self):
        book = ledger()
        make_ready(book)
        book.submit("a0", request(), 0)
        first, second = grant_group(book)
        finish(book, first)
        for kind in ("grant", "input_ready", "executing"):
            book.progress("e1", kind, second)
        before = book.snapshot()
        with self.assertRaises(RuntimeError):
            book.progress("e1", "error", second)
        self.assertEqual(book.snapshot(), before)
        self.assertEqual(book.outstanding["a0"], first.request.key)

    def test_history_and_pending_publications_remain_bounded(self):
        book = ledger()
        make_ready(book)
        for sequence in range(100):
            book.submit("a0", request(sequence=sequence), 0)
            plans = grant_group(book)
            for plan in reversed(plans):
                finish(book, plan)
            self.assertFalse(book.active_parents)
            self.assertFalse(book.pending_child_plans)
            self.assertFalse(book.outstanding)
        self.assertEqual(book.snapshot()["completed_parent_calls"], 100)
        self.assertEqual(book.snapshot()["completed_child_calls"], 200)
        with self.assertRaises(ValueError):
            book.submit("a0", request(sequence=99), 0)
        book.close_client("a0")
        with self.assertRaises(ValueError):
            book.submit("a0", request(sequence=100), 0)

    def test_worker_eof_does_not_release_uncertain_group(self):
        book = ledger()
        make_ready(book)
        pairs = [multiprocessing.Pipe() for _ in range(4)]
        self.addCleanup(
            lambda: [connection.close() for pair in pairs for connection in pair]
        )
        runtime = ControllerRuntime(
            book,
            {"a0": pairs[0][0], "a1": pairs[1][0]},
            {"e0": pairs[2][0], "e1": pairs[3][0]},
        )
        outcome = []

        def run():
            try:
                outcome.append(runtime.run())
            except BaseException as error:
                outcome.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        send_message(pairs[0][1], Message("submit", request=request()))
        plans = [receive_message(pairs[i][1], 2).plan for i in (2, 3)]
        pairs[2][1].close()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0], EOFError)
        self.assertEqual([book.workers[p.worker_id].active for p in plans], plans)
        self.assertEqual(len(book.outstanding), 1)


class FanoutReplyTests(unittest.TestCase):
    def setUp(self):
        self.request = request(tokens=0)
        self.replies = FanoutReplies(self.request, ("e0", "e1"))
        self.plans = {
            worker: ExecutionPlan(self.request, worker, i + 1, 0, 1)
            for i, worker in enumerate(("e0", "e1"))
        }

    def test_zero_token_early_completion_can_precede_other_grant(self):
        for kind in ("grant", "output_ready", "done"):
            self.replies.accept(Message(kind, plan=self.plans["e1"]))
        self.assertIsNone(self.replies.take("grant", "e0"))
        self.assertEqual(len(self.replies.messages), 3)
        for kind in ("grant", "output_ready", "done"):
            self.replies.accept(Message(kind, plan=self.plans["e0"]))
        self.assertEqual(len(self.replies.messages), 6)
        for kind in ("grant", "output_ready", "done"):
            for worker in ("e0", "e1"):
                self.assertEqual(
                    self.replies.take(kind, worker).plan, self.plans[worker]
                )
                self.assertIsNone(self.replies.take(kind, worker))
        self.assertFalse(self.replies.messages)
        self.assertEqual(len(self.replies.plans), 2)

    def test_duplicate_events_remain_rejected_after_cache_consumption(self):
        for kind in ("grant", "output_ready", "done"):
            message = Message(kind, plan=self.plans["e0"])
            self.replies.accept(message)
            self.replies.take(kind, "e0")
            with self.assertRaises(RuntimeError):
                self.replies.accept(message)
        self.assertFalse(self.replies.messages)

    def test_foreign_parent_worker_slot_and_duplicate_plan_id_rejected(self):
        for plan in (
            replace(self.plans["e0"], request=request(sequence=1)),
            replace(self.plans["e0"], worker_id="unknown"),
            replace(self.plans["e0"], slot_id=1),
        ):
            with self.assertRaises(RuntimeError):
                self.replies.accept(Message("grant", plan=plan))
        self.replies.accept(Message("grant", plan=self.plans["e0"]))
        with self.assertRaises(RuntimeError):
            self.replies.accept(
                Message("grant", plan=replace(self.plans["e1"], plan_id=1))
            )
        self.assertEqual(len(self.replies.messages), 1)

    def test_skipped_phase_or_changed_plan_rejected_without_state_advance(self):
        with self.assertRaises(RuntimeError):
            self.replies.accept(Message("output_ready", plan=self.plans["e0"]))
        self.replies.accept(Message("grant", plan=self.plans["e0"]))
        for kind, plan in (
            ("done", self.plans["e0"]),
            ("output_ready", replace(self.plans["e0"], generation=2)),
            ("output_ready", replace(self.plans["e0"], plan_id=99)),
        ):
            with self.assertRaises(RuntimeError):
                self.replies.accept(Message(kind, plan=plan))
        self.replies.accept(Message("output_ready", plan=self.plans["e0"]))

    def test_errors_and_non_execution_messages_fail_closed(self):
        for message in (
            Message("error", request=self.request, detail="rejected"),
            Message("error", plan=self.plans["e0"], detail="failed"),
            Message("closed"),
            Message("input_ready", plan=self.plans["e0"]),
        ):
            with self.assertRaises(RuntimeError):
                self.replies.accept(message)
        for owners in ((), ("e0", "e0"), ("",)):
            with self.assertRaises(ValueError):
                FanoutReplies(self.request, owners)
        with self.assertRaises(ValueError):
            self.replies.take("done", "unknown")
        with self.assertRaises(ValueError):
            self.replies.take("unknown", "e0")


if __name__ == "__main__":
    unittest.main()
