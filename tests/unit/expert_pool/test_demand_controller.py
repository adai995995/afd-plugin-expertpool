# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Demand admission accounts for parents without reserving zero-work owners."""

import multiprocessing
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

from afd_plugin.expert_pool.controller import (
    ControllerClientIdentity,
    ControllerLedger,
    directory_digest,
)
from afd_plugin.expert_pool.controller_service import ControllerRuntime
from afd_plugin.expert_pool.demand import plan_expert_demand
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.fanout_controller import FanoutControllerLedger
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import (
    CallKey,
    CallRequest,
    ExpertDemand,
    Message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory


def identities():
    return tuple(ControllerClientIdentity(f"a{i}", 1, f"domain-{i}") for i in range(2))


def directory():
    return PoolDirectory(
        tuple(
            StaticDirectory(
                "checkpoint",
                1,
                f"e{i}",
                (ExpertPlacement(1, 4, (2 * i, 2 * i + 1)),),
                32,
                2,
                64,
                allow_partial_experts=True,
            )
            for i in range(2)
        ),
        expert_partitioned=True,
    )


def ledger():
    return FanoutControllerLedger(directory(), identities(), demand_aware=True)


def request(client="a0", sequence=0, counts=(8, 8, 0, 0)):
    return CallRequest(
        CallKey(client, 1, sequence),
        "checkpoint",
        1,
        1,
        sum(counts) // 2,
        32,
        2,
        ExpertDemand(counts, ready_ns=100),
    )


def ready(book):
    for worker_id, resident in book.directories.items():
        book.ready(worker_id, directory_digest(resident))


def finish(book, plan):
    for kind in ("grant", "input_ready", "executing", "output_ready", "done"):
        book.progress(plan.worker_id, kind, plan)


class DemandLedgerTests(unittest.TestCase):
    def test_mode_mismatch_rejects_before_consuming_sequence(self):
        aware = ledger()
        broadcast = FanoutControllerLedger(directory(), identities())
        for book, bad in (
            (aware, replace(request(), demand=None)),
            (broadcast, request()),
            (broadcast, request(counts=(0, 0, 0, 0))),
        ):
            with self.assertRaises(ValueError):
                book.submit("a0", bad, 0)
            self.assertEqual(book.last_sequence["a0"], -1)
            self.assertFalse(book.outstanding)
            self.assertEqual(book.pending_count, 0)
        with self.assertRaises(ValueError):
            FanoutControllerLedger(directory(), identities(), demand_aware=1)

    def test_whole_layer_ledger_rejects_demand(self):
        full = StaticDirectory(
            "checkpoint", 1, "e0", (ExpertPlacement(1, 4, (0, 1, 2, 3)),), 32, 2, 64
        )
        book = ControllerLedger(PoolDirectory((full,)), identities())
        with self.assertRaises(ValueError):
            book.submit("a0", request(), 0)
        self.assertEqual(book.last_sequence["a0"], -1)
        self.assertEqual(book.selectors["a0"].next_replica[1], 0)

    def test_single_owner_plan_does_not_reserve_other_worker(self):
        book = ledger()
        ready(book)
        book.submit("a0", request(), 0)
        plan, queue_ms = book.grant(1000000)
        self.assertEqual(queue_ms, 1.0)
        self.assertEqual((plan.worker_id, plan.expert_ids), ("e0", (0, 1)))
        self.assertEqual(plan.num_assignments, 16)
        self.assertEqual(book.workers["e1"].phase, "idle")
        self.assertEqual(book.workers["e1"].generation, 0)
        self.assertIsNone(book.grant(1000001))
        finish(book, plan)
        state = book.snapshot()
        self.assertEqual(state["policy"], "controller-expert-demand-gang")
        self.assertEqual(state["completed_parent_calls"], 1)
        self.assertEqual(state["completed_child_calls"], 1)
        self.assertEqual(state["selected_worker_calls"], {"e0": 1, "e1": 0})
        self.assertEqual(state["skipped_worker_calls"], {"e0": 0, "e1": 1})
        self.assertEqual(
            state["admitted_assignments_by_worker_layer_expert"],
            {"e0": {"1": {"0": 8, "1": 8}}, "e1": {"1": {"2": 0, "3": 0}}},
        )

    def test_unselected_unready_owner_does_not_block_admission(self):
        book = ledger()
        book.ready("e0", directory_digest(book.directories["e0"]))
        book.submit("a0", request(), 0)
        plan = book.grant(1)[0]
        self.assertEqual(plan.worker_id, "e0")
        self.assertFalse(book.workers["e1"].ready)
        finish(book, plan)

    def test_two_clients_with_disjoint_demand_can_both_be_active(self):
        book = ledger()
        ready(book)
        book.submit("a0", request(), 0)
        book.submit("a1", request("a1", counts=(0, 0, 8, 8)), 0)
        first, second = (book.grant(1)[0] for _ in range(2))
        self.assertEqual([first.worker_id, second.worker_id], ["e0", "e1"])
        self.assertEqual(book.snapshot()["active_parent_calls"], 2)
        finish(book, second)
        self.assertEqual(book.workers["e0"].active, first)
        self.assertEqual(set(book.outstanding), {"a0"})
        finish(book, first)
        self.assertEqual(book.snapshot()["completed_by_client"], {"a0": 1, "a1": 1})
        self.assertEqual(book.snapshot()["completed_child_calls"], 2)

    def test_busy_unselected_worker_does_not_block_later_parent(self):
        book = ledger()
        ready(book)
        book.submit("a1", request("a1", counts=(0, 0, 8, 8)), 0)
        busy = book.grant(1)[0]
        book.submit("a0", request(), 2)
        available = book.grant(3)[0]
        self.assertEqual(available.worker_id, "e0")
        self.assertEqual(book.workers["e1"].active, busy)
        self.assertEqual(book.snapshot()["active_parent_calls"], 2)

    def test_selected_gang_is_atomic_and_held_until_parent_drains(self):
        book = ledger()
        ready(book)
        book.submit("a0", request(), 0)
        busy = book.grant(1)[0]
        book.submit("a1", request("a1", counts=(4, 4, 4, 4)), 2)
        self.assertIsNone(book.grant(3))
        self.assertIsNone(book.workers["e1"].active)
        finish(book, busy)
        first = book.grant(4)[0]
        self.assertEqual(book.workers["e1"].phase, "pending_grant")
        second = book.grant(4)[0]
        self.assertEqual((first.num_assignments, second.num_assignments), (8, 8))
        finish(book, first)
        self.assertEqual(book.workers["e0"].phase, "held")
        self.assertEqual(book.snapshot()["completed_parent_calls"], 1)
        finish(book, second)
        self.assertEqual(book.snapshot()["completed_parent_calls"], 2)
        self.assertEqual(book.snapshot()["completed_child_calls"], 3)

    def test_empty_parent_needs_no_ready_worker_or_child_plan(self):
        book = ledger()
        empty = request(counts=(0, 0, 0, 0))
        book.submit("a0", empty, 0)
        state = book.snapshot()
        self.assertEqual(state["empty_parent_calls"], 1)
        self.assertEqual(state["completed_parent_calls"], 1)
        self.assertEqual(state["completed_child_calls"], 0)
        self.assertEqual(state["selected_worker_calls"], {"e0": 0, "e1": 0})
        self.assertEqual(state["skipped_worker_calls"], {"e0": 1, "e1": 1})
        self.assertEqual(book.last_sequence["a0"], 0)
        self.assertFalse(book.outstanding)
        self.assertFalse(book.active_parents)
        self.assertFalse(book.pending_child_plans)
        self.assertFalse(book.pending_demand_plans)
        self.assertIsNone(book.grant(1))
        self.assertEqual(
            [worker.generation for worker in book.workers.values()], [0, 0]
        )
        with self.assertRaises(ValueError):
            book.submit("a0", empty, 2)
        self.assertEqual(book.snapshot()["empty_parent_calls"], 1)
        book.close_client("a0")
        with self.assertRaises(ValueError):
            book.submit("a0", request(sequence=1, counts=(0, 0, 0, 0)), 3)

    def test_invalid_empty_parent_does_not_complete_or_consume_sequence(self):
        book = ledger()
        empty = request(counts=(0, 0, 0, 0))
        for invalid in (
            replace(empty, model_id="other"),
            replace(empty, placement_version=2),
            replace(empty, layer_id=2),
            replace(empty, key=CallKey("a0", 2, 0)),
            replace(empty, demand=ExpertDemand((0, 0, 0), 100)),
        ):
            with self.assertRaises(ValueError):
                book.submit("a0", invalid, 0)
            self.assertEqual(book.last_sequence["a0"], -1)
            self.assertEqual(book.snapshot()["completed_parent_calls"], 0)
        book.submit("a0", request(), 1)
        with self.assertRaises(ValueError):
            book.submit("a0", replace(empty, key=CallKey("a0", 1, 1)), 2)

    def test_changed_assignment_identity_cannot_release_child(self):
        book = ledger()
        ready(book)
        book.submit("a0", request(), 0)
        plan = book.grant(1)[0]
        changed = replace(plan, expert_ids=(1, 0))
        with self.assertRaises(ValueError):
            book.progress("e0", "grant", changed)
        self.assertEqual(book.workers["e0"].active, plan)
        for kind in ("grant", "input_ready", "executing"):
            book.progress("e0", kind, plan)
        before = book.snapshot()
        with self.assertRaises(RuntimeError):
            book.progress("e0", "error", plan)
        self.assertEqual(book.snapshot(), before)

    def test_pending_demand_plan_is_reused_across_busy_worker_feedback(self):
        book = ledger()
        ready(book)
        with patch(
            "afd_plugin.expert_pool.fanout_controller.plan_expert_demand",
            wraps=plan_expert_demand,
        ) as planner:
            book.submit("a0", request(), 0)
            busy = book.grant(1)[0]
            book.submit("a1", request("a1", counts=(4, 4, 4, 4)), 2)
            planning_ms = book.snapshot()["planning_cpu_ms_total"]
            for now in range(3, 53):
                self.assertIsNone(book.grant(now))
            self.assertEqual(planner.call_count, 2)
            self.assertEqual(book.snapshot()["pending_demand_plans"], 1)
            self.assertEqual(book.snapshot()["planning_cpu_ms_total"], planning_ms)
            finish(book, busy)
            first = book.grant(53)[0]
            self.assertEqual(book.snapshot()["pending_demand_plans"], 0)
            second = book.grant(54)[0]
            self.assertEqual(planner.call_count, 2)
            finish(book, first)
            finish(book, second)
        self.assertFalse(book.pending_demand_plans)

    def test_counters_stay_bounded_and_queued_calls_are_not_counted_as_admitted(self):
        book = ledger()
        for sequence in range(50):
            book.submit("a0", request(sequence=sequence, counts=(0, 0, 0, 0)), 0)
        book.submit("a0", request(sequence=50), 0)
        self.assertIsNone(book.grant(1))
        state = book.snapshot()
        self.assertEqual(state["admitted_selected_worker_calls"], 0)
        self.assertEqual(state["admitted_skipped_worker_calls"], 100)
        self.assertEqual(len(book.admitted_assignments), 2)
        self.assertEqual(sum(len(v[1]) for v in book.admitted_assignments.values()), 4)
        ready(book)
        finish(book, book.grant(2)[0])
        self.assertEqual(book.snapshot()["admitted_selected_worker_calls"], 1)
        self.assertEqual(book.snapshot()["completed_parent_calls"], 51)
        self.assertGreater(book.snapshot()["planning_cpu_ms_total"], 0)


class DemandRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.book = ledger()
        self.pairs = {
            identity: multiprocessing.Pipe() for identity in ("a0", "a1", "e0", "e1")
        }
        self.runtime = ControllerRuntime(
            self.book,
            {identity: self.pairs[identity][0] for identity in ("a0", "a1")},
            {identity: self.pairs[identity][0] for identity in ("e0", "e1")},
        )
        self.outcome = []

        def run():
            try:
                self.outcome.append(self.runtime.run())
            except BaseException as error:
                self.outcome.append(error)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup_runtime)

    def cleanup_runtime(self):
        for pair in self.pairs.values():
            for connection in pair:
                connection.close()
        self.thread.join(2)

    def send(self, identity, message):
        send_message(self.pairs[identity][1], message)

    def receive(self, identity):
        return receive_message(self.pairs[identity][1], 2)

    def finish_child(self, plan):
        for kind in ("grant", "input_ready", "executing", "output_ready", "done"):
            self.send(plan.worker_id, Message(kind, plan=plan))
        for kind in ("grant", "output_ready", "done"):
            reply = self.receive(plan.request.key.client_id)
            self.assertEqual(reply.kind, kind)
            self.assertEqual(reply.plan, plan)

    def close_runtime(self):
        for client in ("a0", "a1"):
            self.send(client, Message("close"))
            self.assertEqual(self.receive(client).kind, "closed")
        for worker in ("e0", "e1"):
            self.assertEqual(self.receive(worker).kind, "close")
            self.send(worker, Message("closed"))
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())
        self.assertIsInstance(self.outcome[0], dict)
        return self.outcome[0]

    def test_empty_reply_then_selected_owner_and_parent_counts_over_real_pipes(self):
        empty = request(counts=(0, 0, 0, 0))
        self.send("a0", Message("submit", request=empty))
        reply = self.receive("a0")
        self.assertEqual((reply.kind, reply.request), ("empty_done", empty))
        self.assertFalse(self.pairs["e0"][1].poll(0))
        self.assertFalse(self.pairs["e1"][1].poll(0))
        self.send("a0", Message("submit", request=empty))
        self.assertEqual(self.receive("a0").kind, "error")

        for worker, resident in self.book.directories.items():
            self.send(worker, Message("ready", detail=directory_digest(resident)))
        left = request(sequence=1)
        right = request("a1", counts=(0, 0, 8, 8))
        self.send("a0", Message("submit", request=left))
        self.send("a1", Message("submit", request=right))
        plans = [self.receive(worker).plan for worker in ("e0", "e1")]
        self.assertEqual([plan.request for plan in plans], [left, right])
        self.assertEqual([plan.generation for plan in plans], [1, 1])
        for plan in reversed(plans):
            self.finish_child(plan)
        self.send("a0", Message("status"))
        status = self.receive("a0")
        self.assertEqual(status.kind, "snapshot")
        self.assertEqual(status.metrics["completed"], 2)
        self.assertEqual(status.metrics["completed_parents"], 3)
        result = self.close_runtime()
        self.assertEqual(result["completed_parent_calls"], 3)
        self.assertEqual(result["empty_parent_calls"], 1)
        self.assertEqual(result["completed_child_calls"], 2)
        self.assertEqual(result["completed_by_client"], {"a0": 2, "a1": 1})
        self.assertEqual(result["admitted_selected_worker_calls"], 2)
        self.assertEqual(result["admitted_skipped_worker_calls"], 4)
        self.assertEqual(result["outstanding"], 0)
        self.assertEqual(result["pending_child_plans"], 0)
        self.assertEqual(result["pending_demand_plans"], 0)


if __name__ == "__main__":
    unittest.main()
