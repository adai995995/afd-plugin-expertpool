# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static residency, exactly-once dispatch and live-credit admission contracts."""

import json
import multiprocessing
import threading
import unittest
from dataclasses import replace

from afd_plugin.expert_pool.controller import ControllerClientIdentity, directory_digest
from afd_plugin.expert_pool.controller_service import ControllerRuntime
from afd_plugin.expert_pool.deployment import ExecutionOptions
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.fanout_controller import FanoutControllerLedger
from afd_plugin.expert_pool.fanout_protocol import FanoutReplies
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import (
    CallKey,
    DispatchPlan,
    ExecutionPlan,
    ExpertTask,
    Message,
    decode_message,
    encode_message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.replica_dispatch import (
    WorkerLoad,
    select_replicas,
    validate_dispatch,
)
from tests.unit.expert_pool.test_compact_contract import deployment, directory, request
from tests.unit.expert_pool.test_dispatch_accounting import (
    EXPERT_WEIGHT_BYTES,
    demand_reports,
)
from tools.expert_pool.validate_engines import verify_dispatch


def replicated():
    original = directory()
    # Expert 0 has two copies; experts 1, 2, 3 retain unique owners.
    return PoolDirectory(
        (
            original.workers[0],
            replace(
                original.workers[1], placements=(ExpertPlacement(1, 4, (3, 0, 2)),)
            ),
        ),
        expert_partitioned=True,
        expert_replicated=True,
    )


def ledger(*, warmup=False):
    book = FanoutControllerLedger(
        replicated(),
        tuple(ControllerClientIdentity(f"a{i}", 1, f"d{i}") for i in range(3)),
        demand_aware=True,
        compact_output=True,
        receive_slots=2,
    )
    book.requires_warmup = warmup
    return book


def ready(book):
    for worker, resident in book.directories.items():
        book.ready(
            worker, directory_digest(resident), receive_slots=2, startup_complete=1
        )


class ReplicaSelectionTests(unittest.TestCase):
    def test_overlap_requires_opt_in_and_missing_coverage_is_still_rejected(self):
        pool = replicated()
        self.assertEqual(pool.locations(1, 0), (("e0", 0), ("e1", 1)))
        with self.assertRaisesRegex(ValueError, "unique owner"):
            replace(pool, expert_replicated=False)
        with self.assertRaisesRegex(ValueError, "cover every"):
            replace(pool, workers=(pool.workers[0],))

    def test_replica_configuration_requires_demand_compact_and_pipeline(self):
        options = ExecutionOptions(
            validate_worker_values=False,
            defer_output_sync=True,
            warmup_before_ready=True,
        )
        base = replace(
            deployment(),
            demand_aware=True,
            compact_output=True,
            receive_slots=2,
            execution=options,
            expert_replicated=True,
        )
        self.assertTrue(base.expert_replicated)
        for changes in (
            {"compact_output": False},
            {"receive_slots": 1},
            {"demand_aware": False},
            {"expert_replicated": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(base, **changes)
        with self.assertRaises(ValueError):
            FanoutControllerLedger(
                replicated(), (ControllerClientIdentity("a0", 1, "d"),)
            )
        with self.assertRaises(ValueError):
            ExecutionOptions(warmup_before_ready="true")

    def test_ties_rotate_between_equivalent_resident_copies(self):
        loads = {w: WorkerLoad(2, 0, False, 0) for w in ("e0", "e1")}
        call = request((8, 0, 0, 0))
        for offset, owner in ((0, "e0"), (1, "e1")):
            decision = select_replicas(replicated(), call, loads, offset)
            self.assertEqual(decision.tasks, (ExpertTask(owner, (0,), 8),))
            validate_dispatch(replicated(), decision)

    def test_backlog_and_compute_feedback_only_select_resident_available_copies(self):
        loads = {"e0": WorkerLoad(1, 32, True, 1), "e1": WorkerLoad(2, 0, False, 0)}
        self.assertEqual(
            select_replicas(replicated(), request((8, 0, 0, 0)), loads, 0).owners,
            ("e1",),
        )
        # A free GPU that lacks Expert 1 cannot substitute for its busy owner.
        loads["e0"] = WorkerLoad(0, 32, True, 2)
        self.assertIsNone(
            select_replicas(replicated(), request((0, 8, 0, 0)), loads, 0)
        )
        loads["e0"] = WorkerLoad(1, 0, True, 1)
        self.assertEqual(
            select_replicas(replicated(), request((8, 0, 0, 0)), loads, 0).owners,
            ("e1",),
        )

    def test_co_resident_experts_share_backlog_and_each_demand_is_assigned_once(self):
        loads = {w: WorkerLoad(2, 0, False, 0) for w in ("e0", "e1")}
        decision = select_replicas(replicated(), request((4, 10, 0, 0)), loads, 0)
        self.assertEqual(
            decision.tasks, (ExpertTask("e0", (1,), 10), ExpertTask("e1", (0,), 4))
        )
        self.assertEqual(sum(t.num_assignments for t in decision.tasks), 14)

    def test_dispatch_codec_rejects_duplicate_missing_nonresident_and_extra_fields(
        self,
    ):
        call = request((4, 4, 0, 0))
        decision = DispatchPlan(call, (ExpertTask("e0", (0, 1), 8),))
        message = Message("dispatch", dispatch=decision)
        self.assertEqual(decode_message(encode_message(message)), message)
        for tasks in (
            (ExpertTask("e0", (0,), 4),),
            (ExpertTask("e0", (0, 1), 8), ExpertTask("e1", (0,), 4)),
        ):
            with self.assertRaises(ValueError):
                DispatchPlan(call, tasks)
        with self.assertRaisesRegex(ValueError, "nonresident"):
            validate_dispatch(
                replicated(), DispatchPlan(call, (ExpertTask("e1", (0, 1), 8),))
            )
        for target in ("dispatch", "task"):
            raw = json.loads(encode_message(message))
            record = (
                raw["dispatch"] if target == "dispatch" else raw["dispatch"]["tasks"][0]
            )
            record["unknown"] = 1
            with self.assertRaises(ValueError):
                decode_message(json.dumps(raw).encode())

    def test_worker_grants_cannot_expand_the_announced_expert_subset(self):
        call = request((4, 4, 0, 0))
        decision = DispatchPlan(
            call, (ExpertTask("e0", (1,), 4), ExpertTask("e1", (0,), 4))
        )
        replies = FanoutReplies(
            call, decision.owners, dispatch_plan=decision, receive_slots=2
        )
        bad = ExecutionPlan(call, "e0", 1, 0, 1, (0, 1), 8)
        with self.assertRaises(RuntimeError):
            replies.accept(Message("grant", plan=bad))

    def test_ready_requires_completed_startup_before_any_admission(self):
        book = ledger(warmup=True)
        call = request((8, 0, 0, 0))
        book.submit("a0", call, 0)
        for value in (0, True, 2):
            with self.assertRaises(ValueError):
                book.ready(
                    "e0",
                    directory_digest(book.directories["e0"]),
                    receive_slots=2,
                    startup_complete=value,
                )
            self.assertIsNone(book.grant(1))
        ready(book)
        self.assertIsNotNone(book.grant(2))
        self.assertEqual(book.completed_by_client["a0"], 0)

    def test_controller_replans_at_admission_and_publishes_one_decision(self):
        book = ledger()
        ready(book)
        for index in range(2):
            call = replace(request((8, 0, 0, 0)), key=CallKey(f"a{index}", 1, 0))
            book.submit(f"a{index}", call, 0)
        first = book.grant(1)[0]
        second = book.grant(1)[0]
        self.assertEqual((first.worker_id, second.worker_id), ("e0", "e1"))
        self.assertEqual(len(book.dispatch_notifications), 2)
        for phase in ("grant", "input_ready", "executing"):
            book.progress(first.worker_id, phase, first)
        self.assertTrue(book.worker_loads()["e0"].computing)
        self.assertEqual(book.worker_loads()["e0"].pending_assignments, 8)
        book.progress("e0", "output_ready", first)
        self.assertEqual(book.worker_loads()["e0"].pending_assignments, 0)
        self.assertEqual(book.worker_loads()["e0"].occupied_slots, 1)

    def test_unready_unique_owner_prevents_partial_reservation_or_announcement(self):
        book = ledger()
        book.ready("e0", directory_digest(book.directories["e0"]), receive_slots=2)
        book.submit("a0", request((4, 0, 4, 0)), 0)
        self.assertIsNone(book.grant(1))
        self.assertFalse(book.dispatch_notifications)
        self.assertEqual(book.reserved_count, 0)
        book.ready("e1", directory_digest(book.directories["e1"]), receive_slots=2)
        first = book.grant(2)[0]
        self.assertEqual(book.reserved_count, 2)
        self.assertEqual(len(book.dispatch_notifications), 1)
        self.assertNotEqual(book.grant(3)[0].worker_id, first.worker_id)

    def test_announcement_precedes_child_feedback_over_real_pipes(self):
        book = ledger(warmup=True)
        pairs = {key: multiprocessing.Pipe() for key in (*book.clients, *book.workers)}
        runtime = ControllerRuntime(
            book,
            {key: pairs[key][0] for key in book.clients},
            {key: pairs[key][0] for key in book.workers},
        )
        outcome = []

        def run():
            try:
                outcome.append(runtime.run())
            except BaseException as error:
                outcome.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            for worker, resident in book.directories.items():
                send_message(
                    pairs[worker][1],
                    Message(
                        "ready",
                        detail=directory_digest(resident),
                        metrics={"receive_slots": 2, "startup_complete": 1},
                    ),
                )
            call = request((8, 0, 0, 0))
            send_message(pairs["a0"][1], Message("submit", request=call))
            announced = receive_message(pairs["a0"][1], 2)
            self.assertEqual(announced.kind, "dispatch")
            self.assertEqual(announced.dispatch.request, call)
            owner = announced.dispatch.owners[0]
            plan = receive_message(pairs[owner][1], 2).plan
            for phase in ("grant", "input_ready", "executing", "output_ready", "done"):
                send_message(pairs[owner][1], Message(phase, plan=plan))
            for phase in ("grant", "output_ready", "done"):
                self.assertEqual(receive_message(pairs["a0"][1], 2).kind, phase)
            for client in book.clients:
                send_message(pairs[client][1], Message("close"))
                self.assertEqual(receive_message(pairs[client][1], 2).kind, "closed")
            for worker in book.workers:
                self.assertEqual(receive_message(pairs[worker][1], 2).kind, "close")
                send_message(pairs[worker][1], Message("closed"))
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(outcome[0]["pending_dispatch_notifications"], 0)
            self.assertEqual(outcome[0]["completed_parent_calls"], 1)
        finally:
            for pair in pairs.values():
                for connection in pair:
                    connection.close()
            thread.join(2)


class ReplicaAccountingTests(unittest.TestCase):
    def reports(self):
        pool, before, closed, workers = demand_reports()
        original = pool
        pool = replace(
            pool,
            workers=(
                pool.workers[0],
                replace(
                    pool.workers[1], placements=(ExpertPlacement(1, 4, (1, 3, 0)),)
                ),
            ),
            expert_replicated=True,
        )
        for status in before + [record["status"][0] for record in closed]:
            logical = status["dispatch"]["demand"]["expert_assignments"]["1"]
            status["dispatch"]["worker_expert_assignments"] = {
                worker.worker_id: {
                    "1": {
                        str(expert): logical[expert]
                        for expert in worker.placements[0].expert_ids
                    }
                }
                for worker in original.workers
            }
            status["dispatch"]["worker_expert_assignments"]["e1"]["1"]["0"] = 0
        workers[1]["resident_experts"]["1"].append(0)
        workers[1]["resident_weight_bytes"] += EXPERT_WEIGHT_BYTES
        workers[1]["expert_assignments"]["1"]["0"] = 0
        return pool, before, closed, workers

    def test_extra_resident_copy_does_not_double_assignment_or_execution_counts(self):
        result = verify_dispatch(
            *self.reports(), demand_aware=True, expert_weight_bytes=EXPERT_WEIGHT_BYTES
        )
        self.assertEqual(result["expert_assignments"], 20)
        self.assertEqual(
            result["resident_routed_weight_bytes"], 5 * EXPERT_WEIGHT_BYTES
        )
        self.assertEqual(
            result["expert_assignments_by_worker_layer_expert"]["e1"]["1"]["0"], 0
        )

    def test_agreeing_a_e_counters_cannot_hide_duplicate_expert_execution(self):
        pool, before, closed, workers = self.reports()
        closed[1]["status"][0]["dispatch"]["worker_expert_assignments"]["e1"]["1"][
            "0"
        ] = 2
        workers[1]["expert_assignments"]["1"]["0"] = 2
        with self.assertRaisesRegex(AssertionError, "lost or repeated"):
            verify_dispatch(pool, before, closed, workers, demand_aware=True)
