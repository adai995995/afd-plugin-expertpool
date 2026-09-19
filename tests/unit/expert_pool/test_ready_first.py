# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ready-first placement invariants and real CPU control-channel dispatch.

These tests exercise reservation decisions, not simulated GPU performance.
"""

import json
import multiprocessing
import threading
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from afd_plugin.expert_pool.controller import (
    ControllerClientIdentity,
    ControllerLedger,
    directory_digest,
)
from afd_plugin.expert_pool.controller_service import ControllerRuntime
from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ControllerConfig,
    PoolDeployment,
)
from afd_plugin.expert_pool.directory import PoolDirectory
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

CONTROL_TIMEOUT_S = 2
TO_RETURNING = ("grant", "input_ready", "executing", "output_ready")


def make_directory(
    resident_layers: tuple[tuple[int, ...], ...] = ((1, 2), (1, 2)),
) -> PoolDirectory:
    return PoolDirectory(
        tuple(
            StaticDirectory(
                "checkpoint",
                1,
                f"e{index}",
                tuple(ExpertPlacement(layer, 4, (0, 1, 2, 3)) for layer in layers),
                32,
                2,
                64,
            )
            for index, layers in enumerate(resident_layers)
        )
    )


def make_book(
    policy: str = "ready_first",
    resident_layers: tuple[tuple[int, ...], ...] = ((1, 2), (1, 2)),
    domains: tuple[str, ...] = ("domain-0", "domain-1", "domain-2"),
) -> ControllerLedger:
    return ControllerLedger(
        make_directory(resident_layers),
        tuple(
            ControllerClientIdentity(f"a{index}", 1, domain)
            for index, domain in enumerate(domains)
        ),
        scheduling_policy=policy,
    )


def request(client: str = "a0", sequence: int = 0, layer: int = 1) -> CallRequest:
    return CallRequest(CallKey(client, 1, sequence), "checkpoint", 1, layer, 8, 32, 2)


def ready(book: ControllerLedger, *worker_ids: str) -> None:
    for worker_id in worker_ids or tuple(book.workers):
        book.ready(worker_id, directory_digest(book.directories[worker_id]))


def finish(book: ControllerLedger, plan: ExecutionPlan) -> None:
    for phase in (*TO_RETURNING, "done"):
        book.progress(plan.worker_id, phase, plan)


class ReadyFirstLedgerTests(unittest.TestCase):
    def test_same_trace_can_use_idle_replica_without_waiting_for_bound_target(self):
        for policy in ("round_robin", "ready_first"):
            with self.subTest(policy=policy):
                book = make_book(policy)
                ready(book)
                book.submit("a0", request(), 0)
                busy, _ = book.grant(1)
                self.assertEqual(busy.worker_id, "e0")
                # a2 and a0 have the same initial preference in this two-copy pool.
                book.submit("a2", request("a2"), 2)
                candidate = book.grant(3)
                if policy == "round_robin":
                    self.assertIsNone(candidate)
                    self.assertIsNone(book.workers["e1"].active)
                    finish(book, busy)
                    self.assertEqual(book.grant(4)[0].worker_id, "e0")
                    self.assertEqual(book.snapshot()["busy_replica_bypasses"], 0)
                else:
                    self.assertEqual(candidate[0].worker_id, "e1")
                    self.assertEqual(book.workers["e0"].active, busy)
                    self.assertEqual(book.snapshot()["busy_replica_bypasses"], 1)
                self.assertEqual(
                    book.snapshot()["policy"],
                    f"controller-resident-layer-{policy.replace('_', '-')}",
                )

    def test_waiting_call_can_use_either_copy_only_after_done(self):
        book = make_book()
        ready(book)
        book.submit("a0", request(), 0)
        first, _ = book.grant(1)
        book.submit("a1", request("a1"), 0)
        second, _ = book.grant(1)
        self.assertEqual((first.worker_id, second.worker_id), ("e0", "e1"))
        book.submit("a2", request("a2"), 2)
        self.assertIsNone(book.grant(3))
        for phase in TO_RETURNING:
            book.progress("e1", phase, second)
            self.assertIsNone(book.grant(4))
        self.assertEqual(book.workers["e1"].active, second)
        self.assertEqual(book.snapshot()["busy_replica_bypasses"], 0)
        book.progress("e1", "done", second)
        resumed, queue_ms = book.grant(1_000_002)
        self.assertEqual(resumed.worker_id, "e1")
        self.assertEqual(resumed.request.key.client_id, "a2")
        self.assertEqual(resumed.generation, second.generation + 1)
        self.assertEqual(queue_ms, 1.0)
        self.assertEqual(book.workers["e0"].active, first)
        self.assertEqual(book.snapshot()["busy_replica_bypasses"], 1)

    def test_only_ready_workers_with_the_requested_resident_layer_are_candidates(self):
        book = make_book(resident_layers=((1,), (2,), (1,)))
        ready(book, "e1", "e2")
        book.submit("a0", request(), 0)
        first, _ = book.grant(1)
        self.assertEqual(first.worker_id, "e2")
        self.assertEqual(book.snapshot()["busy_replica_bypasses"], 0)
        book.submit("a1", request("a1"), 2)
        self.assertIsNone(book.grant(3))
        self.assertIsNone(book.workers["e1"].active)
        ready(book, "e0")
        self.assertEqual(book.grant(4)[0].worker_id, "e0")
        with self.assertRaises(ValueError):
            book.submit("a2", request("a2", layer=3), 5)
        self.assertNotIn("a2", book.outstanding)

    def test_blocked_head_does_not_prevent_an_executable_call_in_the_same_domain(self):
        book = make_book(
            resident_layers=((1,), (2,)),
            domains=("busy-domain", "shared-domain", "shared-domain"),
        )
        ready(book)
        book.submit("a0", request(), 0)
        first, _ = book.grant(1)
        book.submit("a1", request("a1"), 2)
        book.submit("a2", request("a2", layer=2), 3)
        executable, _ = book.grant(4)
        self.assertEqual(executable.request.key.client_id, "a2")
        self.assertEqual(executable.worker_id, "e1")
        self.assertEqual(book.pending_count, 1)
        finish(book, first)
        self.assertEqual(book.grant(5)[0].request.key.client_id, "a1")

    def test_domain_rotation_is_not_weighted_by_number_of_clients(self):
        book = make_book(domains=("many", "many", "many", "one"))
        ready(book)
        for client_id in book.clients:
            book.submit(client_id, request(client_id), 0)
        first, _ = book.grant(1)
        second, _ = book.grant(1)
        self.assertEqual(first.request.key.client_id, "a0")
        self.assertEqual(second.request.key.client_id, "a3")
        self.assertNotEqual(first.worker_id, second.worker_id)
        self.assertIsNone(book.grant(2))
        finish(book, first)
        self.assertEqual(book.grant(3)[0].request.key.client_id, "a1")

    def test_repeated_grants_never_reserve_the_same_slot_twice(self):
        book = make_book(domains=tuple(f"domain-{i}" for i in range(5)))
        ready(book)
        for client_id in book.clients:
            book.submit(client_id, request(client_id), 0)
        first, _ = book.grant(1)
        second, _ = book.grant(1)
        for _ in range(5):
            self.assertIsNone(book.grant(2))
        self.assertEqual({first.worker_id, second.worker_id}, {"e0", "e1"})
        self.assertEqual(book.pending_count, 3)
        self.assertEqual(book.snapshot()["peak_reserved"], 2)
        finish(book, first)
        replacement, _ = book.grant(3)
        self.assertEqual(replacement.worker_id, first.worker_id)
        self.assertEqual(replacement.generation, first.generation + 1)
        self.assertEqual(book.workers[second.worker_id].active, second)

    def test_idle_ties_rotate_per_client_and_layer(self):
        book = make_book(resident_layers=((1, 2), (1, 2), (1, 2)), domains=("domain",))
        ready(book)
        targets = []
        for sequence in range(4):
            book.submit("a0", request(sequence=sequence), sequence)
            plan, _ = book.grant(sequence)
            targets.append(plan.worker_id)
            finish(book, plan)
        self.assertEqual(targets, ["e0", "e1", "e2", "e0"])
        book.submit("a0", request(sequence=4, layer=2), 4)
        self.assertEqual(book.grant(4)[0].worker_id, "e0")

    def test_no_available_worker_does_not_advance_tie_break_order(self):
        book = make_book(
            resident_layers=((1,), (1,), (1,)),
            domains=("d0", "d1", "d2", "d3"),
        )
        ready(book)
        active = []
        for client_id in ("a0", "a1", "a2"):
            book.submit(client_id, request(client_id), 0)
            active.append(book.grant(1)[0])
        self.assertEqual([plan.worker_id for plan in active], ["e0", "e1", "e2"])
        book.submit("a3", request("a3"), 2)
        self.assertIsNone(book.grant(3))
        finish(book, active[0])
        finish(book, active[2])
        self.assertEqual(book.grant(4)[0].worker_id, "e0")

    def test_stale_or_early_feedback_cannot_release_a_reassigned_slot(self):
        book = make_book(resident_layers=((1,),))
        ready(book)
        book.submit("a0", request(), 0)
        original, _ = book.grant(1)
        finish(book, original)
        book.submit("a1", request("a1"), 2)
        current, _ = book.grant(3)
        book.submit("a2", request("a2"), 4)
        for kind, plan in (
            ("done", original),
            ("error", original),
            ("done", current),
            ("error", current),
            ("grant", replace(current, generation=original.generation)),
            ("grant", replace(current, worker_id="foreign")),
        ):
            with self.subTest(kind=kind, plan=plan):
                with self.assertRaises(ValueError):
                    book.progress("e0", kind, plan)
                self.assertEqual(book.workers["e0"].active, current)
                self.assertIsNone(book.grant(5))
        for phase in TO_RETURNING[:-1]:
            book.progress("e0", phase, current)
        book.progress("e0", "error", current)
        replacement, _ = book.grant(6)
        self.assertEqual(replacement.request.key.client_id, "a2")
        with self.assertRaises(ValueError):
            book.progress("e0", "error", current)
        self.assertEqual(book.workers["e0"].active, replacement)

    def test_policy_configuration_roundtrip_and_legacy_defaults(self):
        deployment = PoolDeployment(
            "/checkpoint",
            "checkpoint",
            64,
            (ClientEndpoint("a0", 1, "domain", "/pool/direct.sock", 15000),),
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "deployment.json"
            path.write_text(json.dumps(asdict(deployment)))
            self.assertIsNone(PoolDeployment.read(path).controller)
            for policy in ("round_robin", "ready_first"):
                configured = replace(
                    deployment,
                    controller=ControllerConfig("/pool", scheduling_policy=policy),
                )
                path.write_text(json.dumps(asdict(configured)))
                self.assertEqual(PoolDeployment.read(path), configured)
            legacy = asdict(deployment)
            legacy["controller"] = {"socket_dir": "/pool"}
            path.write_text(json.dumps(legacy))
            self.assertEqual(
                PoolDeployment.read(path).controller.scheduling_policy, "round_robin"
            )
            for invalid in ("unknown", "", None, True, 1, []):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        ControllerConfig("/pool", scheduling_policy=invalid)
                    with self.assertRaises(ValueError):
                        make_book(invalid)
                    legacy["controller"]["scheduling_policy"] = invalid
                    path.write_text(json.dumps(legacy))
                    with self.assertRaises(ValueError):
                        PoolDeployment.read(path)
        default = ControllerLedger(
            make_directory(), (ControllerClientIdentity("a0", 1, "domain"),)
        )
        self.assertEqual(
            default.snapshot()["policy"], "controller-resident-layer-round-robin"
        )


class ReadyFirstWireTests(unittest.TestCase):
    def test_busy_copy_does_not_block_grant_to_another_resident_worker(self):
        book = make_book()
        pairs = [multiprocessing.Pipe() for _ in range(5)]
        for pair in pairs:
            for connection in pair:
                self.addCleanup(connection.close)
        clients = {f"a{i}": pairs[i][0] for i in range(3)}
        workers = {f"e{i}": pairs[i + 3][0] for i in range(2)}
        runtime = ControllerRuntime(book, clients, workers)
        outcome = []

        def run():
            try:
                outcome.append(runtime.run())
            except BaseException as error:
                outcome.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        a0, observer, a2, e0, e1 = [pair[1] for pair in pairs]
        for worker_id, peer in (("e0", e0), ("e1", e1)):
            send_message(
                peer,
                Message("ready", detail=directory_digest(book.directories[worker_id])),
            )
        send_message(a0, Message("submit", request=request()))
        first = receive_message(e0, CONTROL_TIMEOUT_S).plan
        for phase in TO_RETURNING[:-1]:
            send_message(e0, Message(phase, plan=first))
        self.assertEqual(receive_message(a0, CONTROL_TIMEOUT_S).kind, "grant")
        send_message(a2, Message("submit", request=request("a2")))
        second = receive_message(e1, CONTROL_TIMEOUT_S).plan
        self.assertEqual(second.request.key.client_id, "a2")
        for phase in TO_RETURNING[:-1]:
            send_message(e1, Message(phase, plan=second))
        self.assertEqual(receive_message(a2, CONTROL_TIMEOUT_S).kind, "grant")
        send_message(observer, Message("status"))
        snapshot = receive_message(observer, CONTROL_TIMEOUT_S)
        self.assertEqual(snapshot.metrics["reserved"], 2)
        self.assertEqual(snapshot.metrics["pending"], 0)
        self.assertEqual(snapshot.metrics["completed"], 0)
        for worker, client, plan in ((e0, a0, first), (e1, a2, second)):
            for phase in ("output_ready", "done"):
                send_message(worker, Message(phase, plan=plan))
                self.assertEqual(receive_message(client, CONTROL_TIMEOUT_S).kind, phase)
        for client in (a0, observer, a2):
            send_message(client, Message("close"))
            self.assertEqual(receive_message(client, CONTROL_TIMEOUT_S).kind, "closed")
        for worker in (e0, e1):
            self.assertEqual(receive_message(worker, CONTROL_TIMEOUT_S).kind, "close")
            send_message(worker, Message("closed"))
        thread.join(CONTROL_TIMEOUT_S)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0], dict)
        self.assertEqual(outcome[0]["completed_by_client"], {"a0": 1, "a1": 0, "a2": 1})
        self.assertEqual(outcome[0]["outstanding"], 0)
        self.assertEqual(outcome[0]["busy_replica_bypasses"], 1)


if __name__ == "__main__":
    unittest.main()
