# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Reservation invariants and real control-channel ordering without CUDA."""

import json
import multiprocessing
import tempfile
import threading
import unittest
from dataclasses import asdict, replace
from pathlib import Path

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
    Message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory


def ledger(workers=1, clients=2):
    directory = PoolDirectory(
        tuple(
            StaticDirectory(
                "checkpoint",
                1,
                f"e{i}",
                (
                    ExpertPlacement(1, 4, (0, 1, 2, 3)),
                    ExpertPlacement(2, 4, (0, 1, 2, 3)),
                ),
                32,
                2,
                64,
            )
            for i in range(workers)
        )
    )
    return ControllerLedger(
        directory,
        tuple(
            ControllerClientIdentity(f"a{i}", 1, f"domain-{i}") for i in range(clients)
        ),
    )


def request(client="a0", sequence=0, layer=1):
    return CallRequest(CallKey(client, 1, sequence), "checkpoint", 1, layer, 8, 32, 2)


def make_ready(book):
    for worker, directory in book.directories.items():
        book.ready(worker, directory_digest(directory))


def finish(book, plan):
    for phase in ("grant", "input_ready", "executing", "output_ready", "done"):
        book.progress(plan.worker_id, phase, plan)


class ControllerLedgerTests(unittest.TestCase):
    def test_domain_rotation_is_not_weighted_by_client_count(self):
        book = ControllerLedger(
            ledger().directory,
            (
                ControllerClientIdentity("a0", 1, "domain-0"),
                ControllerClientIdentity("a1", 1, "domain-0"),
                ControllerClientIdentity("a2", 1, "domain-1"),
            ),
        )
        make_ready(book)
        for client_id in book.clients:
            book.submit(client_id, request(client_id), 0)
        first, _ = book.grant(1)
        self.assertEqual(first.request.key.client_id, "a0")
        finish(book, first)
        self.assertEqual(book.grant(2)[0].request.key.client_id, "a2")

    def test_controller_deployment_roundtrip_and_socket_collisions(self):
        deployment = PoolDeployment(
            "/checkpoint",
            "checkpoint",
            64,
            (ClientEndpoint("a0", 1, "domain-0", "/private-pool/direct.sock", 15000),),
            controller=ControllerConfig("/private-pool"),
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "deployment.json"
            path.write_text(json.dumps(asdict(deployment)))
            self.assertEqual(PoolDeployment.read(path), deployment)
        self.assertEqual(
            deployment.controller_path("worker", "worker-0"),
            "/private-pool/worker-0.sock",
        )
        with self.assertRaises(ValueError):
            replace(
                deployment,
                clients=(
                    replace(
                        deployment.clients[0],
                        control_path="/private-pool/client-0.sock",
                    ),
                ),
            )
        with self.assertRaises(ValueError):
            replace(deployment, controller=ControllerConfig("/" + "x" * 100))
        with self.assertRaises(ValueError):
            ControllerConfig("relative")

    def test_readiness_requires_exact_resident_directory(self):
        book = ledger()
        book.submit("a0", request(), 0)
        self.assertIsNone(book.grant(1))
        with self.assertRaises(ValueError):
            book.ready("e0", "wrong")
        make_ready(book)
        with self.assertRaises(ValueError):
            make_ready(book)
        self.assertIsNotNone(book.grant(1))

    def test_busy_worker_queues_other_client_without_duplicate_slot(self):
        book = ledger()
        make_ready(book)
        book.submit("a0", request(), 0)
        first, _ = book.grant(1)
        book.submit("a1", request("a1", layer=2), 2)
        self.assertEqual(book.pending_count, 1)
        self.assertIsNone(book.grant(3))
        for phase in ("grant", "input_ready", "executing", "output_ready"):
            book.progress("e0", phase, first)
            self.assertIsNone(book.grant(4))
        book.progress("e0", "done", first)
        second, queue_ms = book.grant(1000002)
        self.assertEqual(queue_ms, 1.0)
        self.assertEqual(second.request.key.client_id, "a1")
        self.assertEqual(second.generation, first.generation + 1)

    def test_stale_duplicate_or_foreign_feedback_cannot_release_new_slot(self):
        book = ledger(2)
        make_ready(book)
        book.submit("a0", request(), 0)
        first, _ = book.grant(1)
        finish(book, first)
        book.submit("a0", request(sequence=1), 2)
        current, _ = book.grant(3)
        for source, kind, plan in (
            (first.worker_id, "done", first),
            (current.worker_id, "done", current),
            (current.worker_id, "grant", replace(current, generation=99)),
            (first.worker_id, "grant", current),
        ):
            with self.assertRaises(ValueError):
                book.progress(source, kind, plan)
            self.assertEqual(book.workers[current.worker_id].active, current)

    def test_value_error_only_releases_after_safe_execution_rejection(self):
        book = ledger()
        make_ready(book)
        book.submit("a0", request(), 0)
        plan, _ = book.grant(1)
        with self.assertRaises(ValueError):
            book.progress("e0", "error", plan)
        for phase in ("grant", "input_ready", "executing"):
            book.progress("e0", phase, plan)
        book.progress("e0", "error", plan)
        self.assertIsNone(book.workers["e0"].active)
        self.assertEqual(book.workers["e0"].rejected, 1)
        self.assertEqual(book.workers["e0"].completed, 0)

    def test_invalid_calls_do_not_consume_sequence_or_round_robin(self):
        book = ledger(2)
        original = request()
        for bad in (
            replace(original, model_id="other"),
            replace(original, placement_version=9),
            replace(original, layer_id=7),
            replace(original, num_tokens=65),
            replace(original, key=CallKey("a1", 1, 0)),
            replace(original, key=CallKey("a0", 2, 0)),
        ):
            with self.assertRaises(ValueError):
                book.submit("a0", bad, 0)
        self.assertEqual(book.last_sequence["a0"], -1)
        make_ready(book)
        book.submit("a0", original, 0)
        self.assertEqual(book.grant(1)[0].worker_id, "e0")

    def test_sessions_are_bounded_and_close_requires_drain(self):
        book = ledger()
        make_ready(book)
        book.submit("a0", request(), 0)
        with self.assertRaises(ValueError):
            book.submit("a0", request(sequence=1), 0)
        with self.assertRaises(ValueError):
            book.close_client("a0")
        finish(book, book.grant(1)[0])
        with self.assertRaises(ValueError):
            book.submit("a0", request(), 0)
        book.close_client("a0")
        with self.assertRaises(ValueError):
            book.submit("a0", request(sequence=1), 0)

    def test_two_workers_can_be_reserved_and_idle_client_is_not_a_barrier(self):
        book = ledger(2)
        make_ready(book)
        book.submit("a0", request(), 0)
        first, _ = book.grant(1)
        book.submit("a1", request("a1"), 0)
        second, _ = book.grant(1)
        self.assertNotEqual(first.worker_id, second.worker_id)
        finish(book, first)
        self.assertEqual(len(book.outstanding), 1)
        book.submit("a0", request(sequence=1, layer=2), 2)
        self.assertIsNotNone(book.grant(3))

    def test_completed_history_is_aggregated_not_retained_per_call(self):
        book = ledger()
        make_ready(book)
        for i in range(500):
            book.submit("a0", request(sequence=i), i)
            finish(book, book.grant(i)[0])
        snapshot = book.snapshot()
        self.assertEqual(snapshot["outstanding"], 0)
        self.assertEqual(snapshot["pending"], 0)
        self.assertEqual(snapshot["workers"]["e0"]["completed_calls"], 500)
        self.assertEqual(snapshot["workers"]["e0"]["layer_calls"], {"1": 500, "2": 0})


class ControllerWireTests(unittest.TestCase):
    def test_busy_worker_does_not_block_admission_or_status(self):
        book = ledger(clients=3)
        pairs = [multiprocessing.Pipe() for _ in range(4)]
        self.addCleanup(lambda: [c.close() for pair in pairs for c in pair])
        clients = {f"a{i}": pairs[i][0] for i in range(3)}
        runtime = ControllerRuntime(book, clients, {"e0": pairs[3][0]})
        outcome = []

        def run():
            try:
                outcome.append(runtime.run())
            except BaseException as error:
                outcome.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        worker = pairs[3][1]
        a0, a1, observer = [pair[1] for pair in pairs[:3]]
        send_message(
            worker, Message("ready", detail=directory_digest(book.directories["e0"]))
        )
        send_message(a0, Message("submit", request=request()))
        first = receive_message(worker, 2).plan
        send_message(worker, Message("grant", plan=first))
        self.assertEqual(receive_message(a0, 2).kind, "grant")
        send_message(worker, Message("input_ready", plan=first))
        send_message(worker, Message("executing", plan=first))
        send_message(a1, Message("submit", request=request("a1", layer=2)))
        for _ in range(10):
            send_message(observer, Message("status"))
            snapshot = receive_message(observer, 2)
            if snapshot.metrics["pending"] == 1:
                break
        self.assertEqual(snapshot.metrics["pending"], 1)
        self.assertEqual(snapshot.metrics["reserved"], 1)
        self.assertFalse(a1.poll(0))
        send_message(worker, Message("output_ready", plan=first))
        self.assertEqual(receive_message(a0, 2).kind, "output_ready")
        self.assertFalse(a1.poll(0))
        send_message(worker, Message("done", plan=first))
        self.assertEqual(receive_message(a0, 2).kind, "done")
        second = receive_message(worker, 2).plan
        self.assertEqual(second.request.key.client_id, "a1")
        for phase in ("grant", "input_ready", "executing", "output_ready", "done"):
            send_message(worker, Message(phase, plan=second))
        for expected in ("grant", "output_ready", "done"):
            self.assertEqual(receive_message(a1, 2).kind, expected)
        for client in (a0, a1, observer):
            send_message(client, Message("close"))
            self.assertEqual(receive_message(client, 2).kind, "closed")
        self.assertEqual(receive_message(worker, 2).kind, "close")
        send_message(worker, Message("closed"))
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0], dict)
        self.assertEqual(outcome[0]["outstanding"], 0)
        self.assertEqual(outcome[0]["completed_by_client"], {"a0": 1, "a1": 1, "a2": 0})

    def test_worker_disconnect_preserves_uncertain_reservation(self):
        book = ledger(clients=1)
        client, client_peer = multiprocessing.Pipe()
        worker, worker_peer = multiprocessing.Pipe()
        for c in (client, client_peer, worker, worker_peer):
            self.addCleanup(c.close)
        make_ready(book)
        book.submit("a0", request(), 0)
        plan, _ = book.grant(1)
        worker_peer.close()
        with self.assertRaises(EOFError):
            ControllerRuntime(book, {"a0": client}, {"e0": worker}).run()
        self.assertEqual(book.workers["e0"].active, plan)


if __name__ == "__main__":
    unittest.main()
