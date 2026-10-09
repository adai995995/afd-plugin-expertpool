# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Direct full-E results return before accounting, with bounded generations."""

import json
import tempfile
import threading
import unittest
from dataclasses import asdict, replace
from multiprocessing import Pipe
from pathlib import Path

import torch

from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ExecutionOptions,
    PoolDeployment,
    WorkerPlacement,
)
from afd_plugin.expert_pool.full_e_client import DirectFullPoolClient
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


class Transport:
    device, peer = torch.device("cpu"), 0

    def __init__(self):
        self.operations = []

    def transfer(self, tensors, *, send):
        self.operations.append((send, tuple(t.shape for t in tensors)))
        if not send:
            tensors[0].fill_(7)
        return 0.0

    def close(self):
        pass


def directory():
    return StaticDirectory(
        "checkpoint", 1, "e0", (ExpertPlacement(1, 4, (0, 1, 2, 3)),), 32, 2, 64
    )


class DirectFullClientTests(unittest.TestCase):
    def test_stale_feedback_on_an_idle_channel_prevents_a_new_dispatch(self):
        reader, writer = Pipe()
        transport = Transport()
        client = DirectFullPoolClient(
            "a", 1, directory(), reader, transport, 1, client_slot=0, receive_slots=2
        )
        stale = ExecutionPlan(
            CallRequest(CallKey("a", 1, 0), "checkpoint", 1, 1, 3, 32, 2), "e0", 0, 0, 1
        )
        send_message(writer, Message("done", plan=stale))
        try:
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                client.execute(
                    1,
                    torch.ones(3, 32, dtype=torch.bfloat16),
                    torch.ones(3, 2),
                    torch.zeros(3, 2, dtype=torch.int32),
                )
            self.assertTrue(client.failed)
            self.assertEqual(transport.operations, [])
            self.assertFalse(writer.poll(0))
        finally:
            reader.close()
            writer.close()

    def test_result_does_not_wait_for_done_and_late_done_matches_previous_call(self):
        reader, writer = Pipe()
        release_done = threading.Event()
        errors = []
        plans = []

        def worker():
            try:
                first = receive_message(writer, 2)
                self.assertEqual(first.kind, "execute")
                plans.append(first.plan)
                send_message(writer, Message("output_ready", plan=first.plan))
                # The successor arrives before first.done: result receipt must
                # not be tied to the worker's completion-accounting message.
                second = receive_message(writer, 2)
                self.assertEqual(second.kind, "execute")
                plans.append(second.plan)
                send_message(
                    writer,
                    Message(
                        "done", plan=first.plan, metrics={"compute_phase_gpu_ms": 0.5}
                    ),
                )
                send_message(writer, Message("output_ready", plan=second.plan))
                if not release_done.wait(2):
                    raise TimeoutError("Client waited for done before returning result")
                send_message(
                    writer,
                    Message(
                        "done", plan=second.plan, metrics={"compute_phase_gpu_ms": 0.7}
                    ),
                )
                self.assertEqual(receive_message(writer, 2).kind, "close")
                send_message(writer, Message("closed"))
            except BaseException as error:
                errors.append(error)

        transport = Transport()
        client = DirectFullPoolClient(
            "a", 1, directory(), reader, transport, 1, client_slot=1, receive_slots=2
        )
        client.set_metrics(True)
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            for sequence, rows in ((0, 3), (9, 5)):
                hidden = torch.ones(rows, 32, dtype=torch.bfloat16)
                weights = torch.full((rows, 2), 0.5, dtype=torch.float32)
                ids = torch.zeros(rows, 2, dtype=torch.int32)
                output, result = client.execute(
                    1, hidden, weights, ids, call_seq=sequence
                )
                self.assertEqual(result.kind, "result")
                self.assertEqual(result.metrics["client_completion_wait_ms"], 0)
                torch.testing.assert_close(output, torch.full_like(hidden, 7))
            self.assertFalse(release_done.is_set())
            self.assertEqual(client.completed_calls, 1)
            self.assertEqual(len(client.events.pending), 1)
            self.assertEqual([p.generation for p in plans], [1, 2])
            self.assertEqual([p.slot_id for p in plans], [1, 1])
            self.assertEqual([p.request.key.call_seq for p in plans], [0, 9])
            release_done.set()
            client.drain_feedback()
            self.assertEqual(client.completed_calls, 2)
            self.assertFalse(client.events.pending)
            self.assertFalse(client.client_metrics)
            self.assertFalse(client.completions)
            self.assertEqual(len(transport.operations), 4)
            client.close()
        finally:
            release_done.set()
            thread.join(3)
            reader.close()
            writer.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_unknown_reply_poisoned_after_dispatch_and_cannot_be_retried(self):
        reader, writer = Pipe()
        client = DirectFullPoolClient(
            "a", 1, directory(), reader, Transport(), 1, client_slot=0, receive_slots=2
        )

        def worker():
            message = receive_message(writer, 2)
            send_message(
                writer,
                Message("output_ready", plan=replace(message.plan, generation=2)),
            )

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        hidden = torch.ones(3, 32, dtype=torch.bfloat16)
        weights = torch.ones(3, 2, dtype=torch.float32)
        ids = torch.zeros(3, 2, dtype=torch.int32)
        try:
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                client.execute(1, hidden, weights, ids)
            self.assertTrue(client.failed)
            with self.assertRaisesRegex(RuntimeError, "closed or failed"):
                client.execute(1, hidden, weights, ids)
        finally:
            thread.join(3)
            reader.close()
            writer.close()

    def test_configuration_requires_reserved_slots_and_trusted_whole_layer_routes(self):
        config = PoolDeployment(
            "/checkpoint",
            "checkpoint",
            64,
            tuple(
                ClientEndpoint(
                    client, 1, client, f"/tmp/{client}.sock", 24001 + i, "e0"
                )
                for i, client in enumerate(("a", "b"))
            ),
            workers=(WorkerPlacement("e0"),),
            receive_slots=2,
            execution=ExecutionOptions(
                validate_client_values=False,
                validate_worker_values=False,
                defer_output_sync=True,
            ),
            full_e_direct=True,
        )
        self.assertTrue(config.local_full_pipeline)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "deployment.json"
            path.write_text(json.dumps(asdict(config)))
            self.assertTrue(PoolDeployment.read(path).full_e_direct)
        for changes in (
            {"receive_slots": 1},
            {"receive_slots": 3},
            {"full_e_direct": 1},
            {"execution": replace(config.execution, validate_client_values=True)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)
