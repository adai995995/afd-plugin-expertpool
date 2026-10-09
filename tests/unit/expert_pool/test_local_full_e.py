# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local admission safety and reduced cross-client pipeline contracts."""

import time
import unittest
from dataclasses import replace
from multiprocessing import Pipe
from types import SimpleNamespace

import torch

from afd_plugin.expert_pool.batch_audit import BatchAudit
from afd_plugin.expert_pool.batching import BatchingOptions, ReadyCall, select_batch
from afd_plugin.expert_pool.client import PoolClient
from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ExecutionOptions,
    PoolDeployment,
    WorkerPlacement,
)
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import (
    BatchExecution,
    BatchSlot,
    CallKey,
    CallRequest,
    ExecutionPlan,
    Message,
    send_message,
)
from afd_plugin.expert_pool.scheduler import LocalSlotScheduler, StaticDirectory


def directory() -> StaticDirectory:
    return StaticDirectory(
        "checkpoint",
        1,
        "e0",
        (ExpertPlacement(1, 4, (0, 1, 2, 3)), ExpertPlacement(2, 4, (0, 1, 2, 3))),
        32,
        2,
        64,
    )


def request(client: str, sequence: int = 0, rows: int = 3) -> CallRequest:
    return CallRequest(CallKey(client, 1, sequence), "checkpoint", 1, 1, rows, 32, 2)


class LocalFullETests(unittest.TestCase):
    def book(self) -> LocalSlotScheduler:
        book = LocalSlotScheduler(directory(), 2)
        for client, domain in (("a", "x"), ("a2", "x"), ("b", "y")):
            book.register(client, 1, domain)
        return book

    def test_capacity_fairness_generation_and_drain_before_reuse(self):
        book = self.book()
        for client in ("a", "a2", "b"):
            book.submit(request(client), 0)
        first, second = book.grant(), book.grant()
        self.assertEqual(
            [first.request.key.client_id, second.request.key.client_id], ["a", "b"]
        )
        self.assertEqual([first.slot_id, second.slot_id], [0, 1])
        self.assertIsNone(book.grant())
        with self.assertRaises(ValueError):
            book.cancel_queued(first.request.key)
        with self.assertRaises(ValueError):
            book.complete(replace(first, generation=first.generation + 1))
        book.complete(second)
        third = book.grant()
        self.assertEqual(third.request.key.client_id, "a2")
        self.assertEqual(third.slot_id, 1)
        self.assertEqual(third.generation, 2)
        with self.assertRaises(ValueError):
            book.complete(second)
        self.assertEqual(book.active_slots[0], first)
        book.complete(first)
        book.complete(third)
        self.assertFalse(book.outstanding)

    def test_rejection_before_grant_and_long_single_client_reuse(self):
        book = self.book()
        for changed in (
            replace(request("a"), model_id="other"),
            replace(request("a"), placement_version=2),
            replace(request("a"), layer_id=3),
            request("unknown"),
        ):
            with self.assertRaises(ValueError):
                book.submit(changed, 0)
        self.assertFalse(book.outstanding)
        for sequence in range(100):
            current = request("a", sequence)
            book.submit(current, sequence)
            with self.assertRaises(ValueError):
                book.submit(current, sequence)
            plan = book.grant()
            self.assertEqual(plan.generation, sequence + 1)
            book.complete(plan)
        self.assertFalse(book.outstanding)

    def test_zero_wait_merges_only_ready_compatible_calls_and_audits_ranges(self):
        book = self.book()
        book.submit(request("a", rows=3), 0)
        book.submit(request("b", rows=5), 0)
        plans = book.grant(), book.grant()
        options = BatchingOptions(2, 128, 0)
        ready = tuple(ReadyCall(plan, 10) for plan in plans)
        decision = select_batch(ready[:1], options, 10)
        self.assertEqual(decision.slot_ids, (0,))
        self.assertEqual(decision.wait_ns, 0)
        self.assertEqual(select_batch(ready, options, 10).slot_ids, (0, 1))
        different = ReadyCall(
            replace(plans[1], request=replace(plans[1].request, layer_id=2)), 10
        )
        self.assertEqual(
            select_batch((ready[0], different), options, 10).slot_ids, (0,)
        )
        audit = BatchAudit()
        audit.record_batch(
            BatchExecution("e0", 1, tuple(BatchSlot.from_plan(p) for p in plans)), plans
        )
        self.assertEqual(
            [m["input_row_range"] for m in audit.records[0]["members"]],
            [[0, 3], [3, 8]],
        )
        self.assertIsNone(audit.records[0]["members"][0]["expert_demand"])
        for plan in reversed(plans):
            audit.record_done(plan)
        self.assertFalse(audit.pending)

    def test_client_accepts_dynamic_slots_but_rejects_stale_grants(self):
        book = self.book()
        book.submit(request("a"), 0)
        book.submit(request("b"), 0)
        book.grant()
        plan = book.grant()
        reader, writer = Pipe()
        try:
            client = PoolClient(
                "b", 1, directory(), reader, SimpleNamespace(peer=0), receive_slots=2
            )
            send_message(writer, Message("grant", plan=plan))
            self.assertEqual(client._reply("grant", plan.request).plan.slot_id, 1)
            send_message(writer, Message("grant", plan=plan))
            with self.assertRaisesRegex(RuntimeError, "Stale"):
                client._reply("grant", plan.request)
        finally:
            reader.close()
            writer.close()

    def test_configuration_is_explicit_and_existing_partition_contract_is_retained(
        self,
    ):
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
                validate_worker_values=False, defer_output_sync=True
            ),
            batching=BatchingOptions(2, 128, 0),
        )
        self.assertTrue(config.local_full_pipeline)
        self.assertTrue(replace(config, batching=BatchingOptions()).local_full_pipeline)
        for change in (
            {"batching": BatchingOptions(2, 128, 1)},
            {"execution": ExecutionOptions()},
            {"compact_output": True},
            {"workers": (WorkerPlacement("e0", expert_ids=(0, 1)),)},
        ):
            with self.assertRaises(ValueError):
                replace(config, **change)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA pipeline lifecycle check")
class FullEPipelineTests(unittest.TestCase):
    def test_merged_execution_does_not_reuse_a_returning_output_slot(self):
        self.check_pipeline(direct=False)

    def test_direct_merge_preserves_returning_slots_and_generations(self):
        self.check_pipeline(direct=True)

    def check_pipeline(self, *, direct):
        from afd_plugin.expert_pool.worker import ExpertWorker, WorkerPeer

        class Transfer:
            def __init__(self, transport, sending):
                self.transport, self.sending = transport, sending
                self.event = torch.cuda.Event()
                self.event.record()

            def query(self):
                return (
                    not self.sending or self.transport.allow_send
                ) and self.event.query()

            def finish(self):
                assert self.query()
                torch.cuda.current_stream().wait_event(self.event)
                return 0.0

        class Transport:
            device, peer = torch.device("cuda", torch.cuda.current_device()), 1

            def __init__(self, factor):
                self.factor, self.allow_send, self.outgoing = factor, False, None

            def post(self, tensors, *, send):
                if send:
                    self.outgoing = tensors[0]
                else:
                    hidden, weights, ids = tensors
                    hidden.fill_(self.factor)
                    weights.fill_(0.5)
                    ids.fill_(self.factor - 1)
                return Transfer(self, send)

        class Executor:
            validate_values = False
            checkpoint_path = "/checkpoint"
            hidden_size, top_k = 32, 2

            def __init__(self, placement):
                self.placement = placement
                self.w13 = torch.empty(1, device="cuda")
                self.rows = []

            def forward(self, hidden, weights, ids):
                self.rows.append(hidden.shape[0])
                return hidden * ((ids.float() + 1) * weights).sum(dim=1, keepdim=True)

        controls = [Pipe() for _ in range(2)]
        transports = [Transport(factor) for factor in (1, 2)]
        resident = directory()
        executors = {p.layer_id: Executor(p) for p in resident.placements}
        try:
            worker = ExpertWorker(
                resident,
                executors,
                tuple(
                    WorkerPeer(client, 1, client, control[0], transport)
                    for client, control, transport in zip(
                        ("a", "b"), controls, transports, strict=True
                    )
                ),
                receive_slots=2,
                full_e_direct=direct,
                batching=BatchingOptions(2, 128, 0),
                execution=ExecutionOptions(
                    validate_worker_values=False,
                    defer_output_sync=True,
                    audit_batch_members=True,
                ),
            )
            pipeline = worker.pipeline

            def admit(current):
                if direct:
                    slot = pipeline.book.slot_ids[current.key.client_id]
                    generation = pipeline.book.generations[current.key.client_id] + 1
                    candidate = ExecutionPlan(
                        current, "e0", current.key.call_seq * 2 + slot, slot, generation
                    )
                    pipeline.book.submit(current.key.client_id, candidate)
                    return pipeline.book.startable()[0]
                pipeline.book.submit(current, 0)
                return pipeline.book.grant()

            for client, rows in (("a", 3), ("b", 5)):
                plan = admit(request(client, rows=rows))
                pipeline._accept(
                    Message("grant", plan=plan, metrics={"controller_queue_ms": 0.0})
                )
            torch.cuda.synchronize()
            pipeline._receive_completions()
            self.assertTrue(pipeline._start_compute())
            torch.cuda.synchronize()
            pipeline._compute_completion()
            self.assertEqual(executors[1].rows, [8])
            self.assertFalse(pipeline._send_completions())
            if direct:
                self.assertEqual(pipeline.book.startable(), ())
                self.assertEqual(len(pipeline.book.active), 2)
            else:
                self.assertIsNone(pipeline.book.grant())
            torch.testing.assert_close(
                transports[1].outgoing,
                torch.full((5, 32), 4.0, device="cuda"),
                check_dtype=False,
            )
            transports[0].allow_send = True
            torch.cuda.synchronize()
            pipeline._send_completions()
            plan = admit(request("a", 1, rows=7))
            self.assertEqual((plan.slot_id, plan.generation), (0, 2))
            pipeline._accept(
                Message("grant", plan=plan, metrics={"controller_queue_ms": 0.0})
            )
            torch.cuda.synchronize()
            pipeline._receive_completions()
            self.assertTrue(pipeline._start_compute())
            torch.cuda.synchronize()
            pipeline._compute_completion()
            # Batch workspace can change while b's prior reduced output is
            # still being sent. Its independent slot output must stay intact.
            torch.testing.assert_close(
                transports[1].outgoing,
                torch.full((5, 32), 4.0, device="cuda"),
                check_dtype=False,
            )
            transports[1].allow_send = True
            deadline = time.monotonic() + 5
            while worker.completed_calls < 3 and time.monotonic() < deadline:
                pipeline._send_completions()
            self.assertEqual(worker.completed_calls, 3)
            if direct:
                self.assertFalse(pipeline.book.active)
                self.assertFalse(pipeline.book.pending)
            else:
                self.assertFalse(pipeline.book.outstanding)
            self.assertEqual(executors[1].rows, [8, 7])
            self.assertEqual(pipeline.audit.summary()["pending_recorded_members"], 0)
            self.assertEqual(worker.output_transfer["sent_bytes"], (3 + 5 + 7) * 32 * 2)
        finally:
            for pair in controls:
                for control in pair:
                    control.close()
