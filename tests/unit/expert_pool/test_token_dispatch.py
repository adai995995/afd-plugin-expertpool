# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA exact packing, assignment coverage and original-A restoration."""

import threading
import unittest
from multiprocessing import Pipe

import torch

from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import CallKey, CallRequest
from afd_plugin.expert_pool.replica_dispatch import WorkerLoad
from afd_plugin.expert_pool.scheduler import StaticDirectory
from afd_plugin.expert_pool.token_dispatch import LocalTokenDispatcher


def directory(replicated=False):
    return PoolDirectory(
        tuple(
            StaticDirectory(
                "checkpoint",
                1,
                f"e{i}",
                (
                    ExpertPlacement(
                        1, 4, tuple(range(4)) if replicated else tuple(range(i, 4, 2))
                    ),
                ),
                32,
                2,
                16,
                allow_partial_experts=True,
            )
            for i in range(2)
        ),
        expert_partitioned=True,
        expert_replicated=replicated,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TokenDispatchTests(unittest.TestCase):
    def test_direct_client_packs_restores_and_reuses_maps_without_wire_grants(self):
        from afd_plugin.expert_pool.client import PoolClient
        from afd_plugin.expert_pool.controller import directory_digest
        from afd_plugin.expert_pool.direct_client import DirectFanoutPoolClient
        from afd_plugin.expert_pool.protocol import (
            Message,
            receive_message,
            send_message,
        )

        device = torch.device("cuda", 0)
        pool = directory(True)
        errors, controls, threads, channels, transports = [], [], [], [], []

        class Transport:
            peer = 0

            def __init__(self):
                self.device = device
                self.input_ready = threading.Event()
                self.payloads = []
                self.result = None

            def transfer(self, tensors, *, send):
                if send:
                    x, wt, ids = tensors
                    self.payloads.append(tuple(t.clone() for t in tensors))
                    self.result = (
                        (
                            x[:, None, :].float()
                            * (ids.clamp_min(0) + 1)[:, :, None]
                            * wt[:, :, None]
                        )
                        .to(torch.bfloat16)
                        .sum(1, dtype=torch.float32)
                    )
                    self.input_ready.set()
                else:
                    tensors[0].copy_(self.result)

            def close(self):
                pass

        def server(control, transport, worker):
            try:
                send_message(
                    control,
                    Message(
                        "ready",
                        detail=directory_digest(worker),
                        metrics={
                            "slot_id": 0,
                            "receive_slots": 2,
                            "packed_input": 1,
                            "partial_reduction": 1,
                        },
                    ),
                )
                while True:
                    message = receive_message(control, 5)
                    if message.kind == "close":
                        send_message(control, Message("closed"))
                        break
                    self.assertEqual(message.kind, "execute")
                    self.assertTrue(message.plan.request.partial_reduction)
                    self.assertTrue(transport.input_ready.wait(5))
                    transport.input_ready.clear()
                    self.assertEqual(message.plan.input_rows, transport.result.shape[0])
                    send_message(control, Message("output_ready", plan=message.plan))
                    send_message(
                        control,
                        Message(
                            "done",
                            plan=message.plan,
                            metrics={
                                "direct_pending_assignments": 0,
                                "direct_computing": 0,
                                "direct_occupied_slots": 0,
                            },
                        ),
                    )
            except BaseException as error:
                errors.append(error)

        client = None
        try:
            for worker in pool.workers:
                a, e = Pipe()
                controls.extend((a, e))
                transport = Transport()
                transports.append(transport)
                channels.append(
                    PoolClient(
                        "a",
                        1,
                        worker,
                        a,
                        transport,
                        5,
                        validate_values=False,
                        receive_slots=2,
                    )
                )
                thread = threading.Thread(
                    target=server, args=(e, transport, worker), daemon=True
                )
                thread.start()
                threads.append(thread)
            client = DirectFanoutPoolClient(
                tuple(channels),
                client_slot=0,
                receive_slots=2,
                expert_replicated=True,
                packed_input=True,
                partial_reduction=True,
            )
            for rows in (4, 3):
                x = torch.arange(
                    rows * 32, device=device, dtype=torch.bfloat16
                ).reshape(rows, 32)
                wt = torch.tensor([[0.25, 0.75]] * rows, device=device)
                ids = torch.tensor(
                    [[0, 2], [1, 3], [0, 1], [2, 0]][:rows],
                    device=device,
                    dtype=torch.int32,
                )
                output, completion = client.execute(1, x, wt, ids)
                expected = (
                    (x[:, None, :].float() * (ids + 1)[:, :, None] * wt[:, :, None])
                    .to(torch.bfloat16)
                    .sum(1, dtype=torch.float32)
                    .to(x.dtype)
                )
                torch.testing.assert_close(output, expected, atol=0, rtol=0)
                self.assertEqual(completion.kind, "result")
            status = client.dispatch_status()
            self.assertEqual(status["controller_roundtrips"], 0)
            self.assertEqual(status["pending_feedback"], 0)
            self.assertEqual(status["output_contract"], "fp32-partial-token-hidden")
            self.assertLess(
                status["input_transfer"]["sent_bytes"],
                status["input_transfer"]["dense_equivalent_bytes"],
            )
            client.close()
        finally:
            for thread in threads:
                thread.join(6)
            for control in controls:
                control.close()
        self.assertEqual(errors, [])
        self.assertTrue(all(not t.is_alive() for t in threads))

    def test_unique_rows_masked_slots_and_exact_original_row_restoration(self):
        device = torch.device("cuda", 0)
        hidden = torch.arange(4 * 32, device=device, dtype=torch.bfloat16).reshape(
            4, 32
        )
        weights = torch.tensor([[0.25, 0.75]] * 4, device=device)
        routes = torch.tensor(
            [[0, 2], [1, 3], [0, 1], [2, 0]], device=device, dtype=torch.int32
        )
        request = CallRequest(CallKey("a", 1, 0), "checkpoint", 1, 1, 4, 32, 2)
        for replicated in (False, True):
            dispatcher = LocalTokenDispatcher(directory(replicated), device)
            loads = {w: WorkerLoad(1, 0, False, 0) for w in dispatcher.owners}
            dispatch, payloads, metrics = dispatcher.prepare(
                request, (hidden, weights, routes), loads, 0
            )
            self.assertEqual([payloads[w][0].shape[0] for w in dispatch.owners], [3, 2])
            self.assertEqual(sum(t.num_assignments for t in dispatch.tasks), 8)
            self.assertEqual(metrics["token_summary_bytes"], 7 * 8)
            cover = torch.zeros_like(routes)
            restored = torch.zeros_like(hidden, dtype=torch.float32)
            for owner in reversed(dispatch.owners):
                x, wt, ids = payloads[owner]
                rows = dispatcher.workspaces[owner].row_ids[: x.shape[0]]
                self.assertTrue(torch.equal(x, hidden[rows]))
                mask = ids >= 0
                cover.index_add_(0, rows, mask.int())
                self.assertTrue(torch.equal(ids[mask], routes[rows][mask]))
                self.assertTrue(torch.equal(wt[mask], weights[rows][mask]))
                self.assertTrue(torch.all(wt[~mask] == 0))
                # Weighted BF16 slots simulate the unchanged kernel contract;
                # the E's partial sum stays FP32, not a rounded BF16 partial.
                slots = (
                    x[:, None, :].float()
                    * (ids.clamp_min(0) + 1)[:, :, None]
                    * wt[:, :, None]
                ).to(torch.bfloat16)
                restored.index_add_(0, rows, slots.sum(1, dtype=torch.float32))
            self.assertTrue(torch.all(cover == 1))
            expected = (
                (
                    hidden[:, None, :].float()
                    * (routes + 1)[:, :, None]
                    * weights[:, :, None]
                )
                .to(torch.bfloat16)
                .sum(1, dtype=torch.float32)
            )
            torch.testing.assert_close(restored, expected, atol=0, rtol=0)

    def test_feedback_only_selects_resident_copies_and_empty_targets_are_skipped(self):
        device = torch.device("cuda", 0)
        dispatcher = LocalTokenDispatcher(directory(True), device)
        loads = {"e0": WorkerLoad(1, 1000, True, 1), "e1": WorkerLoad(1, 0, False, 0)}
        hidden = torch.ones((3, 32), device=device, dtype=torch.bfloat16)
        weights = torch.full((3, 2), 0.5, device=device)
        routes = torch.tensor(
            [[0, 3], [1, 2], [0, 1]], device=device, dtype=torch.int32
        )
        request = CallRequest(CallKey("b", 1, 4), "checkpoint", 1, 1, 3, 32, 2)
        dispatch, payloads, _ = dispatcher.prepare(
            request, (hidden, weights, routes), loads, 1
        )
        self.assertEqual(dispatch.owners, ("e1",))
        self.assertEqual(tuple(payloads), ("e1",))
        self.assertEqual(payloads["e1"][0].shape[0], 3)
        for invalid in (-1, 4):
            bad = routes.clone()
            bad[0, 0] = invalid
            with self.assertRaisesRegex(ValueError, "invalid logical"):
                dispatcher.prepare(request, (hidden, weights, bad), loads, 0)
        empty_request = CallRequest(CallKey("b", 1, 5), "checkpoint", 1, 1, 0, 32, 2)
        empty, payloads, _ = dispatcher.prepare(
            empty_request, (hidden[:0], weights[:0], routes[:0]), loads, 0
        )
        self.assertEqual(empty.tasks, ())
        self.assertEqual(payloads, {})
