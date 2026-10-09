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


def directory(replicated=False, max_tokens=16):
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
                max_tokens,
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
        self._direct_client_roundtrip(1)

    def test_direct_client_roundtrip_splits_the_same_expert_across_copies(self):
        self._direct_client_roundtrip(2)

    def _direct_client_roundtrip(self, active_expert_replicas):
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
                active_expert_replicas=active_expert_replicas,
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
            self.assertEqual(status["active_expert_replicas"], active_expert_replicas)
            if active_expert_replicas > 1:
                self.assertGreater(status["split_expert_calls"], 0)
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

    def test_two_copies_split_stable_ranges_across_blocks_and_buffer_reuse(self):
        device = torch.device("cuda", 0)
        pool = directory(True, max_tokens=300)
        dispatcher = LocalTokenDispatcher(pool, device, active_expert_replicas=2)
        # Both copies are explicitly selected; the less busy one receives the
        # first, possibly larger range. Feedback is not a wait or reservation.
        loads = {"e0": WorkerLoad(1, 0, False, 0), "e1": WorkerLoad(1, 100, True, 1)}
        for rows in (1, 5, 127, 129, 257, 3):
            with self.subTest(rows=rows):
                original_ids = [[r % 4, (r * 3 + 1) % 4] for r in range(rows)]
                counts = [
                    sum(e == expert for row in original_ids for e in row)
                    for expert in range(4)
                ]
                seen = [0] * 4
                expected_owner = []
                for row in original_ids:
                    targets = []
                    for expert in row:
                        targets.append(
                            0 if seen[expert] < (counts[expert] + 1) // 2 else 1
                        )
                        seen[expert] += 1
                    expected_owner.append(targets)
                hidden = torch.arange(
                    rows * 32, device=device, dtype=torch.bfloat16
                ).reshape(rows, 32)
                weights = torch.tensor([[0.25, 0.75]] * rows, device=device)
                ids = torch.tensor(original_ids, dtype=torch.int32, device=device)
                request = CallRequest(
                    CallKey("a", 1, rows), "checkpoint", 1, 1, rows, 32, 2
                )
                plan, payloads, metrics = dispatcher.prepare(
                    request, (hidden, weights, ids), loads, 19
                )
                expected = torch.tensor(expected_owner, device=device)
                cover = torch.zeros_like(ids)
                restored = torch.zeros_like(hidden, dtype=torch.float32)
                for index, owner in enumerate(dispatcher.owners):
                    if owner not in payloads:
                        self.assertFalse(bool((expected == index).any()))
                        continue
                    x, wt, routes = payloads[owner]
                    row_ids = dispatcher.workspaces[owner].row_ids[: len(x)]
                    owned = routes >= 0
                    torch.testing.assert_close(
                        owned, expected[row_ids] == index, atol=0, rtol=0
                    )
                    torch.testing.assert_close(x, hidden[row_ids], atol=0, rtol=0)
                    torch.testing.assert_close(
                        routes[owned], ids[row_ids][owned], atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        wt[owned], weights[row_ids][owned], atol=0, rtol=0
                    )
                    self.assertTrue(bool((wt[~owned] == 0).all()))
                    self.assertTrue(bool(owned.any(1).all()))
                    cover.index_add_(0, row_ids, owned.int())
                    task = plan.task_for(owner)
                    self.assertEqual(task.num_assignments, int(owned.sum()))
                    for item in task.assignment_slices:
                        first = (counts[item.expert_id] + 1) // 2
                        self.assertEqual(item.start, 0 if index == 0 else first)
                        self.assertEqual(
                            item.count,
                            first if index == 0 else counts[item.expert_id] - first,
                        )
                    result = (
                        (
                            x[:, None, :].float()
                            * (routes.clamp_min(0) + 1)[:, :, None]
                            * wt[:, :, None]
                        )
                        .to(torch.bfloat16)
                        .sum(1, dtype=torch.float32)
                    )
                    restored.index_add_(0, row_ids, result)
                torch.testing.assert_close(
                    cover, torch.ones_like(cover), atol=0, rtol=0
                )
                reference = (
                    (
                        hidden[:, None, :].float()
                        * (ids + 1)[:, :, None]
                        * weights[:, :, None]
                    )
                    .to(torch.bfloat16)
                    .sum(1, dtype=torch.float32)
                )
                torch.testing.assert_close(restored, reference, atol=0, rtol=0)
                self.assertEqual(metrics["token_summary_bytes"], 7 * 8)
                self.assertEqual(
                    metrics["client_split_experts"], sum(c > 1 for c in counts)
                )

    def test_two_copy_mode_keeps_singleton_experts_and_singleton_demand_on_one_copy(
        self,
    ):
        from dataclasses import replace

        device = torch.device("cuda", 0)
        full = directory(True)
        pool = PoolDirectory(
            (
                full.workers[0],
                replace(full.workers[1], placements=(ExpertPlacement(1, 4, (0, 2)),)),
            ),
            True,
            True,
        )
        dispatcher = LocalTokenDispatcher(pool, device, active_expert_replicas=2)
        loads = {"e0": WorkerLoad(1, 0, False, 0), "e1": WorkerLoad(1, 100, True, 1)}
        hidden = torch.ones((5, 32), dtype=torch.bfloat16, device=device)
        weights = torch.full((5, 2), 0.5, device=device)
        ids = torch.tensor(
            [[0, 1], [0, 1], [0, 1], [2, 1], [3, 1]], dtype=torch.int32, device=device
        )
        request = CallRequest(CallKey("a", 1, 1), "checkpoint", 1, 1, 5, 32, 2)
        plan, payloads, metrics = dispatcher.prepare(
            request, (hidden, weights, ids), loads, 0
        )
        self.assertEqual(metrics["client_split_experts"], 1)
        self.assertEqual(plan.task_for("e0").expert_ids, (0, 1, 2, 3))
        self.assertEqual(plan.task_for("e1").expert_ids, (0,))
        self.assertEqual(plan.task_for("e1").num_assignments, 1)
        torch.testing.assert_close(
            dispatcher.workspaces["e1"].row_ids[:1],
            torch.tensor([2], device=device),
            atol=0,
            rtol=0,
        )
        self.assertEqual(payloads["e1"][0].shape[0], 1)
        for invalid in (-1, 4):
            bad = ids.clone()
            bad[0, 0] = invalid
            with self.assertRaisesRegex(ValueError, "invalid logical"):
                dispatcher.prepare(request, (hidden, weights, bad), loads, 0)
        empty = replace(request, num_tokens=0)
        plan, payloads, metrics = dispatcher.prepare(
            empty, (hidden[:0], weights[:0], ids[:0]), loads, 0
        )
        self.assertEqual(plan.tasks, ())
        self.assertEqual(payloads, {})
        self.assertEqual(metrics["token_summary_wait_ms"], 0)

    def test_invalid_active_copy_configuration_fails_before_dispatch(self):
        device = torch.device("cuda", 0)
        for value in (0, 3, True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LocalTokenDispatcher(
                    directory(True), device, active_expert_replicas=value
                )
        with self.assertRaises(ValueError):
            LocalTokenDispatcher(directory(False), device, active_expert_replicas=2)

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
