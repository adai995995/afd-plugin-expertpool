# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exact assignment coverage and stable reconstruction across two E copies."""

import unittest
from dataclasses import replace

import torch

from afd_plugin.expert_pool.assignment_slices import route_ids_by_worker
from afd_plugin.expert_pool.controller import directory_digest
from afd_plugin.expert_pool.fanout_protocol import FanoutReplies
from afd_plugin.expert_pool.protocol import (
    AssignmentSlice,
    CallKey,
    DispatchPlan,
    ExpertTask,
    Message,
    decode_message,
    encode_message,
)
from afd_plugin.expert_pool.replica_dispatch import (
    WorkerLoad,
    select_replicas,
    validate_dispatch,
)
from tests.unit.expert_pool.test_compact_contract import request
from tests.unit.expert_pool.test_expert_replicas import ledger, replicated


class AssignmentSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = request((5, 1, 0, 0))
        self.loads = {worker: WorkerLoad(2, 0, False, 0) for worker in ("e0", "e1")}

    def decision(self) -> DispatchPlan:
        decision = select_replicas(
            replicated(), self.parent, self.loads, 0, split_assignments=True
        )
        assert decision is not None
        return decision

    def test_dispatch_splits_one_expert_and_preserves_exact_coverage(self):
        decision = self.decision()
        self.assertEqual(decision.owners, ("e0", "e1"))
        self.assertEqual(
            decision.task_for("e0").assignment_slices,
            (AssignmentSlice(0, 0, 3), AssignmentSlice(1, 0, 1)),
        )
        self.assertEqual(
            decision.task_for("e1").assignment_slices,
            (AssignmentSlice(0, 3, 2),),
        )
        validate_dispatch(replicated(), decision)
        self.assertEqual(sum(task.num_assignments for task in decision.tasks), 6)
        decoded = decode_message(encode_message(Message("dispatch", dispatch=decision)))
        self.assertEqual(decoded.dispatch, decision)

    def test_overlapping_or_missing_slices_are_rejected(self):
        base = self.decision()
        for slices in (
            (AssignmentSlice(0, 2, 2),),
            (AssignmentSlice(0, 4, 1),),
        ):
            with self.subTest(slices=slices), self.assertRaises(ValueError):
                DispatchPlan(
                    self.parent,
                    (
                        base.task_for("e0"),
                        ExpertTask(
                            "e1", (0,), sum(item.count for item in slices), slices
                        ),
                    ),
                )
        with self.assertRaises(ValueError):
            DispatchPlan(
                self.parent,
                (base.task_for("e0"), ExpertTask("e1", (0,), 5)),
            )

    def test_worker_route_masks_cover_every_original_topk_slot_once(self):
        decision = self.decision()
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                routes = torch.tensor(
                    [[0, 1], [0, 0], [0, 0]], dtype=torch.int32, device=device
                )
                by_worker = route_ids_by_worker(routes, decision)
                self.assertEqual(
                    by_worker["e0"].tolist(), [[0, 1], [0, 0], [-1, -1]]
                )
                self.assertEqual(
                    by_worker["e1"].tolist(), [[-1, -1], [-1, -1], [0, 0]]
                )
                coverage = sum(
                    (value >= 0).to(torch.int32) for value in by_worker.values()
                )
                self.assertTrue(torch.equal(coverage, torch.ones_like(coverage)))
                for task in decision.tasks:
                    actual = int((by_worker[task.worker_id] >= 0).sum())
                    self.assertEqual(actual, task.num_assignments)

                # E returns only owned weighted rows; A restores original slots.
                hidden = torch.arange(
                    6, dtype=torch.float32, device=device
                ).reshape(3, 2)
                weights = torch.tensor(
                    [[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]], device=device
                )
                gains = torch.tensor([2.0, 3.0, 5.0, 7.0], device=device)
                baseline = (
                    hidden[:, None, :]
                    * weights[:, :, None]
                    * gains[routes.long()].unsqueeze(-1)
                )
                restored = torch.zeros_like(baseline).reshape(-1, 2)
                for owner_routes in by_worker.values():
                    selected = owner_routes >= 0
                    owner_slots = (
                        hidden[:, None, :]
                        * weights[:, :, None]
                        * gains[owner_routes.clamp_min(0).long()].unsqueeze(-1)
                        * selected[:, :, None]
                    )
                    positions = selected.flatten().nonzero().flatten()
                    packed = owner_slots.reshape(-1, 2).index_select(0, positions)
                    restored.index_copy_(0, positions, packed)
                self.assertTrue(torch.equal(restored.reshape_as(baseline), baseline))

    def test_controller_grants_match_slices_and_account_each_share(self):
        book = ledger()
        book.split_assignments = True
        for worker_id, resident in book.directories.items():
            book.ready(
                worker_id,
                directory_digest(resident),
                receive_slots=2,
                startup_complete=1,
            )
        book.submit("a0", self.parent, 0)
        first = book.grant(1)[0]
        second = book.grant(1)[0]
        self.assertNotEqual(first.worker_id, second.worker_id)
        decision = book.dispatch_notifications.popleft()
        replies = FanoutReplies(
            self.parent, decision.owners, dispatch_plan=decision, receive_slots=2
        )
        for plan in (first, second):
            replies.accept(Message("grant", plan=plan))
            self.assertEqual(
                decode_message(encode_message(Message("grant", plan=plan))).plan,
                plan,
            )
        self.assertEqual(book.admitted_assignments["e0"][1][0], 3)
        self.assertEqual(book.admitted_assignments["e1"][1][0], 2)
        bad = replace(
            first,
            assignment_slices=(AssignmentSlice(0, 0, 2), AssignmentSlice(1, 0, 1)),
            num_assignments=3,
        )
        with self.assertRaises(RuntimeError):
            FanoutReplies(
                self.parent, decision.owners, dispatch_plan=decision, receive_slots=2
            ).accept(Message("grant", plan=bad))

    def test_one_available_copy_keeps_whole_expert_on_that_copy(self):
        self.loads["e1"] = WorkerLoad(0, 0, False, 2)
        decision = select_replicas(
            replicated(), request((6, 0, 0, 0)), self.loads, 0,
            split_assignments=True,
        )
        assert decision is not None
        self.assertEqual(decision.owners, ("e0",))
        self.assertEqual(
            decision.task_for("e0").assignment_slices,
            (AssignmentSlice(0, 0, 6),),
        )

    def test_two_independent_a_calls_share_both_resident_copies(self):
        book = ledger()
        book.split_assignments = True
        for worker_id, resident in book.directories.items():
            book.ready(
                worker_id,
                directory_digest(resident),
                receive_slots=2,
                startup_complete=1,
            )
        book.submit("a0", self.parent, 0)
        book.submit("a1", replace(self.parent, key=CallKey("a1", 1, 0)), 0)
        granted = [book.grant(1)[0] for _ in range(4)]
        self.assertEqual(
            {
                client: {
                    plan.worker_id
                    for plan in granted
                    if plan.request.key.client_id == client
                }
                for client in ("a0", "a1")
            },
            {"a0": {"e0", "e1"}, "a1": {"e0", "e1"}},
        )
        self.assertEqual(book.admitted_assignments["e0"][1][0], 5)
        self.assertEqual(book.admitted_assignments["e1"][1][0], 5)


if __name__ == "__main__":
    unittest.main()
