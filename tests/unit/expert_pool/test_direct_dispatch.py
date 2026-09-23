# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Dedicated-buffer safety and result progress independent of CPU accounting."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from afd_plugin.expert_pool.compact_input import should_pack_task
from afd_plugin.expert_pool.controller import ControllerClientIdentity
from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ControllerConfig,
    ExecutionOptions,
    PoolDeployment,
    WorkerPlacement,
)
from afd_plugin.expert_pool.direct_dispatch import DirectReplies, DirectSlots
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import (
    CallKey,
    CallRequest,
    ExecutionPlan,
    ExpertDemand,
    Message,
    decode_message,
    encode_message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory


def plan(client="a0", generation=1, sequence=0):
    request = CallRequest(
        CallKey(client, 7, sequence),
        "checkpoint",
        1,
        1,
        2,
        32,
        2,
        demand=ExpertDemand((2, 2, 0, 0), 0),
        compact_output=True,
    )
    return ExecutionPlan(
        request,
        "e0",
        sequence * 2 + int(client[-1]),
        int(client[-1]),
        generation,
        (0, 1),
        4,
    )


class DirectSlotsTests(unittest.TestCase):
    def setUp(self):
        self.directory = StaticDirectory(
            "checkpoint",
            1,
            "e0",
            (ExpertPlacement(1, 4, (0, 1)),),
            32,
            2,
            8,
            allow_partial_experts=True,
        )
        self.book = DirectSlots(
            self.directory,
            tuple(
                ControllerClientIdentity(f"a{i}", 7, f"domain-{i}") for i in range(2)
            ),
        )

    def test_clients_own_distinct_slots_and_idle_peer_does_not_block(self):
        a = plan("a1")
        self.book.submit("a1", a)
        self.assertEqual(self.book.startable(), (a,))
        self.book.complete(a)
        self.book.close("a1")
        self.assertEqual(self.book.completed, {"a0": 0, "a1": 1})

    def test_successor_cannot_overwrite_live_buffer(self):
        a, b = plan(), plan(generation=2, sequence=1)
        self.book.submit("a0", a)
        self.book.startable()
        self.book.submit("a0", b)
        self.assertEqual(self.book.startable(), ())
        self.assertEqual(self.book.active["a0"], a)
        with self.assertRaises(ValueError):
            self.book.submit("a0", plan(generation=3, sequence=2))
        self.book.complete(a)
        self.assertEqual(self.book.startable(), (b,))
        with self.assertRaises(ValueError):
            self.book.complete(a)
        self.assertEqual(self.book.active["a0"], b)

    def test_waiting_successor_does_not_block_other_client(self):
        a, b, other = plan(), plan(generation=2, sequence=1), plan("a1")
        self.book.submit("a0", a)
        self.book.startable()
        self.book.submit("a0", b)
        self.book.submit("a1", other)
        self.assertEqual(self.book.startable(), (other,))
        self.book.complete(other)
        self.book.close("a1")
        self.assertEqual(self.book.active["a0"], a)

    def test_wrong_endpoint_slot_worker_version_and_session_are_rejected(self):
        p = plan()
        bad = [
            replace(p, slot_id=1),
            replace(p, worker_id="e1"),
            replace(p, request=replace(p.request, placement_version=2)),
            replace(p, request=replace(p.request, key=CallKey("a0", 8, 0))),
        ]
        for candidate in bad:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                self.book.submit("a0", candidate)
        with self.assertRaises(ValueError):
            self.book.submit("a1", p)
        self.assertFalse(self.book.pending)
        self.assertEqual(self.book.generations["a0"], 0)

    def test_skipped_layer_calls_are_allowed_but_slot_generation_is_contiguous(self):
        p = plan(sequence=9)
        self.book.submit("a0", p)
        self.book.startable()
        self.book.complete(p)
        for candidate in [
            plan(generation=3, sequence=10),
            plan(generation=2, sequence=9),
        ]:
            with self.assertRaises(ValueError):
                self.book.submit("a0", candidate)
        next_plan = plan(generation=2, sequence=24)
        self.book.submit("a0", next_plan)
        self.assertEqual(self.book.startable(), (next_plan,))

    def test_close_requires_actual_completion_and_prevents_future_work(self):
        p = plan()
        self.book.submit("a0", p)
        with self.assertRaises(ValueError):
            self.book.close("a0")
        self.book.startable()
        with self.assertRaises(ValueError):
            self.book.close("a0")
        self.book.complete(p)
        self.book.close("a0")
        with self.assertRaises(ValueError):
            self.book.submit("a0", plan(generation=2, sequence=1))


class DirectReplyTests(unittest.TestCase):
    def setUp(self):
        self.book = DirectReplies()
        self.first = plan()
        self.book.register(self.first)

    def send(self, kind, p=None, owner="e0"):
        self.book.accept(
            owner, decode_message(encode_message(Message(kind, plan=p or self.first)))
        )

    def test_result_allows_next_call_before_previous_done(self):
        self.send("output_ready")
        self.book.received(self.first)
        successor = plan(generation=2, sequence=1)
        self.book.register(successor)
        self.assertEqual(len(self.book.pending), 2)
        self.send("done")
        self.send("output_ready", successor)
        self.book.received(successor)
        self.send("done", successor)
        self.assertFalse(self.book.pending)

    def test_done_is_not_authority_to_skip_gpu_result_receipt(self):
        self.send("output_ready")
        self.send("done")
        with self.assertRaises(ValueError):
            self.book.register(plan(generation=2, sequence=1))
        self.book.received(self.first)
        self.book.register(plan(generation=2, sequence=1))

    def test_duplicate_out_of_order_and_wrong_owner_fail(self):
        with self.assertRaises(RuntimeError):
            self.send("done")
        with self.assertRaises(RuntimeError):
            self.send("output_ready", owner="e1")
        with self.assertRaises(RuntimeError):
            self.send("output_ready", replace(self.first, slot_id=1))
        self.send("output_ready")
        with self.assertRaises(RuntimeError):
            self.send("output_ready")
        self.book.received(self.first)
        self.send("done")
        with self.assertRaises(RuntimeError):
            self.send("done")

    def test_feedback_backlog_is_bounded(self):
        self.send("output_ready")
        self.book.received(self.first)
        second = plan(generation=2, sequence=1)
        self.book.register(second)
        self.send("output_ready", second)
        self.book.received(second)
        with self.assertRaises(ValueError):
            self.book.register(plan(generation=3, sequence=2))

    def test_direct_wire_messages_require_plan_and_roundtrip(self):
        for kind in ["execute", "result"]:
            with self.assertRaises(ValueError):
                Message(kind)
            m = Message(kind, plan=self.first)
            self.assertEqual(decode_message(encode_message(m)), m)

    def test_packed_row_count_is_bound_to_plan_and_original_shape(self):
        original = plan()
        for rows in (0, 1, 3, True):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                replace(original, input_rows=rows)
        packed_request = replace(
            original.request, demand=ExpertDemand((1, 1, 1, 1), 0)
        )
        packed = replace(
            original,
            request=packed_request,
            num_assignments=2,
            input_rows=1,
        )
        self.assertEqual(
            decode_message(encode_message(Message("execute", plan=packed))).plan,
            packed,
        )
        # A may send the original dense rows when packing is enabled but its
        # guaranteed-savings gate declines a sparse task.
        dense_bypass = replace(packed, input_rows=packed_request.num_tokens)
        self.assertEqual(
            decode_message(encode_message(Message("execute", plan=dense_bypass))).plan,
            dense_bypass,
        )
        old_wire = json.loads(encode_message(Message("execute", plan=original)))
        del old_wire["plan"]["input_rows"]
        self.assertEqual(
            decode_message(json.dumps(old_wire).encode()).plan, original
        )


class DirectDeploymentTests(unittest.TestCase):
    def test_pack_admission_requires_guaranteed_row_savings(self):
        self.assertTrue(should_pack_task(4, 8))
        self.assertFalse(should_pack_task(5, 8))
        self.assertFalse(should_pack_task(8, 8))

    def deployment(self):
        return PoolDeployment(
            "/models/checkpoint",
            "checkpoint",
            8,
            tuple(
                ClientEndpoint(
                    f"a{i}", 7, f"domain-{i}", f"/tmp/direct-a{i}.sock", 22000 + i, "e0"
                )
                for i in range(2)
            ),
            execution=ExecutionOptions(
                validate_worker_values=False, defer_output_sync=True
            ),
            workers=(WorkerPlacement("e0", expert_ids=(0, 1, 2, 3)),),
            dispatch_mode="expert_partitioned",
            demand_aware=True,
            compact_output=True,
            receive_slots=2,
            direct_dispatch=True,
        )

    def test_explicit_mode_roundtrips_without_controller(self):
        d = self.deployment()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "deployment.json"
            p.write_text(json.dumps(asdict(d)))
            self.assertEqual(PoolDeployment.read(p), d)
            packed = replace(d, packed_input=True)
            p.write_text(json.dumps(asdict(packed)))
            self.assertEqual(PoolDeployment.read(p), packed)
            old = asdict(d)
            del old["packed_input"]
            p.write_text(json.dumps(old))
            self.assertEqual(PoolDeployment.read(p), d)

    def test_controller_or_shared_slots_cannot_silently_enter_direct_mode(self):
        d = self.deployment()
        for fields in [
            dict(controller=ControllerConfig("/tmp/control", "ready_first")),
            dict(receive_slots=3),
            dict(receive_slots=1),
            dict(demand_aware=False, compact_output=False),
            dict(direct_dispatch=False, packed_input=True),
        ]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(d, **fields)

    def test_pooled_admission_requires_controlled_expert_partitions(self):
        controlled = replace(
            self.deployment(),
            direct_dispatch=False,
            controller=ControllerConfig("/tmp/control", "ready_first"),
            pooled_admission=True,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "deployment.json"
            path.write_text(json.dumps(asdict(controlled)))
            self.assertEqual(PoolDeployment.read(path), controlled)
            legacy = asdict(controlled)
            del legacy["pooled_admission"]
            path.write_text(json.dumps(legacy))
            self.assertFalse(PoolDeployment.read(path).pooled_admission)
        for fields in (
            {"direct_dispatch": True, "controller": None},
            {"controller": ControllerConfig("/tmp/control", "round_robin")},
            {"receive_slots": 1},
            {"demand_aware": False, "compact_output": False},
            {"dispatch_mode": "whole_layer"},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(controlled, **fields)


if __name__ == "__main__":
    unittest.main()
