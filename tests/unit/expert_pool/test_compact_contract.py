# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU output-layout identity and exact compact-row admission contracts."""

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from afd_plugin.expert_pool.controller import (
    ControllerClientIdentity,
    ControllerLedger,
    directory_digest,
)
from afd_plugin.expert_pool.controller_service import serve_controller
from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ControllerConfig,
    PoolDeployment,
    WorkerPlacement,
)
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.fanout_controller import FanoutControllerLedger
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


def request(
    counts: tuple[int, ...] = (3, 1, 0, 2), *, compact_output: bool = True
) -> CallRequest:
    return CallRequest(
        CallKey("a0", 1, 0),
        "checkpoint",
        1,
        1,
        sum(counts) // 2,
        32,
        2,
        ExpertDemand(counts, 100),
        compact_output,
    )


def directory() -> PoolDirectory:
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


def ledger(*, compact_output: bool = True) -> FanoutControllerLedger:
    return FanoutControllerLedger(
        directory(),
        (ControllerClientIdentity("a0", 1, "domain-a"),),
        demand_aware=True,
        compact_output=compact_output,
    )


def deployment(model: str = "/checkpoint") -> PoolDeployment:
    return PoolDeployment(
        model,
        "checkpoint",
        64,
        tuple(
            ClientEndpoint(
                "a0", 1, "domain-a", f"/private/a0-e{i}.sock", 24001 + i, f"e{i}"
            )
            for i in range(2)
        ),
        workers=(
            WorkerPlacement("e0", expert_ids=(0, 1)),
            WorkerPlacement("e1", expert_ids=(2, 3)),
        ),
        controller=ControllerConfig("/private/controller", "ready_first"),
        dispatch_mode="expert_partitioned",
        demand_aware=True,
        compact_output=True,
    )


def make_ready(book: FanoutControllerLedger) -> None:
    for worker_id, resident in book.directories.items():
        book.ready(worker_id, directory_digest(resident))


def finish(book: FanoutControllerLedger, plan: ExecutionPlan) -> None:
    for kind in ("grant", "input_ready", "executing", "output_ready", "done"):
        book.progress(plan.worker_id, kind, plan)


class CompactProtocolTests(unittest.TestCase):
    def test_compact_request_and_reply_roundtrip_bind_layout_and_exact_row_count(self):
        parent = request()
        plan = ExecutionPlan(parent, "e0", 1, 0, 1, (0, 1), 4)
        self.assertNotEqual(plan.num_assignments, parent.num_tokens)
        for message in (
            Message("submit", request=parent),
            Message("grant", plan=plan),
            Message("output_ready", plan=plan),
            Message("done", plan=plan),
            Message("empty_done", request=request((0, 0, 0, 0))),
        ):
            with self.subTest(kind=message.kind):
                self.assertEqual(decode_message(encode_message(message)), message)
        for rows in (None, 0, 3, 5, True):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                replace(plan, num_assignments=rows)

    def test_old_request_json_defaults_to_dense_output_with_or_without_demand(self):
        for parent in (
            request(compact_output=False),
            replace(request(compact_output=False), demand=None),
        ):
            raw = json.loads(encode_message(Message("submit", request=parent)))
            del raw["request"]["compact_output"]
            self.assertEqual(decode_message(json.dumps(raw).encode()).request, parent)
            plan = (
                ExecutionPlan(parent, "e0", 1, 0, 1, (0, 1), 4)
                if parent.demand
                else ExecutionPlan(parent, "e0", 1, 0, 1)
            )
            raw = json.loads(encode_message(Message("grant", plan=plan)))
            del raw["plan"]["request"]["compact_output"]
            self.assertEqual(decode_message(json.dumps(raw).encode()).plan, plan)

    def test_compact_layout_requires_typed_boolean_and_demand(self):
        for value in (0, 1, None, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(request(), compact_output=value)
            raw = json.loads(encode_message(Message("submit", request=request())))
            raw["request"]["compact_output"] = value
            with self.subTest(wire=value), self.assertRaises(ValueError):
                decode_message(json.dumps(raw).encode())
        with self.assertRaises(ValueError):
            replace(request(), demand=None)
        raw = json.loads(encode_message(Message("submit", request=request())))
        raw["request"]["unchecked_output_rows"] = 99
        with self.assertRaises(ValueError):
            decode_message(json.dumps(raw).encode())


class CompactDeploymentTests(unittest.TestCase):
    def test_deployment_json_roundtrip_and_old_field_default(self):
        configured = deployment()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "deployment.json"
            path.write_text(json.dumps(asdict(configured)))
            self.assertEqual(PoolDeployment.read(path), configured)
            raw = asdict(configured)
            del raw["compact_output"]
            path.write_text(json.dumps(raw))
            self.assertEqual(
                PoolDeployment.read(path), replace(configured, compact_output=False)
            )

    def test_invalid_mode_combinations_rejected_before_runtime_startup(self):
        configured = deployment()
        for changes in (
            {"compact_output": 1},
            {"compact_output": 0},
            {"compact_output": "false"},
            {"compact_output": None},
            {"demand_aware": False},
            {"dispatch_mode": "whole_layer"},
            {
                "dispatch_mode": "whole_layer",
                "demand_aware": False,
                "workers": (WorkerPlacement("e0"), WorkerPlacement("e1")),
            },
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(configured, **changes)
        legacy = replace(
            configured,
            dispatch_mode="whole_layer",
            demand_aware=False,
            compact_output=False,
            controller=None,
            workers=(WorkerPlacement("e0"), WorkerPlacement("e1")),
        )
        self.assertFalse(legacy.compact_output)

    def test_controller_launcher_passes_configured_layout_to_ledger(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "first_k_dense_replace": 1,
                        "num_hidden_layers": 2,
                        "moe_layer_freq": 1,
                        "n_routed_experts": 4,
                        "hidden_size": 32,
                        "num_experts_per_tok": 2,
                    }
                )
            )
            configured = replace(
                deployment(str(root)),
                controller=ControllerConfig(str(root), "ready_first"),
            )
            with (
                patch("afd_plugin.expert_pool.controller_service.Listener"),
                patch(
                    "afd_plugin.expert_pool.controller_service.ControllerRuntime"
                ) as runtime,
            ):
                serve_controller(configured)
                book = runtime.call_args.args[0]
                self.assertTrue(book.compact_output)
                self.assertTrue(book.demand_aware)
                self.assertTrue(book.snapshot()["compact_output"])


class CompactAdmissionTests(unittest.TestCase):
    def test_mode_mismatch_rejects_before_consuming_sequence_or_credit(self):
        for enabled in (False, True):
            for counts in ((3, 1, 0, 2), (0, 0, 0, 0)):
                book = ledger(compact_output=enabled)
                before = book.snapshot()
                with (
                    self.subTest(enabled=enabled, counts=counts),
                    self.assertRaises(ValueError),
                ):
                    book.submit("a0", request(counts, compact_output=not enabled), 0)
                self.assertEqual(book.snapshot(), before)
                self.assertEqual(book.last_sequence["a0"], -1)
                self.assertFalse(book.pending_demand_plans)
                book.submit("a0", request(counts, compact_output=enabled), 1)
                self.assertEqual(book.last_sequence["a0"], 0)

    def test_controller_rejects_compact_broadcast_and_nonboolean_configuration(self):
        clients = (ControllerClientIdentity("a0", 1, "domain-a"),)
        for options in (
            {"compact_output": True},
            {"compact_output": 1, "demand_aware": True},
            {"compact_output": "true", "demand_aware": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                FanoutControllerLedger(directory(), clients, **options)
        resident = replace(
            directory().workers[0],
            allow_partial_experts=False,
            placements=(ExpertPlacement(1, 4, (0, 1, 2, 3)),),
        )
        whole_layer = ControllerLedger(PoolDirectory((resident,)), clients)
        with self.assertRaises(ValueError):
            whole_layer.submit("a0", request(), 0)
        self.assertFalse(whole_layer.outstanding)

    def test_selected_gang_preserves_exact_j_and_holds_until_all_children_finish(self):
        book = ledger()
        make_ready(book)
        parent = request()
        book.submit("a0", parent, 0)
        first = book.grant(1)[0]
        self.assertIsNotNone(book.workers["e1"].active)
        second = book.grant(1)[0]
        self.assertEqual((first.num_assignments, second.num_assignments), (4, 2))
        self.assertEqual(
            first.num_assignments + second.num_assignments,
            parent.num_tokens * parent.top_k,
        )
        self.assertTrue(all(plan.request.compact_output for plan in (first, second)))
        finish(book, first)
        self.assertEqual(book.workers["e0"].phase, "held")
        self.assertEqual(book.snapshot()["completed_parent_calls"], 0)
        finish(book, second)
        self.assertFalse(book.outstanding)
        self.assertEqual(book.snapshot()["completed_child_calls"], 2)
        self.assertEqual(book.snapshot()["completed_parent_calls"], 1)

    def test_changed_output_layout_is_not_a_valid_completion_identity(self):
        book = ledger()
        make_ready(book)
        book.submit("a0", request((3, 3, 0, 0)), 0)
        plan = book.grant(1)[0]
        changed = replace(plan, request=replace(plan.request, compact_output=False))
        with self.assertRaises(ValueError):
            book.progress("e0", "grant", changed)
        self.assertEqual(book.workers["e0"].active, plan)
        self.assertEqual(plan.num_assignments, 6)
        self.assertIsNone(book.workers["e1"].active)
        finish(book, plan)

    def test_empty_compact_parent_uses_no_buffer_or_child_plan(self):
        book = ledger()
        book.submit("a0", request((0, 0, 0, 0)), 0)
        self.assertIsNone(book.grant(1))
        self.assertEqual(book.snapshot()["empty_parent_calls"], 1)
        self.assertEqual(book.snapshot()["completed_parent_calls"], 1)
        self.assertEqual(book.snapshot()["completed_child_calls"], 0)
        self.assertFalse(book.outstanding)
        self.assertTrue(all(worker.generation == 0 for worker in book.workers.values()))


if __name__ == "__main__":
    unittest.main()
