# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in deployment and expected-task checks across the demand control boundary."""

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from afd_plugin.expert_pool.demand import plan_expert_demand
from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ControllerConfig,
    PoolDeployment,
    WorkerPlacement,
)
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.fanout_protocol import FanoutReplies
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import (
    CallKey,
    CallRequest,
    ExecutionPlan,
    ExpertDemand,
    Message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory


def deployment():
    return PoolDeployment(
        "/checkpoint",
        "checkpoint",
        64,
        tuple(
            ClientEndpoint(
                "a0", 1, "domain-a", f"/private/a0-e{i}.sock", 24001 + i, f"e{i}"
            )
            for i in range(2)
        ),
        workers=tuple(
            WorkerPlacement(f"e{i}", expert_ids=(2 * i, 2 * i + 1)) for i in range(2)
        ),
        controller=ControllerConfig("/private/controller", "ready_first"),
        dispatch_mode="expert_partitioned",
    )


class DemandDeploymentTests(unittest.TestCase):
    def test_demand_is_disabled_in_partial_and_whole_layer_defaults(self):
        partial = deployment()
        self.assertIs(partial.demand_aware, False)
        whole = replace(
            partial,
            workers=tuple(
                replace(worker, expert_ids=None) for worker in partial.workers
            ),
            dispatch_mode="whole_layer",
            controller=None,
        )
        self.assertIs(whole.demand_aware, False)
        with self.assertRaises(ValueError):
            replace(whole, demand_aware=True)
        self.assertIs(replace(partial, demand_aware=True).demand_aware, True)

    def test_enabled_demand_preserves_partition_controller_requirements(self):
        enabled = replace(deployment(), demand_aware=True)
        for invalid in (
            {"controller": None},
            {"controller": ControllerConfig("/private/controller", "round_robin")},
            {"dispatch_mode": "whole_layer"},
            {"dispatch_mode": "unknown"},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(enabled, **invalid)

    def test_demand_flag_rejects_truthy_and_falsey_non_booleans(self):
        for value in (0, 1, "true", "false", None, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(deployment(), demand_aware=value)

    def test_json_roundtrip_preserves_opt_in_and_omitted_field_defaults_false(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "deployment.json"
            for enabled in (False, True):
                expected = replace(deployment(), demand_aware=enabled)
                path.write_text(json.dumps(asdict(expected)))
                self.assertEqual(PoolDeployment.read(path), expected)
            legacy = asdict(deployment())
            del legacy["demand_aware"]
            path.write_text(json.dumps(legacy))
            self.assertIs(PoolDeployment.read(path).demand_aware, False)
            legacy["demand_aware"] = "false"
            path.write_text(json.dumps(legacy))
            with self.assertRaises(ValueError):
                PoolDeployment.read(path)


class DemandRepliesContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = PoolDirectory(
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
        self.request = CallRequest(
            CallKey("a0", 1, 0),
            "checkpoint",
            1,
            1,
            8,
            32,
            2,
            ExpertDemand((4, 4, 4, 4), ready_ns=100),
        )
        self.dispatch = plan_expert_demand(self.directory, self.request)
        self.replies = FanoutReplies(
            self.request, self.dispatch.owners, dispatch_plan=self.dispatch
        )
        self.plans = {
            task.worker_id: ExecutionPlan(
                self.request,
                task.worker_id,
                index + 1,
                0,
                1,
                task.expert_ids,
                task.num_assignments,
            )
            for index, task in enumerate(self.dispatch.tasks)
        }

    def test_valid_numeric_grant_cannot_change_expected_owner_or_expert_subset(self):
        first = self.plans["e0"]
        for malformed in (
            replace(first, worker_id="e1"),
            replace(first, worker_id="foreign"),
            replace(first, expert_ids=(0,), num_assignments=4),
            replace(first, expert_ids=(2, 3)),
            replace(first, expert_ids=(1, 0)),
        ):
            # Each plan satisfies its own positive-count protocol checks. The
            # A must additionally enforce the independently derived dispatch.
            with self.subTest(plan=malformed), self.assertRaises(RuntimeError):
                self.replies.accept(Message("grant", plan=malformed))
            self.assertFalse(self.replies.plans)
            self.assertFalse(self.replies.messages)
            self.assertEqual(self.replies.next_phase, {"e0": 0, "e1": 0})

    def test_matching_tasks_accept_cross_worker_interleaving(self):
        for worker in ("e1", "e0"):
            for kind in ("grant", "output_ready", "done"):
                self.replies.accept(Message(kind, plan=self.plans[worker]))
        for kind in ("grant", "output_ready", "done"):
            for worker in ("e0", "e1"):
                self.assertEqual(
                    self.replies.take(kind, worker).plan, self.plans[worker]
                )
        self.assertFalse(self.replies.messages)

    def test_demand_replies_require_exact_parent_and_owner_set(self):
        with self.assertRaises(ValueError):
            FanoutReplies(self.request, self.dispatch.owners)
        for owners in (("e0",), ("e1", "e0"), ("e0", "foreign")):
            with self.subTest(owners=owners), self.assertRaises(ValueError):
                FanoutReplies(self.request, owners, dispatch_plan=self.dispatch)
        with self.assertRaises(ValueError):
            FanoutReplies(
                replace(self.request, key=CallKey("a0", 1, 1)),
                self.dispatch.owners,
                dispatch_plan=self.dispatch,
            )

    def test_empty_demand_completes_without_a_child_reply_demultiplexer(self):
        empty = replace(
            self.request, num_tokens=0, demand=ExpertDemand((0, 0, 0, 0), 100)
        )
        dispatch = plan_expert_demand(self.directory, empty)
        self.assertEqual(dispatch.tasks, ())
        self.assertEqual(dispatch.owners, ())
        with self.assertRaises(ValueError):
            FanoutReplies(empty, dispatch.owners, dispatch_plan=dispatch)
        completion = Message("empty_done", request=empty)
        self.assertEqual(completion.request, empty)
        self.assertIsNone(completion.plan)


if __name__ == "__main__":
    unittest.main()
