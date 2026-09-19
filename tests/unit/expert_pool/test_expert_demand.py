# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Demand descriptor, wire identity and static expert-task coverage contracts."""

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from afd_plugin.expert_pool.demand import (
    DispatchPlan,
    ExpertTask,
    plan_expert_demand,
)
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import (
    MAX_DEMAND_EXPERTS,
    CallKey,
    CallRequest,
    ExecutionPlan,
    ExpertDemand,
    Message,
    decode_message,
    encode_message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory


def request(counts: tuple[int, ...] = (2, 1, 0, 1)) -> CallRequest:
    return CallRequest(
        CallKey("a0", 1, 7),
        "checkpoint",
        3,
        1,
        sum(counts) // 2,
        32,
        2,
        ExpertDemand(counts, ready_ns=123),
    )


def directory() -> PoolDirectory:
    return PoolDirectory(
        tuple(
            StaticDirectory(
                "checkpoint",
                3,
                worker_id,
                (ExpertPlacement(1, 4, ids), ExpertPlacement(2, 4, ids)),
                32,
                2,
                64,
                allow_partial_experts=True,
            )
            for worker_id, ids in (("e1", (3, 1)), ("e0", (2, 0)))
        ),
        expert_partitioned=True,
    )


class ExpertDemandProtocolTests(unittest.TestCase):
    def test_counts_are_bounded_exact_nonnegative_integers_and_immutable(self):
        for counts in (
            (),
            [1],
            (True,),
            (-1,),
            (1.0,),
            (0,) * (MAX_DEMAND_EXPERTS + 1),
        ):
            with self.subTest(counts=repr(counts)[:30]), self.assertRaises(ValueError):
                ExpertDemand(counts, 0)
        for ready_ns in (-1, True, 1.0, None):
            with self.subTest(ready_ns=ready_ns), self.assertRaises(ValueError):
                ExpertDemand((0,), ready_ns)
        for version in (0, 2, True, 1.0, "1"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                ExpertDemand((0,), 0, version)
        valid = ExpertDemand((0,) * MAX_DEMAND_EXPERTS, 0)
        self.assertEqual(len(valid.counts), MAX_DEMAND_EXPERTS)
        with self.assertRaises(FrozenInstanceError):
            valid.ready_ns = 1

    def test_request_counts_top_k_assignments_instead_of_distinct_rows(self):
        valid = request((2, 0, 2, 0))
        self.assertEqual(valid.num_tokens, 2)
        self.assertEqual(sum(valid.demand.counts), 4)
        for changes in (
            {"num_tokens": 4},
            {"top_k": 1},
            {"demand": ExpertDemand((1, 0, 1, 0), 0)},
            {"demand": {"counts": (2, 0, 2, 0), "ready_ns": 0}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(valid, **changes)

    def test_demand_submit_grant_and_empty_completion_wire_roundtrip(self):
        parent = request()
        plan = ExecutionPlan(parent, "e1", 4, 0, 2, (1, 3), 2)
        for message in (
            Message("submit", request=parent),
            Message("grant", plan=plan),
            Message("done", plan=plan),
            Message("empty_done", request=request((0, 0, 0, 0))),
        ):
            with self.subTest(kind=message.kind):
                self.assertEqual(decode_message(encode_message(message)), message)

    def test_old_json_missing_optional_fields_remains_compatible(self):
        parent = replace(request(), demand=None)
        plan = ExecutionPlan(parent, "e0", 2, 0, 1)
        submit = json.loads(encode_message(Message("submit", request=parent)))
        del submit["request"]["demand"]
        self.assertEqual(
            decode_message(json.dumps(submit).encode()),
            Message("submit", request=parent),
        )
        grant = json.loads(encode_message(Message("grant", plan=plan)))
        del grant["plan"]["request"]["demand"]
        del grant["plan"]["expert_ids"]
        del grant["plan"]["num_assignments"]
        self.assertEqual(
            decode_message(json.dumps(grant).encode()), Message("grant", plan=plan)
        )

    def test_plan_rejects_missing_foreign_duplicate_or_zero_demand_experts(self):
        plan = ExecutionPlan(request(), "e1", 4, 0, 2, (1, 3), 2)
        for changes in (
            {"expert_ids": ()},
            {"expert_ids": [1, 3]},
            {"expert_ids": (1, 1)},
            {"expert_ids": (-1, 3)},
            {"expert_ids": (1, 4)},
            {"expert_ids": (True, 3)},
            {"expert_ids": (1, 2)},
            {"num_assignments": 0},
            {"num_assignments": 3},
            {"num_assignments": True},
            {"num_assignments": None},
            {"request": replace(request(), demand=None)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(plan, **changes)
        legacy = ExecutionPlan(replace(request(), demand=None), "e0", 2, 0, 1)
        with self.assertRaises(ValueError):
            replace(legacy, expert_ids=(0,))
        with self.assertRaises(ValueError):
            ExecutionPlan(request(), "e0", 2, 0, 1)

    def test_empty_completion_requires_zero_token_demand_and_no_plan(self):
        empty = request((0, 0, 0, 0))
        for kwargs in (
            {},
            {"request": request()},
            {"request": replace(empty, demand=None)},
            {
                "request": empty,
                "plan": ExecutionPlan(request(), "e0", 1, 0, 1, (0,), 2),
            },
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Message("empty_done", **kwargs)

    def test_wire_rejects_extra_fields_at_every_nested_level(self):
        for target in ("message", "plan", "request", "demand", "key"):
            plan = ExecutionPlan(request(), "e0", 1, 0, 1, (0,), 2)
            raw = json.loads(encode_message(Message("grant", plan=plan)))
            targets = {
                "message": raw,
                "plan": raw["plan"],
                "request": raw["plan"]["request"],
                "demand": raw["plan"]["request"]["demand"],
                "key": raw["plan"]["request"]["key"],
            }
            targets[target]["unexpected"] = 1
            with self.subTest(target=target), self.assertRaises(ValueError):
                decode_message(json.dumps(raw).encode())

    def test_wire_rejects_nonarray_counts_and_plan_ids(self):
        for value in (None, "", "0", 2, {"0": 2}):
            for field in ("counts", "expert_ids"):
                plan = ExecutionPlan(request(), "e0", 1, 0, 1, (0,), 2)
                raw = json.loads(encode_message(Message("grant", plan=plan)))
                target = (
                    raw["plan"]["request"]["demand"]
                    if field == "counts"
                    else raw["plan"]
                )
                target[field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    decode_message(json.dumps(raw).encode())


class ExpertDemandPlanningTests(unittest.TestCase):
    def test_only_one_partition_is_selected_when_other_owners_have_no_demand(self):
        for counts, worker_id, ids in (
            ((2, 0, 2, 0), "e0", (0, 2)),
            ((0, 2, 0, 2), "e1", (1, 3)),
        ):
            with self.subTest(counts=counts):
                plan = plan_expert_demand(directory(), request(counts))
                self.assertEqual(plan.owners, (worker_id,))
                self.assertEqual(plan.tasks, (ExpertTask(worker_id, ids, 4),))
                self.assertEqual(plan.task_for(worker_id), plan.tasks[0])
                with self.assertRaises(ValueError):
                    plan.task_for("absent")

    def test_both_owners_receive_only_positive_demand_in_sorted_logical_order(self):
        plan = plan_expert_demand(directory(), request())
        self.assertEqual(plan.owners, ("e0", "e1"))
        self.assertEqual(
            plan.tasks,
            (ExpertTask("e0", (0,), 2), ExpertTask("e1", (1, 3), 2)),
        )
        self.assertEqual(plan.request, request())
        self.assertNotIn(2, plan.task_for("e0").expert_ids)

    def test_empty_demand_has_no_worker_tasks(self):
        plan = plan_expert_demand(directory(), request((0, 0, 0, 0)))
        self.assertEqual(plan.tasks, ())
        self.assertEqual(plan.owners, ())

    def test_planning_rejects_wrong_request_shape_checkpoint_and_descriptor_length(
        self,
    ):
        for changes in (
            {"model_id": "other"},
            {"placement_version": 4},
            {"layer_id": 3},
            {"hidden_size": 64},
            {"demand": None},
            {"demand": ExpertDemand((2, 2), 0)},
            {"demand": ExpertDemand((2, 1, 0, 1, 0), 0)},
            {"num_tokens": 65, "demand": ExpertDemand((65, 65, 0, 0), 0)},
            {"top_k": 1, "demand": ExpertDemand((1, 1, 0, 0), 0)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plan_expert_demand(directory(), replace(request(), **changes))
        full = replace(
            directory().workers[0],
            placements=(ExpertPlacement(1, 4, (0, 1, 2, 3)),),
            allow_partial_experts=False,
        )
        with self.assertRaises(ValueError):
            plan_expert_demand(PoolDirectory((full,)), request())

    def test_dispatch_validation_rejects_missing_duplicate_or_forged_tasks(self):
        valid = plan_expert_demand(directory(), request())
        for tasks in (
            (),
            valid.tasks[:1],
            (ExpertTask("e0", (0,), 2), ExpertTask("e0", (1, 3), 2)),
            (ExpertTask("e0", (0,), 2), ExpertTask("e1", (0, 1, 3), 4)),
            (ExpertTask("e0", (0, 2), 2), ExpertTask("e1", (1, 3), 2)),
            (ExpertTask("e0", (0, 4), 2), ExpertTask("e1", (1, 3), 2)),
            (ExpertTask("e0", (0,), 1), ExpertTask("e1", (1, 3), 2)),
        ):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                DispatchPlan(request(), tasks)
        with self.assertRaises(ValueError):
            DispatchPlan(request(), list(valid.tasks))
        with self.assertRaises(ValueError):
            DispatchPlan(replace(request(), demand=None), valid.tasks)

    def test_expert_task_rejects_mutable_or_invalid_identity_and_counts(self):
        valid = ExpertTask("e0", (0, 2), 4)
        for changes in (
            {"worker_id": ""},
            {"expert_ids": []},
            {"expert_ids": ()},
            {"expert_ids": (0, 0)},
            {"expert_ids": (True,)},
            {"expert_ids": (-1,)},
            {"num_assignments": 0},
            {"num_assignments": True},
            {"num_assignments": 1.0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(valid, **changes)


if __name__ == "__main__":
    unittest.main()
