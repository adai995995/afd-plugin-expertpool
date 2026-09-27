# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Replica acceptance requires residency, exact slices and live joint batching."""

import copy
import unittest
from collections import Counter

from afd_plugin.expert_pool.batch_audit import BatchAudit
from afd_plugin.expert_pool.protocol import (
    AssignmentSlice,
    BatchExecution,
    BatchSlot,
    CallKey,
    CallRequest,
    ExecutionPlan,
    ExpertDemand,
)
from tools.expert_pool.replica_validation import verify_replica_assignments
from tools.expert_pool.validate_shared_pool import verify_records


def evidence(*, split=True, paired=True):
    audit = BatchAudit()
    sequences = Counter()
    plan_id = 0
    for seq in range(1 if split else 2):
        plans_by_worker = {"e0": [], "e1": []}
        for index, demand in enumerate(((5, 1), (3, 1))):
            request = CallRequest(
                CallKey(f"a{index}", 1, seq),
                "checkpoint",
                1,
                1,
                sum(demand) // 2,
                8,
                2,
                ExpertDemand(demand, 0),
                True,
            )
            owner = f"e{(index + seq) % 2}"
            if split:
                ranges = {
                    "e0": ((0, 0, (demand[0] + 1) // 2), (1, 0, demand[1])),
                    "e1": ((0, (demand[0] + 1) // 2, demand[0] // 2),),
                }
            else:
                ranges = {owner: ((0, 0, demand[0]),)}
                ranges["e0"] = (*ranges.get("e0", ()), (1, 0, demand[1]))
            for worker, assignments in ranges.items():
                plan_id += 1
                plans_by_worker[worker].append(
                    ExecutionPlan(
                        request,
                        worker,
                        plan_id,
                        index,
                        seq + 1,
                        expert_ids=tuple(item[0] for item in assignments),
                        num_assignments=sum(item[2] for item in assignments),
                        assignment_slices=tuple(
                            AssignmentSlice(*item) for item in assignments
                        )
                        if split
                        else (),
                    )
                )
        for worker, plans in plans_by_worker.items():
            for group in [(plan,) for plan in plans] if not paired else [tuple(plans)]:
                sequences[worker] += 1
                audit.record_batch(
                    BatchExecution(
                        worker,
                        sequences[worker],
                        tuple(BatchSlot.from_plan(p) for p in group),
                    ),
                    group,
                )
                for plan in group:
                    audit.record_done(plan)
    requests = {
        "numerical_passed": True,
        "initial_status": {"a0": {}, "a1": {}},
        "cases": [
            {
                "name": "paired",
                "active_clients": ["a0", "a1"],
                "windows": {
                    f"a{i}": {
                        "start": 0,
                        "end": 1 if split else 2,
                        "expert_assignments": {
                            "1": {"0": n * (1 if split else 2), "1": 1 if split else 2}
                        },
                    }
                    for i, n in enumerate((5, 3))
                },
            }
        ],
    }
    workers = []
    for worker in ("e0", "e1"):
        records = [r for r in audit.records if r["worker_id"] == worker]
        calls = sum(len(r["members"]) for r in records)
        workers.append(
            {
                "worker_id": worker,
                "resident_experts": {"1": [0, 1] if worker == "e0" else [0]},
                "completed_calls": calls,
                "pipeline": {
                    "active_slots": 0,
                    "batching": {
                        "executions": len(records),
                        "calls": calls,
                        "token_rows": sum(r["token_rows"] for r in records),
                        "calls_per_execution": dict(
                            Counter(str(len(r["members"])) for r in records)
                        ),
                    },
                },
            }
        )
    controller = {
        "pending": 0,
        "outstanding": 0,
        "closed_clients": 2,
        "compute_batches_by_worker": dict(sequences),
    }
    return requests, audit.snapshot(), controller, workers


class ReplicaValidationTests(unittest.TestCase):
    def test_alternate_whole_copies_and_joint_split_batches_have_distinct_evidence(
        self,
    ):
        whole = verify_records(*evidence(split=False), require_replica_selection=True)
        self.assertEqual(whole["live_whole_experts_using_multiple_replicas"], 1)
        self.assertEqual(whole["live_split_expert_calls"], 0)
        split = verify_records(*evidence(), require_assignment_split=True)
        self.assertEqual(split["live_split_expert_calls"], 2)
        self.assertEqual(split["live_split_assignments"], 8)
        self.assertEqual(split["live_split_coalesced_expert_groups"], 2)

    def test_split_without_cross_a_batch_can_only_pass_the_split_stage(self):
        sample = evidence(paired=False)
        result = verify_records(
            *sample, require_cross_client_batch=False, require_assignment_split=True
        )
        self.assertEqual(result["live_split_coalesced_expert_groups"], 0)
        with self.assertRaises(AssertionError):
            verify_records(*sample, require_assignment_split=True)

    def test_equal_totals_with_overlapping_ranges_are_rejected(self):
        requests, audit, _, workers = evidence()
        audit["records"][1]["members"][0]["assignment_ranges"]["0"]["start"] = 2
        with self.assertRaisesRegex(AssertionError, "overlap or leave a gap"):
            verify_replica_assignments(requests, audit["records"], workers)

    def test_missing_child_and_inconsistent_parent_metadata_are_rejected(self):
        for mutation in ("missing", "demand", "layer", "range_count"):
            with self.subTest(mutation=mutation):
                requests, audit, _, workers = evidence()
                record = audit["records"][1]
                if mutation == "missing":
                    record["members"].pop(0)
                elif mutation == "demand":
                    record["members"][0]["expert_demand"]["0"] += 1
                elif mutation == "layer":
                    record["layer_id"] = 2
                else:
                    record["members"][0]["assignment_ranges"]["0"]["count"] += 1
                with self.assertRaises(AssertionError):
                    verify_replica_assignments(requests, audit["records"], workers)

    def test_nonresident_copy_and_duplicate_physical_worker_are_rejected(self):
        requests, audit, _, workers = evidence()
        bad = copy.deepcopy(workers)
        bad[1]["resident_experts"]["1"] = [1]
        with self.assertRaisesRegex(AssertionError, "nonresident"):
            verify_replica_assignments(requests, audit["records"], bad)
        with self.assertRaisesRegex(AssertionError, "Duplicate physical"):
            verify_replica_assignments(
                requests, audit["records"], workers + workers[:1]
            )

    def test_warmup_split_cannot_satisfy_live_split_requirement(self):
        requests, audit, controller, workers = evidence(split=False)
        _, warmup, _, _ = evidence()
        for record in audit["records"]:
            record["batch_sequence"] += 1
            for member in record["members"]:
                member["call_seq"] += 1
                member["plan_id"] += 100
        audit["records"] = warmup["records"] + audit["records"]
        audit["observed_batches"] = audit["recorded_batches"] = len(audit["records"])
        for window in requests["cases"][0]["windows"].values():
            window["start"], window["end"] = 1, 3
        for worker in workers:
            records = [
                r for r in audit["records"] if r["worker_id"] == worker["worker_id"]
            ]
            calls = sum(len(r["members"]) for r in records)
            worker["completed_calls"] = calls
            worker["pipeline"]["batching"] = {
                "executions": len(records),
                "calls": calls,
                "token_rows": sum(r["token_rows"] for r in records),
                "calls_per_execution": dict(
                    Counter(str(len(r["members"])) for r in records)
                ),
            }
            controller["compute_batches_by_worker"][worker["worker_id"]] = len(records)
        sample = requests, audit, controller, workers
        with self.assertRaisesRegex(AssertionError, "No live A call split"):
            verify_records(*sample, require_assignment_split=True)


if __name__ == "__main__":
    unittest.main()
