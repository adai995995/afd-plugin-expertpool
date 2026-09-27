# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Reject warmup-only evidence, wrong assignments and mismatched native outputs."""

import copy
import unittest
from dataclasses import replace

from afd_plugin.expert_pool.batch_audit import BatchAudit
from afd_plugin.expert_pool.protocol import CallKey
from tests.unit.expert_pool.test_batching import group, ready_call
from tools.expert_pool.validate_shared_pool import (
    compare_output,
    endpoints,
    request_window,
    verify_records,
)


def evidence():
    audit = BatchAudit()
    for sequence in (0, 1):
        plans = tuple(
            replace(
                ready_call(index, rows=rows).plan,
                plan_id=sequence * 2 + index + 1,
                request=replace(
                    ready_call(index, rows=rows).plan.request,
                    key=CallKey(f"a{index}", 1, sequence),
                ),
            )
            for index, rows in enumerate((3, 5))
        )
        audit.record_batch(group(*plans, sequence=sequence + 1), plans)
        for plan in reversed(plans):
            audit.record_done(plan)
    requests = {
        "numerical_passed": True,
        "initial_status": {"a0": {}, "a1": {}},
        "cases": [
            {
                "name": "paired",
                "active_clients": ["a0", "a1"],
                "windows": {
                    f"a{index}": {
                        "start": 1,
                        "end": 2,
                        "expert_assignments": {"1": {"0": n, "1": n}},
                    }
                    for index, n in enumerate((3, 5))
                },
            }
        ],
    }
    controller = {
        "pending": 0,
        "outstanding": 0,
        "closed_clients": 2,
        "compute_batches_by_worker": {"e0": 2},
    }
    workers = [
        {
            "worker_id": "e0",
            "completed_calls": 4,
            "client_calls": {"a0": 2, "a1": 2},
            "pipeline": {
                "active_slots": 0,
                "batching": {
                    "executions": 2,
                    "calls": 4,
                    "token_rows": 16,
                    "calls_per_execution": {"2": 2},
                },
            },
        }
    ]
    return requests, audit.snapshot(), controller, workers


class SharedPoolValidationTests(unittest.TestCase):
    def test_warmup_is_excluded_from_live_batch_and_assignment_totals(self):
        result = verify_records(*evidence())
        self.assertEqual(result["live_cross_client_batches"], 1)
        self.assertEqual(result["live_coalesced_expert_groups"], 2)
        self.assertEqual(result["live_expert_assignments"], 16)

    def test_warmup_only_cross_batch_does_not_pass(self):
        requests, audit, controller, workers = evidence()
        audit["records"] = audit["records"][:1]
        audit["observed_batches"] = audit["recorded_batches"] = 1
        with self.assertRaisesRegex(AssertionError, "Router demand"):
            verify_records(requests, audit, controller, workers)

    def test_wrong_expert_counts_fail_even_when_batch_totals_agree(self):
        requests, audit, controller, workers = evidence()
        audit["records"][1]["members"][0]["expert_assignments"]["0"] += 1
        audit["records"][1]["expert_assignments"]["0"] += 1
        with self.assertRaisesRegex(AssertionError, "Router demand"):
            verify_records(requests, audit, controller, workers)

    def test_truncation_incomplete_members_and_worker_count_mismatch_fail(self):
        for change in ("truncate", "incomplete", "wrong_worker", "wrong_client"):
            with self.subTest(change=change):
                requests, audit, controller, workers = evidence()
                if change == "truncate":
                    audit["dropped_batches"] = 1
                elif change == "incomplete":
                    audit["records"][1]["members"][0]["completed"] = False
                elif change == "wrong_worker":
                    workers[0]["pipeline"]["batching"]["executions"] = 1
                else:
                    audit["records"][1]["members"][0]["client_id"] = "unknown"
                with self.assertRaises(AssertionError):
                    verify_records(requests, audit, controller, workers)

    def test_reference_checks_tokens_identity_and_finite_logprobs(self):
        output = {
            "client_id": "a0",
            "token_ids": [1, 2],
            "token_logprobs": [-1.0, -2.0],
        }
        self.assertTrue(compare_output(output, output, 0.05)["passed"])
        for field, value in (("token_ids", [2, 1]), ("token_logprobs", [-1.0, -3.0])):
            changed = copy.deepcopy(output)
            changed[field] = value
            self.assertFalse(compare_output(changed, output, 0.05)["passed"])
        for field, value in (
            ("client_id", "a1"),
            ("token_logprobs", [float("nan"), -2.0]),
        ):
            changed = copy.deepcopy(output)
            changed[field] = value
            with self.assertRaises(AssertionError):
                compare_output(changed, output, 0.05)

    def test_delta_counts_new_layers_and_rejects_reset_counters(self):
        before = {
            "dispatch": {
                "next_call_seq": 10,
                "demand": {"expert_assignments": {"1": [3, 2]}},
            }
        }
        after = {
            "dispatch": {
                "next_call_seq": 12,
                "demand": {"expert_assignments": {"1": [4, 2], "2": [1, 0]}},
            }
        }
        self.assertEqual(
            request_window(before, after)["expert_assignments"],
            {"1": {"0": 1}, "2": {"0": 1}},
        )
        after["dispatch"]["demand"]["expert_assignments"]["1"] = [2, 2]
        with self.assertRaises(AssertionError):
            request_window(before, after)

    def test_endpoints_reject_duplicate_identity_credentials_and_non_http(self):
        self.assertEqual(
            endpoints(["a0=http://localhost:8000/"]), {"a0": "http://localhost:8000"}
        )
        for values in (
            ["a0=file:///tmp/output"],
            ["a0=http://user:pass@localhost"],
            ["a0=http://localhost", "a0=http://localhost:2"],
        ):
            with self.assertRaises(ValueError):
                endpoints(values)
