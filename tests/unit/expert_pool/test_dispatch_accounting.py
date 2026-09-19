# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU validation-report contracts; no GPU execution or performance simulation."""

import unittest
from copy import deepcopy

from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.scheduler import StaticDirectory
from tools.expert_pool.validate_engines import verify_dispatch

EXPERT_WEIGHT_BYTES = 192


def reports(partitioned: bool) -> tuple[PoolDirectory, list, list, list]:
    """Two A clients, one MoE layer and two E workers, with startup calls."""
    directory = PoolDirectory(
        tuple(
            StaticDirectory(
                "checkpoint",
                1,
                f"e{index}",
                (
                    ExpertPlacement(
                        1,
                        4,
                        tuple(range(index, 4, 2)) if partitioned else (0, 1, 2, 3),
                    ),
                ),
                32,
                2,
                64,
                allow_partial_experts=partitioned,
            )
            for index in range(2)
        ),
        expert_partitioned=partitioned,
    )
    child_counts = ((3, 3), (5, 5)) if partitioned else ((2, 1), (2, 3))
    ready, closed = [], []
    for client, parent_calls in enumerate((3, 5)):
        status = {
            "client_id": f"a{client}",
            "layers": {"1": {"calls": 1}},
            "dispatch": {
                "workers": {
                    f"e{worker}": {
                        "calls": int(partitioned or worker == 0),
                        "layer_calls": {"1": int(partitioned or worker == 0)},
                    }
                    for worker in range(2)
                }
            },
        }
        ready.append(status)
        final = deepcopy(status)
        final["layers"]["1"]["calls"] = parent_calls
        for worker in range(2):
            count = child_counts[client][worker]
            final["dispatch"]["workers"][f"e{worker}"] = {
                "calls": count,
                "layer_calls": {"1": count},
            }
        closed.append({"status": [final]})
    workers = []
    for index, worker in enumerate(directory.workers):
        count = sum(child[index] for child in child_counts)
        ids = list(worker.placements[0].expert_ids)
        workers.append(
            {
                "worker_id": worker.worker_id,
                "placement_version": 1,
                "resident_layers": [1],
                "resident_experts": {"1": ids},
                "resident_weight_bytes": len(ids) * EXPERT_WEIGHT_BYTES,
                "completed_calls": count,
                "client_calls": {
                    f"a{i}": row[index] for i, row in enumerate(child_counts)
                },
                "layer_calls": {"1": count},
            }
        )
    return directory, ready, closed, workers


class DispatchAccountingTests(unittest.TestCase):
    def test_whole_layer_counts_each_parent_once_and_excludes_startup_from_requests(
        self,
    ):
        summary = verify_dispatch(
            *reports(False), expert_weight_bytes=EXPERT_WEIGHT_BYTES
        )
        self.assertEqual(summary["dispatch_mode"], "whole_layer")
        self.assertEqual(summary["parent_layer_calls_by_client"], {"a0": 3, "a1": 5})
        self.assertEqual(summary["parent_layer_calls"], 8)
        self.assertEqual(summary["worker_child_calls"], 8)
        self.assertEqual(summary["request_parent_layer_calls"], 6)
        self.assertEqual(summary["request_worker_child_calls"], 6)
        self.assertEqual(summary["resident_routed_weight_bytes"], 1536)
        self.assertEqual(summary["single_model_routed_weight_bytes"], 768)

    def test_fanout_keeps_parent_and_child_counts_separate_with_one_weight_copy(self):
        summary = verify_dispatch(
            *reports(True), expert_weight_bytes=EXPERT_WEIGHT_BYTES
        )
        self.assertEqual(summary["dispatch_mode"], "expert_partitioned")
        self.assertEqual(summary["parent_layer_calls_by_client"], {"a0": 3, "a1": 5})
        self.assertEqual(summary["parent_layer_calls"], 8)
        self.assertEqual(summary["worker_child_calls"], 16)
        self.assertEqual(summary["worker_child_calls_by_worker"], {"e0": 8, "e1": 8})
        self.assertEqual(summary["request_parent_layer_calls"], 6)
        self.assertEqual(summary["request_worker_child_calls"], 12)
        self.assertEqual(summary["resident_routed_weight_bytes"], 768)
        self.assertEqual(summary["single_model_routed_weight_bytes"], 768)

    def test_resident_expert_ids_and_physical_slot_order_are_verified(self):
        for ids in ([2, 0], [0, 1], [0], [0, 2, 4]):
            directory, ready, closed, workers = reports(True)
            workers[0]["resident_experts"] = {"1": ids}
            with (
                self.subTest(ids=ids),
                self.assertRaisesRegex(AssertionError, "expert IDs"),
            ):
                verify_dispatch(
                    directory,
                    ready,
                    closed,
                    workers,
                    expert_weight_bytes=EXPERT_WEIGHT_BYTES,
                )

    def test_reported_weight_bytes_must_match_actual_subset_allocation(self):
        for weight_bytes in (0, 383, 385, 768):
            directory, ready, closed, workers = reports(True)
            workers[0]["resident_weight_bytes"] = weight_bytes
            with (
                self.subTest(weight_bytes=weight_bytes),
                self.assertRaisesRegex(AssertionError, "expert weights"),
            ):
                verify_dispatch(
                    directory,
                    ready,
                    closed,
                    workers,
                    expert_weight_bytes=EXPERT_WEIGHT_BYTES,
                )

    def test_internally_consistent_child_counts_still_must_match_parent_calls(self):
        for partitioned in (False, True):
            directory, ready, closed, workers = reports(partitioned)
            # All A-child and E counters agree, but one execution is duplicated.
            child = closed[0]["status"][0]["dispatch"]["workers"]["e0"]
            child["calls"] += 1
            child["layer_calls"]["1"] += 1
            workers[0]["client_calls"]["a0"] += 1
            workers[0]["completed_calls"] += 1
            workers[0]["layer_calls"]["1"] += 1
            with (
                self.subTest(partitioned=partitioned),
                self.assertRaisesRegex(AssertionError, "lost or duplicated"),
            ):
                verify_dispatch(
                    directory,
                    ready,
                    closed,
                    workers,
                    expert_weight_bytes=EXPERT_WEIGHT_BYTES,
                )

    def test_worker_client_layer_and_total_counters_must_agree(self):
        for field, value in (
            ("client_calls", {"a0": 4, "a1": 5}),
            ("layer_calls", {"1": 9}),
            ("completed_calls", 9),
        ):
            directory, ready, closed, workers = reports(True)
            workers[0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                verify_dispatch(
                    directory,
                    ready,
                    closed,
                    workers,
                    expert_weight_bytes=EXPERT_WEIGHT_BYTES,
                )


if __name__ == "__main__":
    unittest.main()
