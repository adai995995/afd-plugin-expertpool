# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU validation-report contracts; no GPU execution or performance simulation."""

import unittest
from copy import deepcopy

from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.scheduler import StaticDirectory
from tools.expert_pool.validate_engines import verify_demand_controller, verify_dispatch

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


def demand_reports(
    *, all_empty: bool = False
) -> tuple[PoolDirectory, list, list, list]:
    """Each A selects one opposite owner; initial empty calls select neither."""
    pool, ready, closed, workers = reports(True)
    for index, (before, record) in enumerate(zip(ready, closed, strict=True)):
        after = record["status"][0]
        for status, initial in ((before, True), (after, False)):
            parent = status["layers"]["1"]["calls"]
            empty = parent if initial or all_empty else index + 1
            selected = parent - empty
            assignments = [0, 0, 0, 0]
            assignments[index] = selected * 2
            assignments[index + 2] = selected * 2
            status["layers"]["1"]["token_rows"] = selected * 2
            for worker in range(2):
                count = selected if index == worker else 0
                status["dispatch"]["workers"][f"e{worker}"] = {
                    "calls": count,
                    "layer_calls": {"1": count},
                }
            status["dispatch"].update(
                demand_aware=True,
                demand={
                    "parent_layer_calls": {"1": parent},
                    "empty_layer_calls": {"1": empty},
                    "expert_assignments": {"1": assignments},
                    "selected_worker_layer_calls": {
                        f"e{worker}": {"1": selected if worker == index else 0}
                        for worker in range(2)
                    },
                    "summary_bytes_by_expert_count": {"4": 40},
                },
            )
    for index, worker in enumerate(workers):
        selected = 0 if all_empty else index + 2
        worker.update(
            demand_aware=True,
            completed_calls=selected,
            client_calls={
                "a0": selected if index == 0 else 0,
                "a1": selected if index == 1 else 0,
            },
            layer_calls={"1": selected},
            expert_assignments={
                "1": {str(index): selected * 2, str(index + 2): selected * 2}
            },
        )
    return pool, ready, closed, workers


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

    def test_demand_counts_only_selected_children_and_tracks_empty_parents(self):
        summary = verify_dispatch(
            *demand_reports(),
            demand_aware=True,
            expert_weight_bytes=EXPERT_WEIGHT_BYTES,
        )
        self.assertTrue(summary["demand_aware"])
        self.assertEqual(summary["parent_layer_calls"], 8)
        self.assertEqual(summary["empty_parent_calls"], 3)
        self.assertEqual(summary["worker_child_calls"], 5)
        self.assertEqual(summary["worker_child_calls_by_worker"], {"e0": 2, "e1": 3})
        self.assertEqual(summary["skipped_worker_calls"], {"e0": 6, "e1": 5})
        self.assertEqual(summary["request_parent_layer_calls"], 6)
        self.assertEqual(summary["request_empty_parent_calls"], 1)
        self.assertEqual(summary["request_worker_child_calls"], 5)
        self.assertEqual(summary["expert_assignments"], 20)
        self.assertEqual(
            summary["expert_assignments_by_worker_layer_expert"],
            {"e0": {"1": {"0": 4, "2": 4}}, "e1": {"1": {"1": 6, "3": 6}}},
        )

    def test_all_empty_demand_has_no_worker_execution_or_assignments(self):
        summary = verify_dispatch(
            *demand_reports(all_empty=True),
            demand_aware=True,
            expert_weight_bytes=EXPERT_WEIGHT_BYTES,
        )
        self.assertEqual(summary["parent_layer_calls"], 8)
        self.assertEqual(summary["empty_parent_calls"], 8)
        self.assertEqual(summary["worker_child_calls"], 0)
        self.assertEqual(summary["expert_assignments"], 0)
        self.assertEqual(summary["request_empty_parent_calls"], 6)
        self.assertEqual(summary["skipped_worker_calls"], {"e0": 8, "e1": 8})

    def test_demand_parent_empty_assignment_and_selection_reports_are_checked(self):
        for field, value in (
            ("parent_layer_calls", {"1": 4}),
            ("empty_layer_calls", {"1": 4}),
            ("empty_layer_calls", {"1": -1}),
            ("expert_assignments", {"1": [4, 0, 3, 0]}),
            ("expert_assignments", {"1": [4, 0, 4]}),
            ("expert_assignments", {"1": [0, 4, 0, 4]}),
            ("selected_worker_layer_calls", {"e0": {"1": 3}, "e1": {"1": 0}}),
        ):
            pool, ready, closed, workers = demand_reports()
            closed[0]["status"][0]["dispatch"]["demand"][field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(AssertionError),
            ):
                verify_dispatch(pool, ready, closed, workers, demand_aware=True)

    def test_worker_assignments_must_match_all_clients_logical_counts(self):
        for counts in ({"0": 3, "2": 4}, {"0": 4}, {"0": 4, "2": 4, "1": 0}):
            pool, ready, closed, workers = demand_reports()
            workers[0]["expert_assignments"] = {"1": counts}
            with (
                self.subTest(counts=counts),
                self.assertRaisesRegex(AssertionError, "A and E"),
            ):
                verify_dispatch(pool, ready, closed, workers, demand_aware=True)
        with self.assertRaises(AssertionError):
            verify_dispatch(*reports(False), demand_aware=True)

    def test_controller_demand_counts_match_drained_execution(self):
        accounting = verify_dispatch(*demand_reports(), demand_aware=True)
        controlled = {
            "demand_aware": True,
            "empty_parent_calls": 3,
            "selected_worker_calls": {"e0": 2, "e1": 3},
            "skipped_worker_calls": {"e0": 6, "e1": 5},
            "admitted_assignments_by_worker_layer_expert": {
                "e0": {"1": {"0": 4, "2": 4}},
                "e1": {"1": {"1": 6, "3": 6}},
            },
        }
        verify_demand_controller(controlled, accounting)
        for key, value in (
            ("demand_aware", False),
            ("empty_parent_calls", 2),
            ("selected_worker_calls", {"e0": 3, "e1": 3}),
            ("skipped_worker_calls", {"e0": 5, "e1": 5}),
            ("admitted_assignments_by_worker_layer_expert", {}),
        ):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                verify_demand_controller({**controlled, key: value}, accounting)


if __name__ == "__main__":
    unittest.main()
