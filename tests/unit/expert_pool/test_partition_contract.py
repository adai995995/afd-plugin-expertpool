# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static expert partition ownership and opt-in deployment safety contracts."""

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ControllerConfig,
    PoolDeployment,
    WorkerPlacement,
)
from afd_plugin.expert_pool.directory import PoolDirectory, ReplicaSelector
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.scheduler import StaticDirectory, StaticScheduler


def worker(
    worker_id: str,
    expert_ids: tuple[int, ...],
    layer_ids: tuple[int, ...] = (1, 2),
    *,
    partial: bool = True,
) -> StaticDirectory:
    return StaticDirectory(
        "checkpoint",
        1,
        worker_id,
        tuple(ExpertPlacement(layer, 4, expert_ids) for layer in layer_ids),
        32,
        2,
        64,
        allow_partial_experts=partial,
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
            WorkerPlacement("e1", expert_ids=(3, 1)),
            WorkerPlacement("e0", expert_ids=(2, 0)),
        ),
        controller=ControllerConfig("/private/controller", "ready_first"),
        dispatch_mode="expert_partitioned",
    )


def write_model_config(path: Path) -> None:
    (path / "config.json").write_text(
        json.dumps(
            {
                "first_k_dense_replace": 1,
                "num_hidden_layers": 3,
                "moe_layer_freq": 1,
                "n_routed_experts": 4,
                "hidden_size": 32,
                "num_experts_per_tok": 2,
            }
        )
    )


class PartitionDirectoryTests(unittest.TestCase):
    def test_partition_coverage_and_physical_slots_have_single_owners(self):
        directory = PoolDirectory(
            (worker("e1", (3, 1)), worker("e0", (2, 0))),
            expert_partitioned=True,
        )
        self.assertEqual(directory.owners(1), ("e0", "e1"))
        self.assertEqual(directory.owners(2), ("e0", "e1"))
        self.assertEqual(directory.locations(1, 0), (("e0", 1),))
        self.assertEqual(directory.locations(1, 1), (("e1", 1),))
        self.assertEqual(directory.locations(1, 2), (("e0", 0),))
        self.assertEqual(directory.locations(1, 3), (("e1", 0),))
        self.assertEqual(
            directory.placements,
            (ExpertPlacement(1, 4, (0, 1, 2, 3)), ExpertPlacement(2, 4, (0, 1, 2, 3))),
        )

    def test_owners_only_include_resident_layer_partitions(self):
        directory = PoolDirectory(
            (
                worker("e2", (0, 1, 2, 3), (2,)),
                worker("e1", (1, 3), (1,)),
                worker("e0", (0, 2), (1,)),
            ),
            expert_partitioned=True,
        )
        self.assertEqual(directory.owners(1), ("e0", "e1"))
        self.assertEqual(directory.owners(2), ("e2",))
        for layer in (-1, 0, 3, True, "1"):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                directory.owners(layer)
        for layer, expert in ((1, -1), (1, 4), (1, True), (1, "0"), (3, 0)):
            with (
                self.subTest(layer=layer, expert=expert),
                self.assertRaises(ValueError),
            ):
                directory.locations(layer, expert)

    def test_missing_and_overlapping_experts_rejected_per_layer(self):
        for first, second in (
            ((0, 1), (2,)),
            ((0, 1), (1, 2, 3)),
            ((0, 1, 2, 3), (0,)),
        ):
            with (
                self.subTest(first=first, second=second),
                self.assertRaises(ValueError),
            ):
                PoolDirectory(
                    (worker("e0", first), worker("e1", second)),
                    expert_partitioned=True,
                )
        with self.assertRaises(ValueError):
            PoolDirectory(
                (worker("e0", (0, 1), (1, 2)), worker("e1", (2, 3), (1,))),
                expert_partitioned=True,
            )

    def test_foreign_and_repeated_ids_rejected_before_directory_use(self):
        for ids in ((0, 4), (-1, 0), (0, True), (0, 0), ()):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                worker("e0", ids)

    def test_partition_mode_must_match_every_worker(self):
        partial = worker("e0", (0, 1, 2, 3))
        complete = worker("e1", (0, 1, 2, 3), partial=False)
        for workers, partitioned in (
            ((partial,), False),
            ((complete,), True),
            ((partial, complete), True),
            ((partial, complete), False),
        ):
            with self.subTest(partitioned=partitioned), self.assertRaises(ValueError):
                PoolDirectory(workers, expert_partitioned=partitioned)
        for invalid in (0, 1, "true", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(partial, allow_partial_experts=invalid)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PoolDirectory((partial,), expert_partitioned=invalid)

    def test_legacy_schedulers_cannot_silently_execute_partial_layers(self):
        first = worker("e0", (0, 1))
        second = worker("e1", (2, 3))
        with self.assertRaises(ValueError):
            replace(first, allow_partial_experts=False)
        with self.assertRaises(ValueError):
            StaticScheduler(first)
        with self.assertRaises(ValueError):
            ReplicaSelector(PoolDirectory((first, second), expert_partitioned=True))

    def test_whole_layer_replica_mode_remains_compatible(self):
        first = worker("e0", (0, 1, 2, 3), partial=False)
        second = worker("e1", (3, 2, 1, 0), partial=False)
        directory = PoolDirectory((first, second))
        self.assertFalse(directory.expert_partitioned)
        self.assertEqual(directory.locations(1, 1), (("e0", 1), ("e1", 2)))
        self.assertEqual(directory.owners(1), ("e0", "e1"))
        selector = ReplicaSelector(directory)
        self.assertEqual((selector.select(1), selector.select(1)), ("e0", "e1"))
        self.assertIsNone(StaticScheduler(first).grant())


class PartitionDeploymentTests(unittest.TestCase):
    def test_json_roundtrip_builds_all_layer_partitions_and_keeps_slot_order(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_model_config(root)
            expected = deployment(str(root))
            path = root / "deployment.json"
            path.write_text(json.dumps(asdict(expected)))
            restored = PoolDeployment.read(path)
            self.assertEqual(restored, expected)
            pool = restored.pool_directory()
            self.assertTrue(pool.expert_partitioned)
            self.assertEqual(pool.owners(1), ("e0", "e1"))
            self.assertEqual(pool.locations(2, 3), (("e1", 0),))
            self.assertTrue(
                all(worker.allow_partial_experts for worker in pool.workers)
            )
            self.assertEqual(
                tuple(p.expert_ids for p in restored.directory("e0").placements),
                ((2, 0), (2, 0)),
            )

    def test_old_json_defaults_to_whole_layer_without_a_controller(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_model_config(root)
            raw = asdict(deployment(str(root)))
            del raw["dispatch_mode"]
            del raw["controller"]
            for entry in raw["workers"]:
                del entry["expert_ids"]
            path = root / "deployment.json"
            path.write_text(json.dumps(raw))
            restored = PoolDeployment.read(path)
            self.assertEqual(restored.dispatch_mode, "whole_layer")
            self.assertIsNone(restored.controller)
            pool = restored.pool_directory()
            self.assertFalse(pool.expert_partitioned)
            self.assertEqual(pool.locations(1, 2), (("e0", 2), ("e1", 2)))

    def test_partitioned_dispatch_requires_ready_first_controller(self):
        configured = deployment()
        for controller in (None, ControllerConfig("/private/controller")):
            with self.subTest(controller=controller), self.assertRaises(ValueError):
                replace(configured, controller=controller)
        for mode in ("unknown", "", None, True, []):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                replace(configured, dispatch_mode=mode)

    def test_expert_subsets_must_be_explicit_only_for_partitioned_mode(self):
        configured = deployment()
        with self.assertRaises(ValueError):
            replace(configured, dispatch_mode="whole_layer")
        for missing in (0, 1):
            workers = list(configured.workers)
            workers[missing] = replace(workers[missing], expert_ids=None)
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                replace(configured, workers=tuple(workers))
        for ids in ((), (0, 0), (-1,), (True,), [0, 1], (0, "1")):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                WorkerPlacement("e0", expert_ids=ids)

    def test_model_range_and_per_layer_coverage_checked_before_startup(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_model_config(root)
            configured = deployment(str(root))
            for workers in (
                (
                    WorkerPlacement("e0", expert_ids=(0, 1)),
                    WorkerPlacement("e1", expert_ids=(2, 4)),
                ),
                (
                    WorkerPlacement("e0", expert_ids=(0, 1)),
                    WorkerPlacement("e1", expert_ids=(2,)),
                ),
                (
                    WorkerPlacement("e0", expert_ids=(0, 1)),
                    WorkerPlacement("e1", expert_ids=(1, 2, 3)),
                ),
                (
                    WorkerPlacement("e0", (1, 2), (0, 1)),
                    WorkerPlacement("e1", (1,), (2, 3)),
                ),
            ):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    replace(configured, workers=workers).pool_directory()


if __name__ == "__main__":
    unittest.main()
