# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared homes must survive independent clients and directory ordering."""

import unittest
from collections import Counter
from dataclasses import replace

from afd_plugin.expert_pool.directory import PoolDirectory, shared_expert_homes
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.scheduler import StaticDirectory


class SharedHomeTests(unittest.TestCase):
    def directory(self):
        return PoolDirectory(
            tuple(
                StaticDirectory(
                    "checkpoint",
                    3,
                    owner,
                    (
                        ExpertPlacement(1, 8, tuple(range(8))),
                        ExpertPlacement(7, 8, tuple(range(8))),
                    ),
                    32,
                    2,
                    16,
                    True,
                )
                for owner in ("e0", "e1", "e2")
            ),
            True,
            True,
        )

    def test_all_clients_use_same_homes_despite_registration_order(self):
        pool = self.directory()
        expected = shared_expert_homes(pool)
        for order in (pool.workers[::-1], pool.workers[1:] + pool.workers[:1]):
            self.assertEqual(
                shared_expert_homes(replace(pool, workers=order)), expected
            )
        for homes in expected.values():
            counts = Counter(homes)
            self.assertEqual(set(counts), {"e0", "e1", "e2"})
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_heterogeneous_residency_never_selects_an_absent_expert(self):
        full = self.directory()
        workers = tuple(
            replace(
                worker,
                placements=tuple(
                    ExpertPlacement(p.layer_id, 8, experts) for p in worker.placements
                ),
            )
            for worker, experts in zip(
                full.workers, ((0, 1, 2, 3), (2, 3, 4, 5), (0, 6, 7)), strict=True
            )
        )
        pool = replace(full, workers=workers)
        homes = shared_expert_homes(pool)
        for layer, owners in homes.items():
            for expert, owner in enumerate(owners):
                self.assertIn(owner, [w for w, _ in pool.locations(layer, expert)])
            self.assertEqual(owners[1], "e0")
            self.assertEqual(owners[4], "e1")
            self.assertEqual(owners[7], "e2")

    def test_whole_layer_directory_is_not_an_expert_home_directory(self):
        full = self.directory()
        whole = PoolDirectory(
            tuple(replace(w, allow_partial_experts=False) for w in full.workers)
        )
        with self.assertRaises(ValueError):
            shared_expert_homes(whole)
