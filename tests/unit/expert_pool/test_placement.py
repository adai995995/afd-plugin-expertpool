# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only checks of logical/physical identity; no torch/vLLM required."""

import unittest

from afd_plugin.expert_pool.placement import ExpertPlacement


class ExpertPlacementTests(unittest.TestCase):
    def test_noncontiguous_permuted_slots(self) -> None:
        placement = ExpertPlacement(13, 8, (7, 1, 4))
        self.assertEqual(placement.global_to_local(), (-1, 1, -1, -1, 2, -1, -1, 0))

    def test_reject_duplicate_weights_and_out_of_range_ids(self) -> None:
        for ids in ((1, 1), (-1,), (8,), (), (True,)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                ExpertPlacement(1, 8, ids)

    def test_reject_invalid_layer_and_expert_count(self) -> None:
        for layer, count in ((-1, 8), (1, 0), (1, -1)):
            with self.subTest(layer=layer, count=count), self.assertRaises(ValueError):
                ExpertPlacement(layer, count, (0,))


if __name__ == "__main__":
    unittest.main()
