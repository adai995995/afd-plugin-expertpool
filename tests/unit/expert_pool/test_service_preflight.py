# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ensure validation cannot silently share GPUs with another running job."""

import unittest
from unittest.mock import patch

from tools.expert_pool.validate_service import BusyGPUError, check_idle


class ServicePreflightTests(unittest.TestCase):
    def check_inventory(self, inventory: str, processes: str = "") -> list[dict]:
        with patch(
            "tools.expert_pool.validate_service.subprocess.check_output",
            side_effect=[inventory, processes],
        ):
            return check_idle([0, 1, 2])

    def test_idle_selected_gpus_accept_unrelated_busy_gpu(self) -> None:
        states = self.check_inventory(
            "0, GPU-a, 0, 0\n1, GPU-b, 0, 0\n2, GPU-c, 0, 0\n3, GPU-d, 10000, 100\n",
            "GPU-d, 1234\n",
        )
        self.assertEqual([state["index"] for state in states], [0, 1, 2])

    def test_zero_utilization_is_not_free_when_weights_are_resident(self) -> None:
        with self.assertRaises(BusyGPUError):
            self.check_inventory("0, GPU-a, 57000, 0\n1, GPU-b, 0, 0\n2, GPU-c, 0, 0\n")

    def test_process_prevents_launch_even_before_memory_is_reported(self) -> None:
        with self.assertRaises(BusyGPUError):
            self.check_inventory(
                "0, GPU-a, 0, 0\n1, GPU-b, 0, 0\n2, GPU-c, 0, 0\n", "GPU-b, 1234\n"
            )

    def test_utilization_prevents_launch(self) -> None:
        with self.assertRaises(BusyGPUError):
            self.check_inventory("0, GPU-a, 0, 1\n1, GPU-b, 0, 0\n2, GPU-c, 0, 0\n")

    def test_nonexistent_gpu_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.check_inventory("0, GPU-a, 0, 0\n1, GPU-b, 0, 0\n")


if __name__ == "__main__":
    unittest.main()
