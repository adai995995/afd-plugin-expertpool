# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Diagnostic inputs are bounded, owned, private and excluded from normal runs."""

import stat
import tempfile
import unittest
from pathlib import Path

import torch

from afd_plugin.expert_pool.batch_capture import BatchInputCapture
from tests.unit.expert_pool.test_batching import ready_call


class BatchCaptureTests(unittest.TestCase):
    def test_declared_window_capacity_and_owned_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "capture"
            capture = BatchInputCapture(directory, 2, 1)
            plan = ready_call(0).plan
            inputs = (torch.arange(6).reshape(3, 2),)
            capture.capture(1, (plan,), inputs)
            self.assertFalse(capture.samples)
            capture.capture(2, (plan,), inputs)
            inputs[0].zero_()
            capture.capture(3, (plan,), inputs)
            self.assertEqual(len(capture.samples), 1)
            self.assertEqual(capture.used_bytes, 48)
            capture.close()
            record = torch.load(directory / "inputs.pt", weights_only=True)
            self.assertEqual(
                record["samples"][0]["inputs"][0].tolist(), [[0, 1], [2, 3], [4, 5]]
            )
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o700)
            self.assertEqual(
                stat.S_IMODE((directory / "inputs.pt").stat().st_mode), 0o600
            )
            self.assertFalse(capture.samples)

    def test_existing_storage_and_invalid_limits_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                BatchInputCapture(Path(temporary), 0, 1)
            for skip, limit in ((-1, 1), (0, 0), (0, 257)):
                with self.assertRaises(ValueError):
                    BatchInputCapture(Path(temporary) / "new", skip, limit)


if __name__ == "__main__":
    unittest.main()
