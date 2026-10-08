# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit diagnostic capture of bounded, real E inputs for kernel calibration.

Capture uses CUDA clones and perturbs execution. All tensor serialization and
CPU reads occur after the worker drains. Stored inputs contain request data;
the CLI requires a fresh private directory. No weights or credentials are saved.
"""

import os
import stat
from pathlib import Path

import torch

from afd_plugin.expert_pool.protocol import ExecutionPlan

MAX_CAPTURE_BYTES = 128 * 1024 * 1024
MAX_CAPTURE_BATCHES = 256


class BatchInputCapture:
    def __init__(self, directory: Path, skip_calls: int, max_batches: int) -> None:
        if skip_calls < 0 or not 1 <= max_batches <= MAX_CAPTURE_BATCHES:
            raise ValueError("Invalid diagnostic capture window")
        directory.mkdir(mode=0o700, exist_ok=False)
        details = directory.stat()
        if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) != 0o700:
            raise ValueError("Capture storage must be owned and mode 0700")
        self.directory = directory
        self.skip_calls = skip_calls
        self.max_batches = max_batches
        self.samples: list[dict] = []
        self.used_bytes = 0
        self.dropped = 0

    def capture(
        self,
        completed_calls: int,
        plans: tuple[ExecutionPlan, ...],
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        if completed_calls < self.skip_calls or len(self.samples) == self.max_batches:
            return
        size = sum(t.numel() * t.element_size() for t in inputs)
        if self.used_bytes + size > MAX_CAPTURE_BYTES:
            self.dropped += 1
            return
        self.used_bytes += size
        self.samples.append(
            {
                "layer_id": plans[0].request.layer_id,
                "members": [
                    {
                        "client_id": p.request.key.client_id,
                        "call_seq": p.request.key.call_seq,
                        "num_tokens": p.request.num_tokens,
                    }
                    for p in plans
                ],
                "inputs": tuple(t.detach().clone() for t in inputs),
            }
        )

    def close(self) -> None:
        """Call after all compute and sends drain; never during inference."""
        samples = [
            {**s, "inputs": tuple(t.cpu() for t in s["inputs"])} for s in self.samples
        ]
        torch.save(
            {
                "scope": "diagnostic real input capture; capture timing is perturbed",
                "skip_calls": self.skip_calls,
                "used_bytes": self.used_bytes,
                "dropped": self.dropped,
                "samples": samples,
            },
            self.directory / "inputs.pt",
        )
        (self.directory / "inputs.pt").chmod(0o600)
        self.samples.clear()
