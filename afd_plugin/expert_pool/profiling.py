# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded live worker traces; no tensor values, weights or activation dumps."""

import json
from pathlib import Path

import torch

from afd_plugin.expert_pool.protocol import ExecutionPlan


class WorkerProfiler:
    """Capture a declared call window, separately from throughput measurements.

    Call metadata lets the analysis reject native startup/profile calls and
    identify which live batches were actually captured. Profiling perturbs
    execution; trace wall time must not substitute for unprofiled performance.
    """

    def __init__(self, directory: Path, skip_calls: int, active_calls: int) -> None:
        if skip_calls < 0 or active_calls <= 0:
            raise ValueError("Invalid profiler call window")
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.skip_calls = skip_calls
        self.active_calls = active_calls
        self.profiler: torch.profiler.profile | None = None
        self.calls: list[dict] = []
        self.done = False

    def before_call(self, completed: int, plan: ExecutionPlan) -> None:
        if self.done or completed < self.skip_calls:
            return
        if self.profiler is None:
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(self.directory), worker_name="expert-worker", use_gzip=True
                ),
            )
            self.profiler.start()
        self.calls.append(
            {
                "completed_before": completed,
                "client_id": plan.request.key.client_id,
                "sequence": plan.request.key.call_seq,
                "layer_id": plan.request.layer_id,
                "token_rows": plan.request.num_tokens,
            }
        )

    def after_call(self) -> None:
        if self.profiler is not None and len(self.calls) >= self.active_calls:
            self.close()

    def close(self) -> None:
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
            (self.directory / "calls.json").write_text(
                json.dumps({"skip_calls": self.skip_calls, "calls": self.calls}) + "\n"
            )
        self.done = True
