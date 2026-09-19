# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded Expert-demand summaries with an explicit GPU-to-host dependency.

Only a fixed-size count vector crosses to pinned host memory. This collector
does not move routing IDs, weights or activations to the CPU. The event wait is
intentional: dispatch cannot use the counts before their copy has completed.
It is a synchronization cost, not a free or fully asynchronous control path.
"""

import time

import torch

from afd_plugin.expert_pool.protocol import MAX_DEMAND_EXPERTS, ExpertDemand


class ExpertDemandCollector:
    """Reusable fixed-shape histogram buffers for one serialized A instance.

    The owner must allow only one outstanding ``collect`` call, as the current
    A dispatcher does. Events and buffers are reused only after the D2H event
    completes. ``ready_ns`` is a host observation taken when collection starts,
    before this stream consumes the routes; it is not their exact GPU-ready
    timestamp. The explicit wait can include earlier work on the caller's
    stream, while the two CUDA intervals measure the histogram and copy only.
    """

    def __init__(
        self, num_experts: int, device: torch.device, max_assignments: int
    ) -> None:
        if type(num_experts) is not int or not 0 < num_experts <= MAX_DEMAND_EXPERTS:
            raise ValueError("Invalid demand-summary Expert count")
        if type(max_assignments) is not int or max_assignments <= 0:
            raise ValueError("Demand-summary capacity must be a positive integer")
        if device.type != "cuda" or device.index is None:
            raise ValueError("Demand collection requires an explicit CUDA device")
        self.num_experts = num_experts
        self.device = device
        self.max_assignments = max_assignments
        # The final bin records invalid IDs without an out-of-bounds GPU access.
        # Fixed-size scatter avoids bincount's data-dependent output allocation.
        self.counts = torch.empty(num_experts + 1, dtype=torch.int64, device=device)
        self.indices = torch.empty(max_assignments, dtype=torch.int64, device=device)
        self.ones = torch.ones(max_assignments, dtype=torch.int64, device=device)
        self.invalid = torch.empty(max_assignments, dtype=torch.bool, device=device)
        self.upper_invalid = torch.empty_like(self.invalid)
        self.host_counts = torch.empty(
            num_experts + 1, dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.summary_bytes = self.host_counts.numel() * self.host_counts.element_size()
        self.histogram_begin = torch.cuda.Event(enable_timing=True)
        self.histogram_end = torch.cuda.Event(enable_timing=True)
        self.copy_end = torch.cuda.Event(enable_timing=True)

    def collect(self, topk_ids: torch.Tensor) -> tuple[ExpertDemand, dict[str, float]]:
        """Count immutable CUDA INT32 ``[tokens, top_k]`` routes.

        IDs must already be produced on the caller's current CUDA stream (or
        have an explicit producer dependency there). Invalid IDs are reported
        only after copying the summary; no route-value read reaches the CPU.
        Empty input performs neither CUDA work nor a device-to-host transfer.
        """
        ready_ns = time.perf_counter_ns()
        if (
            topk_ids.ndim != 2
            or topk_ids.shape[1] <= 0
            or topk_ids.dtype != torch.int32
            or topk_ids.device != self.device
            or not topk_ids.is_contiguous()
        ):
            raise ValueError("Expected contiguous CUDA INT32 top-k IDs")
        assignments = topk_ids.numel()
        if assignments > self.max_assignments:
            raise ValueError("Demand-summary assignment capacity exceeded")
        if assignments == 0:
            demand = ExpertDemand((0,) * self.num_experts, ready_ns)
            return demand, {
                "demand_histogram_gpu_ms": 0.0,
                "demand_summary_copy_gpu_ms": 0.0,
                "demand_metadata_wait_ms": 0.0,
                "demand_host_ms": (time.perf_counter_ns() - ready_ns) / 1e6,
            }

        stream = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(stream):
            self.histogram_begin.record(stream)
            self.counts.zero_()
            indices = self.indices[:assignments]
            indices.copy_(topk_ids.view(-1))
            invalid = self.invalid[:assignments]
            upper_invalid = self.upper_invalid[:assignments]
            torch.lt(indices, 0, out=invalid)
            torch.ge(indices, self.num_experts, out=upper_invalid)
            invalid.logical_or_(upper_invalid)
            indices.masked_fill_(invalid, self.num_experts)
            self.counts.scatter_add_(0, indices, self.ones[:assignments])
            self.histogram_end.record(stream)
            self.host_counts.copy_(self.counts, non_blocking=True)
            self.copy_end.record(stream)

        wait_started_ns = time.perf_counter_ns()
        self.copy_end.synchronize()
        wait_finished_ns = time.perf_counter_ns()
        # This list is the E+1 INT64 summary in pinned CPU memory, never the
        # original routing tensor. The explicit event wait makes its read safe.
        counts = self.host_counts.tolist()
        if counts[-1]:
            raise ValueError("Router supplied an invalid logical Expert ID")
        if sum(counts) != assignments:
            raise RuntimeError("Demand-summary assignment accounting mismatch")
        demand = ExpertDemand(tuple(counts[:-1]), ready_ns)
        metrics = {
            "demand_histogram_gpu_ms": self.histogram_begin.elapsed_time(
                self.histogram_end
            ),
            "demand_summary_copy_gpu_ms": self.histogram_end.elapsed_time(
                self.copy_end
            ),
            "demand_metadata_wait_ms": (wait_finished_ns - wait_started_ns) / 1e6,
            "demand_host_ms": (time.perf_counter_ns() - ready_ns) / 1e6,
        }
        return demand, metrics
