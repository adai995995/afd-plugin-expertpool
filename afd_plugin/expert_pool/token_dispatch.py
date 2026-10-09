# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local resident selection, exact token packing and a single shape summary.

Each logical Expert selects one resident copy per call. The GPU assigns every
Top-k slot to that copy and deduplicates rows per destination. Only bounded
counts reach the host; IDs, activations and the return row map stay on GPU.
The summary wait is required by this NCCL backend, not claimed asynchronous.
"""

import time
from dataclasses import replace

import torch
from vllm.triton_utils import triton

from afd_plugin.expert_pool.compact_input import ROW_BLOCK_SIZE, _row_indices
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.protocol import (
    CallRequest,
    DispatchPlan,
    ExpertDemand,
    ExpertTask,
)
from afd_plugin.expert_pool.replica_dispatch import WorkerLoad


class TokenInputWorkspace:
    """One destination's reusable payload, row mapping and FP32 reply buffer."""

    def __init__(self, rows: int, hidden: int, top_k: int, device: torch.device):
        self.flags = torch.empty(rows, dtype=torch.int32, device=device)
        self.prefix = torch.empty_like(self.flags)
        self.row_ids = torch.empty(rows, dtype=torch.int64, device=device)
        self.hidden = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
        self.weights = torch.empty((rows, top_k), dtype=torch.float32, device=device)
        self.ids = torch.empty((rows, top_k), dtype=torch.int32, device=device)
        self.output = torch.empty((rows, hidden), dtype=torch.float32, device=device)
        self.mask: torch.Tensor | None = None

    def mark(self, destinations: torch.Tensor, owner: int) -> None:
        rows = destinations.shape[0]
        self.mask = destinations == owner
        self.flags[:rows].copy_(self.mask.any(dim=1))
        torch.cumsum(self.flags[:rows], 0, out=self.prefix[:rows])
        _row_indices[(triton.cdiv(rows, ROW_BLOCK_SIZE),)](
            self.flags, self.prefix, self.row_ids, rows, ROW_BLOCK_SIZE
        )

    def pack(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], rows: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.mask is not None
        indices = self.row_ids[:rows]
        payload = self.hidden[:rows], self.weights[:rows], self.ids[:rows]
        for source, target in zip(inputs, payload, strict=True):
            torch.index_select(source, 0, indices, out=target)
        owned = self.mask.index_select(0, indices)
        payload[2].masked_fill_(~owned, -1)
        payload[1].masked_fill_(~owned, 0)
        return payload


class LocalTokenDispatcher:
    """Compiled per-layer candidate lists with a bounded, local load heuristic.

    Load feedback may be stale. It affects copy selection, never slot ownership.
    No global optimization, new feedback wait or centralized reservation occurs.
    """

    def __init__(self, directory: PoolDirectory, device: torch.device):
        if not directory.expert_partitioned or device.type != "cuda":
            raise ValueError("Token dispatch requires a CUDA Expert directory")
        self.directory = directory
        self.device = device
        self.owners = tuple(sorted(w.worker_id for w in directory.workers))
        self.candidates = {
            p.layer_id: tuple(
                tuple(owner for owner, _ in directory.locations(p.layer_id, e))
                for e in range(p.num_experts)
            )
            for p in directory.placements
        }
        shape = directory.workers[0]
        self.workspaces = {
            owner: TokenInputWorkspace(
                shape.max_tokens, shape.hidden_size, shape.top_k, device
            )
            for owner in self.owners
        }
        count = max(p.num_experts for p in directory.placements)
        # Expert histogram, invalid-ID bin, then unique-row counts per E.
        self.summary = torch.empty(
            count + 1 + len(self.owners), dtype=torch.int64, device=device
        )
        self.host = torch.empty_like(self.summary, device="cpu", pin_memory=True)
        self.begin = torch.cuda.Event(enable_timing=True)
        self.counted = torch.cuda.Event(enable_timing=True)
        self.copied = torch.cuda.Event(enable_timing=True)

    def prepare(
        self,
        request: CallRequest,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        loads: dict[str, WorkerLoad],
        turn: int,
    ) -> tuple[DispatchPlan, dict[str, tuple[torch.Tensor, ...]], dict[str, float]]:
        started = time.perf_counter_ns()
        candidates = self.candidates[request.layer_id]
        num_experts = len(candidates)
        choices = []
        for expert, copies in enumerate(candidates):
            rotated = (
                copies[(expert + turn) % len(copies) :]
                + copies[: (expert + turn) % len(copies)]
            )
            choices.append(
                min(
                    rotated,
                    key=lambda w: (
                        loads[w].pending_assignments,
                        loads[w].computing,
                        loads[w].occupied_slots,
                    ),
                )
            )
        selected_ns = time.perf_counter_ns()
        ids = inputs[2]
        if request.num_tokens == 0:
            empty = replace(
                request,
                demand=ExpertDemand((0,) * num_experts, started),
                compact_output=True,
                partial_reduction=True,
            )
            return DispatchPlan(empty, ()), {}, {"token_summary_wait_ms": 0.0}
        self.begin.record()
        table = torch.tensor(
            [self.owners.index(owner) for owner in choices],
            dtype=torch.int32,
            device=self.device,
        )
        valid = (ids >= 0) & (ids < num_experts)
        indices = ids.long().masked_fill(~valid, num_experts)
        histogram = self.summary[: num_experts + 1]
        histogram.zero_()
        histogram.scatter_add_(0, indices.flatten(), torch.ones_like(indices.flatten()))
        destinations = table[indices.clamp_max(num_experts - 1)].masked_fill(~valid, -1)
        for index, owner in enumerate(self.owners):
            workspace = self.workspaces[owner]
            workspace.mark(destinations, index)
            self.summary[num_experts + 1 + index].copy_(
                workspace.flags[: request.num_tokens].sum()
            )
        self.counted.record()
        length = num_experts + 1 + len(self.owners)
        self.host[:length].copy_(self.summary[:length], non_blocking=True)
        self.copied.record()
        waiting = time.perf_counter_ns()
        self.copied.synchronize()
        waited = time.perf_counter_ns()
        counts = self.host[:length].tolist()
        if counts[num_experts]:
            raise ValueError("Router supplied an invalid logical Expert ID")
        request = replace(
            request,
            demand=ExpertDemand(tuple(counts[:num_experts]), started),
            compact_output=True,
            partial_reduction=True,
        )
        tasks = []
        payloads = {}
        for index, owner in enumerate(self.owners):
            experts = tuple(
                e for e, w in enumerate(choices) if w == owner and counts[e]
            )
            rows = counts[num_experts + 1 + index]
            if not experts:
                if rows:
                    raise RuntimeError("Empty destination has routed rows")
                continue
            task = ExpertTask(owner, experts, sum(counts[e] for e in experts))
            if not 0 < rows <= min(request.num_tokens, task.num_assignments):
                raise RuntimeError("Packed row accounting does not match assignments")
            tasks.append(task)
            payloads[owner] = self.workspaces[owner].pack(inputs, rows)
        dispatch = DispatchPlan(request, tuple(tasks))
        return (
            dispatch,
            payloads,
            {
                "client_local_replica_select_ms": (selected_ns - started) / 1e6,
                "token_summary_wait_ms": (waited - waiting) / 1e6,
                "token_summary_host_ms": (waited - selected_ns) / 1e6,
                "token_summary_gpu_ms": self.begin.elapsed_time(self.counted),
                "token_summary_copy_gpu_ms": self.counted.elapsed_time(self.copied),
                "token_summary_bytes": length * self.host.element_size(),
            },
        )
