# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local resident selection, exact token packing and a single shape summary.

Each logical Expert selects up to two resident copies per call. The GPU assigns
stable, disjoint Top-k ranges and deduplicates rows per destination. Only bounded
counts reach the host; IDs, activations and the return row map stay on GPU.
The summary wait is required by this NCCL backend, not claimed asynchronous.
"""

import time
from dataclasses import replace

import torch
from vllm.triton_utils import tl, triton

from afd_plugin.expert_pool.compact_input import ROW_BLOCK_SIZE, _row_indices
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.protocol import (
    MAX_ACTIVE_EXPERT_REPLICAS,
    AssignmentSlice,
    CallRequest,
    DispatchPlan,
    ExpertDemand,
    ExpertTask,
)
from afd_plugin.expert_pool.replica_dispatch import WorkerLoad

ASSIGNMENT_BLOCK_SIZE = 256


@triton.jit
def _expert_block_counts(ids, counts, slots, stride, block: tl.constexpr):
    expert = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * block + tl.arange(0, block)
    routes = tl.load(ids + offsets, offsets < slots, other=-1)
    matched = (offsets < slots) & (routes == expert)
    tl.store(counts + expert * stride + chunk, tl.sum(matched.to(tl.int32), 0))


@triton.jit
def _split_destinations(
    ids,
    histogram,
    choices,
    prefix,
    destinations,
    slots,
    stride,
    multi_block: tl.constexpr,
    replica_slots: tl.constexpr,
    block: tl.constexpr,
):
    expert = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * block + tl.arange(0, block)
    routes = tl.load(ids + offsets, offsets < slots, other=-1)
    matched = (offsets < slots) & (routes == expert)
    ordinal = tl.cumsum(matched.to(tl.int32), 0) - 1
    if multi_block:
        ordinal += tl.load(prefix + expert * stride + chunk - 1, chunk > 0, other=0)
    first = tl.load(choices + expert * replica_slots)
    second = tl.load(choices + expert * replica_slots + 1)
    count = tl.load(histogram + expert)
    first_count = tl.where(second >= 0, (count + 1) // 2, count)
    destination = tl.where(ordinal < first_count, first, second)
    # Only this Expert's program writes these slots. Invalid IDs stay -1 until
    # the existing single shape-summary copy rejects the call on the host.
    tl.store(destinations + offsets, destination, matched)


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

    def __init__(
        self,
        directory: PoolDirectory,
        device: torch.device,
        *,
        active_expert_replicas: int = 1,
    ):
        if not directory.expert_partitioned or device.type != "cuda":
            raise ValueError("Token dispatch requires a CUDA Expert directory")
        if (
            type(active_expert_replicas) is not int
            or not 1 <= active_expert_replicas <= MAX_ACTIVE_EXPERT_REPLICAS
            or (active_expert_replicas > 1 and not directory.expert_replicated)
        ):
            raise ValueError("Multiple active copies require resident Expert replicas")
        self.directory = directory
        self.device = device
        self.active_expert_replicas = active_expert_replicas
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
        self.split_destinations = self.block_counts = self.block_prefix = None
        if active_expert_replicas > 1:
            self.split_destinations = torch.empty(
                (shape.max_tokens, shape.top_k), dtype=torch.int32, device=device
            )
            blocks = triton.cdiv(shape.max_tokens * shape.top_k, ASSIGNMENT_BLOCK_SIZE)
            self.block_counts = torch.empty(
                (count, blocks), dtype=torch.int32, device=device
            )
            self.block_prefix = torch.empty_like(self.block_counts)
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

            def load_key(worker):
                return (
                    loads[worker].pending_assignments,
                    loads[worker].computing,
                    loads[worker].occupied_slots,
                )

            choices.append(
                (min(rotated, key=load_key),)
                if self.active_expert_replicas == 1
                else tuple(sorted(rotated, key=load_key)[: self.active_expert_replicas])
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
            return (
                DispatchPlan(empty, ()),
                {},
                {"token_summary_wait_ms": 0.0, "client_split_experts": 0.0},
            )
        self.begin.record()
        table = torch.tensor(
            [
                [self.owners.index(owner) for owner in copies]
                + [-1] * (self.active_expert_replicas - len(copies))
                for copies in choices
            ],
            dtype=torch.int32,
            device=self.device,
        )
        valid = (ids >= 0) & (ids < num_experts)
        indices = ids.long().masked_fill(~valid, num_experts)
        histogram = self.summary[: num_experts + 1]
        histogram.zero_()
        histogram.scatter_add_(0, indices.flatten(), torch.ones_like(indices.flatten()))
        if self.active_expert_replicas == 1:
            destinations = table[:, 0][indices.clamp_max(num_experts - 1)].masked_fill(
                ~valid, -1
            )
        else:
            assert self.split_destinations is not None
            assert self.block_counts is not None and self.block_prefix is not None
            slots = ids.numel()
            blocks = triton.cdiv(slots, ASSIGNMENT_BLOCK_SIZE)
            stride = self.block_counts.shape[1]
            if blocks > 1:
                _expert_block_counts[(num_experts, blocks)](
                    ids, self.block_counts, slots, stride, ASSIGNMENT_BLOCK_SIZE
                )
                torch.cumsum(
                    self.block_counts[:num_experts, :blocks],
                    1,
                    out=self.block_prefix[:num_experts, :blocks],
                )
            destinations = self.split_destinations[: request.num_tokens]
            destinations.fill_(-1)
            _split_destinations[(num_experts, blocks)](
                ids,
                histogram,
                table,
                self.block_prefix,
                destinations,
                slots,
                stride,
                blocks > 1,
                self.active_expert_replicas,
                ASSIGNMENT_BLOCK_SIZE,
            )
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
        assignments: dict[str, list[AssignmentSlice]] = {w: [] for w in self.owners}
        split_experts = 0
        for expert, copies in (
            enumerate(choices) if self.active_expert_replicas > 1 else ()
        ):
            count = counts[expert]
            if not count:
                continue
            active = min(count, len(copies))
            split_experts += active > 1
            start = 0
            for index, owner in enumerate(copies[:active]):
                share = (count - start + active - index - 1) // (active - index)
                assignments[owner].append(AssignmentSlice(expert, start, share))
                start += share
        tasks = []
        payloads = {}
        for index, owner in enumerate(self.owners):
            slices = tuple(assignments[owner])
            experts = (
                tuple(item.expert_id for item in slices)
                if self.active_expert_replicas > 1
                else tuple(
                    e
                    for e, copies in enumerate(choices)
                    if copies[0] == owner and counts[e]
                )
            )
            rows = counts[num_experts + 1 + index]
            if not experts:
                if rows:
                    raise RuntimeError("Empty destination has routed rows")
                continue
            task = ExpertTask(
                owner,
                experts,
                sum(item.count for item in slices)
                if self.active_expert_replicas > 1
                else sum(counts[e] for e in experts),
                slices if self.active_expert_replicas > 1 else (),
            )
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
                "client_split_experts": float(split_experts),
                "token_summary_wait_ms": (waited - waiting) / 1e6,
                "token_summary_host_ms": (waited - selected_ns) / 1e6,
                "token_summary_gpu_ms": self.begin.elapsed_time(self.counted),
                "token_summary_copy_gpu_ms": self.counted.elapsed_time(self.copied),
                "token_summary_bytes": length * self.host.element_size(),
            },
        )
