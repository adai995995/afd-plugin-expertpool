# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pack bounded token rows needed by one selected physical Expert worker.

Rows retain their original order, as do all top-k slots within a row. The
existing compact-output protocol can therefore restore the returned slots
without receiving an additional index tensor. The known assignment count J
bounds unique selected rows, so A/E exchange exactly J rows. Unused rows have
zero activations/weights and skip-sentinel routes. No GPU-to-host shape copy is
needed on the dispatch path.
"""

import torch
from vllm.triton_utils import tl, triton

ROW_BLOCK_SIZE = 256
MIN_GUARANTEED_ROW_REDUCTION = 2
MAX_INPUT_TOKENS = 2**31 - 1


def should_pack_task(num_assignments: int, num_tokens: int) -> bool:
    """Require a proven twofold row reduction before paying for a shape sync."""
    if (
        type(num_assignments) is not int
        or type(num_tokens) is not int
        or min(num_assignments, num_tokens) <= 0
    ):
        raise ValueError("Invalid Expert task shape")
    return num_assignments * MIN_GUARANTEED_ROW_REDUCTION <= num_tokens


@triton.jit
def _selected_rows(
    ids, ownership, flags, rows, top_k: tl.constexpr, experts, block: tl.constexpr
):
    row = tl.program_id(0) * block + tl.arange(0, block)
    selected = tl.full((block,), False, tl.int1)
    for slot in range(top_k):
        expert = tl.load(ids + row * top_k + slot, row < rows, other=-1)
        valid = (row < rows) & (expert >= 0) & (expert < experts)
        selected |= tl.load(ownership + expert, valid, other=0)
    tl.store(flags + row, selected.to(tl.int32), row < rows)


@triton.jit
def _row_indices(flags, prefix, indices, rows, block: tl.constexpr):
    row = tl.program_id(0) * block + tl.arange(0, block)
    owned = tl.load(flags + row, row < rows, other=0) != 0
    rank = tl.load(prefix + row, row < rows, other=0) - 1
    tl.store(indices + rank, row, (row < rows) & owned)


@triton.jit
def _copy_hidden(source, destination, indices, prefix, source_rows, packed_rows,
                 hidden_size: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    valid_rows = tl.load(prefix + source_rows - 1)
    valid = (row < packed_rows) & (row < valid_rows)
    original = tl.load(indices + row, valid, other=0)
    value = tl.load(
        source + original.to(tl.int64) * hidden_size + columns,
        valid & (columns < hidden_size),
        other=0,
    )
    tl.store(destination + row.to(tl.int64) * hidden_size + columns, value,
             (row < packed_rows) & (columns < hidden_size))


@triton.jit
def _copy_routes(weights, ids, packed_weights, packed_ids, indices, prefix,
                 source_rows, packed_rows, top_k: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    valid_rows = tl.load(prefix + source_rows - 1)
    valid = (row < packed_rows) & (row < valid_rows)
    original = tl.load(indices + row, valid, other=0)
    offsets = original * top_k + columns
    route = tl.load(ids + offsets, valid & (columns < top_k), other=-1)
    weight = tl.load(weights + offsets, valid & (columns < top_k), other=0)
    tl.store(packed_ids + row * top_k + columns, route,
             (row < packed_rows) & (columns < top_k))
    tl.store(packed_weights + row * top_k + columns, weight,
             (row < packed_rows) & (columns < top_k))


class CompactInputWorkspace:
    """One reusable GPU payload per A/E pair; the caller finishes its send."""

    def __init__(
        self, max_tokens: int, hidden_size: int, top_k: int, device: torch.device
    ):
        if min(max_tokens, hidden_size, top_k) <= 0 or max_tokens > MAX_INPUT_TOKENS:
            raise ValueError("Invalid packed-input capacity")
        if device.type != "cuda" or device.index is None:
            raise ValueError("Packed input requires an explicit CUDA device")
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.device = device
        self.flags = torch.empty(max_tokens, dtype=torch.int32, device=device)
        self.prefix = torch.empty_like(self.flags)
        self.indices = torch.empty(max_tokens, dtype=torch.int64, device=device)
        self.hidden = torch.empty(
            (max_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )
        self.weights = torch.empty(
            (max_tokens, top_k), dtype=torch.float32, device=device
        )
        self.ids = torch.empty((max_tokens, top_k), dtype=torch.int32, device=device)

    @torch.inference_mode()
    def pack(
        self,
        hidden: torch.Tensor,
        weights: torch.Tensor,
        ids: torch.Tensor,
        ownership: torch.Tensor,
        *,
        num_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = hidden.shape[0]
        if (
            hidden.shape != (rows, self.hidden_size)
            or weights.shape != (rows, self.top_k)
            or ids.shape != (rows, self.top_k)
            or rows <= 0
            or rows > self.max_tokens
            or type(num_rows) is not int
            or not 0 < num_rows <= rows
            or hidden.dtype != torch.bfloat16
            or weights.dtype != torch.float32
            or ids.dtype != torch.int32
            or any(
                t.device != self.device or not t.is_contiguous()
                for t in (hidden, weights, ids)
            )
            or ownership.dtype != torch.bool
            or ownership.device != self.device
            or ownership.ndim != 1
            or not ownership.is_contiguous()
        ):
            raise ValueError("Invalid packed-input tensors")
        _selected_rows[(triton.cdiv(rows, ROW_BLOCK_SIZE),)](
            ids,
            ownership,
            self.flags,
            rows,
            self.top_k,
            ownership.numel(),
            ROW_BLOCK_SIZE,
        )
        torch.cumsum(
            self.flags[:rows], dim=0, dtype=torch.int32, out=self.prefix[:rows]
        )
        _row_indices[(triton.cdiv(rows, ROW_BLOCK_SIZE),)](
            self.flags, self.prefix, self.indices, rows, ROW_BLOCK_SIZE
        )
        payload = (
            self.hidden[:num_rows],
            self.weights[:num_rows],
            self.ids[:num_rows],
        )
        _copy_hidden[(num_rows, triton.cdiv(self.hidden_size, ROW_BLOCK_SIZE))](
            hidden,
            payload[0],
            self.indices,
            self.prefix,
            rows,
            num_rows,
            self.hidden_size,
            ROW_BLOCK_SIZE,
        )
        _copy_routes[(num_rows,)](
            weights,
            ids,
            payload[1],
            payload[2],
            self.indices,
            self.prefix,
            rows,
            num_rows,
            self.top_k,
            triton.next_power_of_2(self.top_k),
        )
        return payload
