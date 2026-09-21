# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Stable GPU packing of owned Top-k slots without changing Expert arithmetic.

The host supplies the already validated assignment count. A fixed-length GPU
prefix sum determines the original flattened token/slot order; only BF16 values
are gathered or scattered. There is no dynamic output-size discovery, routing
copy to the host, host synchronization, local reduction, or change to either GEMM.
"""

import torch
from vllm.triton_utils import tl, triton

FLAG_BLOCK_SIZE = 256
COPY_BLOCK_SIZE = 256
COPY_NUM_WARPS = 4
MAX_PREFIX_COUNT = 2**31 - 1


@triton.jit
def _ownership_flags(
    route_ids,
    ownership,
    flags,
    num_slots,
    num_experts,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    expert_ids = tl.load(route_ids + offsets, offsets < num_slots, other=-1)
    valid = (offsets < num_slots) & (expert_ids >= 0) & (expert_ids < num_experts)
    owned = tl.load(ownership + expert_ids, valid, other=0)
    tl.store(flags + offsets, owned.to(tl.int32), offsets < num_slots)


@triton.jit
def _copy_owned_slots(
    source,
    destination,
    flags,
    prefix,
    num_slots,
    num_assignments,
    hidden_size,
    pack: tl.constexpr,
    block_size: tl.constexpr,
):
    slot = tl.program_id(0)
    columns = tl.program_id(1) * block_size + tl.arange(0, block_size)
    owned = tl.load(flags + slot, slot < num_slots, other=0) != 0
    packed_row = tl.load(prefix + slot, slot < num_slots, other=0) - 1
    valid = (
        owned
        & (slot < num_slots)
        & (packed_row >= 0)
        & (packed_row < num_assignments)
        & (columns < hidden_size)
    )
    # Promote before multiplication so large, valid buffers cannot overflow
    # INT32 address arithmetic. No floating-point arithmetic touches the values.
    slot_offsets = slot.to(tl.int64) * hidden_size + columns
    packed_offsets = packed_row.to(tl.int64) * hidden_size + columns
    if pack:
        value = tl.load(source + slot_offsets, valid, other=0)
        tl.store(destination + packed_offsets, value, valid)
    else:
        value = tl.load(source + packed_offsets, valid, other=0)
        tl.store(destination + slot_offsets, value, valid)


class CompactOutputWorkspace:
    """Reusable scratch and BF16 payload storage for one serialized operation.

    ``num_assignments`` must equal the count for the supplied immutable routes
    and ownership map. That value comes from the validated demand summary; this
    helper does not read GPU counts back to verify it. Bounds masks prevent
    out-of-range accesses, but cannot turn an incorrect summary into a correct
    result. Callers retain responsibility for route validity and unique owners.

    Work is enqueued on the caller's current stream for ``device``. A reusable
    event orders workspace accesses across current-stream changes without a
    host synchronization. ``receive_buffer`` inserts that dependency before an
    external receive may overwrite the preceding scatter's source. The caller
    still owns external transfers: finish a send before reusing its payload and
    finish a receive before scattering it. Unowned destination slots are left
    unchanged so disjoint workers can restore one slot tensor.
    """

    def __init__(
        self, max_assignments: int, hidden_size: int, device: torch.device
    ) -> None:
        if (
            type(max_assignments) is not int
            or not 0 < max_assignments <= MAX_PREFIX_COUNT
            or type(hidden_size) is not int
            or hidden_size <= 0
        ):
            raise ValueError("Compact output requires positive, bounded capacities")
        if device.type != "cuda" or device.index is None:
            raise ValueError("Compact output requires an explicit CUDA device")
        self.max_assignments = max_assignments
        self.hidden_size = hidden_size
        self.device = device
        self._flags = torch.empty(max_assignments, dtype=torch.int32, device=device)
        self._prefix = torch.empty_like(self._flags)
        self._packed = torch.empty(
            (max_assignments, hidden_size), dtype=torch.bfloat16, device=device
        )
        self._last_use = torch.cuda.Event()
        self._has_last_use = False

    def _wait_for_reuse(self) -> None:
        if self._has_last_use:
            torch.cuda.current_stream(self.device).wait_event(self._last_use)

    def _record_use(self) -> None:
        self._last_use.record(torch.cuda.current_stream(self.device))
        self._has_last_use = True

    def receive_buffer(self, num_assignments: int) -> torch.Tensor:
        """Return a reusable, contiguous ``[assignments, hidden]`` payload view."""
        if (
            type(num_assignments) is not int
            or not 0 <= num_assignments <= self.max_assignments
        ):
            raise ValueError("Invalid compact-output assignment count")
        if num_assignments:
            self._wait_for_reuse()
        return self._packed[:num_assignments]

    def _validate(
        self,
        slots: torch.Tensor,
        topk_ids: torch.Tensor,
        ownership: torch.Tensor,
        num_assignments: int,
    ) -> int:
        if (
            topk_ids.ndim != 2
            or topk_ids.shape[1] <= 0
            or topk_ids.dtype != torch.int32
            or topk_ids.device != self.device
            or not topk_ids.is_contiguous()
            or ownership.ndim != 1
            or not 0 < ownership.numel() <= MAX_PREFIX_COUNT
            or ownership.dtype != torch.bool
            or ownership.device != self.device
            or not ownership.is_contiguous()
        ):
            raise ValueError("Expected contiguous CUDA INT32 routes and BOOL ownership")
        if (
            slots.shape != (*topk_ids.shape, self.hidden_size)
            or slots.dtype != torch.bfloat16
            or slots.device != self.device
            or not slots.is_contiguous()
        ):
            raise ValueError("Expected contiguous CUDA BF16 [tokens, top-k, hidden]")
        num_slots = topk_ids.numel()
        if (
            num_slots > self.max_assignments
            or type(num_assignments) is not int
            or not 0 <= num_assignments <= num_slots
        ):
            raise ValueError("Compact-output count or route capacity is invalid")
        return num_slots

    def _prepare(
        self, topk_ids: torch.Tensor, ownership: torch.Tensor, num_slots: int
    ) -> None:
        _ownership_flags[(triton.cdiv(num_slots, FLAG_BLOCK_SIZE),)](
            topk_ids,
            ownership,
            self._flags,
            num_slots,
            ownership.numel(),
            block_size=FLAG_BLOCK_SIZE,
        )
        # Both views have a host-known, fixed length. cumsum allocates no
        # data-dependent output and retains ranks across all GPU scan blocks.
        torch.cumsum(
            self._flags[:num_slots],
            dim=0,
            dtype=torch.int32,
            out=self._prefix[:num_slots],
        )

    @torch.inference_mode()
    def pack(
        self,
        slots: torch.Tensor,
        topk_ids: torch.Tensor,
        ownership: torch.Tensor,
        *,
        num_assignments: int,
    ) -> torch.Tensor:
        """Gather owned slots in flattened token/slot order, preserving BF16 bits."""
        num_slots = self._validate(slots, topk_ids, ownership, num_assignments)
        packed = self.receive_buffer(num_assignments)
        if not num_assignments:
            return packed
        if any(
            value.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
            for value in (slots, topk_ids, ownership)
        ):
            raise ValueError("Packing inputs must not alias the compact output")
        with torch.cuda.device(self.device):
            self._prepare(topk_ids, ownership, num_slots)
            _copy_owned_slots[
                (num_slots, triton.cdiv(self.hidden_size, COPY_BLOCK_SIZE))
            ](
                slots,
                packed,
                self._flags,
                self._prefix,
                num_slots,
                num_assignments,
                self.hidden_size,
                pack=True,
                block_size=COPY_BLOCK_SIZE,
                num_warps=COPY_NUM_WARPS,
            )
            self._record_use()
        return packed

    @torch.inference_mode()
    def scatter(
        self,
        packed: torch.Tensor,
        topk_ids: torch.Tensor,
        ownership: torch.Tensor,
        destination: torch.Tensor,
        *,
        num_assignments: int,
    ) -> None:
        """Restore owned slots only; the caller performs the final native moe_sum."""
        num_slots = self._validate(destination, topk_ids, ownership, num_assignments)
        if (
            packed.shape != (num_assignments, self.hidden_size)
            or packed.dtype != torch.bfloat16
            or packed.device != self.device
            or not packed.is_contiguous()
        ):
            raise ValueError("Expected contiguous CUDA BF16 compact payload")
        if not num_assignments:
            return
        if any(
            value.untyped_storage().data_ptr()
            == destination.untyped_storage().data_ptr()
            for value in (packed, topk_ids, ownership)
        ):
            raise ValueError("Scatter inputs must not alias the destination")
        with torch.cuda.device(self.device):
            self._wait_for_reuse()
            self._prepare(topk_ids, ownership, num_slots)
            _copy_owned_slots[
                (num_slots, triton.cdiv(self.hidden_size, COPY_BLOCK_SIZE))
            ](
                packed,
                destination,
                self._flags,
                self._prefix,
                num_slots,
                num_assignments,
                self.hidden_size,
                pack=False,
                block_size=COPY_BLOCK_SIZE,
                num_warps=COPY_NUM_WARPS,
            )
            self._record_use()
