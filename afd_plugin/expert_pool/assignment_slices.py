# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Apply a validated per-expert ordinal split without moving route metadata to CPU.

Assignments are ordered by their flattened token/top-k position within each
logical expert. Stable sorting turns each controller slice into exact GPU
positions. A sends the same activation rows but marks other assignments with
the existing -1 skip sentinel; E's compact output retains those positions.
This is a correctness-first path, not a claim of efficient input transfer.
"""

import torch

from afd_plugin.expert_pool.protocol import DispatchPlan


@torch.inference_mode()
def route_ids_by_worker(
    topk_ids: torch.Tensor, dispatch: DispatchPlan
) -> dict[str, torch.Tensor]:
    if (
        topk_ids.shape != (dispatch.request.num_tokens, dispatch.request.top_k)
        or topk_ids.dtype != torch.int32
        or not topk_ids.is_contiguous()
    ):
        raise ValueError("Assignment splitting requires contiguous INT32 routes")
    if not dispatch.tasks:
        return {}
    owners = dispatch.owners
    selected = [expert for task in dispatch.tasks for expert in task.expert_ids]
    if len(selected) == len(set(selected)):
        return dict.fromkeys(owners, topk_ids)
    if any(not task.assignment_slices for task in dispatch.tasks):
        raise ValueError("Shared Expert routes require assignment slices")

    flat = topk_ids.reshape(-1)
    # Stable order preserves original token/top-k order within each expert.
    ordered_positions = torch.argsort(flat, stable=True)
    bases = []
    offset = 0
    assert dispatch.request.demand is not None
    for count in dispatch.request.demand.counts:
        bases.append(offset)
        offset += count
    masked: dict[str, torch.Tensor] = {}
    for task in dispatch.tasks:
        destination = torch.full_like(flat, -1)
        for item in task.assignment_slices:
            start = bases[item.expert_id] + item.start
            positions = ordered_positions[start : start + item.count]
            destination.index_copy_(0, positions, flat.index_select(0, positions))
        masked[task.worker_id] = destination.reshape_as(topk_ids)
    return masked
