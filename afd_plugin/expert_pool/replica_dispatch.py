# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static resident replicas with admission-time backlog and optional splitting.

Assignment splitting is a correctness-first two-copy policy. Backlog is an
assignment-count heuristic, not a latency/SLO prediction. Every selected
worker must have a free slot; the controller reserves them atomically.
"""

from dataclasses import dataclass

from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.protocol import (
    AssignmentSlice,
    CallRequest,
    DispatchPlan,
    ExpertTask,
)


@dataclass(frozen=True)
class WorkerLoad:
    available_slots: int
    pending_assignments: int
    computing: bool
    occupied_slots: int


def validate_dispatch(directory: PoolDirectory, dispatch: DispatchPlan) -> None:
    for task in dispatch.tasks:
        for expert in task.expert_ids:
            if task.worker_id not in {
                owner
                for owner, _ in directory.locations(dispatch.request.layer_id, expert)
            }:
                raise ValueError("Dispatch selected a nonresident Expert copy")


def select_replicas(
    directory: PoolDirectory,
    request: CallRequest,
    loads: dict[str, WorkerLoad],
    tie_offset: int,
    *,
    split_assignments: bool = False,
) -> DispatchPlan | None:
    if not directory.expert_replicated or request.demand is None:
        raise ValueError("Replica selection requires a replicated demand directory")
    if type(split_assignments) is not bool:
        raise ValueError("Assignment splitting must be an explicit boolean")
    workers = sorted(loads)
    order = workers[tie_offset % len(workers) :] + workers[: tie_offset % len(workers)]
    assigned: dict[str, list[int]] = {}
    slices: dict[str, list[AssignmentSlice]] = {}
    added = dict.fromkeys(workers, 0)
    counts = request.demand.counts
    # Largest-first placement balances the known ready work without reading any
    # activation or routing tensor. Co-resident Experts share one worker load.
    for expert in sorted(
        (i for i, n in enumerate(counts) if n), key=lambda i: (-counts[i], i)
    ):
        candidates = [
            worker
            for worker, _ in directory.locations(request.layer_id, expert)
            if loads[worker].available_slots > 0
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda worker: (
                loads[worker].pending_assignments + added[worker],
                loads[worker].computing,
                loads[worker].occupied_slots,
                order.index(worker),
            ),
        )
        active = candidates[: min(counts[expert], 2 if split_assignments else 1)]
        start = 0
        for index, owner in enumerate(active):
            share = (counts[expert] - start + len(active) - index - 1) // (
                len(active) - index
            )
            assigned.setdefault(owner, []).append(expert)
            if split_assignments:
                slices.setdefault(owner, []).append(
                    AssignmentSlice(expert, start, share)
                )
            added[owner] += share
            start += share
    return DispatchPlan(
        request,
        tuple(
            ExpertTask(
                worker,
                tuple(sorted(experts)),
                added[worker],
                tuple(sorted(slices.get(worker, ()), key=lambda item: item.expert_id)),
            )
            for worker, experts in sorted(assigned.items())
        ),
    )
