# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU demand planning over static, uniquely owned expert partitions."""

from dataclasses import dataclass

from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.protocol import CallRequest


@dataclass(frozen=True)
class ExpertTask:
    worker_id: str
    expert_ids: tuple[int, ...]
    num_assignments: int

    def __post_init__(self) -> None:
        if not isinstance(self.worker_id, str) or not 0 < len(self.worker_id) <= 128:
            raise ValueError("Invalid expert task worker")
        if (
            not isinstance(self.expert_ids, tuple)
            or not self.expert_ids
            or any(type(expert) is not int or expert < 0 for expert in self.expert_ids)
            or len(set(self.expert_ids)) != len(self.expert_ids)
        ):
            raise ValueError("Expert tasks require unique nonnegative expert IDs")
        if type(self.num_assignments) is not int or self.num_assignments <= 0:
            raise ValueError("Expert tasks require positive assignment counts")


@dataclass(frozen=True)
class DispatchPlan:
    request: CallRequest
    tasks: tuple[ExpertTask, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, CallRequest) or self.request.demand is None:
            raise ValueError("Dispatch requires a request with expert demand")
        if not isinstance(self.tasks, tuple) or any(
            not isinstance(task, ExpertTask) for task in self.tasks
        ):
            raise ValueError("Dispatch requires immutable expert tasks")
        if len({task.worker_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Each worker must receive at most one expert task")
        counts = self.request.demand.counts
        assigned: set[int] = set()
        for task in self.tasks:
            if any(expert >= len(counts) for expert in task.expert_ids):
                raise ValueError("Task expert is outside the demand range")
            if assigned.intersection(task.expert_ids):
                raise ValueError("Dispatch repeats a demanded expert")
            if any(counts[expert] == 0 for expert in task.expert_ids):
                raise ValueError("Dispatch includes an expert without demand")
            if task.num_assignments != sum(
                counts[expert] for expert in task.expert_ids
            ):
                raise ValueError("Task count disagrees with expert demand")
            assigned.update(task.expert_ids)
        if assigned != {expert for expert, count in enumerate(counts) if count > 0}:
            raise ValueError("Dispatch must cover every demanded expert exactly once")
        if sum(task.num_assignments for task in self.tasks) != (
            self.request.num_tokens * self.request.top_k
        ):
            raise ValueError("Dispatch assignment accounting is incomplete")

    @property
    def owners(self) -> tuple[str, ...]:
        return tuple(sorted(task.worker_id for task in self.tasks))

    def task_for(self, worker_id: str) -> ExpertTask:
        for task in self.tasks:
            if task.worker_id == worker_id:
                return task
        raise ValueError("Worker has no demanded expert task")


def plan_expert_demand(directory: PoolDirectory, request: CallRequest) -> DispatchPlan:
    """Assign positive logical expert demand to each resident unique owner.

    This is a plan over published placement, not resource admission. The
    controller must still atomically reserve the returned participating owners.
    """
    if not isinstance(directory, PoolDirectory) or not directory.expert_partitioned:
        raise ValueError("Expert demand planning requires unique expert partitions")
    if not isinstance(request, CallRequest) or request.demand is None:
        raise ValueError("Expert demand planning requires a demand descriptor")
    resident_owners = directory.owners(request.layer_id)
    for worker in directory.workers:
        if worker.worker_id in resident_owners:
            worker.validate(request)
    placement = next(
        item for item in directory.placements if item.layer_id == request.layer_id
    )
    if len(request.demand.counts) != placement.num_experts:
        raise ValueError("Demand shape does not match the logical expert count")
    by_worker: dict[str, list[int]] = {}
    for expert, count in enumerate(request.demand.counts):
        if count > 0:
            locations = directory.locations(request.layer_id, expert)
            if len(locations) != 1:
                raise ValueError("Demanded experts require one resident owner")
            by_worker.setdefault(locations[0][0], []).append(expert)
    return DispatchPlan(
        request,
        tuple(
            ExpertTask(
                worker_id,
                tuple(experts),
                sum(request.demand.counts[expert] for expert in experts),
            )
            for worker_id, experts in sorted(by_worker.items())
        ),
    )
