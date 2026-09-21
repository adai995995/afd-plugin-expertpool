# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU demand validation and deterministic resident coverage planning."""

from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.protocol import CallRequest, DispatchPlan, ExpertTask


def plan_expert_demand(directory: PoolDirectory, request: CallRequest) -> DispatchPlan:
    """Build deterministic coverage; replicated admission must choose live owners.

    This is a plan over published placement, not resource admission. The
    controller must still atomically reserve the returned participating owners.
    """
    if not isinstance(directory, PoolDirectory) or not directory.expert_partitioned:
        raise ValueError("Expert demand planning requires explicit expert placement")
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
            if len(locations) != 1 and not directory.expert_replicated:
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
