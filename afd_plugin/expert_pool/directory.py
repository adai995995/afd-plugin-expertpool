# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Immutable multi-worker coverage and CPU-only replica tie breaking.

Whole-layer mode permits replicated layers. Expert mode defaults to disjoint
coverage; explicit expert_replicated placement permits multiple resident copies.
The Controller reserves selected workers before dispatch. No migration or retry.
"""

from collections.abc import Collection
from dataclasses import dataclass

from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.scheduler import StaticDirectory


@dataclass(frozen=True)
class PoolDirectory:
    workers: tuple[StaticDirectory, ...]
    expert_partitioned: bool = False
    expert_replicated: bool = False

    def __post_init__(self) -> None:
        if type(self.expert_replicated) is not bool or (
            self.expert_replicated and not self.expert_partitioned
        ):
            raise ValueError("Expert replicas require explicit partitioned dispatch")
        if type(self.expert_partitioned) is not bool:
            raise ValueError("Expert partitioning must be an explicit boolean")
        if not isinstance(self.workers, tuple) or not self.workers:
            raise ValueError("Pool requires immutable worker directories")
        if any(not isinstance(worker, StaticDirectory) for worker in self.workers):
            raise ValueError("Invalid worker directory")
        if len({worker.worker_id for worker in self.workers}) != len(self.workers):
            raise ValueError("Duplicate worker identity")
        reference = self.workers[0]
        layers: dict[int, int] = {}
        covered: dict[int, set[int]] = {}
        for worker in self.workers:
            if worker.allow_partial_experts != self.expert_partitioned:
                raise ValueError("Pool and worker expert partition modes disagree")
            if (
                worker.model_id,
                worker.version,
                worker.hidden_size,
                worker.top_k,
                worker.max_tokens,
            ) != (
                reference.model_id,
                reference.version,
                reference.hidden_size,
                reference.top_k,
                reference.max_tokens,
            ):
                raise ValueError("Workers must bind one checkpoint, version and shape")
            for placement in worker.placements:
                count = layers.setdefault(placement.layer_id, placement.num_experts)
                if count != placement.num_experts:
                    raise ValueError("Replica expert counts disagree")
                if self.expert_partitioned:
                    resident = covered.setdefault(placement.layer_id, set())
                    if (
                        resident.intersection(placement.expert_ids)
                        and not self.expert_replicated
                    ):
                        raise ValueError("Partitioned experts require a unique owner")
                    resident.update(placement.expert_ids)
        if self.expert_partitioned and any(
            covered[layer] != set(range(count)) for layer, count in layers.items()
        ):
            raise ValueError("Expert partitions must cover every logical expert")

    @property
    def placements(self) -> tuple[ExpertPlacement, ...]:
        # Logical order need not match a worker's physical weight-slot order.
        layers = {
            placement.layer_id: placement.num_experts
            for worker in self.workers
            for placement in worker.placements
        }
        return tuple(
            ExpertPlacement(layer, count, tuple(range(count)))
            for layer, count in sorted(layers.items())
        )

    def locations(self, layer_id: int, expert_id: int) -> tuple[tuple[str, int], ...]:
        """Return resident (worker ID, local weight slot) for a logical expert."""
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError("Invalid logical layer ID")
        locations = []
        for worker in self.workers:
            for placement in worker.placements:
                if placement.layer_id == layer_id:
                    if type(expert_id) is not int or not (
                        0 <= expert_id < placement.num_experts
                    ):
                        raise ValueError("Invalid logical expert ID")
                    if expert_id in placement.expert_ids:
                        locations.append(
                            (worker.worker_id, placement.expert_ids.index(expert_id))
                        )
        if not locations:
            raise ValueError("Layer is absent from the pool")
        return tuple(locations)

    def owners(self, layer_id: int) -> tuple[str, ...]:
        """Return all resident owners in deterministic reservation order."""
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError("Invalid logical layer ID")
        owners = tuple(
            sorted(
                worker.worker_id
                for worker in self.workers
                if any(
                    placement.layer_id == layer_id and placement.expert_ids
                    for placement in worker.placements
                )
            )
        )
        if not owners:
            raise ValueError("Layer is absent from the pool")
        return owners


class ReplicaSelector:
    """Round-robin whole-layer calls among resident copies, without GPU reads.

    Client offsets avoid forcing all clients to start on the same replica.
    Selection is static, not a queue estimate or a service-domain SLO guarantee.
    The caller serializes access and must never retry a dispatched GPU call.
    """

    def __init__(self, directory: PoolDirectory, client_offset: int = 0) -> None:
        if directory.expert_partitioned:
            raise ValueError("Replica selector requires whole-layer coverage")
        if type(client_offset) is not int or client_offset < 0:
            raise ValueError("Client offset must be a nonnegative integer")
        self.candidates: dict[int, tuple[str, ...]] = {
            placement.layer_id: tuple(
                sorted(
                    worker.worker_id
                    for worker in directory.workers
                    if any(p.layer_id == placement.layer_id for p in worker.placements)
                )
            )
            for placement in directory.placements
        }
        self.next_replica = dict.fromkeys(self.candidates, client_offset)

    def select(self, layer_id: int) -> str:
        if type(layer_id) is not int or layer_id not in self.candidates:
            raise ValueError("Requested layer is not resident")
        candidates = self.candidates[layer_id]
        index = self.next_replica[layer_id] % len(candidates)
        self.next_replica[layer_id] = (index + 1) % len(candidates)
        return candidates[index]

    def select_available(
        self, layer_id: int, available_workers: Collection[str]
    ) -> str | None:
        """Advance only after selecting an available resident whole-layer copy.

        The caller must atomically reserve the returned worker before exposing
        the plan. Availability is a slot ledger view, not measured GPU load.
        """
        if type(layer_id) is not int or layer_id not in self.candidates:
            raise ValueError("Requested layer is not resident")
        candidates = self.candidates[layer_id]
        start = self.next_replica[layer_id]
        for offset in range(len(candidates)):
            index = (start + offset) % len(candidates)
            if candidates[index] in available_workers:
                self.next_replica[layer_id] = (index + 1) % len(candidates)
                return candidates[index]
        return None
