# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only authoritative reservations for a static resident Expert Pool.

Whole-layer workers have one slot; partitioned workers may reserve more.
There is one outstanding call per client.
Progress events describe protocol phases, not measured GPU utilization.
"""

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field

from afd_plugin.expert_pool.batching import DISABLED_BATCHING, BatchingOptions
from afd_plugin.expert_pool.directory import PoolDirectory, ReplicaSelector
from afd_plugin.expert_pool.protocol import CallKey, CallRequest, ExecutionPlan
from afd_plugin.expert_pool.scheduler import StaticDirectory

CONTROLLER_POLICIES = ("round_robin", "ready_first")


def directory_digest(directory: StaticDirectory) -> str:
    payload = json.dumps(asdict(directory), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ControllerClientIdentity:
    client_id: str
    session_epoch: int
    domain: str

    def __post_init__(self) -> None:
        CallKey(self.client_id, self.session_epoch, 0)
        if not isinstance(self.domain, str) or not 0 < len(self.domain) <= 128:
            raise ValueError("Invalid service domain")


@dataclass(frozen=True)
class QueuedCall:
    request: CallRequest
    worker_id: str | None
    enqueued_ns: int


@dataclass
class BufferReservation:
    generation: int = 0
    active: ExecutionPlan | None = None
    phase: str = "idle"


@dataclass
class WorkerReservation:
    ready: bool = False
    slots: tuple[BufferReservation, ...] = field(
        default_factory=lambda: (BufferReservation(),)
    )
    lifecycle: str = "offline"
    completed: int = 0
    rejected: int = 0

    @property
    def active(self) -> ExecutionPlan | None:
        return next(
            (slot.active for slot in self.slots if slot.active is not None), None
        )

    @active.setter
    def active(self, plan: ExecutionPlan | None) -> None:
        if len(self.slots) != 1:
            raise ValueError("Reserve a specific slot in a multi-slot worker")
        self.slots[0].active = plan

    @property
    def generation(self) -> int:
        return sum(slot.generation for slot in self.slots)

    @generation.setter
    def generation(self, value: int) -> None:
        if len(self.slots) != 1:
            raise ValueError("Advance a specific slot generation")
        self.slots[0].generation = value

    @property
    def phase(self) -> str:
        if self.lifecycle in {"offline", "closed"}:
            return self.lifecycle
        if len(self.slots) == 1:
            return self.slots[0].phase
        return "busy" if self.active is not None else "idle"

    @phase.setter
    def phase(self, value: str) -> None:
        if len(self.slots) != 1 and value not in {"idle", "offline", "closed"}:
            raise ValueError("Update a specific slot phase")
        self.lifecycle = value
        if len(self.slots) == 1:
            self.slots[0].phase = value

    @property
    def available_slots(self) -> int:
        return sum(slot.active is None for slot in self.slots) if self.ready else 0


class ControllerLedger:
    def __init__(
        self,
        directory: PoolDirectory,
        clients: tuple[ControllerClientIdentity, ...],
        scheduling_policy: str = "round_robin",
    ) -> None:
        self._initialize(directory, clients, scheduling_policy)
        self.selectors = {
            client_id: ReplicaSelector(directory, offset)
            for offset, client_id in enumerate(sorted(self.clients))
        }

    def _initialize(
        self,
        directory: PoolDirectory,
        clients: tuple[ControllerClientIdentity, ...],
        scheduling_policy: str,
    ) -> None:
        """Initialize shared bounded state; subclasses supply dispatch policy."""
        if scheduling_policy not in CONTROLLER_POLICIES:
            raise ValueError("Unknown controller scheduling policy")
        if not clients or len({c.client_id for c in clients}) != len(clients):
            raise ValueError("Controller requires unique clients")
        self.directory = directory
        self.batching = BatchingOptions()
        self.requires_warmup = False
        self.collect_cost_feedback = False
        self.scheduling_policy = scheduling_policy
        self.directories = {d.worker_id: d for d in directory.workers}
        self.clients = {c.client_id: c for c in clients}
        self.last_sequence = dict.fromkeys(self.clients, -1)
        self.outstanding: dict[str, CallKey] = {}
        self.closed: set[str] = set()
        self.pending: dict[str, deque[QueuedCall]] = {
            client.domain: deque() for client in clients
        }
        self.domains = deque(self.pending)
        self.workers = {key: WorkerReservation() for key in self.directories}
        self.completed_by_client = dict.fromkeys(self.clients, 0)
        self.completed_by_worker_client = {
            worker: dict.fromkeys(self.clients, 0) for worker in self.workers
        }
        self.completed_by_worker_layer = {
            worker: dict.fromkeys((p.layer_id for p in d.placements), 0)
            for worker, d in self.directories.items()
        }
        self.plan_sequence = 0
        self.peak_pending = 0
        self.peak_reserved = 0
        self.busy_replica_bypasses = 0

    def ready(
        self,
        worker_id: str,
        fingerprint: str,
        *,
        receive_slots: int = 1,
        batching: BatchingOptions = DISABLED_BATCHING,
        startup_complete: int = 0,
        collect_cost_feedback: int = 0,
    ) -> None:
        worker = self.workers[worker_id]
        if (
            worker.ready
            or fingerprint != directory_digest(self.directories[worker_id])
            or type(receive_slots) is not int
            or receive_slots != len(worker.slots)
            or batching != self.batching
            or type(startup_complete) is not int
            or startup_complete not in (0, 1)
            or (self.requires_warmup and not startup_complete)
            or type(collect_cost_feedback) is not int
            or collect_cost_feedback != int(self.collect_cost_feedback)
        ):
            raise ValueError("Duplicate readiness or mismatched resident directory")
        worker.ready = True
        worker.phase = "idle"

    def submit(self, client_id: str, request: CallRequest, now_ns: int) -> None:
        if request.demand is not None:
            raise ValueError("Whole-layer dispatch does not accept expert demand")
        identity = self.clients[client_id]
        if client_id in self.closed or (
            request.key.client_id,
            request.key.session_epoch,
        ) != (client_id, identity.session_epoch):
            raise ValueError("Wrong or closed controller session")
        if client_id in self.outstanding:
            raise ValueError("Only one outstanding call per client is supported")
        if request.key.call_seq <= self.last_sequence[client_id]:
            raise ValueError("Repeated or stale call sequence")
        candidates = self.selectors[client_id].candidates.get(request.layer_id)
        if not candidates:
            raise ValueError("Requested layer is absent from the pool")
        self.directories[candidates[0]].validate(request)
        # Only the comparison policy binds on admission. Ready-first requests
        # remain logical until a resident slot can be reserved at grant time.
        target = (
            self.selectors[client_id].select(request.layer_id)
            if self.scheduling_policy == "round_robin"
            else None
        )
        self.pending[identity.domain].append(QueuedCall(request, target, now_ns))
        self.outstanding[client_id] = request.key
        self.last_sequence[client_id] = request.key.call_seq
        self.peak_pending = max(self.peak_pending, self.pending_count)

    @property
    def pending_count(self) -> int:
        return sum(len(queue) for queue in self.pending.values())

    @property
    def reserved_count(self) -> int:
        return sum(
            slot.active is not None
            for worker in self.workers.values()
            for slot in worker.slots
        )

    def grant(self, now_ns: int) -> tuple[ExecutionPlan, float] | None:
        available = {
            key
            for key, worker in self.workers.items()
            if worker.ready and worker.active is None
        }
        for _ in range(len(self.domains)):
            domain = self.domains[0]
            self.domains.rotate(-1)
            queue = self.pending[domain]
            for call in queue:
                if self.scheduling_policy == "round_robin":
                    target = call.worker_id
                    if target not in available:
                        continue
                else:
                    selector = self.selectors[call.request.key.client_id]
                    layer = call.request.layer_id
                    candidates = selector.candidates[layer]
                    preferred = candidates[
                        selector.next_replica[layer] % len(candidates)
                    ]
                    target = selector.select_available(layer, available)
                    if target is None:
                        continue
                    if self.workers[preferred].active is not None:
                        self.busy_replica_bypasses += 1
                assert target is not None
                worker = self.workers[target]
                queue.remove(call)
                worker.generation += 1
                self.plan_sequence += 1
                plan = ExecutionPlan(
                    call.request,
                    target,
                    self.plan_sequence,
                    0,
                    worker.generation,
                )
                # Reserve before publishing the plan to either endpoint.
                worker.active = plan
                worker.phase = "reserved"
                self.peak_reserved = max(
                    self.peak_reserved,
                    sum(w.active is not None for w in self.workers.values()),
                )
                return plan, max(0, now_ns - call.enqueued_ns) / 1e6
        return None

    def progress(self, worker_id: str, kind: str, plan: ExecutionPlan) -> None:
        worker = self.workers[worker_id]
        if plan.worker_id != worker_id or worker.active != plan:
            raise ValueError("Stale, duplicate or foreign worker feedback")
        transitions = {
            "grant": ("reserved", "receiving"),
            "input_ready": ("receiving", "input_ready"),
            "executing": ("input_ready", "executing"),
            "output_ready": ("executing", "returning"),
            "done": ("returning", "idle"),
            # Only the drained ValueError path in ExpertWorker emits this.
            "error": ("executing", "idle"),
        }
        if kind not in transitions or worker.phase != transitions[kind][0]:
            raise ValueError("Out-of-order worker progress")
        worker.phase = transitions[kind][1]
        if kind not in {"done", "error"}:
            return
        worker.active = None
        del self.outstanding[plan.request.key.client_id]
        if kind == "error":
            worker.rejected += 1
            return
        worker.completed += 1
        client_id = plan.request.key.client_id
        self.completed_by_client[client_id] += 1
        self.completed_by_worker_client[worker_id][client_id] += 1
        self.completed_by_worker_layer[worker_id][plan.request.layer_id] += 1

    def close_client(self, client_id: str) -> None:
        if client_id in self.closed or client_id in self.outstanding:
            raise ValueError("Client must drain before closing exactly once")
        self.closed.add(client_id)

    def snapshot(self) -> dict:
        return {
            "policy": (
                "controller-resident-layer-" + self.scheduling_policy.replace("_", "-")
            ),
            "scheduling_policy": self.scheduling_policy,
            "busy_replica_bypasses": self.busy_replica_bypasses,
            "pending": self.pending_count,
            "outstanding": len(self.outstanding),
            "closed_clients": len(self.closed),
            "peak_pending": self.peak_pending,
            "peak_reserved": self.peak_reserved,
            "completed_by_client": dict(self.completed_by_client),
            "workers": {
                key: {
                    "ready": worker.ready,
                    "phase": worker.phase,
                    "generation": worker.generation,
                    "available_slots": worker.available_slots,
                    "receive_slots": len(worker.slots),
                    "slots": [
                        {
                            "slot_id": index,
                            "generation": slot.generation,
                            "phase": slot.phase,
                            "active": asdict(slot.active) if slot.active else None,
                        }
                        for index, slot in enumerate(worker.slots)
                    ],
                    "active": asdict(worker.active) if worker.active else None,
                    "completed_calls": worker.completed,
                    "rejected_calls": worker.rejected,
                    "client_calls": dict(self.completed_by_worker_client[key]),
                    "layer_calls": {
                        str(k): v
                        for k, v in self.completed_by_worker_layer[key].items()
                    },
                }
                for key, worker in self.workers.items()
            },
        }
