# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Atomic multi-worker reservations for static expert placement and replicas.

Every owner receives the complete input batch. A worker's completed child
remains reserved until the whole parent drains. Dispatch errors terminate the
deployment; uncertain GPU work is never retried or recycled.
"""

import time
from collections import deque
from dataclasses import asdict

from afd_plugin.expert_pool.batching import (
    DISABLED_BATCHING,
    BatchingOptions,
    compatibility_key,
)
from afd_plugin.expert_pool.controller import (
    BufferReservation,
    ControllerClientIdentity,
    ControllerLedger,
    QueuedCall,
    WorkerReservation,
)
from afd_plugin.expert_pool.cost_feedback import ExecutionCostBook, ExecutionShape
from afd_plugin.expert_pool.demand import DispatchPlan, plan_expert_demand
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.protocol import (
    MAX_RECEIVE_SLOTS,
    BatchExecution,
    BatchSlot,
    CallKey,
    CallRequest,
    ExecutionPlan,
)
from afd_plugin.expert_pool.replica_dispatch import WorkerLoad, select_replicas


class FanoutControllerLedger(ControllerLedger):
    def __init__(
        self,
        directory: PoolDirectory,
        clients: tuple[ControllerClientIdentity, ...],
        scheduling_policy: str = "ready_first",
        *,
        demand_aware: bool = False,
        compact_output: bool = False,
        receive_slots: int = 1,
        batching: BatchingOptions = DISABLED_BATCHING,
        collect_cost_feedback: bool = False,
        split_assignments: bool = False,
    ) -> None:
        if not directory.expert_partitioned or scheduling_policy != "ready_first":
            raise ValueError("Expert fan-out requires partitioned ready-first control")
        if type(demand_aware) is not bool:
            raise ValueError("Demand-aware dispatch must be a boolean")
        if type(compact_output) is not bool or (compact_output and not demand_aware):
            raise ValueError("Compact output requires demand-aware dispatch")
        self._initialize(directory, clients, scheduling_policy)
        if (
            type(receive_slots) is not int
            or not 1 <= receive_slots <= MAX_RECEIVE_SLOTS
            or (receive_slots > 1 and not compact_output)
        ):
            raise ValueError("Multiple receive slots require compact output")
        if directory.expert_replicated and (
            not demand_aware or not compact_output or receive_slots < 2
        ):
            raise ValueError(
                "Replicated experts require controlled compact multi-slot execution"
            )
        if type(split_assignments) is not bool or (
            split_assignments and not directory.expert_replicated
        ):
            raise ValueError("Assignment splitting requires resident Expert replicas")
        self.split_assignments = split_assignments
        self.dispatch_notifications: deque[DispatchPlan] = deque()
        self.replica_selections = 0
        self.receive_slots = receive_slots
        if not isinstance(batching, BatchingOptions):
            raise ValueError("Controller requires typed batching options")
        batching.validate_capacity(receive_slots, directory.workers[0].max_tokens)
        self.batching = batching
        self.workers = {
            key: WorkerReservation(
                slots=tuple(BufferReservation() for _ in range(receive_slots))
            )
            for key in self.directories
        }
        self.demand_aware = demand_aware
        self.compact_output = compact_output
        if type(collect_cost_feedback) is not bool or (
            collect_cost_feedback and (not demand_aware or receive_slots < 2)
        ):
            raise ValueError(
                "Execution cost feedback requires demand-aware multi-slot control"
            )
        self.collect_cost_feedback = collect_cost_feedback
        self.cost_books = (
            {key: ExecutionCostBook() for key in self.workers}
            if collect_cost_feedback
            else {}
        )
        # Retain membership through output sends, even after the compute lane
        # is reused. At most one ticket per occupied receive slot survives.
        self.execution_tickets: dict[int, tuple[ExecutionPlan, ...]] = {}
        self.active_batches: dict[str, BatchExecution | None] = dict.fromkeys(
            self.workers
        )
        self.batch_sequences = dict.fromkeys(self.workers, 0)
        self.batch_calls = dict.fromkeys(self.workers, 0)
        self.batch_tokens = dict.fromkeys(self.workers, 0)
        self.merged_batches = dict.fromkeys(self.workers, 0)
        self.active_parents: dict[CallKey, tuple[ExecutionPlan, ...]] = {}
        self.pending_child_plans: deque[tuple[ExecutionPlan, float]] = deque()
        # Validate immutable demand once per outstanding client. Disjoint
        # placement reuses that plan; replicated placement chooses live owners
        # on admission using authoritative slot and phase feedback.
        self.pending_demand_plans: dict[CallKey, DispatchPlan] = {}
        self.empty_parent_calls = 0
        self.planning_cpu_ns = 0
        self.selected_worker_calls = dict.fromkeys(self.workers, 0)
        self.skipped_worker_calls = dict.fromkeys(self.workers, 0)
        # Static placement bounds the accounting state independently of the
        # number of requests. These are admitted assignments, not completions.
        self.admitted_assignments = {
            worker.worker_id: {
                placement.layer_id: dict.fromkeys(placement.expert_ids, 0)
                for placement in worker.placements
            }
            for worker in directory.workers
        }

    def _plan_demand(self, request: CallRequest) -> DispatchPlan:
        started = time.perf_counter_ns()
        try:
            return plan_expert_demand(self.directory, request)
        finally:
            self.planning_cpu_ns += time.perf_counter_ns() - started

    def _record_owners(self, request: CallRequest, owners: tuple[str, ...]) -> None:
        for worker_id in self.directory.owners(request.layer_id):
            if worker_id in owners:
                self.selected_worker_calls[worker_id] += 1
            else:
                self.skipped_worker_calls[worker_id] += 1

    def submit(self, client_id: str, request: CallRequest, now_ns: int) -> None:
        if request.compact_output != self.compact_output:
            raise ValueError(
                "Output layout does not match the configured dispatch mode"
            )
        if (request.demand is not None) != self.demand_aware:
            raise ValueError(
                "Expert demand does not match the configured dispatch mode"
            )
        identity = self.clients.get(client_id)
        if (
            identity is None
            or client_id in self.closed
            or (
                request.key.client_id,
                request.key.session_epoch,
            )
            != (client_id, identity.session_epoch)
        ):
            raise ValueError("Wrong or closed controller session")
        if client_id in self.outstanding:
            raise ValueError("Only one outstanding call per client is supported")
        if request.key.call_seq <= self.last_sequence[client_id]:
            raise ValueError("Repeated or stale call sequence")
        for owner in self.directory.owners(request.layer_id):
            self.directories[owner].validate(request)
        if self.demand_aware:
            dispatch = self._plan_demand(request)
            if not dispatch.tasks:
                # No GPU transfer is posted. Consume the parent sequence and
                # complete it without touching any worker credit or generation.
                self.last_sequence[client_id] = request.key.call_seq
                self.completed_by_client[client_id] += 1
                self.empty_parent_calls += 1
                self._record_owners(request, ())
                return
            self.pending_demand_plans[request.key] = dispatch
        self.pending[identity.domain].append(QueuedCall(request, None, now_ns))
        self.outstanding[client_id] = request.key
        self.last_sequence[client_id] = request.key.call_seq
        self.peak_pending = max(self.peak_pending, self.pending_count)

    def grant(self, now_ns: int) -> tuple[ExecutionPlan, float] | None:
        if self.pending_child_plans:
            plan, queue_ms = self.pending_child_plans.popleft()
            self.workers[plan.worker_id].slots[plan.slot_id].phase = "reserved"
            return plan, queue_ms
        available = {
            worker_id
            for worker_id, worker in self.workers.items()
            if worker.available_slots
        }
        for _ in range(len(self.domains)):
            domain = self.domains[0]
            self.domains.rotate(-1)
            queue = self.pending[domain]
            for call in queue:
                dispatch = (
                    self.pending_demand_plans[call.request.key]
                    if self.demand_aware
                    else None
                )
                if self.directory.expert_replicated:
                    started = time.perf_counter_ns()
                    dispatch = select_replicas(
                        self.directory,
                        call.request,
                        self.worker_loads(),
                        self.replica_selections,
                        split_assignments=self.split_assignments,
                    )
                    self.planning_cpu_ns += time.perf_counter_ns() - started
                    if dispatch is None:
                        continue
                owners = (
                    dispatch.owners
                    if dispatch is not None
                    else self.directory.owners(call.request.layer_id)
                )
                if not all(owner in available for owner in owners):
                    continue
                queue.remove(call)
                if dispatch is not None:
                    del self.pending_demand_plans[call.request.key]
                if self.directory.expert_replicated:
                    assert dispatch is not None
                    self.dispatch_notifications.append(dispatch)
                    self.replica_selections += 1
                self._record_owners(call.request, owners)
                plans = []
                queue_ms = max(0, now_ns - call.enqueued_ns) / 1e6
                for owner in owners:
                    worker = self.workers[owner]
                    slot_id = next(
                        index
                        for index, slot in enumerate(worker.slots)
                        if slot.active is None
                    )
                    slot = worker.slots[slot_id]
                    slot.generation += 1
                    self.plan_sequence += 1
                    task = dispatch.task_for(owner) if dispatch is not None else None
                    plan = ExecutionPlan(
                        call.request,
                        owner,
                        self.plan_sequence,
                        slot_id,
                        slot.generation,
                        expert_ids=task.expert_ids if task is not None else (),
                        num_assignments=(
                            task.num_assignments if task is not None else None
                        ),
                        assignment_slices=(
                            task.assignment_slices if task is not None else ()
                        ),
                    )
                    if task is not None:
                        assert call.request.demand is not None
                        assigned = self.admitted_assignments[owner][
                            call.request.layer_id
                        ]
                        for expert_id in task.expert_ids:
                            assigned[expert_id] += task.assignments_for(
                                expert_id, call.request.demand
                            )
                    # Reserve every owner before publishing the first child.
                    slot.active = plan
                    slot.phase = "pending_grant"
                    plans.append(plan)
                    self.pending_child_plans.append((plan, queue_ms))
                self.active_parents[call.request.key] = tuple(plans)
                self.peak_reserved = max(
                    self.peak_reserved,
                    self.reserved_count,
                )
                return self.grant(now_ns)
        return None

    def start_batch(self, worker_id: str, batch: BatchExecution) -> None:
        """Validate every member before atomically occupying the compute lane."""
        if (
            not self.batching.enabled
            or worker_id not in self.workers
            or batch.worker_id != worker_id
            or self.active_batches[worker_id] is not None
            or batch.sequence != self.batch_sequences[worker_id] + 1
            or len(batch.members) > self.batching.max_calls
        ):
            raise ValueError("Invalid, stale or overlapping compute batch")
        worker = self.workers[worker_id]
        plans = []
        for member in batch.members:
            if member.slot_id >= len(worker.slots):
                raise ValueError("Batch references an absent receive slot")
            slot = worker.slots[member.slot_id]
            if (
                slot.phase != "input_ready"
                or slot.active is None
                or BatchSlot.from_plan(slot.active) != member
            ):
                raise ValueError("Batch member is stale or its input is not ready")
            plans.append(slot.active)
        tokens = sum(plan.request.num_tokens for plan in plans)
        if (
            len({compatibility_key(plan) for plan in plans}) != 1
            or len({plan.request.key.client_id for plan in plans}) != len(plans)
            or tokens > self.batching.max_tokens
            or any(slot.phase == "executing" for slot in worker.slots)
        ):
            raise ValueError("Batch exceeds capacity or mixes incompatible calls")
        for member in batch.members:
            worker.slots[member.slot_id].phase = "executing"
        self.active_batches[worker_id] = batch
        self.batch_sequences[worker_id] = batch.sequence
        self.batch_calls[worker_id] += len(plans)
        self.batch_tokens[worker_id] += tokens
        self.merged_batches[worker_id] += len(plans) > 1
        if self.collect_cost_feedback:
            for plan in plans:
                self.execution_tickets[plan.plan_id] = tuple(plans)

    def progress(
        self,
        worker_id: str,
        kind: str,
        plan: ExecutionPlan,
        metrics: dict[str, float] | None = None,
    ) -> None:
        worker = self.workers.get(worker_id)
        if (
            worker is None
            or plan.worker_id != worker_id
            or plan.slot_id >= len(worker.slots)
            or worker.slots[plan.slot_id].active != plan
        ):
            raise ValueError("Stale, duplicate or foreign worker feedback")
        slot = worker.slots[plan.slot_id]
        transitions = {
            "grant": ("reserved", "receiving"),
            "input_ready": ("receiving", "input_ready"),
            "executing": ("input_ready", "executing"),
            "output_ready": ("executing", "returning"),
            "done": ("returning", "held"),
        }
        if kind == "error":
            # Even a locally drained rejection can leave siblings in NCCL.
            # The supervisor must terminate all peers, without releasing credit.
            raise RuntimeError("Expert fan-out failed; terminate the deployment")
        if kind not in transitions or slot.phase != transitions[kind][0]:
            raise ValueError("Out-of-order worker progress")
        if kind == "executing" and (
            self.batching.enabled or any(s.phase == "executing" for s in worker.slots)
        ):
            raise ValueError("A worker has only one compute lane")
        if self.collect_cost_feedback:
            if kind == "executing":
                self.execution_tickets[plan.plan_id] = (plan,)
            elif kind == "done":
                self._record_execution_cost(plan, metrics)
        slot.phase = transitions[kind][1]
        if kind == "output_ready" and not any(
            s.phase == "executing" for s in worker.slots
        ):
            self.active_batches[worker_id] = None
        if kind != "done":
            return
        client_id = plan.request.key.client_id
        worker.completed += 1
        self.completed_by_worker_client[worker_id][client_id] += 1
        self.completed_by_worker_layer[worker_id][plan.request.layer_id] += 1
        plans = self.active_parents[plan.request.key]
        if not all(
            self.workers[child.worker_id].slots[child.slot_id].phase == "held"
            for child in plans
        ):
            return
        # Every child has drained its output send. The final reply may now let
        # the A submit the next layer without racing this parent's reservations.
        for child in plans:
            reserved = self.workers[child.worker_id].slots[child.slot_id]
            reserved.active = None
            reserved.phase = "idle"
        del self.active_parents[plan.request.key]
        del self.outstanding[client_id]
        self.completed_by_client[client_id] += 1

    def _record_execution_cost(
        self, plan: ExecutionPlan, metrics: dict[str, float] | None
    ) -> None:
        plans = self.execution_tickets[plan.plan_id]
        anchor = plan == plans[0]
        required = (
            "execution_cost_sample",
            "execution_batch_calls",
            "execution_batch_tokens",
            "compute_phase_cost_gpu_ms",
            "input_merge_phase_cost_gpu_ms",
            "execution_batch_pack_gpu_ms",
        )
        if metrics is None or any(key not in metrics for key in required):
            raise ValueError("Missing physical execution cost feedback")
        if (
            metrics["execution_cost_sample"] != int(anchor)
            or metrics["execution_batch_calls"] != len(plans)
            or metrics["execution_batch_tokens"]
            != sum(p.request.num_tokens for p in plans)
        ):
            raise ValueError("Execution cost feedback has wrong batch membership")
        timings = tuple(metrics[key] for key in required[3:])
        if anchor:
            self.cost_books[plan.worker_id].record(
                ExecutionShape.from_plans(plans), *timings
            )
        elif any(timings):
            raise ValueError("Physical execution cost must be charged exactly once")
        del self.execution_tickets[plan.plan_id]

    def worker_loads(self) -> dict[str, WorkerLoad]:
        return {
            key: WorkerLoad(
                worker.available_slots,
                sum(
                    slot.active.num_assignments or 0
                    for slot in worker.slots
                    if slot.active is not None
                    and slot.phase not in {"returning", "held"}
                ),
                any(slot.phase == "executing" for slot in worker.slots),
                sum(slot.active is not None for slot in worker.slots),
            )
            for key, worker in self.workers.items()
        }

    def snapshot(self) -> dict:
        return {
            **super().snapshot(),
            "policy": (
                "controller-expert-demand-gang"
                if self.demand_aware
                else "controller-expert-partitioned-gang"
            ),
            "demand_aware": self.demand_aware,
            "compact_output": self.compact_output,
            "receive_slots": self.receive_slots,
            "cost_feedback_enabled": self.collect_cost_feedback,
            "active_cost_tickets": len(self.execution_tickets),
            "execution_cost_by_worker": {
                key: book.snapshot() for key, book in self.cost_books.items()
            },
            "expert_replicated": self.directory.expert_replicated,
            "split_assignments": self.split_assignments,
            "replica_selections": self.replica_selections,
            "pending_dispatch_notifications": len(self.dispatch_notifications),
            "worker_loads": {
                key: asdict(value) for key, value in self.worker_loads().items()
            },
            "replica_load_scope": (
                "Authoritative slot/phase and assignment backlog; "
                "not hardware utilization or latency prediction"
            ),
            "batching": asdict(self.batching),
            "active_compute_batches": sum(
                b is not None for b in self.active_batches.values()
            ),
            "compute_batches_by_worker": dict(self.batch_sequences),
            "batched_calls_by_worker": dict(self.batch_calls),
            "batch_tokens_by_worker": dict(self.batch_tokens),
            "merged_batches_by_worker": dict(self.merged_batches),
            "empty_parent_calls": self.empty_parent_calls,
            "planning_cpu_ms_total": self.planning_cpu_ns / 1e6,
            "selected_worker_calls": dict(self.selected_worker_calls),
            "skipped_worker_calls": dict(self.skipped_worker_calls),
            "admitted_selected_worker_calls": sum(self.selected_worker_calls.values()),
            "admitted_skipped_worker_calls": sum(self.skipped_worker_calls.values()),
            "admitted_assignments_by_worker_layer_expert": {
                worker: {
                    str(layer): {str(expert): count for expert, count in counts.items()}
                    for layer, counts in layers.items()
                }
                for worker, layers in self.admitted_assignments.items()
            },
            "active_parent_calls": len(self.active_parents),
            "pending_child_plans": len(self.pending_child_plans),
            "pending_demand_plans": len(self.pending_demand_plans),
            "completed_parent_calls": sum(self.completed_by_client.values()),
            "completed_child_calls": sum(
                worker.completed for worker in self.workers.values()
            ),
        }
