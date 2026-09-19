# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Atomic multi-worker reservations for static, disjoint expert partitions.

Every owner receives the complete input batch. A worker's completed child
remains reserved until the whole parent drains. Dispatch errors terminate the
deployment; uncertain GPU work is never retried or recycled.
"""

from collections import deque

from afd_plugin.expert_pool.controller import (
    ControllerClientIdentity,
    ControllerLedger,
    QueuedCall,
)
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.protocol import CallKey, CallRequest, ExecutionPlan


class FanoutControllerLedger(ControllerLedger):
    def __init__(
        self,
        directory: PoolDirectory,
        clients: tuple[ControllerClientIdentity, ...],
        scheduling_policy: str = "ready_first",
    ) -> None:
        if not directory.expert_partitioned or scheduling_policy != "ready_first":
            raise ValueError("Expert fan-out requires partitioned ready-first control")
        self._initialize(directory, clients, scheduling_policy)
        self.active_parents: dict[CallKey, tuple[ExecutionPlan, ...]] = {}
        self.pending_child_plans: deque[tuple[ExecutionPlan, float]] = deque()

    def submit(self, client_id: str, request: CallRequest, now_ns: int) -> None:
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
        self.pending[identity.domain].append(QueuedCall(request, None, now_ns))
        self.outstanding[client_id] = request.key
        self.last_sequence[client_id] = request.key.call_seq
        self.peak_pending = max(self.peak_pending, self.pending_count)

    def grant(self, now_ns: int) -> tuple[ExecutionPlan, float] | None:
        if self.pending_child_plans:
            plan, queue_ms = self.pending_child_plans.popleft()
            self.workers[plan.worker_id].phase = "reserved"
            return plan, queue_ms
        available = {
            worker_id
            for worker_id, worker in self.workers.items()
            if worker.ready and worker.active is None
        }
        for _ in range(len(self.domains)):
            domain = self.domains[0]
            self.domains.rotate(-1)
            queue = self.pending[domain]
            for call in queue:
                owners = self.directory.owners(call.request.layer_id)
                if not all(owner in available for owner in owners):
                    continue
                queue.remove(call)
                plans = []
                queue_ms = max(0, now_ns - call.enqueued_ns) / 1e6
                for owner in owners:
                    worker = self.workers[owner]
                    worker.generation += 1
                    self.plan_sequence += 1
                    plan = ExecutionPlan(
                        call.request, owner, self.plan_sequence, 0, worker.generation
                    )
                    # Reserve every owner before publishing the first child.
                    worker.active = plan
                    worker.phase = "pending_grant"
                    plans.append(plan)
                    self.pending_child_plans.append((plan, queue_ms))
                self.active_parents[call.request.key] = tuple(plans)
                self.peak_reserved = max(
                    self.peak_reserved,
                    sum(worker.active is not None for worker in self.workers.values()),
                )
                return self.grant(now_ns)
        return None

    def progress(self, worker_id: str, kind: str, plan: ExecutionPlan) -> None:
        worker = self.workers.get(worker_id)
        if worker is None or plan.worker_id != worker_id or worker.active != plan:
            raise ValueError("Stale, duplicate or foreign worker feedback")
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
        if kind not in transitions or worker.phase != transitions[kind][0]:
            raise ValueError("Out-of-order worker progress")
        worker.phase = transitions[kind][1]
        if kind != "done":
            return
        client_id = plan.request.key.client_id
        worker.completed += 1
        self.completed_by_worker_client[worker_id][client_id] += 1
        self.completed_by_worker_layer[worker_id][plan.request.layer_id] += 1
        plans = self.active_parents[plan.request.key]
        if not all(self.workers[child.worker_id].phase == "held" for child in plans):
            return
        # Every child has drained its output send. The final reply may now let
        # the A submit the next layer without racing this parent's reservations.
        for child in plans:
            reserved = self.workers[child.worker_id]
            reserved.active = None
            reserved.phase = "idle"
        del self.active_parents[plan.request.key]
        del self.outstanding[client_id]
        self.completed_by_client[client_id] += 1

    def snapshot(self) -> dict:
        return {
            **super().snapshot(),
            "policy": "controller-expert-partitioned-gang",
            "active_parent_calls": len(self.active_parents),
            "pending_child_plans": len(self.pending_child_plans),
            "completed_parent_calls": sum(self.completed_by_client.values()),
            "completed_child_calls": sum(
                worker.completed for worker in self.workers.values()
            ),
        }
