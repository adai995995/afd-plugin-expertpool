# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded endpoint state for dispatch without a central per-call reservation."""

from dataclasses import dataclass

from afd_plugin.expert_pool.controller import ControllerClientIdentity
from afd_plugin.expert_pool.protocol import ExecutionPlan, Message
from afd_plugin.expert_pool.scheduler import StaticDirectory


class DirectSlots:
    """One immutable GPU slot per client, plus at most one successor header.

    complete() is called only after the worker observes its GPU send event.
    Receiving a newer header never releases the previous GPU allocation.
    """

    def __init__(
        self, directory: StaticDirectory, clients: tuple[ControllerClientIdentity, ...]
    ) -> None:
        if not clients or len({c.client_id for c in clients}) != len(clients):
            raise ValueError("Direct slots require unique registered clients")
        self.directory = directory
        self.clients = {
            c.client_id: c for c in sorted(clients, key=lambda c: c.client_id)
        }
        self.slot_ids = {client: index for index, client in enumerate(self.clients)}
        self.generations = dict.fromkeys(self.clients, 0)
        self.sequences = dict.fromkeys(self.clients, -1)
        self.active: dict[str, ExecutionPlan] = {}
        self.pending: dict[str, ExecutionPlan] = {}
        self.closed: set[str] = set()
        self.completed = dict.fromkeys(self.clients, 0)

    def submit(self, client_id: str, plan: ExecutionPlan) -> None:
        self.directory.validate(plan.request)
        client = self.clients.get(client_id)
        key = plan.request.key
        if (
            client is None
            or client_id in self.closed
            or (key.client_id, key.session_epoch) != (client_id, client.session_epoch)
            or plan.worker_id != self.directory.worker_id
            or plan.slot_id != self.slot_ids[client_id]
            or plan.generation != self.generations[client_id] + 1
            or key.call_seq <= self.sequences[client_id]
            or client_id in self.pending
        ):
            raise ValueError("Mismatched direct identity, generation or capacity")
        self.pending[client_id] = plan
        self.generations[client_id] = plan.generation
        self.sequences[client_id] = key.call_seq

    def startable(self) -> tuple[ExecutionPlan, ...]:
        plans = []
        for client_id in self.clients:
            if client_id in self.pending and client_id not in self.active:
                plan = self.pending.pop(client_id)
                self.active[client_id] = plan
                plans.append(plan)
        return tuple(plans)

    def complete(self, plan: ExecutionPlan) -> None:
        client_id = plan.request.key.client_id
        if self.active.get(client_id) != plan:
            raise ValueError("Completion cannot release another direct generation")
        del self.active[client_id]
        self.completed[client_id] += 1

    def close(self, client_id: str) -> None:
        if (
            client_id not in self.clients
            or client_id in self.closed
            or client_id in self.active
            or client_id in self.pending
        ):
            raise ValueError("Direct endpoint must drain before close")
        self.closed.add(client_id)


@dataclass
class DirectResult:
    plan: ExecutionPlan
    ready: bool = False
    received: bool = False
    done: bool = False


class DirectReplies:
    """Result availability and asynchronous accounting are separate events."""

    def __init__(self) -> None:
        self.pending: dict[tuple[str, int], DirectResult] = {}
        self.generations: dict[str, int] = {}

    def register(self, plan: ExecutionPlan) -> None:
        owner = plan.worker_id
        previous = [s for (w, _), s in self.pending.items() if w == owner]
        if (
            plan.generation != self.generations.get(owner, 0) + 1
            or any(not state.received for state in previous)
            or len(previous) >= 2
        ):
            raise ValueError("Direct caller exceeded its result/feedback capacity")
        self.generations[owner] = plan.generation
        self.pending[(owner, plan.generation)] = DirectResult(plan)

    def accept(self, owner: str, message: Message) -> None:
        plan = message.plan
        if plan is None or plan.worker_id != owner:
            raise RuntimeError("Direct reply came from a different endpoint")
        state = self.pending.get((owner, plan.generation))
        if state is None or state.plan != plan:
            raise RuntimeError("Direct reply has an unknown call or generation")
        if message.kind == "output_ready" and not state.ready:
            state.ready = True
        elif message.kind == "done" and state.ready and not state.done:
            state.done = True
        else:
            raise RuntimeError("Duplicate or out-of-order direct reply")
        self._reap(state)

    def is_ready(self, plan: ExecutionPlan) -> bool:
        return self.pending[(plan.worker_id, plan.generation)].ready

    def received(self, plan: ExecutionPlan) -> None:
        state = self.pending[(plan.worker_id, plan.generation)]
        if state.plan != plan or not state.ready or state.received:
            raise RuntimeError("Result receipt does not match a ready direct plan")
        state.received = True
        self._reap(state)

    def _reap(self, state: DirectResult) -> None:
        if state.received and state.done:
            del self.pending[(state.plan.worker_id, state.plan.generation)]
