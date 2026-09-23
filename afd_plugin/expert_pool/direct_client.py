# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local physical routing with direct endpoint replies and deferred accounting."""

import time
from contextlib import ExitStack
from multiprocessing.connection import wait

import torch

from afd_plugin.expert_pool.client import PoolClient
from afd_plugin.expert_pool.compact_input import (
    CompactInputWorkspace,
    should_pack_task,
)
from afd_plugin.expert_pool.compact_output import selection_ownership
from afd_plugin.expert_pool.controller import directory_digest
from afd_plugin.expert_pool.direct_dispatch import DirectReplies
from afd_plugin.expert_pool.fanout_client import FanoutPoolClient
from afd_plugin.expert_pool.fanout_protocol import FanoutReplies
from afd_plugin.expert_pool.metrics import CallMetrics
from afd_plugin.expert_pool.protocol import (
    CallRequest,
    DispatchPlan,
    ExecutionPlan,
    Message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.replica_client import ReplicaPoolClient
from afd_plugin.expert_pool.replica_dispatch import WorkerLoad, select_replicas


class DirectFanoutPoolClient(FanoutPoolClient):
    def __init__(
        self,
        channels: tuple[PoolClient, ...],
        *,
        client_slot: int,
        receive_slots: int,
        expert_replicated: bool = False,
        packed_input: bool = False,
    ) -> None:
        super().__init__(
            channels,
            None,
            demand_aware=True,
            compact_output=True,
            receive_slots=receive_slots,
            expert_replicated=expert_replicated,
        )
        if not 0 <= client_slot < receive_slots:
            raise ValueError("Direct client requires its own fixed slot")
        if type(packed_input) is not bool:
            raise ValueError("Packed input must be an explicit boolean")
        self.client_slot = client_slot
        self.packed_input = packed_input
        self.input_workspaces: dict[str, CompactInputWorkspace] = {}
        self.input_transfer = {"sent_bytes": 0, "dense_equivalent_bytes": 0}
        self.packed_tasks = 0
        self.dense_tasks = 0
        self.events = DirectReplies()
        self.loads = {w: WorkerLoad(1, 0, False, 0) for w in self.channels}
        self.generations = dict.fromkeys(self.channels, 0)
        self.feedback_messages = 0
        self.connections = {c.control: w for w, c in self.channels.items()}
        for channel in self.channels.values():
            message = receive_message(channel.control, self.timeout_s)
            if (
                message.kind != "ready"
                or message.detail != directory_digest(channel.directory)
                or message.metrics.get("slot_id") != client_slot
                or message.metrics.get("receive_slots") != receive_slots
                or bool(message.metrics.get("packed_input")) != packed_input
            ):
                raise RuntimeError(
                    "Direct worker did not bind the expected slot/directory"
                )

    def _drain(self, *, block: bool = False) -> None:
        ready = wait(list(self.connections), timeout=self.timeout_s if block else 0)
        if block and not ready:
            raise TimeoutError("Direct endpoint reply timed out")
        for connection in ready:
            while connection.poll(0):
                owner = self.connections[connection]
                message = receive_message(connection, 0)
                self.events.accept(owner, message)
                if message.kind == "done":
                    self.feedback_messages += 1
                    self.loads[owner] = WorkerLoad(
                        1,
                        int(message.metrics["direct_pending_assignments"]),
                        bool(message.metrics["direct_computing"]),
                        int(message.metrics["direct_occupied_slots"]),
                    )
                    channel = self.channels[owner]
                    if channel.metrics is not None:
                        plan = message.plan
                        channel.metrics.record(
                            plan.request.layer_id,
                            plan.request.num_tokens,
                            message.metrics,
                        )

    def _drain_all(self) -> None:
        while self.events.pending:
            self._drain(block=True)

    def _submit(self, request: CallRequest) -> None:
        self._drain()

    def _empty_reply(self, request: CallRequest) -> Message:
        return Message("empty_done", request=request)

    def _admit(
        self,
        request: CallRequest,
        dispatch_plan: DispatchPlan | None,
        owners: tuple[str, ...],
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        tuple[str, ...],
        FanoutReplies,
        dict[str, Message],
        dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        if self.directory.expert_replicated:
            dispatch_plan = select_replicas(
                self.directory, request, self.loads, self.sequence + self.client_slot
            )
            assert dispatch_plan is not None
        # A feedback sample is a one-decision hint, never persistent authority
        # over another client's slot or a reason to wait for a new sample.
        self.loads = {w: WorkerLoad(1, 0, False, 0) for w in self.channels}
        assert dispatch_plan is not None
        owners = dispatch_plan.owners
        self.last_dispatch = dispatch_plan
        pack_started = time.perf_counter_ns()
        payloads = {}
        for owner in owners:
            task = dispatch_plan.task_for(owner)
            # Unique selected rows cannot exceed selected assignments. Avoid
            # a per-owner GPU→CPU shape wait unless this bound alone proves
            # at least a twofold input-row reduction.
            should_pack = (
                self.packed_input
                and should_pack_task(task.num_assignments, request.num_tokens)
            )
            if should_pack:
                if owner not in self.input_workspaces:
                    channel = self.channels[owner]
                    self.input_workspaces[owner] = CompactInputWorkspace(
                        channel.directory.max_tokens,
                        channel.directory.hidden_size,
                        channel.directory.top_k,
                        channel.transport.device,
                    )
                ownership = (
                    selection_ownership(
                        task.expert_ids,
                        self.layer_expert_counts[request.layer_id],
                        inputs[0].device,
                    )
                    if self.directory.expert_replicated
                    else self.ownership[(owner, request.layer_id)]
                )
                payloads[owner] = self.input_workspaces[owner].pack(
                    *inputs, ownership, num_rows=task.num_assignments
                )
                self.packed_tasks += 1
            else:
                payloads[owner] = inputs
                self.dense_tasks += 1
            self.input_transfer["sent_bytes"] += sum(
                tensor.numel() * tensor.element_size() for tensor in payloads[owner]
            )
            self.input_transfer["dense_equivalent_bytes"] += sum(
                tensor.numel() * tensor.element_size() for tensor in inputs
            )
        self.input_pack_ms = (
            (time.perf_counter_ns() - pack_started) / 1e6
            if self.packed_input
            else 0.0
        )
        replies = FanoutReplies(
            request,
            owners,
            dispatch_plan=dispatch_plan,
            receive_slots=self.receive_slots,
        )
        grants = {}
        for owner in owners:
            task = dispatch_plan.task_for(owner)
            self.generations[owner] += 1
            plan = ExecutionPlan(
                request=request,
                worker_id=owner,
                plan_id=(request.key.call_seq * self.receive_slots + self.client_slot)
                * len(self.channels)
                + tuple(self.channels).index(owner),
                slot_id=self.client_slot,
                generation=self.generations[owner],
                expert_ids=task.expert_ids,
                num_assignments=task.num_assignments,
                input_rows=payloads[owner][0].shape[0] if self.packed_input else None,
            )
            self.events.register(plan)
            # Local plan metadata reuses the existing result assembly contract.
            # This is not an E acknowledgement and no grant is sent on the wire.
            grants[owner] = Message("grant", plan=plan)
            replies.accept(grants[owner])
            replies.take("grant", owner)
            send_message(self.channels[owner].control, Message("execute", plan=plan))
        return owners, replies, grants, payloads

    def _ready_output_owner(self, replies: FanoutReplies, remaining: list[str]) -> str:
        while True:
            for owner in remaining:
                if self.events.is_ready(replies.plans[owner]):
                    return owner
            self._drain(block=True)

    def _output_received(self, plan: ExecutionPlan) -> None:
        self.events.received(plan)

    def _finish_replies(
        self, replies: FanoutReplies, owners: tuple[str, ...]
    ) -> dict[str, Message]:
        self._drain()  # Accounting can finish after the current layer returns.
        return {}

    def _dispatch_status(self) -> dict:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Drain the model call before reading direct status")
        try:
            self._drain_all()
            return ReplicaPoolClient.dispatch_status(self)
        finally:
            self.lock.release()

    def dispatch_status(self) -> dict:
        result = super().dispatch_status()
        result.update(
            policy="direct-local-expert-dispatch",
            direct_dispatch=True,
            packed_input=self.packed_input,
            input_transfer=dict(self.input_transfer),
            packed_tasks=self.packed_tasks,
            dense_tasks=self.dense_tasks,
            feedback_messages=self.feedback_messages,
            pending_feedback=len(self.events.pending),
            controller_roundtrips=0,
        )
        return result

    def set_metrics(self, enabled: bool) -> None:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Drain the model call before resetting direct metrics")
        try:
            self._drain_all()
            self.metrics = CallMetrics() if enabled else None
            for channel in self.channels.values():
                channel.set_metrics(enabled)
        finally:
            self.lock.release()

    def close(self) -> None:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Drain the model call before closing direct endpoints")
        try:
            if self.closed:
                return
            if not self.failed:
                self._drain_all()
                for channel in self.channels.values():
                    send_message(channel.control, Message("close"))
                for channel in self.channels.values():
                    if (
                        receive_message(channel.control, self.timeout_s).kind
                        != "closed"
                    ):
                        raise RuntimeError("Unexpected direct close reply")
            with ExitStack() as cleanup:
                for channel in self.channels.values():
                    cleanup.callback(channel.control.close)
                    cleanup.callback(channel.transport.close)
                    channel.closed = True
            self.closed = True
        finally:
            self.lock.release()
