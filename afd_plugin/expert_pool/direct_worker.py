# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Direct A/E endpoint loop; GPU execution is shared with the legacy pipeline."""

from dataclasses import replace
from multiprocessing.connection import wait
from typing import TYPE_CHECKING

import torch

from afd_plugin.expert_pool.batch_audit import MAX_AUDIT_RECORDS, BatchAudit
from afd_plugin.expert_pool.controller import ControllerClientIdentity, directory_digest
from afd_plugin.expert_pool.direct_dispatch import DirectSlots
from afd_plugin.expert_pool.pipeline_worker import WorkerPipeline
from afd_plugin.expert_pool.progress_wait import ProgressWaiter
from afd_plugin.expert_pool.protocol import (
    BatchExecution,
    BatchSlot,
    Message,
    receive_message,
    send_message,
)

if TYPE_CHECKING:
    from afd_plugin.expert_pool.worker import ExpertWorker


class DirectWorkerPipeline(WorkerPipeline):
    def __init__(self, worker: "ExpertWorker", receive_slots: int) -> None:
        super().__init__(worker, receive_slots)
        identities = tuple(
            ControllerClientIdentity(p.client_id, p.session_epoch, p.domain)
            for p in worker.peers.values()
        )
        self.book = DirectSlots(worker.directory, identities)
        self.audit = (
            BatchAudit(MAX_AUDIT_RECORDS)
            if (worker.full_e_direct or worker.partial_reduction)
            and worker.execution.audit_batch_members
            else None
        )
        self.progress_waiter = ProgressWaiter(
            tuple(p.control for p in worker.peers.values())
        )

    def _notify(self, message: Message) -> None:
        # Input progress and batch execution remain local state. No central
        # grant, stage notification or completion acknowledgement is required.
        if message.kind in {"batch_executing", "executing"} and self.audit is not None:
            plans = tuple(slot.plan for slot in self.computing)
            assert all(plan is not None for plan in plans)
            batch = message.batch or BatchExecution(
                self.worker.directory.worker_id,
                self.batch_executions,
                tuple(BatchSlot.from_plan(plan) for plan in plans),
            )
            self.audit.record_batch(batch, plans)
            for slot in self.computing:
                self.audit.record_ready_timing(
                    slot.plan,
                    slot.started_ns,
                    slot.ready_ns,
                    slot.compute_submitted_ns,
                )
        if message.kind not in {"output_ready", "done"}:
            return
        assert message.plan is not None
        if message.kind == "done":
            # The inherited pipeline calls this only after send.finish().
            self.book.complete(message.plan)
            if self.audit is not None:
                self.audit.record_done(message.plan, message.metrics)
            message = replace(
                message,
                metrics={
                    **message.metrics,
                    "direct_pending_assignments": sum(
                        p.num_assignments
                        if p.num_assignments is not None
                        else p.request.num_tokens * p.request.top_k
                        for p in (
                            *self.book.active.values(),
                            *self.book.pending.values(),
                        )
                    ),
                    "direct_occupied_slots": len(self.book.active),
                    "direct_computing": int(bool(self.computing)),
                },
            )
        peer = self.worker.peers[message.plan.request.key.client_id]
        send_message(peer.control, message)

    @torch.inference_mode()
    def run(self) -> None:
        connections = {p.control: p for p in self.worker.peers.values()}
        for peer in connections.values():
            send_message(
                peer.control,
                Message(
                    "ready",
                    detail=directory_digest(self.worker.directory),
                    metrics={
                        "slot_id": self.book.slot_ids[peer.client_id],
                        "receive_slots": len(self.slots),
                        "startup_complete": int(self.worker.startup["completed"]),
                        "packed_input": int(self.worker.packed_input),
                        "full_e_direct": int(self.worker.full_e_direct),
                        "partial_reduction": int(self.worker.partial_reduction),
                    },
                ),
            )
        while connections:
            progressed = (
                self._start_compute() if self.collecting_since_ns is not None else False
            )
            progressed = self._receive_completions() or progressed
            progressed = self._compute_completion() or progressed
            progressed = self._send_completions() or progressed
            active = bool(self.book.active or self.book.pending)
            for connection in wait(list(connections), timeout=0 if active else None):
                peer = connections[connection]
                message = receive_message(connection, 0)
                if message.kind == "close":
                    self.book.close(peer.client_id)
                    send_message(connection, Message("closed"))
                    del connections[connection]
                    self.progress_waiter = ProgressWaiter(tuple(connections))
                elif message.kind == "execute" and message.plan is not None:
                    self.book.submit(peer.client_id, message.plan)
                else:
                    raise RuntimeError("Direct worker expected execute or close")
                progressed = True
            for plan in self.book.startable():
                # Reuse the exact receive/compute/packing implementation; its
                # grant notification is suppressed by _notify above.
                self._accept(
                    Message("grant", plan=plan, metrics={"controller_queue_ms": 0.0})
                )
                progressed = True
            # Observe all newly posted receives before choosing the zero-wait
            # batch. No cross-A layer barrier or collection delay is added.
            progressed = self._receive_completions() or progressed
            progressed = self._start_compute() or progressed
            if connections and not progressed:
                self.progress_waiter.wait(self.next_wait_s)

    def snapshot(self) -> dict:
        return {
            **super().snapshot(),
            "direct_dispatch": self.worker.direct_dispatch,
            "full_e_direct": self.worker.full_e_direct,
            "partial_reduction": self.worker.partial_reduction,
            "output_layout": (
                "fp32-partial-token-hidden"
                if self.worker.partial_reduction
                else "reduced-token-hidden"
                if self.worker.full_e_direct
                else "compact-assignments"
            ),
            "packed_input": self.worker.packed_input,
            "direct_active": len(self.book.active),
            "direct_pending": len(self.book.pending),
            "direct_closed_clients": len(self.book.closed),
            "direct_completed_by_client": dict(self.book.completed),
            "direct_slot_clients": list(self.book.clients),
            "batch_audit": self.audit.snapshot()
            if self.audit is not None
            else {"enabled": False},
        }
