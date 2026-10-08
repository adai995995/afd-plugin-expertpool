# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local full-E admission around the shared asynchronous GPU pipeline."""

import time
from collections import Counter
from multiprocessing.connection import wait
from typing import TYPE_CHECKING

import torch

from afd_plugin.expert_pool.batch_audit import MAX_AUDIT_RECORDS, BatchAudit
from afd_plugin.expert_pool.controller import directory_digest
from afd_plugin.expert_pool.pipeline_worker import PipelineSlot, WorkerPipeline
from afd_plugin.expert_pool.progress_wait import ProgressWaiter
from afd_plugin.expert_pool.protocol import (
    BatchExecution,
    BatchSlot,
    ExecutionPlan,
    Message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.scheduler import LocalSlotScheduler

if TYPE_CHECKING:
    from afd_plugin.expert_pool.worker import ExpertWorker


class LocalWorkerPipeline(WorkerPipeline):
    def __init__(self, worker: "ExpertWorker", receive_slots: int) -> None:
        super().__init__(worker, receive_slots)
        if not worker.local_full_pipeline or worker.batching.max_wait_us:
            raise ValueError(
                "Local pipeline requires complete E coverage and zero wait"
            )
        domains = Counter(peer.domain for peer in worker.peers.values())
        self.book = LocalSlotScheduler(
            worker.directory, receive_slots, max(domains.values())
        )
        for peer in worker.peers.values():
            self.book.register(peer.client_id, peer.session_epoch, peer.domain)
        self.audit = (
            BatchAudit(MAX_AUDIT_RECORDS)
            if worker.execution.audit_batch_members
            else None
        )

    def _validate_plan(self, plan: ExecutionPlan, slot: PipelineSlot) -> None:
        if (
            self.book.active_slots.get(plan.slot_id) != plan
            or plan.generation != slot.generation + 1
        ):
            raise ValueError("Plan does not own this local slot generation")

    def _notify(self, message: Message) -> None:
        if message.kind in {"batch_executing", "executing"}:
            if self.audit is not None:
                plans = tuple(slot.plan for slot in self.computing)
                assert all(plan is not None for plan in plans)
                batch = message.batch or BatchExecution(
                    self.worker.directory.worker_id,
                    self.batch_executions,
                    tuple(BatchSlot.from_plan(plan) for plan in plans),
                )
                self.audit.record_batch(batch, plans)
            return
        if message.kind not in {"grant", "output_ready", "done"}:
            return
        assert message.plan is not None
        if message.kind == "done":
            # Called only after output transfer.finish(); buffer reuse before
            # this point would corrupt a different A's pending receive.
            self.book.complete(message.plan)
            if self.audit is not None:
                self.audit.record_done(message.plan)
        peer = self.worker.peers[message.plan.request.key.client_id]
        send_message(peer.control, message)

    @torch.inference_mode()
    def run(self) -> None:
        connections = {peer.control: peer for peer in self.worker.peers.values()}
        for peer in connections.values():
            send_message(
                peer.control,
                Message(
                    "ready",
                    detail=directory_digest(self.worker.directory),
                    metrics={
                        "receive_slots": len(self.slots),
                        "startup_complete": int(self.worker.startup["completed"]),
                    },
                ),
            )
        while connections:
            # Observe every ingress event before choosing an execution. There
            # is no intentional collection wait or cross-A layer barrier.
            progressed = self._receive_completions()
            progressed = self._compute_completion() or progressed
            progressed = self._send_completions() or progressed
            for connection in wait(
                list(connections), timeout=0 if self.book.outstanding else None
            ):
                peer = connections[connection]
                message = receive_message(connection, 0)
                if message.kind == "close":
                    if peer.client_id in self.book.outstanding:
                        raise RuntimeError("Client closed with undrained work")
                    send_message(connection, Message("closed"))
                    del connections[connection]
                    self.progress_waiter = ProgressWaiter(tuple(connections))
                elif message.kind == "submit" and message.request is not None:
                    request = message.request
                    try:
                        if (request.key.client_id, request.key.session_epoch) != (
                            peer.client_id,
                            peer.session_epoch,
                        ):
                            raise ValueError("Call does not belong to this endpoint")
                        self.book.submit(request, time.perf_counter_ns())
                    except ValueError as error:
                        send_message(
                            connection,
                            Message("error", request=request, detail=str(error)),
                        )
                else:
                    raise RuntimeError("Local worker expected submit or close")
                progressed = True
            # Fixed membership and one message per ready endpoint bound the
            # host control work. Admission does not wait for the compute lane.
            while (plan := self.book.grant()) is not None:
                queue_ms = (
                    time.perf_counter_ns() - self.book.enqueued_ns[plan.slot_id]
                ) / 1e6
                self._accept(
                    Message(
                        "grant", plan=plan, metrics={"controller_queue_ms": queue_ms}
                    )
                )
                progressed = True
            progressed = self._receive_completions() or progressed
            progressed = self._start_compute() or progressed
            if connections and not progressed:
                self.progress_waiter.wait(self.next_wait_s)

    def snapshot(self) -> dict:
        return {
            **super().snapshot(),
            "local_full_e": True,
            "output_layout": "reduced-token-hidden",
            "admission": "local-domain-round-robin",
            "pending_calls": len(self.book.outstanding) - len(self.book.active_slots),
            "batch_audit": self.audit.snapshot()
            if self.audit is not None
            else {"enabled": False},
        }
