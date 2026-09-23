# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Direct A/E endpoint loop; GPU execution is shared with the legacy pipeline."""

from dataclasses import replace
from multiprocessing.connection import wait
from typing import TYPE_CHECKING

import torch

from afd_plugin.expert_pool.controller import ControllerClientIdentity, directory_digest
from afd_plugin.expert_pool.direct_dispatch import DirectSlots
from afd_plugin.expert_pool.pipeline_worker import WorkerPipeline
from afd_plugin.expert_pool.progress_wait import ProgressWaiter
from afd_plugin.expert_pool.protocol import Message, receive_message, send_message

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
        self.progress_waiter = ProgressWaiter(
            tuple(p.control for p in worker.peers.values())
        )

    def _notify(self, message: Message) -> None:
        # Input progress and batch execution remain local state. No central
        # grant, stage notification or completion acknowledgement is required.
        if message.kind not in {"output_ready", "done"}:
            return
        assert message.plan is not None
        if message.kind == "done":
            # The inherited pipeline calls this only after send.finish().
            self.book.complete(message.plan)
            message = replace(
                message,
                metrics={
                    **message.metrics,
                    "direct_pending_assignments": sum(
                        p.num_assignments
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
            progressed = self._start_compute() or progressed
            if connections and not progressed:
                self.progress_waiter.wait(self.next_wait_s)

    def snapshot(self) -> dict:
        return {
            **super().snapshot(),
            "direct_dispatch": True,
            "packed_input": self.worker.packed_input,
            "direct_active": len(self.book.active),
            "direct_pending": len(self.book.pending),
            "direct_closed_clients": len(self.book.closed),
            "direct_completed_by_client": dict(self.book.completed),
            "direct_slot_clients": list(self.book.clients),
        }
