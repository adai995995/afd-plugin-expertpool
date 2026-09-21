# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded input/output slots around one serialized Expert compute stream.

The controller owns admission. CUDA events own transfer/compute completion.
Each live slot belongs to a different A client, whose NCCL channel has at most
one posted transfer. No executor workspace is used by two compute streams.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from typing import TYPE_CHECKING

import torch

from afd_plugin.connectors.gpu.pool import PoolTransfer
from afd_plugin.expert_pool.compact_output import CompactOutputWorkspace
from afd_plugin.expert_pool.controller import directory_digest
from afd_plugin.expert_pool.protocol import (
    ExecutionPlan,
    Message,
    receive_message,
    send_message,
)

if TYPE_CHECKING:
    from afd_plugin.expert_pool.worker import ExpertWorker

PROGRESS_POLL_S = 0.0002


@dataclass
class PipelineSlot:
    hidden: torch.Tensor
    weights: torch.Tensor
    ids: torch.Tensor
    workspace: CompactOutputWorkspace
    begin: torch.cuda.Event
    compute_end: torch.cuda.Event
    packed_end: torch.cuda.Event
    input_ready: torch.cuda.Event
    generation: int = 0
    plan: ExecutionPlan | None = None
    phase: str = "idle"
    transfer: PoolTransfer | None = None
    outgoing: torch.Tensor | None = None
    result: torch.Tensor | None = None
    started_ns: int = 0
    ready_ns: int = 0
    compute_submitted_ns: int = 0
    queue_ms: float = 0.0
    receive_ms: float = 0.0
    completed: int = 0


class WorkerPipeline:
    def __init__(self, worker: ExpertWorker, receive_slots: int) -> None:
        if worker.controller is None or worker.output_workspace is None:
            raise ValueError("Pipeline requires controlled compact Expert partitions")
        if (
            worker.audit_inputs
            or worker.profiler is not None
            or any(executor.validate_values for executor in worker.executors.values())
        ):
            raise ValueError(
                "Pipeline requires trusted routes and no per-call auditor/profiler"
            )
        self.worker = worker
        self.control = worker.controller
        self.compute_stream = torch.cuda.Stream(device=worker.device)
        self.slots = []
        for index in range(receive_slots):
            self.slots.append(
                PipelineSlot(
                    worker.hidden if index == 0 else torch.empty_like(worker.hidden),
                    worker.weights if index == 0 else torch.empty_like(worker.weights),
                    worker.ids if index == 0 else torch.empty_like(worker.ids),
                    worker.output_workspace
                    if index == 0
                    else CompactOutputWorkspace(
                        worker.directory.max_tokens * worker.directory.top_k,
                        worker.directory.hidden_size,
                        worker.device,
                    ),
                    *(torch.cuda.Event(enable_timing=True) for _ in range(3)),
                    torch.cuda.Event(),
                )
            )
        initialized = torch.cuda.Event()
        initialized.record(torch.cuda.current_stream(worker.device))
        self.compute_stream.wait_event(initialized)
        self.computing: PipelineSlot | None = None
        self.peak_occupied = 0
        self.peak_ready = 0
        self.receives_during_compute = 0
        self.ready_during_compute = 0
        self.compute_during_receive = 0

    def _accept(self, message: Message) -> None:
        plan = message.plan
        if message.kind != "grant" or plan is None or plan.slot_id >= len(self.slots):
            raise RuntimeError("Expected a reserved pipeline slot")
        slot = self.slots[plan.slot_id]
        if slot.plan is not None or any(
            other.plan is not None
            and other.plan.request.key.client_id == plan.request.key.client_id
            for other in self.slots
        ):
            raise RuntimeError("Slot or client channel is still occupied")
        self.worker.validate_controlled_plan(
            plan, slot.generation + 1, receive_slots=len(self.slots)
        )
        slot.plan = plan
        slot.generation = plan.generation
        slot.phase = "receiving"
        slot.queue_ms = message.metrics["controller_queue_ms"]
        slot.started_ns = time.perf_counter_ns()
        self.peak_occupied = max(
            self.peak_occupied, sum(other.plan is not None for other in self.slots)
        )
        # A sees the grant before either peer can block waiting for payload.
        send_message(self.control, Message("grant", plan=plan, metrics=message.metrics))
        peer = self.worker.peers[plan.request.key.client_id]
        rows = plan.request.num_tokens
        if self.computing is not None and not self.computing.packed_end.query():
            self.receives_during_compute += 1
        # The caller's stream is the ingress/default stream, not the compute
        # stream. Receiving a new input must not wait for unrelated GEMMs.
        slot.transfer = peer.transport.post(
            (slot.hidden[:rows], slot.weights[:rows], slot.ids[:rows]), send=False
        )

    def _receive_completions(self) -> bool:
        progressed = False
        for slot in self.slots:
            if slot.phase != "receiving":
                continue
            assert slot.transfer is not None and slot.plan is not None
            if not slot.transfer.query():
                continue
            slot.receive_ms = slot.transfer.finish()
            progressed = True
            slot.input_ready.record(torch.cuda.current_stream(self.worker.device))
            slot.transfer = None
            slot.ready_ns = time.perf_counter_ns()
            slot.phase = "input_ready"
            if self.computing is not None and not self.computing.packed_end.query():
                self.ready_during_compute += 1
            send_message(self.control, Message("input_ready", plan=slot.plan))
        self.peak_ready = max(
            self.peak_ready, sum(slot.phase == "input_ready" for slot in self.slots)
        )
        return progressed

    def _compute_completion(self) -> bool:
        slot = self.computing
        if slot is None or not slot.packed_end.query():
            return False
        assert slot.plan is not None and slot.outgoing is not None
        slot.phase = "returning"
        send_message(self.control, Message("output_ready", plan=slot.plan))
        peer = self.worker.peers[slot.plan.request.key.client_id]
        with torch.cuda.stream(self.compute_stream):
            slot.transfer = peer.transport.post((slot.outgoing,), send=True)
        # All GEMM/scratch/packing work is complete. Another slot may compute
        # while this one retains its independent output during the send.
        self.computing = None
        return True

    def _send_completions(self) -> bool:
        worker = self.worker
        progressed = False
        for slot in self.slots:
            if slot.phase != "returning":
                continue
            assert slot.transfer is not None and slot.plan is not None
            if not slot.transfer.query():
                continue
            send_ms = slot.transfer.finish()
            progressed = True
            plan = slot.plan
            assert slot.outgoing is not None and slot.result is not None
            worker.output_transfer["sent_bytes"] += (
                slot.outgoing.numel() * slot.outgoing.element_size()
            )
            worker.output_transfer["dense_equivalent_bytes"] += (
                slot.result.numel() * slot.result.element_size()
            )
            worker.completed_calls += 1
            worker.client_calls[plan.request.key.client_id] += 1
            worker.layer_calls[plan.request.layer_id] += 1
            assert plan.request.demand is not None
            for expert in plan.expert_ids:
                worker.expert_assignments[str(plan.request.layer_id)][str(expert)] += (
                    plan.request.demand.counts[expert]
                )
            metrics = {
                "admitted_queue_ms": slot.queue_ms,
                "receive_gpu_ms": slot.receive_ms,
                "compute_phase_gpu_ms": slot.begin.elapsed_time(slot.compute_end),
                "output_pack_gpu_ms": slot.compute_end.elapsed_time(slot.packed_end),
                "send_gpu_ms": send_ms,
                "input_audit_host_ms": 0.0,
                "worker_ready_queue_ms": (slot.compute_submitted_ns - slot.ready_ns)
                / 1e6,
                "worker_service_ms": (time.perf_counter_ns() - slot.started_ns) / 1e6,
            }
            # Only completed output sends return credit. The Controller may
            # still hold this slot until the other children of the parent drain.
            send_message(self.control, Message("done", plan=plan, metrics=metrics))
            slot.plan = None
            slot.transfer = None
            slot.outgoing = slot.result = None
            slot.phase = "idle"
            slot.completed += 1
        return progressed

    def _start_compute(self) -> bool:
        if self.computing is not None:
            return False
        ready = [slot for slot in self.slots if slot.phase == "input_ready"]
        if not ready:
            return False
        slot = min(ready, key=lambda entry: entry.plan.plan_id)
        plan = slot.plan
        assert plan is not None and plan.num_assignments is not None
        rows = plan.request.num_tokens
        slot.phase = "executing"
        slot.compute_submitted_ns = time.perf_counter_ns()
        self.computing = slot
        if any(other.phase == "receiving" for other in self.slots):
            self.compute_during_receive += 1
        send_message(self.control, Message("executing", plan=plan))
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(slot.input_ready)
            slot.begin.record()
            slot.result = self.worker.executors[plan.request.layer_id].forward_slots(
                slot.hidden[:rows], slot.weights[:rows], slot.ids[:rows]
            )
            slot.compute_end.record()
            slot.outgoing = slot.workspace.pack(
                slot.result,
                slot.ids[:rows],
                self.worker.ownership[plan.request.layer_id],
                num_assignments=plan.num_assignments,
            )
            slot.packed_end.record()
        return True

    @torch.inference_mode()
    def run(self) -> None:
        send_message(
            self.control,
            Message(
                "ready",
                detail=directory_digest(self.worker.directory),
                metrics={"receive_slots": len(self.slots)},
            ),
        )
        while True:
            active = any(slot.plan is not None for slot in self.slots)
            progressed = False
            if wait([self.control], timeout=0 if active else None):
                message = receive_message(self.control, 0)
                if message.kind == "close":
                    if active:
                        raise RuntimeError("Cannot close an undrained pipeline")
                    send_message(self.control, Message("closed"))
                    return
                self._accept(message)
                progressed = True
            # Observe ready events before waiting. A busy receive/compute/send
            # must not impose a fixed sleep on work that has already completed.
            progressed = self._receive_completions() or progressed
            progressed = self._compute_completion() or progressed
            progressed = self._send_completions() or progressed
            progressed = self._start_compute() or progressed
            if not progressed:
                wait([self.control], timeout=PROGRESS_POLL_S)

    def snapshot(self) -> dict:
        return {
            "enabled": True,
            "receive_slots": len(self.slots),
            "compute_lanes": 1,
            "active_slots": sum(slot.plan is not None for slot in self.slots),
            "peak_occupied_slots": self.peak_occupied,
            "peak_ready_slots": self.peak_ready,
            "slot_completed_calls": [slot.completed for slot in self.slots],
            "slot_generations": [slot.generation for slot in self.slots],
            "receives_posted_while_compute_pending": self.receives_during_compute,
            "input_ready_while_compute_pending": self.ready_during_compute,
            "compute_started_while_receive_pending": self.compute_during_receive,
            "observation_scope": "Host-observed pending work; not GPU overlap",
            "input_buffer_bytes": sum(
                tensor.numel() * tensor.element_size()
                for slot in self.slots
                for tensor in (slot.hidden, slot.weights, slot.ids)
            ),
        }
