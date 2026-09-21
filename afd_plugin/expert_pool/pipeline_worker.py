# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded input/output slots around one serialized Expert compute stream.

The controller owns admission. CUDA events own transfer/compute completion.
Each live slot belongs to a different A client, whose NCCL channel has at most
one posted transfer. No executor workspace is used by two compute streams.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from multiprocessing.connection import wait
from typing import TYPE_CHECKING

import torch

from afd_plugin.connectors.gpu.pool import PoolTransfer
from afd_plugin.expert_pool.batching import ReadyCall, select_batch
from afd_plugin.expert_pool.compact_output import (
    CompactOutputWorkspace,
    selection_ownership,
)
from afd_plugin.expert_pool.controller import directory_digest
from afd_plugin.expert_pool.protocol import (
    BatchExecution,
    BatchSlot,
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
    packed_end: torch.cuda.Event
    input_ready: torch.cuda.Event
    generation: int = 0
    plan: ExecutionPlan | None = None
    phase: str = "idle"
    transfer: PoolTransfer | None = None
    outgoing: torch.Tensor | None = None
    result: torch.Tensor | None = None
    task_ownership: torch.Tensor | None = None
    started_ns: int = 0
    ready_ns: int = 0
    compute_submitted_ns: int = 0
    queue_ms: float = 0.0
    receive_ms: float = 0.0
    completed: int = 0
    compute_ms: float = 0.0
    compute_cost_ms: float = 0.0
    merge_cost_ms: float = 0.0
    pack_ms: float = 0.0
    batch_calls: int = 1
    batch_tokens: int = 0
    cost_sample: int = 0
    batch_pack_ms: float = 0.0


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
        self.batching = worker.batching
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
                    *(torch.cuda.Event(enable_timing=True) for _ in range(2)),
                    torch.cuda.Event(),
                )
            )
        self.batch_inputs = (
            tuple(
                torch.empty(
                    (self.batching.max_tokens, tensor.shape[1]),
                    dtype=tensor.dtype,
                    device=worker.device,
                )
                for tensor in (worker.hidden, worker.weights, worker.ids)
            )
            if self.batching.enabled
            else ()
        )
        self.merge_begin = torch.cuda.Event(enable_timing=True)
        self.compute_begin = torch.cuda.Event(enable_timing=True)
        self.compute_end = torch.cuda.Event(enable_timing=True)
        initialized = torch.cuda.Event()
        initialized.record(torch.cuda.current_stream(worker.device))
        self.compute_stream.wait_event(initialized)
        self.computing: tuple[PipelineSlot, ...] = ()
        self.batch_executions = 0
        self.batch_calls = 0
        self.batch_tokens = 0
        self.batch_histogram = dict.fromkeys(range(1, self.batching.max_calls + 1), 0)
        self.max_batch_tokens = 0
        self.compute_gpu_ms = 0.0
        self.merge_gpu_ms = 0.0
        self.collecting_since_ns: int | None = None
        self.collection_wait_ms = 0.0
        self.max_collection_wait_ms = 0.0
        self.max_collection_overrun_ms = 0.0
        self.next_wait_s = PROGRESS_POLL_S
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
        if self.computing and not self.computing[-1].packed_end.query():
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
            if self.computing and not self.computing[-1].packed_end.query():
                self.ready_during_compute += 1
            send_message(self.control, Message("input_ready", plan=slot.plan))
        self.peak_ready = max(
            self.peak_ready, sum(slot.phase == "input_ready" for slot in self.slots)
        )
        return progressed

    def _compute_completion(self) -> bool:
        if not self.computing or not self.computing[-1].packed_end.query():
            return False
        compute_ms = self.compute_begin.elapsed_time(self.compute_end)
        merge_ms = self.merge_begin.elapsed_time(self.compute_begin)
        self.compute_gpu_ms += compute_ms
        self.merge_gpu_ms += merge_ms
        pack_times = [
            slot.begin.elapsed_time(slot.packed_end) for slot in self.computing
        ]
        for index, slot in enumerate(self.computing):
            assert slot.plan is not None and slot.outgoing is not None
            slot.compute_ms = compute_ms
            # Per-call compute latency spans the whole batch. Physical cost is
            # charged once, to its first member, and once in the worker summary.
            slot.compute_cost_ms = compute_ms if index == 0 else 0.0
            slot.merge_cost_ms = merge_ms if index == 0 else 0.0
            slot.pack_ms = pack_times[index]
            if self.worker.execution.collect_cost_feedback:
                slot.cost_sample = int(index == 0)
                slot.batch_pack_ms = sum(pack_times) if index == 0 else 0.0
            slot.phase = "returning"
            send_message(self.control, Message("output_ready", plan=slot.plan))
            peer = self.worker.peers[slot.plan.request.key.client_id]
            with torch.cuda.stream(self.compute_stream):
                slot.transfer = peer.transport.post((slot.outgoing,), send=True)
        # All GEMM/scratch/packing work is complete. Another slot may compute
        # while this one retains its independent output during the send.
        self.computing = ()
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
                "compute_phase_gpu_ms": slot.compute_ms,
                "compute_phase_cost_gpu_ms": slot.compute_cost_ms,
                "input_merge_phase_cost_gpu_ms": slot.merge_cost_ms,
                "output_pack_gpu_ms": slot.pack_ms,
                "execution_batch_calls": slot.batch_calls,
                "execution_batch_tokens": slot.batch_tokens,
                "send_gpu_ms": send_ms,
                "input_audit_host_ms": 0.0,
                "worker_ready_queue_ms": (slot.compute_submitted_ns - slot.ready_ns)
                / 1e6,
                "worker_service_ms": (time.perf_counter_ns() - slot.started_ns) / 1e6,
            }
            # Only completed output sends return credit. The Controller may
            # still hold this slot until the other children of the parent drain.
            if worker.execution.collect_cost_feedback:
                metrics.update(
                    execution_cost_sample=slot.cost_sample,
                    execution_batch_pack_gpu_ms=slot.batch_pack_ms,
                )
            send_message(self.control, Message("done", plan=plan, metrics=metrics))
            slot.plan = None
            slot.transfer = None
            slot.outgoing = slot.result = slot.task_ownership = None
            slot.phase = "idle"
            slot.completed += 1
        return progressed

    def _start_compute(self) -> bool:
        self.next_wait_s = PROGRESS_POLL_S
        if self.computing:
            return False
        now_ns = time.perf_counter_ns()
        ready = tuple(
            ReadyCall(slot.plan, slot.ready_ns)
            for slot in self.slots
            if slot.phase == "input_ready" and slot.plan is not None
        )
        decision = select_batch(
            ready,
            self.batching,
            now_ns,
            # If every slot (or every A's only outstanding call) is ready,
            # incompatible queued work must execute before a batch can grow.
            can_grow=len(ready) < min(len(self.slots), len(self.worker.peers)),
        )
        if decision is None:
            return False
        if decision.wait_ns:
            if self.collecting_since_ns is None:
                self.collecting_since_ns = now_ns
            self.next_wait_s = min(PROGRESS_POLL_S, decision.wait_ns / 1e9)
            return False
        if self.collecting_since_ns is not None:
            waited_ms = (now_ns - self.collecting_since_ns) / 1e6
            self.collection_wait_ms += waited_ms
            self.max_collection_wait_ms = max(self.max_collection_wait_ms, waited_ms)
            self.max_collection_overrun_ms = max(
                self.max_collection_overrun_ms,
                max(0, now_ns - decision.deadline_ns) / 1e6,
            )
            self.collecting_since_ns = None
        selected = tuple(self.slots[index] for index in decision.slot_ids)
        plans = tuple(slot.plan for slot in selected)
        assert all(plan is not None for plan in plans)
        self.batch_executions += 1
        self.batch_calls += len(selected)
        self.batch_tokens += decision.num_tokens
        self.batch_histogram[len(selected)] += 1
        self.max_batch_tokens = max(self.max_batch_tokens, decision.num_tokens)
        for slot in selected:
            slot.phase = "executing"
            slot.compute_submitted_ns = now_ns
            slot.batch_calls = len(selected)
            slot.batch_tokens = decision.num_tokens
        self.computing = selected
        if any(other.phase == "receiving" for other in self.slots):
            self.compute_during_receive += 1
        if self.batching.enabled:
            send_message(
                self.control,
                Message(
                    "batch_executing",
                    batch=BatchExecution(
                        self.worker.directory.worker_id,
                        self.batch_executions,
                        tuple(BatchSlot.from_plan(plan) for plan in plans),
                    ),
                ),
            )
        else:
            send_message(self.control, Message("executing", plan=plans[0]))
        with torch.cuda.stream(self.compute_stream):
            for slot in selected:
                self.compute_stream.wait_event(slot.input_ready)
            self.merge_begin.record()
            if len(selected) == 1:
                inputs = tuple(
                    t[: decision.num_tokens]
                    for t in (selected[0].hidden, selected[0].weights, selected[0].ids)
                )
            else:
                inputs = tuple(t[: decision.num_tokens] for t in self.batch_inputs)
                offset = 0
                for slot, plan in zip(selected, plans, strict=True):
                    end = offset + plan.request.num_tokens
                    for dest, source in zip(
                        inputs, (slot.hidden, slot.weights, slot.ids), strict=True
                    ):
                        dest[offset:end].copy_(source[: end - offset])
                    offset = end
            self.compute_begin.record()
            layer = plans[0].request.layer_id
            assignment_mask = None
            if self.worker.expert_replicated:
                masks = []
                for slot, plan in zip(selected, plans, strict=True):
                    slot.task_ownership = selection_ownership(
                        plan.expert_ids,
                        self.worker.ownership[layer].numel(),
                        self.worker.device,
                    )
                    masks.append(
                        slot.task_ownership[slot.ids[: plan.request.num_tokens].long()]
                    )
                assignment_mask = (
                    masks[0] if len(masks) == 1 else torch.cat(masks, dim=0)
                )
            result = self.worker.executors[layer].forward_slots(
                *inputs, assignment_mask=assignment_mask
            )
            self.compute_end.record()
            offset = 0
            for slot, plan in zip(selected, plans, strict=True):
                rows = plan.request.num_tokens
                slot.result = result[offset : offset + rows]
                offset += rows
                slot.begin.record()
                slot.outgoing = slot.workspace.pack(
                    slot.result,
                    slot.ids[:rows],
                    slot.task_ownership
                    if self.worker.expert_replicated
                    else self.worker.ownership[layer],
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
                metrics={
                    "receive_slots": len(self.slots),
                    "startup_complete": int(self.worker.startup["completed"]),
                    "collect_cost_feedback": int(
                        self.worker.execution.collect_cost_feedback
                    ),
                    "batch_max_calls": self.batching.max_calls,
                    "batch_max_tokens": self.batching.max_tokens,
                    "batch_max_wait_us": self.batching.max_wait_us,
                },
            ),
        )
        while True:
            active = any(slot.plan is not None for slot in self.slots)
            # Honor an expired collection deadline before posting more ingress.
            # NCCL posting and host scheduling are not preemptible: report any
            # resulting deadline overrun instead of promising a hard SLO.
            progressed = (
                self._start_compute() if self.collecting_since_ns is not None else False
            )
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
                wait([self.control], timeout=self.next_wait_s)

    def snapshot(self) -> dict:
        return {
            "enabled": True,
            "receive_slots": len(self.slots),
            "compute_lanes": 1,
            "batching": {
                **asdict(self.batching),
                "executions": self.batch_executions,
                "calls": self.batch_calls,
                "token_rows": self.batch_tokens,
                "calls_per_execution": dict(self.batch_histogram),
                "max_executed_tokens": self.max_batch_tokens,
                "compute_phase_gpu_ms_total": self.compute_gpu_ms,
                "input_merge_phase_gpu_ms_total": self.merge_gpu_ms,
                "timing_scope": (
                    "CUDA event intervals include host submission gaps; "
                    "not active-kernel GPU cost"
                ),
                "collection_wait_ms_total": self.collection_wait_ms,
                "collection_wait_ms_max": self.max_collection_wait_ms,
                "collection_deadline_overrun_ms_max": self.max_collection_overrun_ms,
                "wait_scope": (
                    "Host elapsed while collecting, including control/communication "
                    "progress; excludes compute queue; not a hard realtime bound"
                ),
                "input_buffer_bytes": sum(
                    t.numel() * t.element_size() for t in self.batch_inputs
                ),
            },
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
