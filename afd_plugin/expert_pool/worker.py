# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Task-driven single-GPU Expert service with one reusable buffer slot.

The worker owns no Attention, Router, shared branch, request scheduler or KV.
The input auditor is opt-in and intentionally expensive; it is for correctness
validation and must be disabled when measuring service performance.
"""

import hashlib
import time
from collections import Counter
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait

import torch

from afd_plugin.connectors.gpu.pool import PoolTransport
from afd_plugin.expert_pool.controller import directory_digest
from afd_plugin.expert_pool.deployment import ExecutionOptions
from afd_plugin.expert_pool.executor import ExpertExecutor
from afd_plugin.expert_pool.profiling import WorkerProfiler
from afd_plugin.expert_pool.protocol import (
    ExecutionPlan,
    Message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory, StaticScheduler


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class WorkerPeer:
    client_id: str
    session_epoch: int
    domain: str
    control: Connection
    transport: PoolTransport


class ExpertWorker:
    def __init__(
        self,
        directory: StaticDirectory,
        executors: dict[int, ExpertExecutor],
        peers: tuple[WorkerPeer, ...],
        *,
        audit_inputs: bool = False,
        execution: ExecutionOptions | None = None,
        profiler: WorkerProfiler | None = None,
        controller: Connection | None = None,
    ) -> None:
        if not peers or len({peer.client_id for peer in peers}) != len(peers):
            raise ValueError("Worker requires unique client endpoints")
        if set(executors) != {p.layer_id for p in directory.placements}:
            raise ValueError("Loaded layers do not match the directory")
        self.device = peers[0].transport.device
        if any(
            peer.transport.device != self.device or peer.transport.peer != 1
            for peer in peers
        ):
            raise ValueError("All worker channels must use its resident GPU")
        for placement in directory.placements:
            executor = executors[placement.layer_id]
            if (
                executor.placement != placement
                or executor.w13.device != self.device
                or executor.hidden_size != directory.hidden_size
                or executor.top_k != directory.top_k
            ):
                raise ValueError(
                    "Loaded executor does not match the published placement"
                )
        if len({executor.checkpoint_path for executor in executors.values()}) != 1:
            raise ValueError("The initial worker supports one immutable checkpoint")
        self.directory = directory
        self.executors = executors
        self.peers = {peer.client_id: peer for peer in peers}
        # One outstanding call per declared client bounds the queues even when
        # a service domain has more than the original four-client default.
        domain_clients = Counter(peer.domain for peer in peers)
        self.scheduler = StaticScheduler(
            directory, max_pending_per_domain=max(domain_clients.values())
        )
        for peer in peers:
            self.scheduler.register(peer.client_id, peer.session_epoch, peer.domain)
        self.audit_inputs = audit_inputs
        self.execution = execution or ExecutionOptions()
        self.profiler = profiler
        self.controller = controller
        self.controller_generation = 0
        self.compute_events = (
            tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
            if self.execution.reuse_cuda_events
            else None
        )
        self.hidden = torch.empty(
            (directory.max_tokens, directory.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.weights = torch.empty(
            (directory.max_tokens, directory.top_k),
            dtype=torch.float32,
            device=self.device,
        )
        self.ids = torch.empty_like(self.weights, dtype=torch.int32)
        self.output = torch.empty_like(self.hidden)
        self.completed_calls = 0
        self.client_calls = dict.fromkeys(self.peers, 0)
        self.layer_calls = dict.fromkeys(self.executors, 0)

    @torch.inference_mode()
    def _execute(
        self, plan: ExecutionPlan, controller_queue_ms: float | None = None
    ) -> None:
        peer = self.peers[plan.request.key.client_id]
        control = self.controller if self.controller is not None else peer.control
        admitted_queue_ms = (
            controller_queue_ms
            if controller_queue_ms is not None
            else (time.perf_counter_ns() - self.scheduler.active_enqueued_ns) / 1e6
        )
        started = time.perf_counter_ns()
        send_message(
            control,
            Message(
                "grant",
                plan=plan,
                metrics=(
                    {"controller_queue_ms": controller_queue_ms}
                    if controller_queue_ms is not None
                    else {}
                ),
            ),
        )
        rows = plan.request.num_tokens
        hidden, weights, ids = self.hidden[:rows], self.weights[:rows], self.ids[:rows]
        receive_ms = peer.transport.transfer((hidden, weights, ids), send=False)
        if self.controller is not None:
            send_message(control, Message("input_ready", plan=plan))
        audit_started = time.perf_counter_ns()
        digests = (
            {
                "hidden": tensor_digest(hidden),
                "weights": tensor_digest(weights),
                "ids": tensor_digest(ids),
            }
            if self.audit_inputs
            else {}
        )
        audit_ms = (time.perf_counter_ns() - audit_started) / 1e6
        begin, end = self.compute_events or (
            torch.cuda.Event(enable_timing=True) for _ in range(2)
        )
        begin.record()
        if self.controller is not None:
            send_message(control, Message("executing", plan=plan))
        try:
            result = self.executors[plan.request.layer_id](hidden, weights, ids)
        except ValueError as error:
            # All incoming transfers have completed. No output send was posted,
            # so a matched error can safely release the slot. CUDA failures must
            # escape and terminate the launch instead of reusing suspect memory.
            torch.cuda.current_stream(self.device).synchronize()
            send_message(control, Message("error", plan=plan, detail=str(error)))
            if self.controller is None:
                self.scheduler.complete(plan)
            return
        self.output[:rows].copy_(result)
        end.record()
        if not self.execution.defer_output_sync:
            end.synchronize()
        # output_ready authorizes posting a receive; it is not a completion
        # credit. The transfer stream waits for the producer event, and its
        # finished event still completes before the slot can be released.
        send_message(control, Message("output_ready", plan=plan))
        send_ms = peer.transport.transfer((self.output[:rows],), send=True)
        metrics = {
            "admitted_queue_ms": admitted_queue_ms,
            "receive_gpu_ms": receive_ms,
            "compute_phase_gpu_ms": begin.elapsed_time(end),
            "send_gpu_ms": send_ms,
            "input_audit_host_ms": audit_ms,
            "worker_service_ms": (time.perf_counter_ns() - started) / 1e6,
        }
        # The send event is complete: a later call may now reuse the slot. Do
        # not release on host enqueue or on receipt of a CPU completion alone.
        if self.controller is None:
            self.scheduler.complete(plan)
        self.completed_calls += 1
        self.client_calls[plan.request.key.client_id] += 1
        self.layer_calls[plan.request.layer_id] += 1
        send_message(
            control, Message("done", plan=plan, metrics=metrics, digests=digests)
        )

    def _run_controlled(self) -> None:
        control = self.controller
        assert control is not None
        send_message(control, Message("ready", detail=directory_digest(self.directory)))
        while True:
            wait([control])
            message = receive_message(control, 0)
            if message.kind == "close":
                send_message(control, Message("closed"))
                return
            if message.kind != "grant" or message.plan is None:
                raise RuntimeError("Worker expected a reserved controller plan")
            plan = message.plan
            self.directory.validate(plan.request)
            peer = self.peers.get(plan.request.key.client_id)
            if (
                peer is None
                or peer.session_epoch != plan.request.key.session_epoch
                or plan.worker_id != self.directory.worker_id
                or plan.slot_id != 0
                or plan.generation != self.controller_generation + 1
            ):
                raise RuntimeError("Stale or mismatched controller reservation")
            self.controller_generation = plan.generation
            if self.profiler is not None:
                self.profiler.before_call(self.completed_calls, plan)
            self._execute(plan, message.metrics["controller_queue_ms"])
            if self.profiler is not None:
                self.profiler.after_call()

    def run(self) -> None:
        connections = {peer.control: peer for peer in self.peers.values()}
        try:
            if self.controller is not None:
                self._run_controlled()
                return
            while connections:
                timeout = 0 if self.scheduler.outstanding else None
                for connection in wait(list(connections), timeout=timeout):
                    peer = connections[connection]
                    message = receive_message(connection, 0)
                    if message.kind == "close":
                        if peer.client_id in self.scheduler.outstanding:
                            raise RuntimeError("Client closed with undrained work")
                        send_message(connection, Message("closed"))
                        del connections[connection]
                        continue
                    if message.kind != "submit" or message.request is None:
                        raise RuntimeError("Worker expected a submit or close message")
                    request = message.request
                    try:
                        if (request.key.client_id, request.key.session_epoch) != (
                            peer.client_id,
                            peer.session_epoch,
                        ):
                            raise ValueError("Call does not belong to this endpoint")
                        self.scheduler.submit(request, time.perf_counter_ns())
                    except ValueError as error:
                        send_message(
                            connection,
                            Message("error", request=request, detail=str(error)),
                        )
                plan = self.scheduler.grant()
                if plan is not None:
                    if self.profiler is not None:
                        self.profiler.before_call(self.completed_calls, plan)
                    self._execute(plan)
                    if self.profiler is not None:
                        self.profiler.after_call()
        finally:
            try:
                if self.profiler is not None:
                    self.profiler.close()
            finally:
                if self.controller is not None:
                    self.controller.close()
                for peer in self.peers.values():
                    peer.transport.close()
                    peer.control.close()
