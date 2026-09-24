# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Task-driven single-GPU Expert service with bounded reusable buffer slots.

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
from afd_plugin.expert_pool.batching import DISABLED_BATCHING, BatchingOptions
from afd_plugin.expert_pool.compact_output import CompactOutputWorkspace
from afd_plugin.expert_pool.controller import directory_digest
from afd_plugin.expert_pool.deployment import ExecutionOptions
from afd_plugin.expert_pool.executor import ExpertExecutor
from afd_plugin.expert_pool.profiling import WorkerProfiler
from afd_plugin.expert_pool.protocol import (
    MAX_RECEIVE_SLOTS,
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
        demand_aware: bool = False,
        compact_output: bool = False,
        receive_slots: int = 1,
        batching: BatchingOptions = DISABLED_BATCHING,
        expert_replicated: bool = False,
        direct_dispatch: bool = False,
        packed_input: bool = False,
    ) -> None:
        if not peers or len({peer.client_id for peer in peers}) != len(peers):
            raise ValueError("Worker requires unique client endpoints")
        if type(direct_dispatch) is not bool or (
            direct_dispatch
            and (
                controller is not None
                or not directory.allow_partial_experts
                or not compact_output
                or not demand_aware
                or receive_slots < 2
                or receive_slots != len(peers)
            )
        ):
            raise ValueError("Direct worker requires dedicated compact client slots")
        self.direct_dispatch = direct_dispatch
        if type(packed_input) is not bool or (packed_input and not direct_dispatch):
            raise ValueError("Packed input requires direct dispatch")
        self.packed_input = packed_input
        if (
            directory.allow_partial_experts
            and controller is None
            and not direct_dispatch
        ):
            raise ValueError("Expert partitions require authoritative gang control")
        if type(demand_aware) is not bool or (
            demand_aware
            and (
                not directory.allow_partial_experts
                or (controller is None and not direct_dispatch)
            )
        ):
            raise ValueError("Expert demand requires controlled expert partitions")
        self.demand_aware = demand_aware
        if type(compact_output) is not bool or (compact_output and not demand_aware):
            raise ValueError("Compact output requires expert-demand dispatch")
        self.compact_output = compact_output
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
        self.scheduler: StaticScheduler | None = None
        if not directory.allow_partial_experts:
            # One outstanding call per declared client bounds the queues even
            # when a service domain has more than four clients.
            domain_clients = Counter(peer.domain for peer in peers)
            self.scheduler = StaticScheduler(
                directory, max_pending_per_domain=max(domain_clients.values())
            )
            for peer in peers:
                self.scheduler.register(peer.client_id, peer.session_epoch, peer.domain)
        self.audit_inputs = audit_inputs
        self.execution = execution or ExecutionOptions()
        if (
            type(receive_slots) is not int
            or not 1 <= receive_slots <= MAX_RECEIVE_SLOTS
            or (
                receive_slots > 1
                and (
                    not compact_output
                    or self.execution.validate_worker_values
                    or not self.execution.defer_output_sync
                )
            )
        ):
            raise ValueError("Multiple receive slots require trusted compact execution")
        self.receive_slots = receive_slots
        if not isinstance(batching, BatchingOptions):
            raise ValueError("Worker requires typed batching options")
        batching.validate_capacity(receive_slots, directory.max_tokens)
        self.batching = batching
        if type(expert_replicated) is not bool or (
            expert_replicated and (receive_slots < 2 or not compact_output)
        ):
            raise ValueError("Expert replicas require compact multi-slot execution")
        self.expert_replicated = expert_replicated
        self.startup = {
            "enabled": self.execution.warmup_before_ready,
            "completed": False,
            "host_ms": 0.0,
            "kernel_shapes": [],
            "transport_channels": 0,
        }
        self.prepared = False
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
        self.output_workspace = (
            CompactOutputWorkspace(
                directory.max_tokens * directory.top_k,
                directory.hidden_size,
                self.device,
            )
            if compact_output
            else None
        )
        self.ownership = (
            {
                placement.layer_id: torch.tensor(
                    [slot >= 0 for slot in placement.global_to_local()],
                    dtype=torch.bool,
                    device=self.device,
                )
                for placement in directory.placements
            }
            if compact_output
            else {}
        )
        self.output = (
            None
            if compact_output
            else (
                torch.empty(
                    (directory.max_tokens, directory.top_k, directory.hidden_size),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                if directory.allow_partial_experts
                else torch.empty_like(self.hidden)
            )
        )
        self.pack_events = (
            tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
            if compact_output and self.execution.reuse_cuda_events
            else None
        )
        self.output_transfer = {"sent_bytes": 0, "dense_equivalent_bytes": 0}
        self.completed_calls = 0
        self.client_calls = dict.fromkeys(self.peers, 0)
        self.layer_calls = dict.fromkeys(self.executors, 0)
        self.expert_assignments = {
            str(p.layer_id): dict.fromkeys((str(e) for e in p.expert_ids), 0)
            for p in directory.placements
        }
        self.pipeline = None
        if receive_slots > 1:
            # Lazy import avoids the Worker/typed pipeline ownership cycle.
            if direct_dispatch:
                from afd_plugin.expert_pool.direct_worker import DirectWorkerPipeline

                self.pipeline = DirectWorkerPipeline(self, receive_slots)
            else:
                from afd_plugin.expert_pool.pipeline_worker import WorkerPipeline

                self.pipeline = WorkerPipeline(self, receive_slots)

    def pipeline_status(self) -> dict:
        return (
            self.pipeline.snapshot()
            if self.pipeline is not None
            else {"enabled": False, "receive_slots": 1, "compute_lanes": 1}
        )

    @torch.inference_mode()
    def _execute(
        self, plan: ExecutionPlan, controller_queue_ms: float | None = None
    ) -> None:
        peer = self.peers[plan.request.key.client_id]
        control = self.controller if self.controller is not None else peer.control
        if controller_queue_ms is not None:
            admitted_queue_ms = controller_queue_ms
        else:
            assert self.scheduler is not None
            admitted_queue_ms = (
                time.perf_counter_ns() - self.scheduler.active_enqueued_ns
            ) / 1e6
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
            executor = self.executors[plan.request.layer_id]
            result = (
                executor.forward_slots(hidden, weights, ids)
                if self.directory.allow_partial_experts
                else executor(hidden, weights, ids)
            )
        except ValueError as error:
            # All incoming transfers have completed. No output send was posted,
            # so a whole-layer error can safely release its individual slot.
            # Partitioned siblings may still be in NCCL and require fail-stop.
            # CUDA failures always terminate without reusing suspect memory.
            torch.cuda.current_stream(self.device).synchronize()
            send_message(control, Message("error", plan=plan, detail=str(error)))
            if self.directory.allow_partial_experts:
                raise RuntimeError(
                    "Expert fan-out failed; terminate the deployment"
                ) from error
            if self.controller is None:
                assert self.scheduler is not None
                self.scheduler.complete(plan)
            return
        pack_metrics = {}
        pack_begin = pack_end = None
        if self.compact_output:
            end.record()
            assert self.output_workspace is not None
            assert plan.num_assignments is not None
            pack_begin, pack_end = self.pack_events or (
                torch.cuda.Event(enable_timing=True) for _ in range(2)
            )
            pack_begin.record()
            outgoing = self.output_workspace.pack(
                result,
                ids,
                self.ownership[plan.request.layer_id],
                num_assignments=plan.num_assignments,
            )
            pack_end.record()
            output_event = pack_end
        else:
            assert self.output is not None
            self.output[:rows].copy_(result)
            outgoing = self.output[:rows]
            end.record()
            output_event = end
        if not self.execution.defer_output_sync:
            output_event.synchronize()
        # output_ready authorizes posting a receive; it is not a completion
        # credit. The transfer stream waits for the producer event, and its
        # finished event still completes before the slot can be released.
        send_message(control, Message("output_ready", plan=plan))
        send_ms = peer.transport.transfer((outgoing,), send=True)
        if pack_begin is not None and pack_end is not None:
            pack_metrics["output_pack_gpu_ms"] = pack_begin.elapsed_time(pack_end)
        self.output_transfer["sent_bytes"] += outgoing.numel() * outgoing.element_size()
        self.output_transfer["dense_equivalent_bytes"] += (
            result.numel() * result.element_size()
        )
        metrics = {
            **pack_metrics,
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
            assert self.scheduler is not None
            self.scheduler.complete(plan)
        self.completed_calls += 1
        self.client_calls[plan.request.key.client_id] += 1
        self.layer_calls[plan.request.layer_id] += 1
        if self.demand_aware:
            assert plan.request.demand is not None
            for expert in plan.expert_ids:
                self.expert_assignments[str(plan.request.layer_id)][str(expert)] += (
                    plan.assignments_for(expert)
                )
        send_message(
            control, Message("done", plan=plan, metrics=metrics, digests=digests)
        )

    def _run_controlled(self) -> None:
        if self.pipeline is not None:
            self.pipeline.run()
            return
        control = self.controller
        assert control is not None
        send_message(
            control,
            Message(
                "ready",
                detail=directory_digest(self.directory),
                metrics={"startup_complete": int(self.startup["completed"])},
            ),
        )
        while True:
            wait([control])
            message = receive_message(control, 0)
            if message.kind == "close":
                send_message(control, Message("closed"))
                return
            if message.kind != "grant" or message.plan is None:
                raise RuntimeError("Worker expected a reserved controller plan")
            plan = message.plan
            self.validate_controlled_plan(plan, self.controller_generation + 1)
            self.controller_generation = plan.generation
            if self.profiler is not None:
                self.profiler.before_call(self.completed_calls, plan)
            self._execute(plan, message.metrics["controller_queue_ms"])
            if self.profiler is not None:
                self.profiler.after_call()

    def validate_controlled_plan(
        self, plan: ExecutionPlan, generation: int, *, receive_slots: int = 1
    ) -> None:
        self.directory.validate(plan.request)
        if self.demand_aware != (plan.request.demand is not None):
            raise RuntimeError("Worker and plan demand modes disagree")
        if self.compact_output != plan.request.compact_output:
            raise RuntimeError("Worker and plan output layouts disagree")
        if (plan.input_rows is not None) != self.packed_input:
            raise RuntimeError("Worker and plan input layouts disagree")
        if self.demand_aware:
            placement = self.executors[plan.request.layer_id].placement
            demand = plan.request.demand
            assert demand is not None
            if len(demand.counts) != placement.num_experts:
                raise RuntimeError("Plan demand has the wrong expert count")
            resident_demand = tuple(
                sorted(e for e in placement.expert_ids if demand.counts[e])
            )
            if (
                not set(plan.expert_ids) <= set(resident_demand)
                if self.expert_replicated
                else plan.expert_ids != resident_demand
            ):
                raise RuntimeError("Plan does not match resident expert demand")
        peer = self.peers.get(plan.request.key.client_id)
        if (
            peer is None
            or peer.session_epoch != plan.request.key.session_epoch
            or plan.worker_id != self.directory.worker_id
            or plan.slot_id >= receive_slots
            or plan.generation != generation
        ):
            raise RuntimeError("Stale or mismatched controller reservation")

    @torch.inference_mode()
    def prepare(self) -> None:
        """Complete explicit startup work before publishing schedulable Ready.

        Shapes are representative, not an exhaustive JIT guarantee. No request
        identities, routing counts or admission credits are consumed by warmup.
        """
        if self.prepared:
            return
        started = time.perf_counter_ns()
        if self.execution.warmup_before_ready:
            if not all(peer.transport.warmup_completed for peer in self.peers.values()):
                raise RuntimeError("All A/E channels must warm before worker Ready")
            shapes = sorted(
                {1, self.directory.max_tokens, self.batching.max_tokens} - {0}
            )
            rows = max(shapes)
            hidden = torch.zeros(
                (rows, self.directory.hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
            weights = torch.full(
                (rows, self.directory.top_k),
                1 / self.directory.top_k,
                dtype=torch.float32,
                device=self.device,
            )
            for layer, executor in self.executors.items():
                ids = (
                    torch.tensor(
                        [
                            executor.placement.expert_ids[
                                i % len(executor.placement.expert_ids)
                            ]
                            for i in range(self.directory.top_k)
                        ],
                        dtype=torch.int32,
                        device=self.device,
                    )
                    .expand(rows, -1)
                    .contiguous()
                )
                for tokens in shapes:
                    mask = (
                        self.ownership[layer][ids[:tokens].long()]
                        if self.expert_replicated
                        else None
                    )
                    result = (
                        executor.forward_slots(
                            hidden[:tokens],
                            weights[:tokens],
                            ids[:tokens],
                            assignment_mask=mask,
                        )
                        if self.directory.allow_partial_experts
                        else executor(hidden[:tokens], weights[:tokens], ids[:tokens])
                    )
                    if self.output_workspace is not None:
                        packed_rows = min(tokens, self.directory.max_tokens)
                        self.output_workspace.pack(
                            result[:packed_rows],
                            ids[:packed_rows],
                            self.ownership[layer],
                            num_assignments=packed_rows * self.directory.top_k,
                        )
            torch.cuda.synchronize(self.device)
            self.startup.update(
                completed=True, kernel_shapes=shapes, transport_channels=len(self.peers)
            )
        self.startup["host_ms"] = (time.perf_counter_ns() - started) / 1e6
        self.prepared = True

    def run(self) -> None:
        self.prepare()
        connections = {peer.control: peer for peer in self.peers.values()}
        try:
            if self.direct_dispatch:
                assert self.pipeline is not None
                self.pipeline.run()
                return
            if self.controller is not None:
                self._run_controlled()
                return
            assert self.scheduler is not None
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
