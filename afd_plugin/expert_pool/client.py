# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A blocking per-instance client; other instances can submit independently."""

import threading
import time
from multiprocessing.connection import Connection

import torch

from afd_plugin.connectors.gpu.pool import PoolTransport
from afd_plugin.expert_pool.metrics import CallMetrics
from afd_plugin.expert_pool.protocol import (
    CallKey,
    CallRequest,
    ExecutionPlan,
    Message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory


class CallRejectedError(ValueError):
    """Worker rejected a call without leaving pending GPU transfers."""


class PoolClient:
    def __init__(
        self,
        client_id: str,
        session_epoch: int,
        directory: StaticDirectory,
        control: Connection,
        transport: PoolTransport,
        timeout_s: float = 60,
        *,
        validate_values: bool = True,
    ) -> None:
        CallKey(client_id, session_epoch, 0)
        if transport.peer != 0:
            raise ValueError("Client transport must target the worker rank")
        self.client_id = client_id
        self.session_epoch = session_epoch
        self.directory = directory
        self.control = control
        self.transport = transport
        self.timeout_s = timeout_s
        self.validate_values = validate_values
        self.sequence = 0
        self.lock = threading.Lock()
        self.failed = False
        self.closed = False
        self.metrics: CallMetrics | None = None

    def set_metrics(self, enabled: bool) -> None:
        """Reset aggregates between drained trials, never during a model call."""
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Drain the client before changing metrics")
        try:
            self.metrics = CallMetrics() if enabled else None
        finally:
            self.lock.release()

    def _reply(
        self,
        kind: str,
        request: CallRequest,
        plan: ExecutionPlan | None = None,
    ) -> Message:
        message = receive_message(self.control, self.timeout_s)
        if message.kind == "error":
            if (plan is None and message.request == request) or (
                plan is not None and message.plan == plan
            ):
                raise CallRejectedError(message.detail)
            raise RuntimeError("Error reply belongs to another call")
        if message.kind != kind or message.plan is None:
            raise RuntimeError("Unexpected control reply")
        if message.plan.request != request:
            raise RuntimeError("Reply belongs to another call")
        if plan is not None and message.plan != plan:
            raise RuntimeError("Stale plan or buffer generation")
        if (
            message.plan.worker_id != self.directory.worker_id
            or message.plan.slot_id != 0
        ):
            raise RuntimeError("Unexpected worker or buffer slot")
        return message

    @torch.inference_mode()
    def execute(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        call_seq: int | None = None,
    ) -> tuple[torch.Tensor, Message]:
        """Return an owned output tensor and a matching completion record.

        Inputs must be ready on the caller's current CUDA stream. An error or
        timeout after dispatch poisons the channel unless the worker explicitly
        reports a safely rejected call. A supervisor must terminate a hung NCCL
        launch; a control timeout alone cannot cancel a GPU operation.
        """
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("This client already has an outstanding operation")
        issued = False
        try:
            validation_started = time.perf_counter_ns()
            if self.closed or self.failed:
                raise RuntimeError("Client is closed or failed")
            sequence = self.sequence if call_seq is None else call_seq
            if type(sequence) is not int or sequence < self.sequence:
                raise ValueError("Call sequence must increase on each worker channel")
            if hidden_states.ndim != 2:
                raise ValueError("Expected a two-dimensional activation tensor")
            shape = (hidden_states.shape[0], self.directory.top_k)
            if (
                hidden_states.ndim != 2
                or hidden_states.shape[1] != self.directory.hidden_size
                or hidden_states.dtype != torch.bfloat16
                or topk_ids.shape != shape
                or topk_ids.dtype != torch.int32
                or topk_weights.shape != shape
                or topk_weights.dtype != torch.float32
                or any(
                    t.device != self.transport.device or not t.is_contiguous()
                    for t in (hidden_states, topk_weights, topk_ids)
                )
            ):
                raise ValueError("Invalid activation or external routing payload")
            request = CallRequest(
                CallKey(self.client_id, self.session_epoch, sequence),
                self.directory.model_id,
                self.directory.version,
                layer_id,
                hidden_states.shape[0],
                self.directory.hidden_size,
                self.directory.top_k,
            )
            self.directory.validate(request)
            if self.validate_values:
                placement = next(
                    p for p in self.directory.placements if p.layer_id == layer_id
                )
                if bool(((topk_ids < 0) | (topk_ids >= placement.num_experts)).any()):
                    raise ValueError("Invalid logical expert ID")
                if not bool(torch.isfinite(topk_weights).all()) or bool(
                    (topk_weights < 0).any()
                ):
                    raise ValueError("Invalid routing weight")
            started = time.perf_counter_ns()
            self.sequence = sequence + 1
            issued = True
            send_message(self.control, Message("submit", request=request))
            grant = self._reply("grant", request)
            granted = time.perf_counter_ns()
            plan = grant.plan
            assert plan is not None
            self.transport.transfer((hidden_states, topk_weights, topk_ids), send=True)
            inputs_sent = time.perf_counter_ns()
            self._reply("output_ready", request, plan)
            output_ready = time.perf_counter_ns()
            output = torch.empty_like(hidden_states)
            self.transport.transfer((output,), send=False)
            output_received = time.perf_counter_ns()
            completion = self._reply("done", request, plan)
            finished = time.perf_counter_ns()
            metrics = {
                **completion.metrics,
                "client_validation_ms": (started - validation_started) / 1e6,
                "client_admission_wait_ms": (granted - started) / 1e6,
                "client_input_transfer_wall_ms": (inputs_sent - granted) / 1e6,
                "client_output_ready_wait_ms": (output_ready - inputs_sent) / 1e6,
                "client_output_transfer_wall_ms": (output_received - output_ready)
                / 1e6,
                "client_completion_wait_ms": (finished - output_received) / 1e6,
                "client_roundtrip_ms": (finished - started) / 1e6,
            }
            if self.metrics is not None:
                self.metrics.record(layer_id, hidden_states.shape[0], metrics)
            return output, Message(
                "done",
                plan=plan,
                metrics=metrics,
                digests=completion.digests,
            )
        except CallRejectedError:
            raise
        except BaseException:
            if issued:
                self.failed = True
            raise
        finally:
            self.lock.release()

    def close(self) -> None:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Drain the outstanding call before closing")
        try:
            if self.closed:
                return
            try:
                if not self.failed:
                    send_message(self.control, Message("close"))
                    if receive_message(self.control, self.timeout_s).kind != "closed":
                        raise RuntimeError("Unexpected close reply")
            finally:
                try:
                    self.transport.close()
                finally:
                    self.control.close()
                    self.closed = True
        finally:
            self.lock.release()
