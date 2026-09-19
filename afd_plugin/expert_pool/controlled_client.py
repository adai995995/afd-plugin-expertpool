# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Per-A control session with direct GPU transfers to the granted E worker."""

import time
from contextlib import ExitStack
from multiprocessing.connection import Connection

import torch

from afd_plugin.expert_pool.client import CallRejectedError, PoolClient
from afd_plugin.expert_pool.protocol import (
    CallRequest,
    ExecutionPlan,
    Message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.replica_client import ReplicaPoolClient


class ControlledPoolClient(ReplicaPoolClient):
    def __init__(self, channels: tuple[PoolClient, ...], control: Connection) -> None:
        super().__init__(channels)
        self.control = control
        self.timeout_s = channels[0].timeout_s

    def _reply(
        self, kind: str, request: CallRequest, plan: ExecutionPlan | None = None
    ) -> Message:
        message = receive_message(self.control, self.timeout_s)
        if message.kind == "error":
            if (plan is None and message.request == request) or (
                plan is not None and message.plan == plan
            ):
                raise CallRejectedError(message.detail)
            raise RuntimeError("Controller error belongs to another call")
        if message.kind != kind or message.plan is None:
            raise RuntimeError("Unexpected controller reply")
        received = message.plan
        if received.request != request or (plan is not None and received != plan):
            raise RuntimeError("Mismatched call, plan or buffer generation")
        if received.worker_id not in self.channels or received.slot_id != 0:
            raise RuntimeError("Unknown worker or buffer slot")
        self.channels[received.worker_id].directory.validate(request)
        return message

    @torch.inference_mode()
    def execute(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, Message]:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("This A instance already has an outstanding operation")
        issued = False
        try:
            validation_started = time.perf_counter_ns()
            if self.closed or self.failed:
                raise RuntimeError("Pool client is closed or failed")
            candidates = self.selector.candidates.get(layer_id)
            if not candidates:
                raise ValueError("Requested layer is absent from the pool")
            # Only metadata validation uses a candidate; the Controller alone
            # selects and reserves the actual execution target.
            request = self.channels[candidates[0]].prepare_request(
                layer_id, hidden_states, topk_weights, topk_ids, self.sequence
            )
            started = time.perf_counter_ns()
            self.sequence += 1
            issued = True
            send_message(self.control, Message("submit", request=request))
            grant = self._reply("grant", request)
            granted = time.perf_counter_ns()
            plan = grant.plan
            assert plan is not None
            channel = self.channels[plan.worker_id]
            channel.transport.transfer(
                (hidden_states, topk_weights, topk_ids), send=True
            )
            inputs_sent = time.perf_counter_ns()
            self._reply("output_ready", request, plan)
            output_ready = time.perf_counter_ns()
            output = torch.empty_like(hidden_states)
            channel.transport.transfer((output,), send=False)
            output_received = time.perf_counter_ns()
            completion = self._reply("done", request, plan)
            finished = time.perf_counter_ns()
            metrics = {
                **grant.metrics,
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
            self.worker_calls[plan.worker_id] += 1
            self.worker_layer_calls[plan.worker_id][layer_id] += 1
            if channel.metrics is not None:
                channel.metrics.record(layer_id, hidden_states.shape[0], metrics)
            if self.metrics is not None:
                self.metrics.record(layer_id, hidden_states.shape[0], metrics)
            return output, Message(
                "done", plan=plan, metrics=metrics, digests=completion.digests
            )
        except CallRejectedError:
            raise
        except BaseException:
            if issued:
                self.failed = True
            raise
        finally:
            self.lock.release()

    def dispatch_status(self) -> dict:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Drain this A before requesting controller status")
        try:
            result = super().dispatch_status()
            result["policy"] = "controller-resident-layer-round-robin"
            if not self.closed and not self.failed:
                send_message(self.control, Message("status"))
                reply = receive_message(self.control, self.timeout_s)
                if reply.kind != "snapshot":
                    self.failed = True
                    raise RuntimeError("Unexpected controller status reply")
                result["controller"] = reply.metrics
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.lock.release()

    def close(self) -> None:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Drain this A before closing")
        try:
            if self.closed:
                return
            try:
                if not self.failed:
                    send_message(self.control, Message("close"))
                    if receive_message(self.control, self.timeout_s).kind != "closed":
                        raise RuntimeError("Unexpected controller close reply")
            finally:
                try:
                    with ExitStack() as cleanup:
                        for channel in self.channels.values():
                            cleanup.callback(channel.control.close)
                            cleanup.callback(channel.transport.close)
                            channel.closed = True
                finally:
                    self.control.close()
                    self.closed = True
        finally:
            self.lock.release()
