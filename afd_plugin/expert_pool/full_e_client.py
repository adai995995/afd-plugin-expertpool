# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Whole-layer direct dispatch with result receipt separate from accounting.

Startup reserves a transport slot, not Expert weights or computation. An E
may queue one successor header until its previous output send actually drains.
"""

import time
from multiprocessing.connection import Connection

import torch

from afd_plugin.connectors.gpu.pool import PoolTransport
from afd_plugin.expert_pool.client import PoolClient
from afd_plugin.expert_pool.direct_dispatch import DirectReplies
from afd_plugin.expert_pool.protocol import (
    ExecutionPlan,
    Message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory

MAX_PENDING_FULL_E_RESULTS = 2


class DirectFullPoolClient(PoolClient):
    def __init__(
        self,
        client_id: str,
        session_epoch: int,
        directory: StaticDirectory,
        control: Connection,
        transport: PoolTransport,
        timeout_s: float = 60,
        *,
        client_slot: int,
        validate_values: bool = False,
        receive_slots: int = 1,
    ) -> None:
        super().__init__(
            client_id,
            session_epoch,
            directory,
            control,
            transport,
            timeout_s,
            validate_values=validate_values,
            receive_slots=receive_slots,
        )
        if (
            type(client_slot) is not int
            or not 0 <= client_slot < receive_slots
            or directory.allow_partial_experts
            or validate_values
        ):
            raise ValueError(
                "Direct full E requires a reserved slot and trusted routes"
            )
        self.client_slot = client_slot
        self.generation = 0
        self.events = DirectReplies()
        self.client_metrics: dict[int, dict[str, float]] = {}
        self.completions: dict[int, Message] = {}
        self.completed_calls = 0

    def _record_done(self, completion: Message, metrics: dict[str, float]) -> None:
        self.completed_calls += 1
        if self.metrics is not None:
            plan = completion.plan
            assert plan is not None
            self.metrics.record(
                plan.request.layer_id,
                plan.request.num_tokens,
                {**completion.metrics, **metrics},
            )

    def _drain(self, *, block: bool = False) -> None:
        first = block
        while first or self.control.poll(0):
            message = receive_message(self.control, self.timeout_s if first else 0)
            first = False
            if message.kind == "error":
                raise RuntimeError(
                    "Direct full E rejected an issued GPU call: " + message.detail
                )
            self.events.accept(self.directory.worker_id, message)
            if message.kind == "done":
                assert message.plan is not None
                generation = message.plan.generation
                metrics = self.client_metrics.pop(generation, None)
                if metrics is None:
                    self.completions[generation] = message
                else:
                    self._record_done(message, metrics)

    def _drain_all(self) -> None:
        while self.events.pending:
            if any(not state.received for state in self.events.pending.values()):
                raise RuntimeError(
                    "Drain the result before draining completion feedback"
                )
            self._drain(block=True)
        if self.client_metrics or self.completions:
            raise RuntimeError("Full E completion accounting did not drain")

    def drain_feedback(self) -> None:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Drain the model call before reading feedback")
        try:
            if not self.failed:
                self._drain_all()
        except BaseException:
            self.failed = True
            raise
        finally:
            self.lock.release()

    def set_metrics(self, enabled: bool) -> None:
        self.drain_feedback()
        super().set_metrics(enabled)

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
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("This client already has an outstanding operation")
        protocol_started = False
        try:
            validation_started = time.perf_counter_ns()
            if self.closed or self.failed:
                raise RuntimeError("Client is closed or failed")
            sequence = self.sequence if call_seq is None else call_seq
            if type(sequence) is not int or sequence < self.sequence:
                raise ValueError("Call sequence must increase on each worker channel")
            request = self.prepare_request(
                layer_id, hidden_states, topk_weights, topk_ids, sequence
            )
            started = time.perf_counter_ns()
            # Unknown asynchronous feedback invalidates the channel even if
            # this call has not yet issued its own GPU transfer.
            protocol_started = True
            self._drain()
            # Normally the previous done is already queued. This only bounds
            # unusually delayed feedback, never waits once per returned result.
            while len(self.events.pending) >= MAX_PENDING_FULL_E_RESULTS:
                self._drain(block=True)
            plan = ExecutionPlan(
                request,
                self.directory.worker_id,
                sequence * self.receive_slots + self.client_slot,
                self.client_slot,
                self.generation + 1,
            )
            self.events.register(plan)
            self.generation += 1
            self.sequence = sequence + 1
            send_message(self.control, Message("execute", plan=plan))
            dispatched = time.perf_counter_ns()
            self.transport.transfer((hidden_states, topk_weights, topk_ids), send=True)
            inputs_sent = time.perf_counter_ns()
            while not self.events.is_ready(plan):
                self._drain(block=True)
            output_ready = time.perf_counter_ns()
            output = torch.empty_like(hidden_states)
            self.transport.transfer((output,), send=False)
            output_received = time.perf_counter_ns()
            self.events.received(plan)
            metrics = {
                "client_validation_ms": (started - validation_started) / 1e6,
                "client_admission_wait_ms": (dispatched - started) / 1e6,
                "client_input_transfer_wall_ms": (inputs_sent - dispatched) / 1e6,
                "client_output_ready_wait_ms": (output_ready - inputs_sent) / 1e6,
                "client_output_transfer_wall_ms": (output_received - output_ready)
                / 1e6,
                "client_completion_wait_ms": 0.0,
                "client_roundtrip_ms": (output_received - started) / 1e6,
            }
            completion = self.completions.pop(plan.generation, None)
            if completion is None:
                self.client_metrics[plan.generation] = metrics
            else:
                self._record_done(completion, metrics)
            # Only poll feedback that has already arrived; the returned result
            # is usable regardless of whether E's CPU has observed its send.
            self._drain()
            return output, Message("result", plan=plan, metrics=metrics)
        except BaseException:
            if protocol_started or self.events.pending:
                self.failed = True
            raise
        finally:
            self.lock.release()

    def close(self) -> None:
        self.drain_feedback()
        super().close()
