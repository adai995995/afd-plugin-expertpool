# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static expert partition dispatch with one native Top-k reduction on A.

The first version broadcasts the complete batch to every resident owner and
returns unreduced slots. It preserves routing and avoids CPU route reads, but
does not yet pack selected rows or optimize communication volume.
"""

import time
from multiprocessing.connection import Connection

import torch
from vllm import _custom_ops as ops

from afd_plugin.expert_pool.client import PoolClient
from afd_plugin.expert_pool.controlled_client import ControlledPoolClient
from afd_plugin.expert_pool.fanout_protocol import FanoutReplies
from afd_plugin.expert_pool.protocol import Message, receive_message, send_message


class FanoutPoolClient(ControlledPoolClient):
    def __init__(
        self,
        channels: tuple[PoolClient, ...],
        control: Connection,
        scheduling_policy: str = "ready_first",
    ) -> None:
        if scheduling_policy != "ready_first":
            raise ValueError("Expert partitions require ready-first gang admission")
        self._initialize_channels(channels, expert_partitioned=True)
        self.control = control
        self.timeout_s = channels[0].timeout_s
        self.scheduling_policy = scheduling_policy
        # Immutable placement metadata is uploaded once. No router values are
        # fetched from CUDA during dispatch; indexing below stays on the GPU.
        self.ownership = {
            (worker.worker_id, placement.layer_id): torch.tensor(
                [slot >= 0 for slot in placement.global_to_local()],
                dtype=torch.bool,
                device=channels[0].transport.device,
            )
            for worker in self.directory.workers
            for placement in worker.placements
        }

    def _fanout_reply(
        self, replies: FanoutReplies, kind: str, worker_id: str
    ) -> Message:
        while (message := replies.take(kind, worker_id)) is None:
            incoming = receive_message(self.control, self.timeout_s)
            replies.accept(incoming)
            if incoming.kind == "grant":
                assert incoming.plan is not None
                self.channels[incoming.plan.worker_id].directory.validate(
                    incoming.plan.request
                )
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
            owners = self.directory.owners(layer_id)
            request = self.channels[owners[0]].prepare_request(
                layer_id, hidden_states, topk_weights, topk_ids, self.sequence
            )
            replies = FanoutReplies(request, owners)
            started = time.perf_counter_ns()
            self.sequence += 1
            issued = True
            send_message(self.control, Message("submit", request=request))
            grants = {
                owner: self._fanout_reply(replies, "grant", owner) for owner in owners
            }
            granted = time.perf_counter_ns()
            for owner in owners:
                self.channels[owner].transport.transfer(
                    (hidden_states, topk_weights, topk_ids), send=True
                )
            inputs_sent = time.perf_counter_ns()
            for owner in owners:
                self._fanout_reply(replies, "output_ready", owner)
            output_ready = time.perf_counter_ns()
            slots = hidden_states.new_zeros(
                (request.num_tokens, request.top_k, request.hidden_size)
            )
            partial = torch.empty_like(slots)
            route_indices = topk_ids.long()
            for owner in owners:
                self.channels[owner].transport.transfer((partial,), send=False)
                mask = self.ownership[(owner, layer_id)][route_indices].unsqueeze(-1)
                # Each original slot has exactly one owner. Do not locally
                # sum BF16 outputs and then add rounded partition sums.
                torch.where(mask, partial, slots, out=slots)
            output_received = time.perf_counter_ns()
            output = torch.empty_like(hidden_states)
            if request.num_tokens:
                ops.moe_sum(slots, output)
            completions = {
                owner: self._fanout_reply(replies, "done", owner) for owner in owners
            }
            finished = time.perf_counter_ns()
            metrics = {
                "client_validation_ms": (started - validation_started) / 1e6,
                "client_admission_wait_ms": (granted - started) / 1e6,
                "client_input_transfer_wall_ms": (inputs_sent - granted) / 1e6,
                "client_output_ready_wait_ms": (output_ready - inputs_sent) / 1e6,
                "client_output_transfer_wall_ms": (output_received - output_ready)
                / 1e6,
                "client_completion_wait_ms": (finished - output_received) / 1e6,
                # Final A-side merge is enqueued on its current CUDA stream.
                # Avoid a metrics-only synchronization in the model hot path;
                # this host interval is not output-ready GPU latency.
                "client_host_roundtrip_ms": (finished - started) / 1e6,
            }
            for owner in owners:
                self.worker_calls[owner] += 1
                self.worker_layer_calls[owner][layer_id] += 1
                channel = self.channels[owner]
                if channel.metrics is not None:
                    channel.metrics.record(
                        layer_id,
                        request.num_tokens,
                        {**grants[owner].metrics, **completions[owner].metrics},
                    )
            if self.metrics is not None:
                self.metrics.record(layer_id, request.num_tokens, metrics)
            # The public return represents the parent; its representative
            # child plan is metadata only, never a buffer-release authority.
            return output, Message(
                "done", plan=completions[owners[-1]].plan, metrics=metrics
            )
        except BaseException:
            if issued:
                # A child may already have posted NCCL work. Any failure after
                # submit requires teardown; another owner is never a retry.
                self.failed = True
            raise
        finally:
            self.lock.release()

    def dispatch_status(self) -> dict:
        result = super().dispatch_status()
        result["policy"] = "controller-expert-partitioned-gang"
        result["output_contract"] = "weighted-top-k-slots"
        return result
