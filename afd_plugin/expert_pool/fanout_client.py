# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static expert partition dispatch with one native Top-k reduction on A.

Inputs remain complete batches and outputs retain original Top-k slots. The
optional demand path copies only a bounded Expert count vector to the host,
then submits to the owners with positive demand. Payloads stay on the GPU.
"""

import time
from dataclasses import replace
from multiprocessing.connection import Connection

import torch
from vllm import _custom_ops as ops

from afd_plugin.expert_pool.client import PoolClient
from afd_plugin.expert_pool.compact_output import CompactOutputWorkspace
from afd_plugin.expert_pool.controlled_client import ControlledPoolClient
from afd_plugin.expert_pool.demand import plan_expert_demand
from afd_plugin.expert_pool.demand_gpu import ExpertDemandCollector
from afd_plugin.expert_pool.fanout_protocol import FanoutReplies
from afd_plugin.expert_pool.protocol import Message, receive_message, send_message


class FanoutPoolClient(ControlledPoolClient):
    def __init__(
        self,
        channels: tuple[PoolClient, ...],
        control: Connection,
        scheduling_policy: str = "ready_first",
        *,
        demand_aware: bool = False,
        compact_output: bool = False,
    ) -> None:
        if scheduling_policy != "ready_first":
            raise ValueError("Expert partitions require ready-first gang admission")
        self._initialize_channels(channels, expert_partitioned=True)
        self.control = control
        self.timeout_s = channels[0].timeout_s
        self.scheduling_policy = scheduling_policy
        if type(demand_aware) is not bool:
            raise ValueError("Expert demand must be an explicit boolean")
        self.demand_aware = demand_aware
        if type(compact_output) is not bool or (compact_output and not demand_aware):
            raise ValueError("Compact output requires expert-demand dispatch")
        self.compact_output = compact_output
        directory = channels[0].directory
        self.output_workspace = (
            CompactOutputWorkspace(
                directory.max_tokens * directory.top_k,
                directory.hidden_size,
                channels[0].transport.device,
            )
            if compact_output
            else None
        )
        self.output_transfer = {"received_bytes": 0, "dense_equivalent_bytes": 0}
        self.worker_output_transfer = {
            worker_id: {"received_bytes": 0, "dense_equivalent_bytes": 0}
            for worker_id in self.channels
        }
        self.layer_expert_counts = {
            placement.layer_id: placement.num_experts
            for placement in self.directory.placements
        }
        self.collectors = (
            {
                count: ExpertDemandCollector(
                    count,
                    channels[0].transport.device,
                    channels[0].directory.max_tokens * channels[0].directory.top_k,
                )
                for count in set(self.layer_expert_counts.values())
            }
            if demand_aware
            else {}
        )
        self.parent_layer_calls = dict.fromkeys(self.layer_expert_counts, 0)
        self.empty_layer_calls = dict.fromkeys(self.layer_expert_counts, 0)
        self.expert_assignments = {
            layer: [0] * count for layer, count in self.layer_expert_counts.items()
        }
        self.selected_worker_layer_calls = {
            worker: dict.fromkeys(layers, 0)
            for worker, layers in self.worker_layer_calls.items()
        }
        # Immutable placement metadata is uploaded once. Lookup indices stay
        # on the GPU; the optional demand path copies only its count summary.
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
            started = time.perf_counter_ns()
            demand_metrics = {}
            dispatch_plan = None
            if self.demand_aware:
                try:
                    demand, demand_metrics = self.collectors[
                        self.layer_expert_counts[layer_id]
                    ].collect(topk_ids)
                except ValueError:
                    # Metadata rejection or invalid-ID rejection after the
                    # completed summary copy posts no controller/NCCL work.
                    raise
                except BaseException:
                    # GPU failure or interruption may leave collector events
                    # in flight even though no request has been submitted yet.
                    self.failed = True
                    raise
                request = replace(
                    request, demand=demand, compact_output=self.compact_output
                )
                planning_started = time.perf_counter_ns()
                dispatch_plan = plan_expert_demand(self.directory, request)
                owners = dispatch_plan.owners
                demand_metrics["client_demand_plan_ms"] = (
                    time.perf_counter_ns() - planning_started
                ) / 1e6
            submitted = time.perf_counter_ns()
            self.sequence += 1
            issued = True
            send_message(self.control, Message("submit", request=request))
            if not owners:
                completion = receive_message(self.control, self.timeout_s)
                if completion.kind != "empty_done" or completion.request != request:
                    raise RuntimeError("Empty demand completion does not match request")
                finished = time.perf_counter_ns()
                self.parent_layer_calls[layer_id] += 1
                self.empty_layer_calls[layer_id] += 1
                metrics = {
                    **demand_metrics,
                    "client_validation_ms": (started - validation_started) / 1e6,
                    "client_admission_wait_ms": (finished - submitted) / 1e6,
                    "client_host_roundtrip_ms": (finished - started) / 1e6,
                }
                if self.metrics is not None:
                    self.metrics.record(layer_id, 0, metrics)
                return torch.empty_like(hidden_states), Message(
                    "empty_done", request=request, metrics=metrics
                )
            replies = FanoutReplies(request, owners, dispatch_plan=dispatch_plan)
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
            partial = None if self.compact_output else torch.empty_like(slots)
            route_indices = None if self.compact_output else topk_ids.long()
            output_restore_submit_ns = 0
            output_bytes = {}
            for owner in owners:
                if self.compact_output:
                    assert self.output_workspace is not None
                    plan = grants[owner].plan
                    assert plan is not None and plan.num_assignments is not None
                    received = self.output_workspace.receive_buffer(
                        plan.num_assignments
                    )
                else:
                    assert partial is not None
                    received = partial
                self.channels[owner].transport.transfer((received,), send=False)
                restore_started = time.perf_counter_ns()
                if self.compact_output:
                    self.output_workspace.scatter(
                        received,
                        topk_ids,
                        self.ownership[(owner, layer_id)],
                        slots,
                        num_assignments=plan.num_assignments,
                    )
                else:
                    mask = self.ownership[(owner, layer_id)][route_indices].unsqueeze(
                        -1
                    )
                    # Each original slot has exactly one owner. Do not locally
                    # sum BF16 outputs and then add rounded partition sums.
                    torch.where(mask, received, slots, out=slots)
                output_restore_submit_ns += time.perf_counter_ns() - restore_started
                output_bytes[owner] = received.numel() * received.element_size()
            output_received = time.perf_counter_ns()
            output = torch.empty_like(hidden_states)
            if request.num_tokens:
                ops.moe_sum(slots, output)
            completions = {
                owner: self._fanout_reply(replies, "done", owner) for owner in owners
            }
            finished = time.perf_counter_ns()
            metrics = {
                **demand_metrics,
                "client_validation_ms": (started - validation_started) / 1e6,
                "client_admission_wait_ms": (granted - submitted) / 1e6,
                "client_input_transfer_wall_ms": (inputs_sent - granted) / 1e6,
                "client_output_ready_wait_ms": (output_ready - inputs_sent) / 1e6,
                "client_output_transfer_wall_ms": (output_received - output_ready)
                / 1e6,
                # Host submission only: no metrics-only CUDA synchronization.
                "client_output_restore_submit_ms": output_restore_submit_ns / 1e6,
                "client_completion_wait_ms": (finished - output_received) / 1e6,
                # Final A-side merge is enqueued on its current CUDA stream.
                # Avoid a metrics-only synchronization in the model hot path;
                # this host interval is not output-ready GPU latency.
                "client_host_roundtrip_ms": (finished - started) / 1e6,
            }
            for owner in owners:
                dense_bytes = slots.numel() * slots.element_size()
                self.worker_output_transfer[owner]["received_bytes"] += output_bytes[
                    owner
                ]
                self.worker_output_transfer[owner]["dense_equivalent_bytes"] += (
                    dense_bytes
                )
                self.output_transfer["received_bytes"] += output_bytes[owner]
                self.output_transfer["dense_equivalent_bytes"] += dense_bytes
                self.worker_calls[owner] += 1
                self.worker_layer_calls[owner][layer_id] += 1
                self.selected_worker_layer_calls[owner][layer_id] += 1
                channel = self.channels[owner]
                if channel.metrics is not None:
                    channel.metrics.record(
                        layer_id,
                        request.num_tokens,
                        {**grants[owner].metrics, **completions[owner].metrics},
                    )
            if self.metrics is not None:
                self.metrics.record(layer_id, request.num_tokens, metrics)
            self.parent_layer_calls[layer_id] += 1
            if request.demand is not None:
                counts = self.expert_assignments[layer_id]
                for expert, assignments in enumerate(request.demand.counts):
                    counts[expert] += assignments
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
        result["policy"] = (
            "controller-expert-demand-gang"
            if self.demand_aware
            else "controller-expert-partitioned-gang"
        )
        result["output_contract"] = (
            "compact-weighted-top-k-slots"
            if self.compact_output
            else "weighted-top-k-slots"
        )
        result["compact_output"] = self.compact_output
        result["output_transfer"] = dict(self.output_transfer)
        for worker_id, counters in self.worker_output_transfer.items():
            result["workers"][worker_id]["output_transfer"] = dict(counters)
        result["demand_aware"] = self.demand_aware
        if self.demand_aware:
            result["demand"] = {
                "parent_layer_calls": {
                    str(k): v for k, v in self.parent_layer_calls.items()
                },
                "empty_layer_calls": {
                    str(k): v for k, v in self.empty_layer_calls.items()
                },
                "expert_assignments": {
                    str(k): list(v) for k, v in self.expert_assignments.items()
                },
                "selected_worker_layer_calls": {
                    w: {str(k): v for k, v in layers.items()}
                    for w, layers in self.selected_worker_layer_calls.items()
                },
                "summary_bytes_by_expert_count": {
                    str(count): collector.summary_bytes
                    for count, collector in self.collectors.items()
                },
            }
        return result
