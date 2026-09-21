#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Four-GPU demand-dispatch correctness check with a real CPU Controller.

Two A-side clients and two E workers execute one real DeepSeek MoE layer over
NCCL. A clients retain native weights for validation; they are not complete
Attention/KV services. Natural Router cases and explicitly controlled boundary
routes are reported separately. Input checks and local references perturb this
run: neither host intervals nor summary timings are a performance result.
No weights, activations or raw routing tensors are written to disk.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import socket
import time
import traceback
import uuid
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING

import afd_plugin
from afd_plugin.expert_pool.controller import ControllerClientIdentity
from afd_plugin.expert_pool.controller_service import ControllerRuntime
from afd_plugin.expert_pool.deployment import ExecutionOptions
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.fanout_controller import FanoutControllerLedger
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import MAX_RECEIVE_SLOTS
from afd_plugin.expert_pool.scheduler import StaticDirectory
from tools.expert_pool.validate_engines import (
    process_environment,
    stop_children,
    verify_pipeline,
)
from tools.expert_pool.validate_service import (
    BusyGPUError,
    Launch,
    check_idle,
    gpu_runtime,
    receive_record,
    send_record,
)

if TYPE_CHECKING:
    import torch

    from afd_plugin.connectors.gpu.pool import PoolTransport

CLIENT_COUNT = 2
WORKER_COUNT = 2
GPU_COUNT = CLIENT_COUNT + WORKER_COUNT
DEFAULT_TIMEOUT_S = 120
DEFAULT_ATOL = 0.01
DEFAULT_RTOL = 0.02
BF16_OUTPUT_ELEMENT_BYTES = 2


@dataclass(frozen=True)
class DemandLaunch:
    runtime: Launch
    directory: PoolDirectory
    ports: tuple[tuple[int, int], tuple[int, int]]
    compact_output: bool = False
    receive_slots: int = 1
    pipeline_trace_dir: str | None = None


class TransferRecorder:
    """Per-channel validation instrumentation; no global transport patch."""

    def __init__(self, transport: PoolTransport) -> None:
        self.transport = transport
        self.device = transport.device
        self.peer = transport.peer
        self.counters = dict.fromkeys(
            ("send_calls", "receive_calls", "send_bytes", "receive_bytes"), 0
        )

    def transfer(self, tensors: tuple[torch.Tensor, ...], *, send: bool) -> float:
        elapsed = self.transport.transfer(tensors, send=send)
        direction = "send" if send else "receive"
        self.counters[f"{direction}_calls"] += 1
        self.counters[f"{direction}_bytes"] += sum(
            tensor.numel() * tensor.element_size() for tensor in tensors
        )
        return elapsed

    def close(self) -> None:
        self.transport.close()


def controller_process(
    launch: DemandLaunch,
    clients: tuple[Connection, Connection],
    workers: tuple[Connection, Connection],
    supervisor: Connection,
) -> None:
    try:
        os.setsid()
        send_record(supervisor, {"kind": "started", "pid": os.getpid()})
        ledger = FanoutControllerLedger(
            launch.directory,
            tuple(
                ControllerClientIdentity(f"client-{index}", 1, f"domain-{index}")
                for index in range(CLIENT_COUNT)
            ),
            demand_aware=True,
            compact_output=launch.compact_output,
            receive_slots=launch.receive_slots,
        )
        runtime = ControllerRuntime(
            ledger,
            {f"client-{index}": connection for index, connection in enumerate(clients)},
            {f"worker-{index}": connection for index, connection in enumerate(workers)},
        )
        send_record(supervisor, {"kind": "ready", "role": "controller"})
        send_record(supervisor, {"kind": "finished", "ledger": runtime.run()})
    except BaseException:
        send_record(supervisor, {"kind": "error", "detail": traceback.format_exc()})
        raise
    finally:
        for connection in (*clients, *workers, supervisor):
            connection.close()


def worker_process(
    launch: DemandLaunch,
    gpu: int,
    worker_index: int,
    peers: tuple[Connection, Connection],
    control: Connection,
    supervisor: Connection,
) -> None:
    try:
        process_environment(gpu)
        send_record(supervisor, {"kind": "started", "pid": os.getpid()})
        with gpu_runtime(launch.runtime, gpu) as (checkpoint, _):
            import torch

            from afd_plugin.connectors.gpu.pool import PoolTransport
            from afd_plugin.expert_pool.executor import ExpertExecutor
            from afd_plugin.expert_pool.worker import ExpertWorker, WorkerPeer

            class TracedTransport(PoolTransport):
                # Validation-only annotations bind each NCCL launch to its
                # direction. The production hot path has no profiler ranges.
                def post(self, tensors, *, send):
                    with torch.profiler.record_function(
                        "pool.output.send" if send else "pool.input.receive"
                    ):
                        return super().post(tensors, send=send)

            transport_type = (
                TracedTransport
                if launch.pipeline_trace_dir is not None
                else PoolTransport
            )
            device = torch.device("cuda", 0)
            directory = launch.directory.workers[worker_index]
            transports = tuple(
                transport_type(
                    "127.0.0.1", row[worker_index], 0, device, launch.runtime.timeout_s
                )
                for row in launch.ports
            )
            worker = ExpertWorker(
                directory,
                {
                    placement.layer_id: ExpertExecutor(
                        checkpoint,
                        placement,
                        device,
                        validate_values=launch.receive_slots == 1,
                    )
                    for placement in directory.placements
                },
                tuple(
                    WorkerPeer(f"client-{index}", 1, f"domain-{index}", peer, transport)
                    for index, (peer, transport) in enumerate(
                        zip(peers, transports, strict=True)
                    )
                ),
                controller=control,
                demand_aware=True,
                compact_output=launch.compact_output,
                receive_slots=launch.receive_slots,
                execution=ExecutionOptions(
                    validate_worker_values=launch.receive_slots == 1,
                    reuse_cuda_events=True,
                    defer_output_sync=True,
                ),
            )
            send_record(supervisor, {"kind": "ready", "role": directory.worker_id})
            profiling = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                )
                if launch.pipeline_trace_dir is not None
                else nullcontext()
            )
            with profiling as profile:
                worker.run()
            if launch.pipeline_trace_dir is not None:
                profile.export_chrome_trace(
                    str(Path(launch.pipeline_trace_dir) / f"{directory.worker_id}.json")
                )
            send_record(
                supervisor,
                {
                    "kind": "finished",
                    "worker_id": directory.worker_id,
                    "completed_calls": worker.completed_calls,
                    "client_calls": dict(worker.client_calls),
                    "layer_calls": dict(worker.layer_calls),
                    "expert_assignments": worker.expert_assignments,
                    "compact_output": worker.compact_output,
                    "output_transfer": dict(worker.output_transfer),
                    "pipeline": worker.pipeline_status(),
                },
            )
    except BaseException:
        send_record(supervisor, {"kind": "error", "detail": traceback.format_exc()})
        raise
    finally:
        supervisor.close()


def client_process(
    launch: DemandLaunch,
    gpu: int,
    client_index: int,
    peers: tuple[Connection, Connection],
    control: Connection,
    supervisor: Connection,
) -> None:
    try:
        process_environment(gpu)
        send_record(supervisor, {"kind": "started", "pid": os.getpid()})
        with gpu_runtime(launch.runtime, gpu) as (checkpoint, config):
            import torch
            from vllm.forward_context import set_forward_context

            from afd_plugin.connectors.gpu.pool import PoolTransport
            from afd_plugin.expert_pool.client import PoolClient
            from afd_plugin.expert_pool.demand_gpu import ExpertDemandCollector
            from afd_plugin.expert_pool.fanout_client import FanoutPoolClient
            from afd_plugin.expert_pool.router import DeepseekPoolRouter
            from tools.expert_pool.validate_layer import (
                compare,
                expect_rejection,
                load_native,
            )

            device = torch.device("cuda", 0)
            directory = launch.runtime.directory
            layer = directory.placements[0].layer_id
            num_experts = directory.placements[0].num_experts
            transports = tuple(
                TransferRecorder(
                    PoolTransport(
                        "127.0.0.1", port, 1, device, launch.runtime.timeout_s
                    )
                )
                for port in launch.ports[client_index]
            )
            client = FanoutPoolClient(
                tuple(
                    PoolClient(
                        f"client-{client_index}",
                        1,
                        worker,
                        peer,
                        transport,
                        launch.runtime.timeout_s,
                        # Exercise the collector's own invalid-ID handling.
                        # Worker-side value validation remains enabled.
                        validate_values=False,
                    )
                    for worker, peer, transport in zip(
                        launch.directory.workers, peers, transports, strict=True
                    )
                ),
                control,
                demand_aware=True,
                compact_output=launch.compact_output,
                receive_slots=launch.receive_slots,
            )
            collector = ExpertDemandCollector(
                num_experts, device, directory.max_tokens * directory.top_k
            )
            with torch.inference_mode():
                native = load_native(checkpoint, layer, config)
                router = DeepseekPoolRouter(checkpoint, layer, device)
                # Invalid IDs must fail in collection, before any submit/transfer.
                valid_hidden = torch.zeros(
                    (1, directory.hidden_size), dtype=torch.bfloat16, device=device
                )
                valid_weights = torch.full(
                    (1, directory.top_k),
                    1.0 / directory.top_k,
                    dtype=torch.float32,
                    device=device,
                )
                saved_hidden, saved_weights = (
                    valid_hidden.clone(),
                    valid_weights.clone(),
                )
                for invalid_id in (-1, num_experts, torch.iinfo(torch.int32).min):
                    invalid = torch.full(
                        (1, directory.top_k),
                        invalid_id,
                        dtype=torch.int32,
                        device=device,
                    )
                    saved = invalid.clone()
                    expect_rejection(
                        lambda payload=invalid: collector.collect(payload),
                        "invalid demand ID",
                    )
                    sequence = client.sequence
                    expect_rejection(
                        lambda payload=invalid: client.execute(
                            layer, valid_hidden, valid_weights, payload
                        ),
                        "invalid demand rejected by the real A client",
                    )
                    if (
                        client.failed
                        or client.sequence != sequence
                        or any(any(t.counters.values()) for t in transports)
                    ):
                        raise AssertionError(
                            "Rejected demand poisoned A or posted work"
                        )
                    compare(invalid, saved, 0, 0)
                compare(valid_hidden, saved_hidden, 0, 0)
                compare(valid_weights, saved_weights, 0, 0)
                excessive = torch.zeros(
                    (directory.max_tokens + 1, directory.top_k),
                    dtype=torch.int32,
                    device=device,
                )
                expect_rejection(
                    lambda: collector.collect(excessive), "demand capacity exceeded"
                )
                if client.sequence or any(any(t.counters.values()) for t in transports):
                    raise AssertionError(
                        "Invalid demand posted a request or GPU transfer"
                    )
                send_record(
                    supervisor,
                    {
                        "kind": "ready",
                        "role": client.client_id,
                        "invalid_demand_rejected": True,
                        "real_client_invalid_demand_rejected_without_posting": True,
                    },
                )
                while True:
                    command = receive_record(supervisor, launch.runtime.timeout_s * 4)
                    if command["op"] == "close":
                        status = client.dispatch_status()
                        client.close()
                        send_record(supervisor, {"kind": "finished", "status": status})
                        return
                    if command["op"] != "prepare":
                        raise ValueError("Unknown demand validation command")
                    case = command["case"]
                    tokens = case["tokens"]
                    hidden = torch.randn(
                        tokens,
                        directory.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                        generator=torch.Generator(device=device).manual_seed(
                            case["seed"]
                        ),
                    )
                    weights, ids = router(hidden)
                    natural = case["routes"] == "native"
                    if not natural and tokens:
                        if case["routes"] == "even":
                            choices = tuple(range(0, num_experts, 2))[: directory.top_k]
                        elif case["routes"] == "odd":
                            choices = tuple(range(1, num_experts, 2))[: directory.top_k]
                        elif case["routes"] == "mixed":
                            choices = tuple(
                                i if i % 2 == 0 else num_experts - i
                                for i in range(directory.top_k)
                            )
                        else:
                            raise ValueError("Unknown controlled routing case")
                        positions = (
                            torch.arange(directory.top_k, device=device)[None, :]
                            + torch.arange(tokens, device=device)[:, None]
                        ) % directory.top_k
                        ids = torch.tensor(choices, dtype=torch.int32, device=device)[
                            positions
                        ].contiguous()
                    if tokens:
                        if natural:
                            logits, _ = native.gate(hidden)
                            native_weights, native_ids = (
                                native.experts.router.select_experts(
                                    hidden, logits, topk_indices_dtype=torch.int32
                                )
                            )
                            compare(ids, native_ids, 0, 0)
                            compare(weights, native_weights, 0, 0)
                        with set_forward_context(None, config, num_tokens=tokens):
                            expected = native.experts.routed_experts.forward_modular(
                                hidden, weights, ids
                            )
                    else:
                        expected = torch.empty_like(hidden)
                    original = tuple(value.clone() for value in (hidden, weights, ids))
                    # An independent CPU reference is permitted only in this
                    # validation harness, outside the dispatch interval. Raw IDs
                    # are neither included in reports nor saved to disk.
                    reference = Counter(ids.flatten().cpu().tolist())
                    expected_counts = tuple(
                        reference.get(i, 0) for i in range(num_experts)
                    )
                    collect_start = time.perf_counter_ns()
                    demand, collection_metrics = collector.collect(ids)
                    if (
                        demand.counts != expected_counts
                        or sum(demand.counts) != tokens * directory.top_k
                        or not collect_start
                        <= demand.ready_ns
                        <= time.perf_counter_ns()
                    ):
                        raise AssertionError(
                            "GPU demand summary differs from reference"
                        )
                    if not tokens and any(
                        collection_metrics[key]
                        for key in (
                            "demand_histogram_gpu_ms",
                            "demand_summary_copy_gpu_ms",
                            "demand_metadata_wait_ms",
                        )
                    ):
                        raise AssertionError(
                            "Empty demand performed a GPU summary phase"
                        )
                    selected = tuple(
                        worker.worker_id
                        for worker in launch.directory.workers
                        if any(
                            expected_counts[e] for e in worker.placements[0].expert_ids
                        )
                    )
                    before = [dict(transport.counters) for transport in transports]
                    send_record(supervisor, {"kind": "prepared"})
                    if (
                        receive_record(supervisor, launch.runtime.timeout_s)["op"]
                        != "execute"
                    ):
                        raise ValueError("Expected execution release")
                    started_ns = time.perf_counter_ns()
                    actual, completion = client.execute(layer, hidden, weights, ids)
                    finished_ns = time.perf_counter_ns()
                    if tokens and completion.plan is None:
                        raise AssertionError(
                            "Nonempty demand lacks a completed child plan"
                        )
                    request = (
                        completion.request
                        if completion.plan is None
                        else completion.plan.request
                    )
                    if (
                        completion.kind != ("done" if tokens else "empty_done")
                        or request is None
                        or request.demand is None
                        or request.demand.counts != expected_counts
                        or request.compact_output is not launch.compact_output
                    ):
                        raise AssertionError(
                            "Completed request lost its demand summary"
                        )
                    traffic = {}
                    for worker, transport, previous in zip(
                        launch.directory.workers, transports, before, strict=True
                    ):
                        difference = {
                            key: value - previous[key]
                            for key, value in transport.counters.items()
                        }
                        active = int(worker.worker_id in selected)
                        assignments = sum(
                            expected_counts[expert]
                            for expert in worker.placements[0].expert_ids
                        )
                        expected_traffic = {
                            "send_calls": active,
                            "receive_calls": active,
                            "send_bytes": active
                            * (
                                hidden.numel() * hidden.element_size()
                                + weights.numel() * weights.element_size()
                                + ids.numel() * ids.element_size()
                            ),
                            "receive_bytes": (
                                assignments
                                if launch.compact_output
                                else active * tokens * directory.top_k
                            )
                            * directory.hidden_size
                            * hidden.element_size(),
                        }
                        if difference != expected_traffic:
                            raise AssertionError(
                                "Selected/zero-hit worker transfer accounting differs"
                            )
                        traffic[worker.worker_id] = difference
                    if launch.compact_output and sum(
                        worker["receive_bytes"] for worker in traffic.values()
                    ) != (
                        tokens
                        * directory.top_k
                        * directory.hidden_size
                        * hidden.element_size()
                    ):
                        raise AssertionError(
                            "Compact return lost or repeated routed slots"
                        )
                    for value, saved in zip(
                        (hidden, weights, ids), original, strict=True
                    ):
                        compare(value, saved, 0, 0)
                    send_record(
                        supervisor,
                        {
                            "kind": "case",
                            "case": {
                                **case,
                                "client_id": client.client_id,
                                "layer_id": layer,
                                "started_ns": started_ns,
                                "finished_ns": finished_ns,
                                "routing_kind": "fresh native Router"
                                if natural
                                else "controlled boundary IDs; native Router weights",
                                "counts": list(expected_counts),
                                "counts_match_exactly": True,
                                "selected_workers": list(selected),
                                "transfer_accounting": traffic,
                                "compact_output": launch.compact_output,
                                "dense_equivalent_output_bytes": len(selected)
                                * tokens
                                * directory.top_k
                                * directory.hidden_size
                                * hidden.element_size(),
                                "inputs_unchanged": True,
                                "native_routes_exact": natural and tokens > 0,
                                "completion": asdict(completion),
                                "summary_bytes": collector.summary_bytes
                                if tokens
                                else 0,
                                "standalone_collection_metrics": collection_metrics,
                                "routed": compare(
                                    actual,
                                    expected,
                                    launch.runtime.atol,
                                    launch.runtime.rtol,
                                ),
                            },
                        },
                    )
    except BaseException:
        send_record(supervisor, {"kind": "error", "detail": traceback.format_exc()})
        raise
    finally:
        supervisor.close()


def run(
    launch: DemandLaunch,
    gpus: list[int],
    batches: list[int],
    seeds: list[int],
    progress_path: Path,
) -> dict:
    context = multiprocessing.get_context("spawn")
    data = [[context.Pipe() for _ in range(WORKER_COUNT)] for _ in range(CLIENT_COUNT)]
    client_control = [context.Pipe() for _ in range(CLIENT_COUNT)]
    worker_control = [context.Pipe() for _ in range(WORKER_COUNT)]
    supervisors = [context.Pipe() for _ in range(GPU_COUNT + 1)]
    children = [
        context.Process(
            target=client_process,
            args=(
                launch,
                gpus[i],
                i,
                tuple(pair[0] for pair in data[i]),
                client_control[i][0],
                supervisors[i][1],
            ),
            name=f"demand-client-{i}",
        )
        for i in range(CLIENT_COUNT)
    ] + [
        context.Process(
            target=worker_process,
            args=(
                launch,
                gpus[CLIENT_COUNT + i],
                i,
                tuple(data[c][i][1] for c in range(CLIENT_COUNT)),
                worker_control[i][0],
                supervisors[CLIENT_COUNT + i][1],
            ),
            name=f"demand-worker-{i}",
        )
        for i in range(WORKER_COUNT)
    ]
    children.append(
        context.Process(
            target=controller_process,
            args=(
                launch,
                tuple(pair[1] for pair in client_control),
                tuple(pair[1] for pair in worker_control),
                supervisors[-1][1],
            ),
            name="demand-controller",
        )
    )
    owned_groups: set[int] = set()
    all_pipes = (
        [pair for row in data for pair in row]
        + client_control
        + worker_control
        + supervisors
    )
    records = []
    overlaps = 0
    try:
        for child in children:
            child.start()
        for pair in all_pipes[: -len(supervisors)]:
            for connection in pair:
                connection.close()
        for _, child_end in supervisors:
            child_end.close()
        for index, (supervisor, _) in enumerate(supervisors):
            record = receive_record(supervisor, launch.runtime.timeout_s)
            if record["kind"] != "started" or record["pid"] != children[index].pid:
                raise RuntimeError(
                    "Validation child did not acknowledge its process group"
                )
            owned_groups.add(record["pid"])
        for supervisor, _ in supervisors:
            if (
                receive_record(supervisor, launch.runtime.timeout_s * 4)["kind"]
                != "ready"
            ):
                raise RuntimeError("Demand validation process did not become ready")
        matrix = [
            ("natural", tokens, seed, "native", "native")
            for seed in seeds
            for tokens in batches
        ]
        matrix += [
            (name, max(batches), seeds[-1], left, right)
            for name, left, right in (
                ("same_even", "even", "even"),
                ("same_odd", "odd", "odd"),
                ("opposite", "even", "odd"),
                ("opposite_swapped", "odd", "even"),
                ("mixed", "mixed", "mixed"),
            )
        ]
        matrix.append(("empty", 0, seeds[-1], "native", "native"))
        for phase, tokens, seed, left, right in matrix:
            for index, routes in enumerate((left, right)):
                send_record(
                    supervisors[index][0],
                    {
                        "op": "prepare",
                        "case": {
                            "phase": phase,
                            "tokens": tokens,
                            "seed": seed + index,
                            "routes": routes,
                        },
                    },
                )
            for supervisor, _ in supervisors[:CLIENT_COUNT]:
                if (
                    receive_record(supervisor, launch.runtime.timeout_s * 3)["kind"]
                    != "prepared"
                ):
                    raise RuntimeError(
                        "Client did not prepare its independent reference"
                    )
            for supervisor, _ in supervisors[:CLIENT_COUNT]:
                send_record(supervisor, {"op": "execute"})
            pair_results = []
            for supervisor, _ in supervisors[:CLIENT_COUNT]:
                record = receive_record(supervisor, launch.runtime.timeout_s * 3)
                if record["kind"] != "case":
                    raise RuntimeError("Unexpected demand validation result")
                pair_results.append(record["case"])
            first, second = pair_results
            if (
                tokens
                and first["started_ns"] < second["finished_ns"]
                and second["started_ns"] < first["finished_ns"]
            ):
                overlaps += 1
            records.extend(pair_results)
            with progress_path.open("a") as progress:
                for record in pair_results:
                    progress.write(json.dumps(record, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "tokens": tokens,
                        "passed_cases": len(pair_results),
                    }
                ),
                flush=True,
            )
        if not overlaps:
            raise AssertionError("No overlapping A-side calls were established")
        for supervisor, _ in supervisors[:CLIENT_COUNT]:
            send_record(supervisor, {"op": "close"})
        final = [
            receive_record(supervisor, launch.runtime.timeout_s)
            for supervisor, _ in supervisors
        ]
        if any(record["kind"] != "finished" for record in final):
            raise RuntimeError("Demand validation shutdown did not finish")
        ledger = final[-1]["ledger"]
        if (
            ledger["pending"]
            or ledger["outstanding"]
            or ledger["completed_parent_calls"] != len(records)
        ):
            raise AssertionError("Controller parent accounting did not drain")
        layer = str(launch.runtime.directory.placements[0].layer_id)
        for index, worker in enumerate(launch.directory.workers):
            report = final[CLIENT_COUNT + index]
            expected_calls = sum(
                worker.worker_id in case["selected_workers"] for case in records
            )
            expected_counts = {
                str(expert): sum(case["counts"][expert] for case in records)
                for expert in worker.placements[0].expert_ids
            }
            if (
                report["completed_calls"] != expected_calls
                or report["expert_assignments"][layer] != expected_counts
            ):
                raise AssertionError(
                    "Worker completed demand differs from submitted counts"
                )
            expected_bytes = sum(
                case["transfer_accounting"][worker.worker_id]["receive_bytes"]
                for case in records
            )
            dense_bytes = sum(
                case["tokens"]
                * launch.runtime.directory.top_k
                * launch.runtime.directory.hidden_size
                * BF16_OUTPUT_ELEMENT_BYTES
                for case in records
                if worker.worker_id in case["selected_workers"]
            )
            if (
                report["compact_output"] is not launch.compact_output
                or report["output_transfer"]["sent_bytes"] != expected_bytes
                or report["output_transfer"]["dense_equivalent_bytes"] != dense_bytes
            ):
                raise AssertionError(
                    "Worker output bytes differ from recorded transfers"
                )
            if (
                ledger["selected_worker_calls"][worker.worker_id] != expected_calls
                or ledger["admitted_assignments_by_worker_layer_expert"][
                    worker.worker_id
                ][layer]
                != expected_counts
            ):
                raise AssertionError("Controller demand differs from worker completion")
        for index, client in enumerate(final[:CLIENT_COUNT]):
            own = [case for case in records if case["client_id"] == f"client-{index}"]
            counts = client["status"]["demand"]
            expected = [
                sum(case["counts"][e] for case in own)
                for e in range(len(own[0]["counts"]))
            ]
            if (
                counts["parent_layer_calls"][layer] != len(own)
                or counts["expert_assignments"][layer] != expected
            ):
                raise AssertionError(
                    "A-side demand accounting differs from validation cases"
                )
            expected_bytes = sum(
                traffic["receive_bytes"]
                for case in own
                for traffic in case["transfer_accounting"].values()
            )
            dense_bytes = sum(case["dense_equivalent_output_bytes"] for case in own)
            if (
                client["status"]["compact_output"] is not launch.compact_output
                or client["status"]["output_transfer"]["received_bytes"]
                != expected_bytes
                or client["status"]["output_transfer"]["dense_equivalent_bytes"]
                != dense_bytes
            ):
                raise AssertionError("A output bytes differ from recorded transfers")
            for worker in launch.directory.workers:
                channel = client["status"]["workers"][worker.worker_id][
                    "output_transfer"
                ]
                channel_actual = sum(
                    case["transfer_accounting"][worker.worker_id]["receive_bytes"]
                    for case in own
                )
                channel_dense = sum(
                    case["tokens"]
                    * launch.runtime.directory.top_k
                    * launch.runtime.directory.hidden_size
                    * BF16_OUTPUT_ELEMENT_BYTES
                    for case in own
                    if worker.worker_id in case["selected_workers"]
                )
                if (
                    channel["received_bytes"] != channel_actual
                    or channel["dense_equivalent_bytes"] != channel_dense
                ):
                    raise AssertionError(
                        "A channel bytes differ from recorded transfers"
                    )
        if launch.receive_slots > 1:
            verify_pipeline(
                final[CLIENT_COUNT : CLIENT_COUNT + WORKER_COUNT],
                ledger,
                launch.receive_slots,
            )
        for child in children:
            child.join(20)
            if child.exitcode != 0:
                raise RuntimeError("Validation child did not exit cleanly")
        return {
            "cases": records,
            "clients": final[:CLIENT_COUNT],
            "workers": final[CLIENT_COUNT:GPU_COUNT],
            "controller": ledger,
            "overlapping_call_pairs": overlaps,
            "all_demand_and_transfer_counts_match": True,
            "compact_output": launch.compact_output,
            "output_transfer": {
                "received_bytes": sum(
                    traffic["receive_bytes"]
                    for case in records
                    for traffic in case["transfer_accounting"].values()
                ),
                "dense_equivalent_bytes": sum(
                    case["dense_equivalent_output_bytes"] for case in records
                ),
            },
        }
    finally:
        stop_children(children, owned_groups)
        for pair in all_pipes:
            for connection in pair:
                connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpus", nargs=GPU_COUNT, type=int, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19])
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Return only this worker's routed Top-k slots; input payload stays dense",
    )
    parser.add_argument("--receive-slots", type=int, default=1)
    parser.add_argument("--pipeline-trace-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.receive_slots <= MAX_RECEIVE_SLOTS or (
        args.receive_slots > 1 and not args.compact_output
    ):
        parser.error("Multiple receive slots require compact output")
    if args.pipeline_trace_dir is not None:
        if args.receive_slots == 1 or args.pipeline_trace_dir.exists():
            parser.error(
                "Select a new private trace directory for multi-slot validation"
            )
        args.pipeline_trace_dir.mkdir(parents=True)
    if len(set(args.gpus)) != GPU_COUNT or min(args.gpus) < 0 or min(args.batches) <= 0:
        parser.error("Select four distinct GPUs and positive batch sizes")
    if args.timeout <= 0 or any(
        not math.isfinite(v) or v < 0 for v in (args.atol, args.rtol)
    ):
        parser.error("Invalid timeout or tolerances")
    if args.atol > DEFAULT_ATOL or args.rtol > DEFAULT_RTOL:
        parser.error("This correctness test does not permit relaxed tolerances")
    if args.output.exists():
        parser.error("Choose a new private output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "scope": (
            "Real 2A-client + 2E-worker + CPU-Controller one-layer validation; "
            "no online A/KV service or performance claim"
        ),
        "input_kind": (
            "Seeded synthetic hidden states; natural Router cases and "
            "explicitly controlled routing boundaries"
        ),
        "validation_instrumentation": (
            "A-side native references and CPU routing-count checks; "
            "transfer counters are local instrumentation"
        ),
        "demand_timing_limitations": (
            "Pinned count copy explicitly synchronizes; "
            "ready_ns is a host observation, not exact GPU readiness"
        ),
        "model": str(args.model.resolve()),
        "physical_gpus": args.gpus,
        "compact_output": args.compact_output,
        "receive_slots": args.receive_slots,
        "pipeline_trace_dir": str(args.pipeline_trace_dir)
        if args.pipeline_trace_dir
        else None,
        "tolerances_declared_before_execution": {"atol": args.atol, "rtol": args.rtol},
    }
    try:
        source_root = Path(__file__).resolve().parents[2]
        if Path(afd_plugin.__file__).resolve().parent != source_root / "afd_plugin":
            raise RuntimeError("Python imported a different AFD source tree")
        manifest = source_root / "source_manifest.json"
        if manifest.exists():
            report["source_manifest"] = json.loads(manifest.read_text())
            for name, expected in report["source_manifest"]["files"].items():
                if (
                    hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise RuntimeError("Synced source changed since its manifest")
        with args.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report["initial_gpu_state"] = check_idle(args.gpus)
            raw = json.loads((args.model / "config.json").read_text())
            if (
                not raw["first_k_dense_replace"]
                <= args.layer
                < raw["num_hidden_layers"]
                or args.layer % raw["moe_layer_freq"]
            ):
                raise ValueError("Select an existing MoE layer")
            num_experts, top_k = raw["n_routed_experts"], raw["num_experts_per_tok"]
            if num_experts // WORKER_COUNT < top_k or num_experts % WORKER_COUNT:
                raise ValueError(
                    "Boundary cases require equally sized disjoint partitions"
                )
            model_id = uuid.uuid4().hex
            workers = tuple(
                StaticDirectory(
                    model_id,
                    1,
                    f"worker-{index}",
                    (
                        ExpertPlacement(
                            args.layer,
                            num_experts,
                            tuple(reversed(range(index, num_experts, WORKER_COUNT))),
                        ),
                    ),
                    raw["hidden_size"],
                    top_k,
                    max(args.batches),
                    allow_partial_experts=True,
                )
                for index in range(WORKER_COUNT)
            )
            reference = StaticDirectory(
                model_id,
                1,
                "reference",
                (ExpertPlacement(args.layer, num_experts, tuple(range(num_experts))),),
                raw["hidden_size"],
                top_k,
                max(args.batches),
            )
            sockets = [socket.socket() for _ in range(CLIENT_COUNT * WORKER_COUNT)]
            try:
                for reserved in sockets:
                    reserved.bind(("127.0.0.1", 0))
                ports = tuple(reserved.getsockname()[1] for reserved in sockets)
            finally:
                for reserved in sockets:
                    reserved.close()
            launch = DemandLaunch(
                Launch(
                    str(args.model.resolve()),
                    reference,
                    ports[:WORKER_COUNT],
                    args.timeout,
                    args.atol,
                    args.rtol,
                ),
                PoolDirectory(workers, expert_partitioned=True),
                (ports[:WORKER_COUNT], ports[WORKER_COUNT:]),
                compact_output=args.compact_output,
                receive_slots=args.receive_slots,
                pipeline_trace_dir=(
                    str(args.pipeline_trace_dir) if args.pipeline_trace_dir else None
                ),
            )
            progress_path = args.output.with_suffix(".calls.jsonl")
            progress_path.touch(exist_ok=False)
            report["call_records"] = str(progress_path)
            report.update(
                run(launch, args.gpus, args.batches, args.seeds, progress_path)
            )
            report["status"] = "passed"
    except BusyGPUError as error:
        report["status"] = "blocked_preflight"
        report["error"] = str(error)
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RESULT {report['status']}: {args.output}", flush=True)
    return (
        0
        if report["status"] == "passed"
        else 2
        if report["status"] == "blocked_preflight"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
