#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Single-host multi-A/multi-E pilot against equal-budget full TP1 replicas.

Two independent service domains, deterministic offered traffic, optional
per-domain SLOs. Native defaults retain engine optimizations; the matched
profile is diagnostic. Finite traces include fill/drain, not steady-state
capacity. This tool does not implement cross-host Pool communication.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import random
import socket
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict
from multiprocessing.connection import Connection
from pathlib import Path

import afd_plugin
from afd_plugin.expert_pool.batching import BatchingOptions
from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ControllerConfig,
    ExecutionOptions,
    PoolDeployment,
    WorkerPlacement,
)
from tools.expert_pool.benchmark_workload import (
    arrival_offsets,
    assignments,
    domain_replicas,
    mixed_lengths,
    open_loop_assignments,
    summarize,
    verify_cost_accounting,
)
from tools.expert_pool.benchmark_workload import (
    percentile as percentile,
)
from tools.expert_pool.direct_accounting import verify_direct_pipeline
from tools.expert_pool.validate_engines import (
    controller_entry,
    expert_entry,
    process_environment,
    stop_children,
)
from tools.expert_pool.validate_service import (
    BusyGPUError,
    check_idle,
    send_record,
)
from tools.expert_pool.validate_service import (
    receive_record as receive_validation_record,
)

START_DELAY_NS = 1_000_000_000
MAX_BENCHMARK_RECORD_BYTES = 16 * 1024 * 1024
FIXTURE_SENTENCES = (
    "A library stores books and makes them available to readers.",
    "An experiment compares observations under specified conditions.",
    "The programmer checks inputs before calculating the result.",
    "A train carries passengers between cities along a railway.",
    "A table presents measurements in rows and columns.",
    "A garden needs sunlight, water and suitable soil.",
)


def receive_record(connection: Connection, timeout_s: float) -> dict:
    # Bounded local supervisor reports include many shape buckets and streamed
    # request timestamps. The per-layer control protocol retains its small cap.
    return receive_validation_record(
        connection, timeout_s, max_bytes=MAX_BENCHMARK_RECORD_BYTES
    )


def fixtures(model: Path, max_count: int) -> dict[int, list[list[int]]]:
    # Parent remains GPU-free; tokenization is outside every timed request.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    random_source = random.Random(19)
    result = {128: [], 512: []}
    for index in range(2 * max_count):
        text = f"Read passage {index} and continue it. " + " ".join(
            random_source.choice(FIXTURE_SENTENCES) for _ in range(100)
        )
        tokens = tokenizer.encode(text, add_special_tokens=True)
        for length in result:
            if len(tokens) < length:
                raise ValueError("Fixture did not reach the requested input length")
            result[length].append(tokens[:length])
    return result


async def engine_loop(
    model: str,
    deployment: str | None,
    client_id: str,
    control: Connection,
    config: dict,
    profile_dir: str | None,
) -> None:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    from afd_plugin.expert_pool import register_expert_pool

    options = {}
    if deployment or config["native_profile"] == "matched":
        options.update(
            enforce_eager=True,
            async_scheduling=False,
            kernel_config={"moe_backend": "triton"},
        )
    if deployment:
        register_expert_pool()
        options.update(
            worker_cls="afd_plugin.v1.worker.pool_attention_worker.PoolAttentionWorker",
            hf_overrides={"architectures": ["PoolDeepseekV2ForCausalLM"]},
            additional_config={
                "expert_pool": {"deployment": deployment, "client_id": client_id}
            },
        )
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=model,
            dtype="bfloat16",
            seed=0,
            max_model_len=config["max_model_len"],
            max_num_batched_tokens=config["max_batch_tokens"],
            max_num_seqs=config["max_sequences"],
            kv_cache_memory_bytes=(
                config["kv_cache_mib"] * 1024 * 1024 if config["kv_cache_mib"] else None
            ),
            gpu_memory_utilization=config["gpu_memory_utilization"],
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            disable_log_stats=True,
            profiler_config=(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": profile_dir,
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": False,
                    "ignore_frontend": True,
                    "max_iterations": 4,
                }
                if profile_dir
                else {}
            ),
            **options,
        )
    )
    try:
        send_record(control, {"kind": "ready", "client_id": client_id})
        while True:
            command = await asyncio.to_thread(
                receive_record, control, config["timeout"]
            )
            if command["kind"] == "close":
                status = (
                    await engine.collective_rpc("pool_status") if deployment else None
                )
                if deployment:
                    await engine.collective_rpc("close_pool")
                send_record(control, {"kind": "closed", "pool_status": status})
                break
            if command["kind"] == "prepare":
                if deployment:
                    await engine.collective_rpc(
                        "pool_set_metrics", args=(command["measure"],)
                    )
                send_record(control, {"kind": "prepared"})
                continue
            if command["kind"] != "bench":
                raise ValueError("Unexpected benchmark command")
            if command["trace"]:
                await engine.start_profile(command["trial"])
            await asyncio.sleep(
                max(0, (command["start_ns"] - time.perf_counter_ns()) / 1e9)
            )
            records = []

            async def request(item: dict, command=command, records=records) -> None:
                planned = command["start_ns"] + item["arrival_offset_ns"]
                if command["open_loop"]:
                    await asyncio.sleep(
                        max(0, (planned - time.perf_counter_ns()) / 1e9)
                    )
                submitted = time.perf_counter_ns()
                started = planned if command["open_loop"] else submitted
                request_id = f"{client_id}-{command['trial']}-{item['index']}"
                params = SamplingParams(
                    temperature=0,
                    max_tokens=config["output_tokens"],
                    ignore_eos=True,
                    detokenize=False,
                    output_kind=RequestOutputKind.DELTA,
                )
                tokens, chunks = [], []

                async def consume() -> None:
                    finished = False
                    async for output in engine.generate(
                        {"prompt_token_ids": item["tokens"]},
                        params,
                        request_id,
                    ):
                        new_tokens = list(output.outputs[0].token_ids)
                        if new_tokens:
                            chunks.append((time.perf_counter_ns(), len(new_tokens)))
                            tokens.extend(new_tokens)
                        finished = output.finished
                    if (
                        not finished
                        or len(tokens) != config["output_tokens"]
                        or not chunks
                    ):
                        raise RuntimeError("Incomplete generated output")

                status, error = "completed", None
                try:
                    # Offered arrival owns the deadline, including generator lag.
                    remaining = config["request_timeout"] - (submitted - started) / 1e9
                    if remaining <= 0:
                        raise asyncio.TimeoutError()
                    await asyncio.wait_for(consume(), timeout=remaining)
                except asyncio.TimeoutError:
                    status, error = "timeout", "Request deadline exceeded"
                    await engine.abort(request_id)
                ended = time.perf_counter_ns()
                record = {
                    "request_id": request_id,
                    "index": item["index"],
                    "domain": command["domain"],
                    "replica": client_id,
                    "input_tokens": len(item["tokens"]),
                    "output_tokens": len(tokens),
                    "status": status,
                    "error": error,
                    "started_ns": started,
                    "submitted_ns": submitted,
                    "finished_ns": ended,
                    "submission_lag_ms": (submitted - started) / 1e6,
                    "e2e_ms": (ended - started) / 1e6,
                    "stream_chunks": chunks,
                }
                if status == "completed":
                    record.update(
                        ttft_ms=(chunks[0][0] - started) / 1e6,
                        tpot_ms=(chunks[-1][0] - chunks[0][0])
                        / 1e6
                        / (len(tokens) - 1),
                        output_sha256=hashlib.sha256(
                            json.dumps(tokens).encode()
                        ).hexdigest(),
                    )
                records.append(record)

            if command["open_loop"]:
                await asyncio.gather(*(request(item) for item in command["requests"]))
            else:
                cursor = iter(command["requests"])

                async def slot(cursor=cursor) -> None:
                    for item in cursor:
                        await request(item)

                await asyncio.gather(*(slot() for _ in range(command["concurrency"])))
            if command["trace"]:
                await engine.stop_profile()
            status = await engine.collective_rpc("pool_status") if deployment else None
            send_record(
                control,
                {"kind": "bench_done", "records": records, "pool_status": status},
            )
    finally:
        engine.shutdown(timeout=10)


def engine_process(
    model: str,
    gpu: int,
    deployment: str | None,
    client_id: str,
    config: dict,
    profile_dir: str | None,
    control: Connection,
) -> None:
    process_environment(gpu)
    send_record(control, {"kind": "spawned", "pid": os.getpid()})
    try:
        asyncio.run(
            engine_loop(model, deployment, client_id, control, config, profile_dir)
        )
    except BaseException:
        send_record(control, {"kind": "error", "detail": traceback.format_exc()})
        raise
    finally:
        control.close()


def make_deployment(args: argparse.Namespace, temporary: Path) -> PoolDeployment:
    experts = json.loads((args.model / "config.json").read_text())["n_routed_experts"]
    if args.expert_workers > experts or args.replicated_experts > experts:
        raise ValueError("Expert worker/replica counts exceed logical expert count")
    placements = tuple(
        WorkerPlacement(
            f"worker-{i}",
            expert_ids=tuple(
                sorted(
                    set(range(i, experts, args.expert_workers))
                    | set(range(args.replicated_experts))
                )
            ),
        )
        for i in range(args.expert_workers)
    )
    sockets = [
        socket.socket() for _ in range(args.attention_workers * args.expert_workers)
    ]
    try:
        for reserved in sockets:
            reserved.bind(("127.0.0.1", 0))
        domains = {
            replica: domain
            for domain, replicas in enumerate(
                domain_replicas(args.attention_workers, 0)
            )
            for replica in replicas
        }
        endpoints = tuple(
            ClientEndpoint(
                f"client-{a}",
                1,
                f"domain-{domains[a]}",
                str(temporary / f"a{a}e{e}.sock"),
                sockets[a * args.expert_workers + e].getsockname()[1],
                f"worker-{e}",
            )
            for a in range(args.attention_workers)
            for e in range(args.expert_workers)
        )
    finally:
        for reserved in sockets:
            reserved.close()
    return PoolDeployment(
        str(args.model.resolve()),
        uuid.uuid4().hex,
        args.max_batch_tokens,
        endpoints,
        args.timeout,
        ExecutionOptions(**args.execution_options),
        workers=placements,
        controller=(
            None
            if args.direct_dispatch
            else ControllerConfig(str(temporary), "ready_first")
        ),
        direct_dispatch=args.direct_dispatch,
        dispatch_mode="expert_partitioned",
        demand_aware=True,
        compact_output=True,
        receive_slots=args.receive_slots,
        batching=BatchingOptions(
            args.batch_max_calls, args.batch_max_tokens, args.batch_wait_us
        ),
        expert_replicated=bool(args.replicated_experts),
    )


def run_mode(
    args: argparse.Namespace, mode: str, inputs: dict, report: dict, progress: Path
) -> None:
    context = multiprocessing.get_context("spawn")
    children, connections, groups = [], [], set()

    def start(target, *arguments) -> Connection:
        parent, child = context.Pipe()
        connections.extend((parent, child))
        process = context.Process(target=target, args=(*arguments, child))
        children.append(process)
        process.start()
        child.close()
        if receive_record(parent, args.timeout) != {
            "kind": "spawned",
            "pid": process.pid,
        }:
            raise RuntimeError("Missing process-group startup acknowledgement")
        groups.add(process.pid)
        return parent

    try:
        with tempfile.TemporaryDirectory(prefix="pool-bench-") as temporary:
            path = Path(temporary) / "deployment.json"
            workers, controller = [], None
            replica_count = args.attention_workers if mode == "pool" else len(args.gpus)
            if mode == "pool":
                deployment = make_deployment(args, Path(temporary))
                path.write_text(json.dumps(asdict(deployment)))
                if deployment.controller is not None:
                    controller = start(controller_entry, str(path))
                workers = [
                    start(
                        expert_entry,
                        str(path),
                        args.gpus[args.attention_workers + e],
                        f"worker-{e}",
                    )
                    for e in range(args.expert_workers)
                ]
                report["pool_deployment"] = asdict(deployment)
            clients = [
                start(
                    engine_process,
                    str(args.model.resolve()),
                    args.gpus[i],
                    str(path) if mode == "pool" else None,
                    f"client-{i}",
                    vars(args),
                    str(args.profile_dir / mode / f"client-{i}")
                    if args.profile_dir
                    else None,
                )
                for i in range(replica_count)
            ]
            for client in clients:
                if receive_record(client, args.timeout)["kind"] != "ready":
                    raise RuntimeError("Engine failed startup")
            print(f"{mode}: {replica_count} engines ready", flush=True)
            lengths = ["mixed"] if args.mixed_inputs else args.input_lengths
            loads = args.rps if args.load_mode == "open" else args.concurrency
            max_count = len(inputs[128]) // 2
            slos = (
                {d: (args.ttft_slo_ms[d], args.tpot_slo_ms[d]) for d in range(2)}
                if args.ttft_slo_ms
                else None
            )
            for repeat in range(args.repeats):
                cases = [(length, load) for length in lengths for load in loads]
                if repeat % 2:
                    cases.reverse()
                for length, load in cases:
                    count = (
                        args.min_requests
                        if args.load_mode == "open"
                        else max(args.min_requests, args.waves * load)
                    )
                    plans = (
                        open_loop_assignments(replica_count, count, repeat)
                        if args.load_mode == "open"
                        else assignments(mode, load, count, repeat, replica_count)
                    )
                    offsets = {
                        d: arrival_offsets(
                            count, load / 2, args.arrival, 19 + repeat * 2 + d
                        )
                        if args.load_mode == "open"
                        else [0] * count
                        for d in range(2)
                    }
                    trial_id = f"{mode}-r{repeat}-in{length}-{args.load_mode}{load}"
                    lengths_by_domain = {
                        d: mixed_lengths(count, args.input_lengths, 41 + repeat * 2 + d)
                        for d in range(2)
                    }
                    measured = []
                    for measure in (False, True):
                        for plan in plans:
                            send_record(
                                clients[plan["replica"]],
                                {"kind": "prepare", "measure": measure},
                            )
                        for plan in plans:
                            if (
                                receive_record(clients[plan["replica"]], args.timeout)[
                                    "kind"
                                ]
                                != "prepared"
                            ):
                                raise RuntimeError("Benchmark preparation failed")
                        start_ns = time.perf_counter_ns() + START_DELAY_NS
                        workload = []
                        for plan in plans:
                            indices = (
                                plan["indices"]
                                if measure
                                else plan["indices"][: max(2, plan["concurrency"] * 2)]
                            )
                            requests = []
                            for index in indices:
                                token_length = (
                                    lengths_by_domain[plan["domain"]][index]
                                    if length == "mixed"
                                    else length
                                )
                                item = {
                                    "index": index,
                                    "tokens": inputs[token_length][
                                        index + plan["domain"] * max_count
                                    ],
                                    "arrival_offset_ns": offsets[plan["domain"]][index]
                                    if measure
                                    else 0,
                                }
                                requests.append(item)
                                workload.append({"domain": plan["domain"], **item})
                            send_record(
                                clients[plan["replica"]],
                                {
                                    "kind": "bench",
                                    "trial": trial_id
                                    + ("-measure" if measure else "-warmup"),
                                    "start_ns": start_ns,
                                    "measure": measure,
                                    "trace": bool(
                                        args.profile_dir
                                        and measure
                                        and repeat == 0
                                        and (length, load) == cases[0]
                                    ),
                                    "domain": plan["domain"],
                                    "concurrency": max(1, plan["concurrency"]),
                                    "open_loop": measure and args.load_mode == "open",
                                    "requests": requests,
                                },
                            )
                        received = [
                            receive_record(clients[plan["replica"]], args.timeout)
                            for plan in plans
                        ]
                        if any(
                            response["kind"] != "bench_done" for response in received
                        ):
                            raise RuntimeError("Malformed benchmark response")
                        if measure:
                            measured = received
                        elif any(
                            r["status"] != "completed"
                            for response in received
                            for r in response["records"]
                        ):
                            raise RuntimeError("Warmup requests failed")
                    records = [r for response in measured for r in response["records"]]
                    expected = {(d, i) for d in range(2) for i in range(count)}
                    if {(r["domain"], r["index"]) for r in records} != expected or len(
                        records
                    ) != len(expected):
                        raise AssertionError("Dropped or duplicated offered requests")
                    window = (
                        min(r["started_ns"] for r in records),
                        max(r["finished_ns"] for r in records),
                    )
                    result = {
                        "mode": mode,
                        "trial": trial_id,
                        "repeat": repeat,
                        "input_tokens": length,
                        "output_tokens": args.output_tokens,
                        "load_mode": args.load_mode,
                        "load": load,
                        "target_total_rps": load if args.load_mode == "open" else None,
                        "planned_arrival_span_s": (
                            (
                                max(r["started_ns"] for r in records)
                                - min(r["started_ns"] for r in records)
                            )
                            / 1e9
                            if args.load_mode == "open"
                            else None
                        ),
                        "gpu_budget": len(args.gpus),
                        "profiling_run": args.profile_dir is not None,
                        "trace_sha256": hashlib.sha256(
                            json.dumps(
                                sorted(
                                    workload, key=lambda r: (r["domain"], r["index"])
                                ),
                                sort_keys=True,
                            ).encode()
                        ).hexdigest(),
                        "plans": plans,
                        "summary": summarize(records, window=window, slos=slos),
                        "domains": {
                            str(d): summarize(
                                [r for r in records if r["domain"] == d],
                                window=window,
                                slos=slos,
                            )
                            for d in range(2)
                        },
                        "records": records,
                        "pool_status": [
                            s
                            for response in measured
                            for s in (response["pool_status"] or [])
                        ],
                    }
                    with progress.open("a") as output:
                        output.write(json.dumps(result, allow_nan=False) + "\n")
                    report["trials"].append(result)
                    print(
                        f"{trial_id}: "
                        f"{result['summary']['output_throughput_tps']:.2f} tok/s",
                        flush=True,
                    )
            for client in clients:
                send_record(client, {"kind": "close"})
            closed = []
            for client in clients:
                response = receive_record(client, args.timeout)
                if response["kind"] != "closed":
                    raise RuntimeError("Engine did not close cleanly")
                closed.extend(response["pool_status"] or [])
            if workers:
                report["workers"] = [
                    receive_record(worker, args.timeout) for worker in workers
                ]
            if controller is not None:
                report["controller"] = receive_record(controller, args.timeout)
                if args.execution_options["collect_cost_feedback"]:
                    report["cost_accounting"] = verify_cost_accounting(
                        report["controller"], report["workers"]
                    )
            if mode == "pool" and args.direct_dispatch:
                report["direct_accounting"] = verify_direct_pipeline(
                    report["workers"], closed, args.receive_slots
                )
            for child in children:
                child.join(args.timeout)
                if child.exitcode != 0:
                    raise RuntimeError("Benchmark child did not exit cleanly")
    finally:
        stop_children(children, groups)
        for connection in connections:
            connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--attention-workers", type=int, default=2)
    parser.add_argument("--expert-workers", type=int, default=2)
    parser.add_argument(
        "--modes", nargs="+", choices=("native", "pool"), default=["native", "pool"]
    )
    parser.add_argument(
        "--native-profile", choices=("optimized", "matched"), default="optimized"
    )
    parser.add_argument("--load-mode", choices=("closed", "open"), default="open")
    parser.add_argument(
        "--rps",
        type=float,
        nargs="+",
        default=[2, 4, 8],
        help="Total offered RPS, split equally between domains",
    )
    parser.add_argument(
        "--arrival", choices=("steady", "poisson", "burst"), default="steady"
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--input-lengths", type=int, nargs="+", choices=(128, 512), default=[128, 512]
    )
    parser.add_argument("--mixed-inputs", action="store_true")
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--ttft-slo-ms", type=float, nargs=2)
    parser.add_argument("--tpot-slo-ms", type=float, nargs=2)
    parser.add_argument("--execution-options", type=json.loads, default={})
    parser.add_argument("--receive-slots", type=int, default=2)
    parser.add_argument("--direct-dispatch", action="store_true")
    parser.add_argument("--batch-max-calls", type=int, default=2)
    parser.add_argument("--batch-max-tokens", type=int, default=1024)
    parser.add_argument("--batch-wait-us", type=int, default=2000)
    parser.add_argument(
        "--replicated-experts",
        type=int,
        default=0,
        help="First N experts resident on every E; duplicates explicitly counted",
    )
    parser.add_argument("--max-batch-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument(
        "--kv-cache-mib",
        type=int,
        default=0,
        help="0 lets the engine size KV from its memory budget",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--min-requests", type=int, default=12, help="Requests per domain per case"
    )
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--request-timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        args.execution_options = asdict(
            ExecutionOptions(
                **{
                    "validate_client_values": False,
                    "validate_worker_values": False,
                    "reuse_cuda_events": True,
                    "defer_output_sync": True,
                    "warmup_before_ready": True,
                    "collect_cost_feedback": True,
                    **args.execution_options,
                }
            )
        )
        BatchingOptions(
            args.batch_max_calls, args.batch_max_tokens, args.batch_wait_us
        ).validate_capacity(args.receive_slots, args.max_batch_tokens)
        if (
            args.attention_workers < 2
            or args.attention_workers % 2
            or args.expert_workers < 1
        ):
            raise ValueError(
                "Use an even number of A workers for two equal domains and at "
                "least one E"
            )
        if (
            len(set(args.gpus)) != len(args.gpus)
            or len(args.gpus) != args.attention_workers + args.expert_workers
            or min(args.gpus) < 0
        ):
            raise ValueError("Distinct GPUs must match the A+E budget")
        if (
            len(set(args.modes)) != len(args.modes)
            or args.replicated_experts < 0
            or args.kv_cache_mib < 0
        ):
            raise ValueError("Invalid mode or memory/replica configuration")
        if args.direct_dispatch and args.receive_slots != args.attention_workers:
            raise ValueError("Direct dispatch needs one dedicated slot per A worker")
        if (
            min(
                args.repeats,
                args.min_requests,
                args.waves,
                args.timeout,
                args.max_batch_tokens,
                args.max_sequences,
            )
            <= 0
            or args.output_tokens < 2
        ):
            raise ValueError("Counts must be positive and output length at least two")
        if min(args.concurrency) <= 0 or max(args.concurrency) > args.max_sequences:
            raise ValueError("Domain concurrency exceeds configured sequence capacity")
        if max(args.input_lengths) + args.output_tokens > args.max_model_len:
            raise ValueError("Context capacity is smaller than the requested fixture")
        if (
            not 0 < args.gpu_memory_utilization < 1
            or not math.isfinite(args.request_timeout)
            or args.request_timeout <= 0
        ):
            raise ValueError("Invalid memory fraction or request timeout")
        if any(not math.isfinite(v) or v <= 0 for v in args.rps):
            raise ValueError("Offered RPS must be finite and positive")
        if bool(args.ttft_slo_ms) != bool(args.tpot_slo_ms) or any(
            not math.isfinite(v) or v <= 0
            for v in (args.ttft_slo_ms or []) + (args.tpot_slo_ms or [])
        ):
            raise ValueError(
                "Specify positive TTFT and TPOT limits for both domains, or neither"
            )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error("Choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "input_kind": (
            "Distinct fixed-length tokenized synthetic natural-language "
            "inputs; identical domain/index traces across deployments"
        ),
        "timing_scope": (
            "AsyncLLM streaming; excludes tokenization/HTTP; open-loop TTFT "
            "starts at planned arrival and includes submission lag"
        ),
        "tpot_scope": (
            "Per-request mean (last nonempty chunk - first nonempty "
            "chunk)/(output tokens - 1); not individual-token tail latency"
        ),
        "routing": (
            "Static round-robin within independent domains; odd native replica"
            " advantage swaps by repetition"
        ),
        "native_config": (
            "Full TP1 replicas on every budgeted GPU; native engine defaults "
            "unless matched profile selected"
        ),
        "pool_config": (
            "Configurable A/KV and E workers on one host; static placement and"
            " optional physical replicas"
        ),
        "limits": [
            (
                "Finite pilot includes fill/drain, not steady-state capacity; "
                "synthetic fixtures are not a representative dataset"
            ),
            (
                "Per-domain Goodput uses the same global window; timeouts count as"
                " SLO misses; absent SLO means no Goodput claim"
            ),
            (
                "Cost feedback includes initialization and warmup; event intervals"
                " include host submission gaps, not active GEMM cost"
            ),
            (
                "Profiling runs are diagnostic; matched eager/Triton is not the "
                "primary optimized native baseline"
            ),
            (
                "No cross-host Pool transport or strict E compute quota isolation "
                "is implemented by this benchmark"
            ),
        ],
        "trials": [],
    }
    progress = args.output.with_suffix(".trials.jsonl")
    try:
        root = Path(__file__).resolve().parents[2]
        if Path(afd_plugin.__file__).resolve().parent != root / "afd_plugin":
            raise RuntimeError("Wrong source tree imported; set PYTHONPATH")
        manifest_path = root / "source_manifest.json"
        if manifest_path.exists():
            report["source_manifest"] = json.loads(manifest_path.read_text())
            for name, expected in report["source_manifest"]["files"].items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Source changed after synchronization")
        with args.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            max_count = (
                args.min_requests
                if args.load_mode == "open"
                else max(args.min_requests, args.waves * max(args.concurrency))
            )
            inputs = fixtures(args.model, max_count)
            report["input_sha256"] = hashlib.sha256(
                json.dumps(inputs).encode()
            ).hexdigest()
            progress.touch(exist_ok=False)
            report["initial_gpu_states"] = []
            for mode in args.modes:
                report["initial_gpu_states"].append(
                    {"mode": mode, "gpus": check_idle(args.gpus)}
                )
                run_mode(args, mode, inputs, report, progress)
            if set(args.modes) == {"native", "pool"}:
                traces = {
                    mode: sorted(
                        t["trace_sha256"] for t in report["trials"] if t["mode"] == mode
                    )
                    for mode in args.modes
                }
                if traces["native"] != traces["pool"]:
                    raise AssertionError(
                        "Native and Pool offered different workload traces"
                    )
        report["status"] = "passed"
    except BusyGPUError as error:
        report.update(status="blocked_preflight", error=str(error))
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"RESULT {report['status']}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
