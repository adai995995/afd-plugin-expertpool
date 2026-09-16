#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Closed-loop, equal-GPU-budget pilot measurement of the eager Pool prototype.

Pool: two independent A/KV engines and one E GPU. Native: three complete TP1
replicas assigned 2+1 to the same two service domains, swapped between repeats.
Both use eager/Triton, identical fixed-length token inputs and output lengths,
and identical domain concurrency limits. This is an engine-level diagnostic,
not a tuned native-default comparison, HTTP benchmark or SLO capacity result.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import multiprocessing
import os
import random
import socket
import tempfile
import time
import traceback
import uuid
from collections.abc import Iterator
from dataclasses import asdict
from multiprocessing.connection import Connection
from pathlib import Path

import afd_plugin
from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ExecutionOptions,
    PoolDeployment,
)
from tools.expert_pool.validate_engines import (
    expert_process,
    process_environment,
    stop_children,
)
from tools.expert_pool.validate_service import (
    BusyGPUError,
    check_idle,
    receive_record,
    send_record,
)

MAX_BATCH_TOKENS = 512
MAX_MODEL_LEN = 1024
MAX_SEQUENCES = 16
KV_CACHE_BYTES = 512 * 1024 * 1024
OUTPUT_TOKENS = 32
START_DELAY_NS = 500_000_000
FIXTURE_SENTENCES = (
    "A library stores books and makes them available to readers.",
    "An experiment compares observations under specified conditions.",
    "The programmer checks inputs before calculating the result.",
    "A train carries passengers between cities along a railway.",
    "A table presents measurements in rows and columns.",
    "A garden needs sunlight, water and suitable soil.",
)


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("A percentile requires samples and a fraction in [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("No completed requests")
    window_s = (
        max(r["finished_ns"] for r in records) - min(r["started_ns"] for r in records)
    ) / 1e9
    if window_s <= 0:
        raise ValueError("Nonpositive observation window")
    summary = {
        "requests": len(records),
        "window_s": window_s,
        "request_throughput_rps": len(records) / window_s,
        "output_throughput_tps": sum(r["output_tokens"] for r in records) / window_s,
    }
    for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [r[name] for r in records]
        summary[name] = {
            "mean": sum(values) / len(values),
            "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
        }
    return summary


def assignments(mode: str, concurrency: int, count: int, repeat: int) -> list[dict]:
    """Bound domain concurrency exactly; split requests evenly within a domain."""
    if concurrency < 1 or count < concurrency:
        raise ValueError("Need enough requests for each positive concurrency")
    if mode == "pool":
        domains = ((0,), (1,))
    elif mode == "native":
        domains = ((0, 1), (2,)) if repeat % 2 == 0 else ((0,), (1, 2))
    else:
        raise ValueError("Unknown deployment mode")
    plans = []
    for domain, replicas in enumerate(domains):
        # At domain concurrency one, only one replica can be active at a time.
        active = replicas[: min(len(replicas), concurrency)]
        for position, replica in enumerate(active):
            slots = concurrency // len(active) + int(
                position < concurrency % len(active)
            )
            plans.append(
                {
                    "replica": replica,
                    "domain": domain,
                    "concurrency": slots,
                    "indices": list(range(position, count, len(active))),
                }
            )
    return plans


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
    timeout_s: int,
    profile_dir: str | None,
) -> None:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    from afd_plugin.expert_pool import register_expert_pool

    options = {}
    if deployment:
        register_expert_pool()
        options = {
            "worker_cls": (
                "afd_plugin.v1.worker.pool_attention_worker.PoolAttentionWorker"
            ),
            "hf_overrides": {"architectures": ["PoolDeepseekV2ForCausalLM"]},
            "additional_config": {
                "expert_pool": {"deployment": deployment, "client_id": client_id}
            },
        }
    engine = AsyncLLM.from_engine_args(
        AsyncEngineArgs(
            model=model,
            dtype="bfloat16",
            enforce_eager=True,
            seed=0,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=MAX_BATCH_TOKENS,
            max_num_seqs=MAX_SEQUENCES,
            kv_cache_memory_bytes=KV_CACHE_BYTES,
            gpu_memory_utilization=0.4,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            async_scheduling=False,
            kernel_config={"moe_backend": "triton"},
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
            command = await asyncio.to_thread(receive_record, control, timeout_s)
            if command["kind"] == "close":
                if deployment:
                    await engine.collective_rpc("close_pool")
                send_record(control, {"kind": "closed"})
                break
            if command["kind"] != "bench":
                raise ValueError("Unexpected benchmark command")
            if deployment:
                await engine.collective_rpc(
                    "pool_set_metrics", args=(command["measure"],)
                )
            if command["trace"]:
                await engine.start_profile(command["trial"])
            delay = (command["start_ns"] - time.perf_counter_ns()) / 1e9
            if delay > 0:
                await asyncio.sleep(delay)
            cursor = iter(command["requests"])
            records = []

            async def slot(
                cursor: Iterator[dict], command: dict, records: list[dict]
            ) -> None:
                for item in cursor:
                    request_id = f"{client_id}-{command['trial']}-{item['index']}"
                    params = SamplingParams(
                        temperature=0,
                        max_tokens=OUTPUT_TOKENS,
                        ignore_eos=True,
                        detokenize=False,
                        output_kind=RequestOutputKind.DELTA,
                    )
                    started = time.perf_counter_ns()
                    tokens = []
                    chunks = []
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
                    ended = time.perf_counter_ns()
                    if not finished or len(tokens) != OUTPUT_TOKENS or not chunks:
                        raise RuntimeError("Incomplete or malformed generated output")
                    records.append(
                        {
                            "request_id": request_id,
                            "index": item["index"],
                            "domain": command["domain"],
                            "replica": client_id,
                            "input_tokens": len(item["tokens"]),
                            "output_tokens": len(tokens),
                            "started_ns": started,
                            "finished_ns": ended,
                            "ttft_ms": (chunks[0][0] - started) / 1e6,
                            "tpot_ms": (chunks[-1][0] - chunks[0][0])
                            / 1e6
                            / (len(tokens) - 1),
                            "e2e_ms": (ended - started) / 1e6,
                            "stream_chunks": chunks,
                            "output_sha256": hashlib.sha256(
                                json.dumps(tokens).encode()
                            ).hexdigest(),
                        }
                    )

            await asyncio.gather(
                *(slot(cursor, command, records) for _ in range(command["concurrency"]))
            )
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
    timeout_s: int,
    profile_dir: str | None,
    control: Connection,
) -> None:
    process_environment(gpu)
    send_record(control, {"kind": "spawned", "pid": os.getpid()})
    try:
        asyncio.run(
            engine_loop(model, deployment, client_id, control, timeout_s, profile_dir)
        )
    except BaseException:
        send_record(control, {"kind": "error", "detail": traceback.format_exc()})
        raise
    finally:
        control.close()


def benchmark_expert_process(
    path: str, gpu: int, profile_dir: Path | None, control: Connection
) -> None:
    expert_process(path, gpu, control, profile_dir=profile_dir)


def run_mode(
    args: argparse.Namespace,
    mode: str,
    inputs: dict[int, list[list[int]]],
    report: dict,
    progress_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    children, connections = [], []
    groups = set()

    def start(target, *arguments) -> Connection:
        parent, child = context.Pipe()
        connections.extend((parent, child))
        process = context.Process(target=target, args=(*arguments, child))
        children.append(process)
        process.start()
        child.close()
        spawned = receive_record(parent, args.timeout)
        if spawned != {"kind": "spawned", "pid": process.pid}:
            raise RuntimeError("Missing process-group startup acknowledgement")
        groups.add(process.pid)
        return parent

    try:
        with tempfile.TemporaryDirectory(prefix="pool-bench-") as temporary:
            path = Path(temporary) / "deployment.json"
            worker = None
            if mode == "pool":
                sockets = [socket.socket() for _ in range(2)]
                try:
                    for reserved in sockets:
                        reserved.bind(("127.0.0.1", 0))
                    endpoints = tuple(
                        ClientEndpoint(
                            f"client-{i}",
                            1,
                            f"domain-{i}",
                            str(Path(temporary) / f"a{i}.sock"),
                            reserved.getsockname()[1],
                        )
                        for i, reserved in enumerate(sockets)
                    )
                finally:
                    for reserved in sockets:
                        reserved.close()
                deployment = PoolDeployment(
                    str(args.model.resolve()),
                    uuid.uuid4().hex,
                    MAX_BATCH_TOKENS,
                    endpoints,
                    args.timeout,
                    ExecutionOptions(**args.execution_options),
                )
                path.write_text(json.dumps(asdict(deployment)))
                worker = start(
                    benchmark_expert_process,
                    str(path),
                    args.gpus[2],
                    args.profile_dir / "pool" / "expert"
                    if args.profile_dir is not None
                    else None,
                )
            clients = [
                start(
                    engine_process,
                    str(args.model.resolve()),
                    args.gpus[i],
                    str(path) if mode == "pool" else None,
                    f"client-{i}",
                    args.timeout,
                    str(args.profile_dir / mode / f"client-{i}")
                    if args.profile_dir is not None
                    else None,
                )
                for i in range(2 if mode == "pool" else 3)
            ]
            for client in clients:
                receive_record(client, args.timeout)
            print(f"{mode}: engines ready", flush=True)
            for repeat in range(args.repeats):
                cases = [
                    (length, concurrency)
                    for length in inputs
                    for concurrency in args.concurrency
                ]
                if repeat % 2:
                    cases.reverse()
                for length, concurrency in cases:
                    count = max(args.min_requests, args.waves * concurrency)
                    plans = assignments(mode, concurrency, count, repeat)
                    trial_id = f"{mode}-r{repeat}-in{length}-c{concurrency}"
                    measured = []
                    # Warm the actual input/batch shape with the same streaming
                    # output path. Warmup results never enter measured quantiles.
                    for measure in (False, True):
                        start_ns = time.perf_counter_ns() + START_DELAY_NS
                        for plan in plans:
                            indices = (
                                plan["indices"]
                                if measure
                                else plan["indices"][: plan["concurrency"] * 2]
                            )
                            send_record(
                                clients[plan["replica"]],
                                {
                                    "kind": "bench",
                                    "trial": trial_id
                                    + ("-measure" if measure else "-warmup"),
                                    "start_ns": start_ns,
                                    "measure": measure,
                                    "trace": (
                                        args.profile_dir is not None
                                        and measure
                                        and repeat == 0
                                        and (length, concurrency) == cases[0]
                                    ),
                                    "domain": plan["domain"],
                                    "concurrency": plan["concurrency"],
                                    "requests": [
                                        {
                                            "index": index,
                                            "tokens": inputs[length][
                                                index
                                                + plan["domain"]
                                                * (len(inputs[length]) // 2)
                                            ],
                                        }
                                        for index in indices
                                    ],
                                },
                            )
                        received = [
                            receive_record(clients[plan["replica"]], args.timeout)
                            for plan in plans
                        ]
                        if measure:
                            measured = received
                    records = [r for response in measured for r in response["records"]]
                    expected = {
                        (domain, index) for domain in range(2) for index in range(count)
                    }
                    if {(r["domain"], r["index"]) for r in records} != expected or len(
                        records
                    ) != len(expected):
                        raise AssertionError("Dropped or duplicated benchmark requests")
                    result = {
                        "mode": mode,
                        "trial": trial_id,
                        "repeat": repeat,
                        "input_tokens": length,
                        "output_tokens": OUTPUT_TOKENS,
                        "domain_concurrency": concurrency,
                        "gpu_budget": 3,
                        "profiling_run": args.profile_dir is not None,
                        "plans": plans,
                        "summary": summarize(records),
                        "domains": {
                            str(domain): summarize(
                                [r for r in records if r["domain"] == domain]
                            )
                            for domain in range(2)
                        },
                        "records": records,
                        "pool_status": [
                            s
                            for response in measured
                            for s in (response["pool_status"] or [])
                        ],
                    }
                    with progress_path.open("a") as output:
                        output.write(json.dumps(result, allow_nan=False) + "\n")
                    report["trials"].append(result)
                    print(
                        f"{trial_id}: "
                        f"{result['summary']['output_throughput_tps']:.2f} tok/s",
                        flush=True,
                    )
            for client in clients:
                send_record(client, {"kind": "close"})
            for client in clients:
                receive_record(client, args.timeout)
            if worker is not None:
                report["worker"] = receive_record(worker, args.timeout)
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
    parser.add_argument("--gpus", type=int, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument(
        "--modes", nargs="+", choices=("native", "pool"), default=["native", "pool"]
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--input-lengths", type=int, nargs="+", choices=(128, 512), default=[128, 512]
    )
    parser.add_argument("--execution-options", type=json.loads, default={})
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--min-requests", type=int, default=12)
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    try:
        ExecutionOptions(**args.execution_options)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if (
        len(set(args.gpus)) != 3
        or min(args.gpus) < 0
        or len(set(args.modes)) != len(args.modes)
    ):
        parser.error("Choose three distinct GPUs and unique modes")
    if min(args.concurrency) <= 0 or max(args.concurrency) > MAX_SEQUENCES:
        parser.error("Concurrency exceeds the supported sequence capacity")
    if min(args.repeats, args.min_requests, args.waves, args.timeout) <= 0:
        parser.error("Counts and timeouts must be positive")
    if args.output.exists():
        parser.error("Choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "input_kind": "Fixed-length tokenized synthetic natural-language fixtures",
        "timing_scope": "AsyncLLM streaming; excludes tokenization and HTTP",
        "load_model": "Closed-loop domain concurrency; each slot refills on completion",
        "native_config": "Three full TP1 replicas, 2+1 domains swapped by repetition",
        "pool_config": "Two independent A/KV engines and one serial E worker",
        "limits": [
            "Both eager/Triton; native default graphs/async scheduling not benchmarked",
            "No SLO threshold supplied; no goodput or max-capacity claim",
            "Execution ablations are explicit in config.execution_options",
            "Profiling runs are diagnostic and must not enter performance comparisons",
            "Worker admitted queue excludes unread messages; "
            "client admission includes control overhead",
            "E compute event includes validation, output copy and launch gaps; "
            "not pure GEMM",
            "Closed-loop makespan includes fill/drain; "
            "two repeats are a pilot, not a confidence interval",
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
            max_count = max(args.min_requests, args.waves * max(args.concurrency))
            inputs = fixtures(args.model, max_count)
            inputs = {length: inputs[length] for length in args.input_lengths}
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
