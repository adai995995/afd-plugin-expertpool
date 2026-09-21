#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Full-model validation: two native AsyncLLM A/KV engines, resident E workers.

The last GPU first runs the unmodified model as a correctness reference and
exits before E workers start. It can then be reused by an E worker. Requests
use public synthetic fixtures, real tokenization, prefill, decode and routing.
No replay, forced routing, weight dumps or activation dumps are used. This is
an engine integration test, not an HTTP benchmark or equal-budget speed test.
"""

import argparse
import asyncio
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import signal
import socket
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict
from multiprocessing.connection import Connection
from pathlib import Path

import afd_plugin
from afd_plugin.expert_pool.controller import CONTROLLER_POLICIES
from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ControllerConfig,
    ExecutionOptions,
    PoolDeployment,
    WorkerPlacement,
)
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.protocol import MAX_RECEIVE_SLOTS
from tools.expert_pool.validate_service import (
    BusyGPUError,
    check_idle,
    receive_record,
    send_record,
)

MAX_BATCH_TOKENS = 256
MAX_MODEL_LEN = 1024
KV_CACHE_BYTES = 256 * 1024 * 1024
GENERATED_TOKENS = 16
BF16_WEIGHT_ELEMENT_BYTES = 2
BF16_OUTPUT_ELEMENT_BYTES = 2
EXPERT_PROJECTION_COUNT = 3
PROMPTS = (
    "The capital of France is",
    "A triangle has three sides. A square has",
    "Water freezes at zero degrees Celsius. " * 48 + "Summarize this in one sentence:",
    "The following sequence contains even numbers: 2, 4, 6, 8, 10. " * 12
    + "The next even number is",
)


def process_environment(gpu: int) -> None:
    # Each child owns its process group, including the native EngineCore child.
    # The supervisor may stop only these groups on timeout or test failure.
    os.setsid()
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "VLLM_PLUGINS": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
        }
    )


async def engine_loop(
    model: str,
    deployment_path: str | None,
    client_id: str,
    supervisor: Connection,
    timeout_s: int,
) -> None:
    # Imports follow per-process CUDA selection. Native vLLM owns model workers,
    # scheduling and KV allocation; this harness only submits requests.
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    from afd_plugin.expert_pool import register_expert_pool

    options = {}
    if deployment_path is not None:
        register_expert_pool()
        options = {
            "worker_cls": (
                "afd_plugin.v1.worker.pool_attention_worker.PoolAttentionWorker"
            ),
            "hf_overrides": {"architectures": ["PoolDeepseekV2ForCausalLM"]},
            "additional_config": {
                "expert_pool": {
                    "deployment": deployment_path,
                    "client_id": client_id,
                }
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
            max_num_seqs=8,
            kv_cache_memory_bytes=KV_CACHE_BYTES,
            gpu_memory_utilization=0.4,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            async_scheduling=False,
            kernel_config={"moe_backend": "triton"},
            disable_log_stats=True,
            **options,
        )
    )
    try:
        status = await engine.collective_rpc("pool_status") if deployment_path else None
        send_record(
            supervisor,
            {
                "kind": "ready",
                "client_id": client_id,
                "pid": os.getpid(),
                "status": status,
            },
        )

        async def request(case: int, request_id: str) -> dict:
            started = time.perf_counter_ns()
            params = SamplingParams(
                temperature=0,
                max_tokens=GENERATED_TOKENS,
                ignore_eos=True,
                logprobs=0,
                prompt_logprobs=0,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            final = None
            async for output in engine.generate(PROMPTS[case], params, request_id):
                final = output
            if final is None or not final.finished:
                raise RuntimeError("Native engine did not finish the request")
            output = final.outputs[0]
            if final.prompt_token_ids is None or final.prompt_logprobs is None:
                raise RuntimeError("Missing teacher-forced prompt logprobs")
            if output.logprobs is None:
                raise RuntimeError("Missing generated token logprobs")
            return {
                "case": case,
                "request_id": request_id,
                "started_ns": started,
                "finished_ns": time.perf_counter_ns(),
                "prompt_token_ids": final.prompt_token_ids,
                "prompt_logprobs": [
                    None if probabilities is None else probabilities[token].logprob
                    for token, probabilities in zip(
                        final.prompt_token_ids, final.prompt_logprobs, strict=True
                    )
                ],
                "token_ids": list(output.token_ids),
                "logprobs": [
                    probabilities[token].logprob
                    for token, probabilities in zip(
                        output.token_ids, output.logprobs, strict=True
                    )
                ],
            }

        while True:
            command = await asyncio.to_thread(receive_record, supervisor, timeout_s)
            if command["kind"] == "close":
                if deployment_path:
                    status = await engine.collective_rpc("pool_status")
                    await engine.collective_rpc("close_pool")
                send_record(supervisor, {"kind": "closed", "status": status})
                break
            if command["kind"] != "generate":
                raise ValueError("Unexpected validation command")
            # Identical entry path to the native API server. Concurrent requests
            # are independently admitted and batched by this engine's scheduler.
            results = await asyncio.gather(
                *(
                    request(case, f"{client_id}-{command['phase']}-{index}")
                    for index, case in enumerate(command["cases"])
                )
            )
            for result in results:
                result["phase"] = command["phase"]
            send_record(
                supervisor,
                {"kind": "results", "cases": results, "phase": command["phase"]},
            )
    finally:
        engine.shutdown(timeout=10)


def engine_process(
    model: str,
    gpu: int,
    deployment: str | None,
    client_id: str,
    supervisor: Connection,
    timeout_s: int,
) -> None:
    process_environment(gpu)
    send_record(supervisor, {"kind": "spawned", "pid": os.getpid()})
    try:
        asyncio.run(engine_loop(model, deployment, client_id, supervisor, timeout_s))
    except BaseException:
        send_record(supervisor, {"kind": "error", "detail": traceback.format_exc()})
        raise
    finally:
        supervisor.close()


def expert_process(
    path: str,
    gpu: int,
    supervisor: Connection,
    profile_dir: Path | None = None,
    worker_id: str | None = None,
) -> None:
    process_environment(gpu)
    send_record(supervisor, {"kind": "spawned", "pid": os.getpid()})
    try:
        from afd_plugin.expert_pool.service import serve

        report = serve(
            PoolDeployment.read(Path(path)),
            profile_dir=profile_dir,
            worker_id=worker_id,
        )
        send_record(supervisor, {"kind": "finished", **report})
    except BaseException:
        send_record(supervisor, {"kind": "error", "detail": traceback.format_exc()})
        raise
    finally:
        supervisor.close()


def compare_request(reference: dict, candidate: dict, tolerance: float) -> dict:
    """Compare identical teacher-forced prefixes, separately from free decoding."""
    if reference["prompt_token_ids"] != candidate["prompt_token_ids"]:
        raise AssertionError("Tokenization or fixed-prefix inputs changed")
    errors = []
    for expected, actual in zip(
        reference["prompt_logprobs"], candidate["prompt_logprobs"], strict=True
    ):
        if (expected is None) != (actual is None):
            raise AssertionError("Prompt probability coverage changed")
        if expected is not None:
            if not math.isfinite(expected) or not math.isfinite(actual):
                raise AssertionError("Nonfinite fixed-prefix logprob")
            errors.append(abs(expected - actual))
    if not errors:
        raise AssertionError("No fixed-prefix probabilities were checked")
    tokens_exact = reference["token_ids"] == candidate["token_ids"]
    if len(candidate["token_ids"]) != GENERATED_TOKENS:
        raise AssertionError("Request produced an unexpected output length")
    generated_error = None
    if tokens_exact:
        generated_error = max(
            abs(a - b)
            for a, b in zip(reference["logprobs"], candidate["logprobs"], strict=True)
        )
        if not all(math.isfinite(value) for value in candidate["logprobs"]):
            raise AssertionError("Nonfinite generated logprob")
    return {
        "case": candidate["case"],
        "tokens_exact": tokens_exact,
        "teacher_forced_tokens": len(errors),
        "prompt_logprob_max_abs": max(errors),
        "generated_logprob_max_abs": generated_error,
        "passed": tokens_exact
        and max(errors) <= tolerance
        and generated_error is not None
        and generated_error <= tolerance,
    }


def stop_children(children: list[multiprocessing.Process], groups: set[int]) -> None:
    for child in children:
        if child.pid is None:
            continue
        # Acknowledged groups belong to this launch, even if their leader has
        # already exited after an error. Clean up its remaining EngineCore too.
        try:
            if child.pid in groups:
                os.killpg(child.pid, signal.SIGTERM)
            elif child.is_alive():
                child.terminate()
        except ProcessLookupError:
            pass
    for child in children:
        if child.pid is not None:
            child.join(5)
            if child.pid in groups or child.is_alive():
                try:
                    if child.pid in groups:
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
                except ProcessLookupError:
                    pass
                child.join(5)


def run(args: argparse.Namespace, report: dict) -> None:
    context = multiprocessing.get_context("spawn")
    children = []
    pipes = []
    groups = set()

    def start(target, *arguments) -> Connection:
        parent, child = context.Pipe()
        pipes.extend((parent, child))
        process = context.Process(target=target, args=(*arguments, child))
        children.append(process)
        process.start()
        child.close()
        spawned = receive_record(parent, args.timeout)
        if spawned != {"kind": "spawned", "pid": process.pid}:
            raise RuntimeError("Child did not acknowledge its isolated process group")
        groups.add(process.pid)
        return parent

    # One wrapper accommodates the engine's timeout argument while keeping the
    # supervisor Connection in a consistent position for the start helper.
    try:
        with tempfile.TemporaryDirectory(prefix="pool-online-") as temporary:
            root = Path(temporary)
            sockets = [socket.socket() for _ in range(2 * args.expert_workers)]
            try:
                for reserved in sockets:
                    reserved.bind(("127.0.0.1", 0))
                endpoints = tuple(
                    ClientEndpoint(
                        f"client-{index % 2}",
                        1,
                        f"domain-{index % 2}",
                        str(root / f"e{index // 2}-a{index % 2}.sock"),
                        reserved.getsockname()[1],
                        f"worker-{index // 2}",
                    )
                    for index, reserved in enumerate(sockets)
                )
            finally:
                for reserved in sockets:
                    reserved.close()
            model_config = json.loads((args.model / "config.json").read_text())
            moe_layers = tuple(
                layer
                for layer in range(
                    model_config["first_k_dense_replace"],
                    model_config["num_hidden_layers"],
                )
                if layer % model_config["moe_layer_freq"] == 0
            )
            deployment = PoolDeployment(
                str(args.model.resolve()),
                uuid.uuid4().hex,
                MAX_BATCH_TOKENS,
                endpoints,
                args.timeout,
                ExecutionOptions(**args.execution_options),
                tuple(
                    WorkerPlacement(
                        f"worker-{index}",
                        (
                            moe_layers[index :: args.expert_workers]
                            if args.placement == "partitioned"
                            else None
                        ),
                        expert_ids=(
                            tuple(
                                range(
                                    index,
                                    model_config["n_routed_experts"],
                                    args.expert_workers,
                                )
                            )
                            if args.placement == "expert_partitioned"
                            else None
                        ),
                    )
                    for index in range(args.expert_workers)
                ),
                controller=(
                    ControllerConfig(str(root), args.controller_policy)
                    if args.controller
                    else None
                ),
                dispatch_mode=(
                    "expert_partitioned"
                    if args.placement == "expert_partitioned"
                    else "whole_layer"
                ),
                demand_aware=args.expert_demand,
                compact_output=args.compact_output,
                receive_slots=args.receive_slots,
            )
            directory = deployment.pool_directory()
            report["placement"] = asdict(directory)
            path = root / "deployment.json"
            path.write_text(json.dumps(asdict(deployment)))
            # Reference uses exactly the same local checkpoint and eager kernel
            # configuration. Its private E weights exist only for verification.
            reference = start(
                engine_entry,
                str(args.model),
                args.gpus[-1],
                None,
                "reference",
                args.timeout,
            )
            receive_record(reference, args.timeout)
            golden = {}
            for case in range(len(PROMPTS)):
                send_record(
                    reference,
                    {"kind": "generate", "phase": "independent", "cases": [case]},
                )
                golden[("independent", case)] = receive_record(reference, args.timeout)[
                    "cases"
                ][0]
            # Native BF16 execution is batch-sensitive. Collect the same paired
            # workload as each A engine; never widen tolerance to fit a failure
            # against a singleton-only reference.
            for repeat in range(args.repeats):
                phase = f"parallel-{repeat}"
                for index in range(2):
                    send_record(
                        reference,
                        {
                            "kind": "generate",
                            "phase": phase,
                            "cases": [index, index + 2],
                        },
                    )
                    for result in receive_record(reference, args.timeout)["cases"]:
                        golden[(phase, result["case"])] = result
            report["reference"] = list(golden.values())
            report["native_batch_sensitivity"] = [
                compare_request(
                    golden[("independent", case)], result, args.logprob_atol
                )
                for (phase, case), result in golden.items()
                if phase != "independent"
            ]
            print("Native reference requests completed", flush=True)
            send_record(reference, {"kind": "close"})
            receive_record(reference, args.timeout)
            children[0].join(args.timeout)
            if children[0].exitcode != 0:
                raise RuntimeError("Reference engine did not exit cleanly")
            report["post_reference_gpu_state"] = check_idle(args.gpus)
            controller = start(controller_entry, str(path)) if args.controller else None
            workers = [
                start(expert_entry, str(path), args.gpus[index + 2], worker_id)
                for index, worker_id in enumerate(deployment.worker_ids)
            ]
            clients = [
                start(
                    engine_entry,
                    str(args.model),
                    args.gpus[index],
                    str(path),
                    client_id,
                    args.timeout,
                )
                for index, client_id in enumerate(deployment.client_ids)
            ]
            ready = [receive_record(client, args.timeout) for client in clients]
            report["engines_ready"] = ready
            print("Both independent A/KV engines are ready", flush=True)
            statuses = [item["status"][0] for item in ready]
            if len({status["pid"] for status in statuses}) != 2:
                raise AssertionError("Expected separate native model worker processes")
            if any(status["routed_parameter_bytes"] for status in statuses):
                raise AssertionError("A engine allocated routed expert parameters")
            results = []
            # Each engine progresses while the other receives no requests.
            for index, client in enumerate(clients):
                send_record(
                    client,
                    {"kind": "generate", "phase": "independent", "cases": [index]},
                )
                results.extend(receive_record(client, args.timeout)["cases"])
            overlaps = []
            for repeat in range(args.repeats):
                for index, client in enumerate(clients):
                    send_record(
                        client,
                        {
                            "kind": "generate",
                            "phase": f"parallel-{repeat}",
                            "cases": [index, index + 2],
                        },
                    )
                paired = [
                    receive_record(client, args.timeout)["cases"] for client in clients
                ]
                overlap = sum(
                    max(a["started_ns"], b["started_ns"])
                    < min(a["finished_ns"], b["finished_ns"])
                    for a in paired[0]
                    for b in paired[1]
                )
                if not overlap:
                    raise AssertionError(
                        "Concurrent phase did not overlap between engines"
                    )
                overlaps.append(overlap)
                for group in paired:
                    results.extend(group)
                print(f"Concurrent request repeat {repeat + 1} completed", flush=True)
            report["requests"] = results
            report["overlapping_request_pairs_per_repeat"] = overlaps
            report["comparisons"] = [
                compare_request(
                    golden[(item["phase"], item["case"])], item, args.logprob_atol
                )
                for item in results
            ]
            for client in clients:
                send_record(client, {"kind": "close"})
            closed = [receive_record(client, args.timeout) for client in clients]
            report["engines_closed"] = closed
            report["workers"] = [
                receive_record(worker, args.timeout) for worker in workers
            ]
            if len(report["workers"]) == 1:
                report["worker"] = report["workers"][0]
            for before, after in zip(
                statuses, (item["status"][0] for item in closed), strict=True
            ):
                if set(after["layers"]) != {
                    str(p.layer_id) for p in directory.placements
                }:
                    raise AssertionError("Full-model MoE layer coverage is incomplete")
                if any(
                    after["layers"][layer]["calls"] <= before["layers"][layer]["calls"]
                    for layer in before["layers"]
                ):
                    raise AssertionError("An MoE layer did not execute real requests")
            report["dispatch_accounting"] = verify_dispatch(
                directory,
                statuses,
                closed,
                report["workers"],
                require_each_replica=(
                    not args.controller or args.controller_policy == "round_robin"
                ),
                expert_weight_bytes=(
                    model_config["hidden_size"]
                    * model_config["moe_intermediate_size"]
                    * EXPERT_PROJECTION_COUNT
                    * BF16_WEIGHT_ELEMENT_BYTES
                ),
                demand_aware=args.expert_demand,
            )
            if args.expert_demand:
                report["output_transfer_accounting"] = verify_output_transfer(
                    directory,
                    statuses,
                    closed,
                    report["workers"],
                    compact_output=args.compact_output,
                )
            if controller is not None:
                controlled = receive_record(controller, args.timeout)
                report["controller"] = controlled
                if args.receive_slots > 1:
                    report["pipeline_accounting"] = verify_pipeline(
                        report["workers"], controlled, args.receive_slots
                    )
                if controlled["scheduling_policy"] != args.controller_policy or any(
                    item["status"][0]["dispatch"]["policy"] != controlled["policy"]
                    for item in ready + closed
                ):
                    raise AssertionError("Controller and clients disagree on policy")
                if (
                    controlled["pending"]
                    or controlled["outstanding"]
                    or controlled["closed_clients"] != len(clients)
                ):
                    raise AssertionError(
                        "Controller did not drain all clients and reservations"
                    )
                if (
                    controlled["completed_by_client"]
                    != report["dispatch_accounting"]["parent_layer_calls_by_client"]
                ):
                    raise AssertionError(
                        "Controller parent completions disagree with A layer calls"
                    )
                for worker in report["workers"]:
                    counters = controlled["workers"][worker["worker_id"]]
                    if counters["active"] is not None or any(
                        counters[key] != worker[key]
                        for key in ("completed_calls", "client_calls", "layer_calls")
                    ):
                        raise AssertionError(
                            "Controller and GPU worker counters disagree"
                        )
                if args.expert_demand:
                    verify_demand_controller(controlled, report["dispatch_accounting"])
            for child in children[1:]:
                child.join(args.timeout)
                if child.exitcode != 0:
                    raise RuntimeError("A Pool process did not exit cleanly")
            if not all(item["passed"] for item in report["comparisons"]):
                raise AssertionError(
                    "Full-model correctness tolerance failed; inspect comparisons"
                )
    finally:
        stop_children(children, groups)
        for connection in pipes:
            connection.close()


def engine_entry(
    model: str,
    gpu: int,
    deployment: str | None,
    client_id: str,
    timeout_s: int,
    supervisor: Connection,
) -> None:
    engine_process(model, gpu, deployment, client_id, supervisor, timeout_s)


def expert_entry(path: str, gpu: int, worker_id: str, supervisor: Connection) -> None:
    expert_process(path, gpu, supervisor, worker_id=worker_id)


def controller_entry(path: str, supervisor: Connection) -> None:
    os.setsid()
    send_record(supervisor, {"kind": "spawned", "pid": os.getpid()})
    try:
        from afd_plugin.expert_pool.controller_service import serve_controller

        report = serve_controller(PoolDeployment.read(Path(path)))
        send_record(supervisor, {"kind": "finished", **report})
    except BaseException:
        send_record(supervisor, {"kind": "error", "detail": traceback.format_exc()})
        raise
    finally:
        supervisor.close()


def verify_dispatch(
    directory: PoolDirectory,
    ready: list[dict],
    closed: list[dict],
    workers: list[dict],
    *,
    require_each_replica: bool = True,
    expert_weight_bytes: int | None = None,
    demand_aware: bool = False,
) -> dict:
    """Reconcile parent calls with worker children and actual resident weights.

    A broadcast parent executes on every owner; a demand parent executes only
    on selected owners, and an empty parent has no children. A whole-layer
    parent executes on one replica. None of these counts are user requests.
    """
    if demand_aware and not directory.expert_partitioned:
        raise AssertionError("Expert demand requires partitioned dispatch")
    after = [item["status"][0] for item in closed]
    by_worker = {item["worker_id"]: item for item in workers}
    if set(by_worker) != {worker.worker_id for worker in directory.workers}:
        raise AssertionError("Worker report coverage changed")
    for worker in directory.workers:
        actual = by_worker[worker.worker_id]
        if actual["placement_version"] != worker.version or set(
            actual["resident_layers"]
        ) != {placement.layer_id for placement in worker.placements}:
            raise AssertionError("Worker loaded the wrong placement")
        if actual["resident_experts"] != {
            str(placement.layer_id): list(placement.expert_ids)
            for placement in worker.placements
        }:
            raise AssertionError("Worker loaded the wrong expert IDs or slot order")
        if expert_weight_bytes is not None and actual["resident_weight_bytes"] != (
            expert_weight_bytes
            * sum(len(placement.expert_ids) for placement in worker.placements)
        ):
            raise AssertionError("Worker allocated more or fewer expert weights")
        calls = 0
        for before, status in zip(ready, after, strict=True):
            start = before["dispatch"]["workers"][worker.worker_id]
            end = status["dispatch"]["workers"][worker.worker_id]
            if end["calls"] != actual["client_calls"][status["client_id"]]:
                raise AssertionError("Client/worker completion counts disagree")
            if (
                require_each_replica
                and not demand_aware
                and any(
                    end["layer_calls"][str(placement.layer_id)]
                    <= start["layer_calls"][str(placement.layer_id)]
                    for placement in worker.placements
                )
            ):
                raise AssertionError("A client did not exercise each resident replica")
            calls += end["calls"]
        if calls != actual["completed_calls"]:
            raise AssertionError("Worker completion accounting is incomplete")
        for placement in worker.placements:
            layer = str(placement.layer_id)
            if actual["layer_calls"][layer] != sum(
                status["dispatch"]["workers"][worker.worker_id]["layer_calls"][layer]
                for status in after
            ):
                raise AssertionError("Layer completion accounting disagrees")
    for status in after:
        for layer, counters in status["layers"].items():
            owners = directory.owners(int(layer))
            child_counts = {
                worker_id: worker["layer_calls"].get(layer, 0)
                for worker_id, worker in status["dispatch"]["workers"].items()
            }
            if any(count for key, count in child_counts.items() if key not in owners):
                raise AssertionError("A layer executed on a nonresident worker")
            if demand_aware:
                # Per-expert metadata below determines which owners participated.
                continue
            if directory.expert_partitioned:
                if any(child_counts[owner] != counters["calls"] for owner in owners):
                    raise AssertionError("A partition child was lost or duplicated")
            elif counters["calls"] != sum(child_counts.values()):
                raise AssertionError("A whole-layer call was lost or duplicated")
    parents = {
        status["client_id"]: sum(layer["calls"] for layer in status["layers"].values())
        for status in after
    }
    children = {worker["worker_id"]: worker["completed_calls"] for worker in workers}
    demand_summary = (
        verify_expert_demand(directory, ready, after, workers) if demand_aware else {}
    )
    return {
        "dispatch_mode": (
            "expert_partitioned" if directory.expert_partitioned else "whole_layer"
        ),
        "demand_aware": demand_aware,
        "count_scope": "Layer calls including engine initialization, not user requests",
        "parent_layer_calls": sum(parents.values()),
        "parent_layer_calls_by_client": parents,
        "worker_child_calls": sum(children.values()),
        "worker_child_calls_by_worker": children,
        "request_parent_layer_calls": sum(
            status["layers"][layer]["calls"] - before["layers"][layer]["calls"]
            for before, status in zip(ready, after, strict=True)
            for layer in status["layers"]
        ),
        "request_worker_child_calls": sum(
            status["dispatch"]["workers"][worker]["calls"]
            - before["dispatch"]["workers"][worker]["calls"]
            for before, status in zip(ready, after, strict=True)
            for worker in status["dispatch"]["workers"]
        ),
        "resident_routed_weight_bytes": sum(
            worker["resident_weight_bytes"] for worker in workers
        ),
        "single_model_routed_weight_bytes": (
            expert_weight_bytes
            * sum(placement.num_experts for placement in directory.placements)
            if expert_weight_bytes is not None
            else None
        ),
        **demand_summary,
    }


def verify_expert_demand(
    directory: PoolDirectory,
    ready: list[dict],
    after: list[dict],
    workers: list[dict],
) -> dict:
    """Validate selected children and assignment totals from independent reports."""
    logical_counts = {
        str(placement.layer_id): placement.num_experts
        for placement in directory.placements
    }
    worker_layers = {
        worker.worker_id: {str(p.layer_id): p.expert_ids for p in worker.placements}
        for worker in directory.workers
    }
    top_k = directory.workers[0].top_k
    for status in ready + after:
        if status["dispatch"].get("demand_aware") is not True:
            raise AssertionError("A report does not enable expert demand")
        demand = status["dispatch"]["demand"]
        if demand["parent_layer_calls"] != {
            layer: counters["calls"] for layer, counters in status["layers"].items()
        }:
            raise AssertionError("Demand parent counters disagree with model layers")
        if any(
            set(demand[field]) != set(logical_counts)
            for field in (
                "parent_layer_calls",
                "empty_layer_calls",
                "expert_assignments",
            )
        ) or set(demand["selected_worker_layer_calls"]) != set(worker_layers):
            raise AssertionError("Demand report coverage is incomplete")
        for worker_id, layers in worker_layers.items():
            selected = demand["selected_worker_layer_calls"][worker_id]
            completed = status["dispatch"]["workers"][worker_id]
            if set(selected) != set(layers) or selected != completed["layer_calls"]:
                raise AssertionError("Selected and completed worker layers disagree")
            if sum(selected.values()) != completed["calls"]:
                raise AssertionError("Selected worker total is inconsistent")
        for layer, expert_count in logical_counts.items():
            parent = demand["parent_layer_calls"][layer]
            empty = demand["empty_layer_calls"][layer]
            assignments = demand["expert_assignments"][layer]
            if (
                type(parent) is not int
                or type(empty) is not int
                or not 0 <= empty <= parent
                or not isinstance(assignments, list)
                or len(assignments) != expert_count
                or any(type(count) is not int or count < 0 for count in assignments)
            ):
                raise AssertionError("Invalid demand parent or assignment counts")
            if sum(assignments) != status["layers"][layer]["token_rows"] * top_k:
                raise AssertionError(
                    "Demand assignments do not cover routed token rows"
                )
            selected_total = 0
            for owner in directory.owners(int(layer)):
                count = demand["selected_worker_layer_calls"][owner][layer]
                owned_assignments = sum(
                    assignments[expert] for expert in worker_layers[owner][layer]
                )
                if (
                    type(count) is not int
                    or not 0 <= count <= parent - empty
                    or owned_assignments < count
                    or (count == 0) != (owned_assignments == 0)
                ):
                    raise AssertionError(
                        "Selected child count contradicts expert demand"
                    )
                selected_total += count
            if selected_total < parent - empty:
                raise AssertionError("A nonempty parent is missing all child calls")
    expected_assignments = {
        worker_id: {
            layer: {
                str(expert): sum(
                    status["dispatch"]["demand"]["expert_assignments"][layer][expert]
                    for status in after
                )
                for expert in experts
            }
            for layer, experts in layers.items()
        }
        for worker_id, layers in worker_layers.items()
    }
    for worker in workers:
        if (
            worker.get("demand_aware") is not True
            or worker["expert_assignments"]
            != (expected_assignments[worker["worker_id"]])
        ):
            raise AssertionError("A and E expert assignment reports disagree")
    empty_calls = sum(
        sum(status["dispatch"]["demand"]["empty_layer_calls"].values())
        for status in after
    )
    return {
        "empty_parent_calls": empty_calls,
        "expert_assignments_by_worker_layer_expert": expected_assignments,
        "expert_assignments": sum(
            sum(counts.values())
            for layers in expected_assignments.values()
            for counts in layers.values()
        ),
        "skipped_worker_calls": {
            worker_id: sum(
                status["layers"][layer]["calls"]
                - status["dispatch"]["workers"][worker_id]["layer_calls"][layer]
                for status in after
                for layer in layers
            )
            for worker_id, layers in worker_layers.items()
        },
        "request_empty_parent_calls": sum(
            status["dispatch"]["demand"]["empty_layer_calls"][layer]
            - before["dispatch"]["demand"]["empty_layer_calls"][layer]
            for before, status in zip(ready, after, strict=True)
            for layer in logical_counts
        ),
    }


def verify_demand_controller(controlled: dict, accounting: dict) -> None:
    if controlled.get("demand_aware") is not True or any(
        controlled[reported] != accounting[expected]
        for reported, expected in (
            ("empty_parent_calls", "empty_parent_calls"),
            ("selected_worker_calls", "worker_child_calls_by_worker"),
            ("skipped_worker_calls", "skipped_worker_calls"),
            (
                "admitted_assignments_by_worker_layer_expert",
                "expert_assignments_by_worker_layer_expert",
            ),
        )
    ):
        raise AssertionError("Controller and completed demand accounting disagree")


def verify_output_transfer(
    directory: PoolDirectory,
    ready: list[dict],
    closed: list[dict],
    workers: list[dict],
    *,
    compact_output: bool,
) -> dict:
    """Reconcile payload bytes with A assignments and independent E reports.

    Dense-equivalent bytes count only the workers selected by real calls. They
    exclude skipped owners, allocation capacity and network protocol overhead.
    Startup is checked in the totals and subtracted from request-only counters.
    """
    if not directory.expert_partitioned or type(compact_output) is not bool:
        raise AssertionError("Output accounting requires explicit partition mode")
    resident = {worker.worker_id: worker for worker in directory.workers}
    worker_reports = {worker["worker_id"]: worker for worker in workers}
    if set(worker_reports) != set(resident) or len(worker_reports) != len(workers):
        raise AssertionError("Output worker report coverage is inconsistent")
    after = [item["status"][0] for item in closed]
    before_ids = [status["client_id"] for status in ready]
    after_ids = [status["client_id"] for status in after]
    if before_ids != after_ids or len(set(before_ids)) != len(before_ids):
        raise AssertionError("Output client report identities changed")

    phases = []
    for statuses in (ready, after):
        per_client = {}
        for status in statuses:
            dispatch = status["dispatch"]
            if (
                dispatch.get("compact_output") is not compact_output
                or dispatch.get("demand_aware") is not True
                or set(dispatch["workers"]) != set(resident)
            ):
                raise AssertionError("A output mode or worker coverage disagrees")
            per_worker = {}
            for worker_id, worker in resident.items():
                output = dispatch["workers"][worker_id]["output_transfer"]
                actual = output["received_bytes"]
                dense = output["dense_equivalent_bytes"]
                row_bytes = (
                    worker.top_k * worker.hidden_size * BF16_OUTPUT_ELEMENT_BYTES
                )
                useful = (
                    sum(
                        dispatch["demand"]["expert_assignments"][str(p.layer_id)][
                            expert
                        ]
                        for p in worker.placements
                        for expert in p.expert_ids
                    )
                    * worker.hidden_size
                    * BF16_OUTPUT_ELEMENT_BYTES
                )
                dense_upper = (
                    sum(
                        status["layers"][str(p.layer_id)]["token_rows"]
                        for p in worker.placements
                    )
                    * row_bytes
                )
                if (
                    any(
                        type(value) is not int or value < 0 for value in (actual, dense)
                    )
                    or not useful <= dense <= dense_upper
                    or dense % row_bytes
                    or (dense == 0) != (useful == 0)
                    or actual != (useful if compact_output else dense)
                ):
                    raise AssertionError("A output bytes contradict routed assignments")
                per_worker[worker_id] = {
                    "received_bytes": actual,
                    "dense_equivalent_bytes": dense,
                    "useful_assignment_bytes": useful,
                }
            for field in ("received_bytes", "dense_equivalent_bytes"):
                total = dispatch["output_transfer"][field]
                if type(total) is not int or total != sum(
                    output[field] for output in per_worker.values()
                ):
                    raise AssertionError("A output total differs from worker channels")
            per_client[status["client_id"]] = per_worker
        phases.append(per_client)

    before, after_counts = phases
    for client_id, channels in after_counts.items():
        for worker_id, output in channels.items():
            if any(
                output[field] < before[client_id][worker_id][field] for field in output
            ):
                raise AssertionError("Output counters regressed after engine warmup")
    by_worker = {}
    for worker_id, report in worker_reports.items():
        actual = sum(
            channels[worker_id]["received_bytes"] for channels in after_counts.values()
        )
        dense = sum(
            channels[worker_id]["dense_equivalent_bytes"]
            for channels in after_counts.values()
        )
        output = report["output_transfer"]
        if (
            report.get("compact_output") is not compact_output
            or type(output["sent_bytes"]) is not int
            or type(output["dense_equivalent_bytes"]) is not int
            or output["sent_bytes"] != actual
            or output["dense_equivalent_bytes"] != dense
        ):
            raise AssertionError("A received bytes and E sent bytes disagree")
        by_worker[worker_id] = {
            "received_bytes": actual,
            "sent_bytes": output["sent_bytes"],
            "dense_equivalent_bytes": dense,
        }
    actual = sum(worker["received_bytes"] for worker in by_worker.values())
    dense = sum(worker["dense_equivalent_bytes"] for worker in by_worker.values())
    startup_actual = sum(
        output["received_bytes"]
        for channels in before.values()
        for output in channels.values()
    )
    startup_dense = sum(
        output["dense_equivalent_bytes"]
        for channels in before.values()
        for output in channels.values()
    )
    return {
        "compact_output": compact_output,
        "count_scope": "Payload bytes including startup; request deltas exclude warmup",
        "received_bytes": actual,
        "sent_bytes": sum(worker["sent_bytes"] for worker in by_worker.values()),
        "dense_equivalent_bytes": dense,
        "avoided_output_bytes": dense - actual,
        "request_received_bytes": actual - startup_actual,
        "request_dense_equivalent_bytes": dense - startup_dense,
        "by_worker": by_worker,
    }


def verify_pipeline(workers: list[dict], controller: dict, receive_slots: int) -> dict:
    """Check each physical slot drained exactly its granted generations."""
    if controller["receive_slots"] != receive_slots:
        raise AssertionError("Controller slot capacity differs from deployment")
    peak = 0
    for worker in workers:
        pipeline = worker["pipeline"]
        reserved = controller["workers"][worker["worker_id"]]
        slots = reserved["slots"]
        if (
            not pipeline["enabled"]
            or pipeline["receive_slots"] != receive_slots
            or pipeline["compute_lanes"] != 1
            or pipeline["active_slots"] != 0
            or not 1 <= pipeline["peak_occupied_slots"] <= receive_slots
            or len(slots) != receive_slots
            or any(
                slot["active"] is not None or slot["phase"] != "idle" for slot in slots
            )
            or pipeline["slot_generations"] != [slot["generation"] for slot in slots]
            or pipeline["slot_completed_calls"] != pipeline["slot_generations"]
            or sum(pipeline["slot_completed_calls"]) != worker["completed_calls"]
        ):
            raise AssertionError("Pipeline slot lifecycle or accounting disagrees")
        peak = max(peak, pipeline["peak_occupied_slots"])
    if peak <= 1:
        raise AssertionError("Validation did not exercise simultaneous live slots")
    return {
        "receive_slots": receive_slots,
        "peak_occupied_slots": peak,
        "slot_generations_and_completions_match": True,
        "all_slots_drained": True,
        "scope": "Concurrent live buffers; GPU overlap requires a device trace",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        required=True,
        help="A0, A1, E GPUs; minimum four, last reused after native reference",
    )
    parser.add_argument("--expert-workers", type=int, default=1)
    parser.add_argument(
        "--controller",
        action="store_true",
        help="Route control messages through the authoritative CPU controller",
    )
    parser.add_argument(
        "--controller-policy",
        choices=CONTROLLER_POLICIES,
        default="round_robin",
        help="Bind on admission, or select an available resident copy at grant time",
    )
    parser.add_argument(
        "--placement",
        choices=("replicated", "partitioned", "expert_partitioned"),
        default="replicated",
    )
    parser.add_argument(
        "--expert-demand",
        action="store_true",
        help="Dispatch expert partitions only to owners with positive routed demand",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Return compact assignment slots; requires --expert-demand",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--logprob-atol", type=float, default=0.05)
    parser.add_argument("--execution-options", type=json.loads, default={})
    parser.add_argument("--receive-slots", type=int, default=1)
    args = parser.parse_args()
    if not args.controller and args.controller_policy != "round_robin":
        parser.error("--controller-policy requires --controller")
    if args.placement == "expert_partitioned" and (
        not args.controller or args.controller_policy != "ready_first"
    ):
        parser.error(
            "Expert partitions require --controller --controller-policy ready_first"
        )
    if args.expert_demand and args.placement != "expert_partitioned":
        parser.error("--expert-demand requires --placement expert_partitioned")
    if args.compact_output and not args.expert_demand:
        parser.error("--compact-output requires --expert-demand")
    try:
        execution = ExecutionOptions(**args.execution_options)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if not 1 <= args.receive_slots <= MAX_RECEIVE_SLOTS or (
        args.receive_slots > 1
        and (
            not args.compact_output
            or execution.validate_worker_values
            or not execution.defer_output_sync
        )
    ):
        parser.error(
            "Multiple slots require compact output and trusted deferred execution"
        )
    required_gpus = max(4, 2 + args.expert_workers)
    if (
        args.expert_workers < 1
        or len(args.gpus) != required_gpus
        or len(set(args.gpus)) != required_gpus
        or min(args.gpus) < 0
    ):
        parser.error("Select distinct GPUs: max(4, 2 + expert-workers)")
    if (
        args.timeout <= 0
        or args.repeats <= 0
        or not math.isfinite(args.logprob_atol)
        or args.logprob_atol < 0
    ):
        parser.error("Invalid timeout, repetitions or logprob tolerance")
    if args.output.exists():
        parser.error("Choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "model": str(args.model.resolve()),
        "gpus": args.gpus,
        "scope": "Native AsyncLLM request integration; no HTTP/performance claim",
        "logprob_atol_declared_before_execution": args.logprob_atol,
        "generation_tokens_required_exact": True,
        "repeats": args.repeats,
        "execution_options": args.execution_options,
        "expert_workers": args.expert_workers,
        "placement_mode": args.placement,
        "dispatch_mode": (
            "expert_partitioned"
            if args.placement == "expert_partitioned"
            else "whole_layer"
        ),
        "demand_aware": args.expert_demand,
        "compact_output": args.compact_output,
        "receive_slots": args.receive_slots,
        "controller_enabled": args.controller,
        "controller_policy": args.controller_policy if args.controller else None,
    }
    try:
        root = Path(__file__).resolve().parents[2]
        if Path(afd_plugin.__file__).resolve().parent != root / "afd_plugin":
            raise RuntimeError("Python imported another source tree; set PYTHONPATH")
        manifest = root / "source_manifest.json"
        if manifest.exists():
            report["source_manifest"] = json.loads(manifest.read_text())
            for name, expected in report["source_manifest"]["files"].items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Source snapshot changed after synchronization")
        with args.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report["initial_gpu_state"] = check_idle(args.gpus)
            run(args, report)
        report["status"] = "passed"
    except BusyGPUError as error:
        report.update(status="blocked_preflight", error=str(error))
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        traceback.print_exc()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RESULT {report['status']}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
