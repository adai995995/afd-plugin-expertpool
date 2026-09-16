#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Full-model validation: two native AsyncLLM A/KV engines, one shared E GPU.

A fourth GPU runs the unmodified model as a correctness reference. Requests
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
from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ExecutionOptions,
    PoolDeployment,
)
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
    path: str, gpu: int, supervisor: Connection, profile_dir: Path | None = None
) -> None:
    process_environment(gpu)
    send_record(supervisor, {"kind": "spawned", "pid": os.getpid()})
    try:
        from afd_plugin.expert_pool.service import serve

        report = serve(PoolDeployment.read(Path(path)), profile_dir=profile_dir)
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
            sockets = [socket.socket() for _ in range(2)]
            try:
                for reserved in sockets:
                    reserved.bind(("127.0.0.1", 0))
                endpoints = tuple(
                    ClientEndpoint(
                        f"client-{index}",
                        1,
                        f"domain-{index}",
                        str(root / f"a{index}.sock"),
                        reserved.getsockname()[1],
                    )
                    for index, reserved in enumerate(sockets)
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
            path = root / "deployment.json"
            path.write_text(json.dumps(asdict(deployment)))
            # Reference uses exactly the same local checkpoint and eager kernel
            # configuration. Its private E weights exist only for verification.
            reference = start(
                engine_entry,
                str(args.model),
                args.gpus[3],
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
            worker = start(expert_process, str(path), args.gpus[2])
            clients = [
                start(
                    engine_entry,
                    str(args.model),
                    args.gpus[index],
                    str(path),
                    endpoint.client_id,
                    args.timeout,
                )
                for index, endpoint in enumerate(endpoints)
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
            report["worker"] = receive_record(worker, args.timeout)
            for before, after in zip(
                statuses, (item["status"][0] for item in closed), strict=True
            ):
                if set(after["layers"]) != {
                    str(p.layer_id) for p in deployment.directory().placements
                }:
                    raise AssertionError("Full-model MoE layer coverage is incomplete")
                if any(
                    after["layers"][layer]["calls"] <= before["layers"][layer]["calls"]
                    for layer in before["layers"]
                ):
                    raise AssertionError("An MoE layer did not execute real requests")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        nargs=4,
        type=int,
        required=True,
        help="A0, A1, E worker, native reference physical GPUs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--logprob-atol", type=float, default=0.05)
    parser.add_argument("--execution-options", type=json.loads, default={})
    args = parser.parse_args()
    try:
        ExecutionOptions(**args.execution_options)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if len(set(args.gpus)) != 4 or min(args.gpus) < 0:
        parser.error("Select four distinct GPUs")
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
