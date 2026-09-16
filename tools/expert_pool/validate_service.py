#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Three-GPU, two-client/one-worker correctness test with real NCCL transfers.

Clients contain native reference weights only for validation. They are not full
Attention/KV services. No weights, activations or routes are saved to disk.
Timing includes correctness instrumentation and is not a performance result.
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
import subprocess
import tempfile
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING

import afd_plugin
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import MAX_CONTROL_BYTES
from afd_plugin.expert_pool.scheduler import StaticDirectory

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.expert_pool.checkpoint import DeepseekCheckpoint

DEFAULT_TIMEOUT_S = 120
IDLE_MEMORY_LIMIT_MIB = 128


class BusyGPUError(RuntimeError):
    """Preflight found other work; no validation process was started."""


@dataclass(frozen=True)
class Launch:
    model: str
    directory: StaticDirectory
    ports: tuple[int, int]
    timeout_s: int
    atol: float
    rtol: float


def send_record(connection: Connection, record: dict) -> None:
    connection.send_bytes(json.dumps(record, allow_nan=False).encode())


def receive_record(connection: Connection, timeout_s: float) -> dict:
    if not connection.poll(timeout_s):
        raise TimeoutError("Validation child exceeded its time budget")
    record = json.loads(connection.recv_bytes(MAX_CONTROL_BYTES * 16))
    if record.get("kind") == "error":
        raise RuntimeError(record["detail"])
    return record


@contextmanager
def gpu_runtime(
    launch: Launch, physical_gpu: int
) -> Iterator[tuple[DeepseekCheckpoint, VllmConfig]]:
    # These imports must follow the per-process device selection. The parent
    # and spawn bootstrap stay CPU-only and must not initialize CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    from afd_plugin.expert_pool.checkpoint import DeepseekCheckpoint

    torch.cuda.set_device(0)
    checkpoint = DeepseekCheckpoint(Path(launch.model))
    config = EngineArgs(
        model=launch.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=4096,
        max_num_batched_tokens=max(4096, launch.directory.max_tokens),
        kernel_config={"moe_backend": "triton"},
    ).create_engine_config()
    with tempfile.TemporaryDirectory(prefix="pool-rendezvous-") as temp:
        try:
            with set_current_vllm_config(config):
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    local_rank=0,
                    distributed_init_method=f"file://{temp}/torch_init",
                    backend="nccl",
                )
                ensure_model_parallel_initialized(1, 1)
                init_workspace_manager(torch.device("cuda", 0))
                yield checkpoint, config
        finally:
            reset_workspace_manager()
            destroy_model_parallel()
            destroy_distributed_environment()


def worker_process(
    launch: Launch,
    physical_gpu: int,
    controls: tuple[Connection, Connection],
    supervisor: Connection,
) -> None:
    try:
        with gpu_runtime(launch, physical_gpu) as (checkpoint, _):
            import torch

            from afd_plugin.connectors.gpu.pool import PoolTransport
            from afd_plugin.expert_pool.executor import ExpertExecutor
            from afd_plugin.expert_pool.worker import ExpertWorker, WorkerPeer

            device = torch.device("cuda", 0)
            transports = [
                PoolTransport("127.0.0.1", port, 0, device, launch.timeout_s)
                for port in launch.ports
            ]
            executors = {
                placement.layer_id: ExpertExecutor(checkpoint, placement, device)
                for placement in launch.directory.placements
            }
            peers = tuple(
                WorkerPeer(f"client-{index}", 1, f"domain-{index}", control, transport)
                for index, (control, transport) in enumerate(
                    zip(controls, transports, strict=True)
                )
            )
            worker = ExpertWorker(launch.directory, executors, peers, audit_inputs=True)
            send_record(supervisor, {"kind": "ready", "role": "worker"})
            worker.run()
            send_record(
                supervisor,
                {
                    "kind": "finished",
                    "completed_calls": worker.completed_calls,
                    "resident_weight_bytes": sum(
                        e.weight_storage_bytes for e in executors.values()
                    ),
                    "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                },
            )
    except BaseException:
        send_record(supervisor, {"kind": "error", "detail": traceback.format_exc()})
        raise
    finally:
        supervisor.close()


def client_process(
    launch: Launch,
    physical_gpu: int,
    client_index: int,
    control: Connection,
    supervisor: Connection,
) -> None:
    try:
        with gpu_runtime(launch, physical_gpu) as (checkpoint, config):
            import torch
            from vllm.forward_context import set_forward_context

            from afd_plugin.connectors.gpu.pool import PoolTransport
            from afd_plugin.expert_pool.client import PoolClient
            from afd_plugin.expert_pool.router import DeepseekPoolRouter
            from afd_plugin.expert_pool.worker import tensor_digest
            from tools.expert_pool.validate_layer import compare, load_native

            device = torch.device("cuda", 0)
            transport = PoolTransport(
                "127.0.0.1",
                launch.ports[client_index],
                1,
                device,
                launch.timeout_s,
            )
            client = PoolClient(
                f"client-{client_index}",
                1,
                launch.directory,
                control,
                transport,
                launch.timeout_s,
            )
            with torch.inference_mode():
                references = {
                    p.layer_id: load_native(checkpoint, p.layer_id, config)
                    for p in launch.directory.placements
                }
                routers = {
                    p.layer_id: DeepseekPoolRouter(checkpoint, p.layer_id, device)
                    for p in launch.directory.placements
                }
                send_record(
                    supervisor, {"kind": "ready", "role": f"client-{client_index}"}
                )
                while True:
                    command = receive_record(supervisor, launch.timeout_s * 4)
                    if command["op"] == "close":
                        client.close()
                        send_record(supervisor, {"kind": "finished"})
                        return
                    if command["op"] != "cases":
                        raise ValueError("Unknown validation command")
                    time.sleep(command.get("delay_s", 0))
                    cases = []
                    for case in command["cases"]:
                        layer, tokens, seed = (
                            case["layer"],
                            case["tokens"],
                            case["seed"],
                        )
                        generator = torch.Generator(device=device).manual_seed(seed)
                        hidden = torch.randn(
                            tokens,
                            launch.directory.hidden_size,
                            device=device,
                            dtype=torch.bfloat16,
                            generator=generator,
                        )
                        weights, ids = routers[layer](hidden)
                        native = references[layer]
                        if tokens:
                            logits, _ = native.gate(hidden)
                            native_weights, native_ids = (
                                native.experts.router.select_experts(
                                    hidden,
                                    logits,
                                    topk_indices_dtype=torch.int32,
                                )
                            )
                            compare(ids, native_ids, 0, 0)
                            compare(weights, native_weights, 0, 0)
                            with set_forward_context(None, config, num_tokens=tokens):
                                expected = (
                                    native.experts.routed_experts.forward_modular(
                                        hidden, weights, ids
                                    )
                                )
                                expected_moe = native(hidden)
                            shared = (
                                native.shared_experts(hidden)
                                if native.shared_experts is not None
                                else torch.zeros_like(hidden)
                            )
                        else:
                            expected = expected_moe = shared = torch.zeros_like(hidden)
                        digests = {
                            "hidden": tensor_digest(hidden),
                            "weights": tensor_digest(weights),
                            "ids": tensor_digest(ids),
                        }
                        start_ns = time.perf_counter_ns()
                        actual, completion = client.execute(layer, hidden, weights, ids)
                        finish_ns = time.perf_counter_ns()
                        if completion.digests != digests:
                            raise AssertionError(
                                "GPU input or routing payload changed in transit"
                            )
                        if digests != {
                            "hidden": tensor_digest(hidden),
                            "weights": tensor_digest(weights),
                            "ids": tensor_digest(ids),
                        }:
                            raise AssertionError("Client payload was mutated")
                        cases.append(
                            {
                                **case,
                                "phase": command["phase"],
                                "client_id": client.client_id,
                                "start_ns": start_ns,
                                "finish_ns": finish_ns,
                                "plan": asdict(completion.plan),
                                "metrics": completion.metrics,
                                "input_and_route_digests_exact": True,
                                "native_routes_exact": True if tokens else None,
                                "routed": compare(
                                    actual, expected, launch.atol, launch.rtol
                                ),
                                "full_moe": compare(
                                    actual + shared,
                                    expected_moe,
                                    launch.atol,
                                    launch.rtol,
                                ),
                            }
                        )
                    send_record(supervisor, {"kind": "cases", "cases": cases})
    except BaseException:
        send_record(supervisor, {"kind": "error", "detail": traceback.format_exc()})
        raise
    finally:
        supervisor.close()


def check_idle(gpus: list[int]) -> list[dict]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    states = []
    for row in rows:
        index, gpu_uuid, memory, utilization = (
            value.strip() for value in row.split(",")
        )
        if int(index) in gpus:
            states.append(
                {
                    "index": int(index),
                    "uuid": gpu_uuid,
                    "memory_mib": int(memory),
                    "utilization": int(utilization),
                }
            )
    if {state["index"] for state in states} != set(gpus):
        raise ValueError("A selected physical GPU does not exist")
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader",
        ],
        text=True,
    ).splitlines()
    occupied = {row.split(",")[0].strip() for row in processes}
    if any(
        state["memory_mib"] > IDLE_MEMORY_LIMIT_MIB
        or state["utilization"]
        or state["uuid"] in occupied
        for state in states
    ):
        raise BusyGPUError(
            "Selected GPUs are occupied; no child processes were started"
        )
    return states


def run_validation(
    launch: Launch,
    gpus: list[int],
    batches: list[int],
    seeds: list[int],
    progress_path: Path,
) -> dict:
    context = multiprocessing.get_context("spawn")
    data_pipes = [context.Pipe() for _ in range(2)]
    supervisor_pipes = [context.Pipe() for _ in range(3)]
    clients = [pair[0] for pair in supervisor_pipes[:2]]
    children = [
        context.Process(
            target=client_process,
            args=(
                launch,
                gpus[index],
                index,
                data_pipes[index][0],
                supervisor_pipes[index][1],
            ),
            name=f"pool-validation-client-{index}",
        )
        for index in range(2)
    ]
    children.append(
        context.Process(
            target=worker_process,
            args=(
                launch,
                gpus[2],
                tuple(pair[1] for pair in data_pipes),
                supervisor_pipes[2][1],
            ),
            name="pool-validation-worker",
        )
    )
    cases = []
    try:
        for child in children:
            child.start()
        for pair in data_pipes:
            for connection in pair:
                connection.close()
        for parent, child_end in supervisor_pipes:
            child_end.close()
            if receive_record(parent, launch.timeout_s * 3)["kind"] != "ready":
                raise RuntimeError("Validation process did not become ready")
        layers = [placement.layer_id for placement in launch.directory.placements]

        def phase(
            name: str, commands: dict[int, list[dict]], delay_second: float = 0
        ) -> list[dict]:
            for index, items in commands.items():
                send_record(
                    clients[index],
                    {
                        "op": "cases",
                        "phase": name,
                        "cases": items,
                        "delay_s": delay_second if index == 1 else 0,
                    },
                )
            results = []
            for index in commands:
                record = receive_record(clients[index], launch.timeout_s * 3)
                if record["kind"] != "cases":
                    raise RuntimeError("Unexpected validation phase result")
                results.extend(record["cases"])
            cases.extend(results)
            with progress_path.open("a") as progress:
                for result in results:
                    progress.write(json.dumps(result, allow_nan=False) + "\n")
            print(json.dumps({"phase": name, "passed_cases": len(results)}), flush=True)
            return results

        warmup = [{"layer": layer, "tokens": batches[0], "seed": 0} for layer in layers]
        phase("warmup", {0: warmup, 1: warmup})
        phase("client_1_idle", {0: warmup})
        phase("client_0_idle", {1: list(reversed(warmup))})
        matrix = [
            {"layer": layer, "tokens": batch, "seed": seed}
            for seed in seeds
            for layer in layers
            for batch in batches
        ]
        concurrent = phase(
            "concurrent_independent_order",
            {
                0: matrix,
                1: [{**case, "seed": case["seed"] + 1000} for case in reversed(matrix)],
            },
        )
        overlaps = sum(
            left["start_ns"] < right["finish_ns"]
            and right["start_ns"] < left["finish_ns"]
            for left in concurrent
            if left["client_id"] == "client-0"
            for right in concurrent
            if right["client_id"] == "client-1"
        )
        if not overlaps:
            raise AssertionError("The run did not establish overlapping client calls")
        delayed = phase(
            "delayed_client", {0: warmup[:1], 1: warmup[-1:]}, delay_second=1
        )
        first = next(case for case in delayed if case["client_id"] == "client-0")
        second = next(case for case in delayed if case["client_id"] == "client-1")
        if first["finish_ns"] >= second["start_ns"]:
            raise AssertionError(
                "Other client did not finish before the delayed submission"
            )
        empty = [{"layer": layers[0], "tokens": 0, "seed": 0}]
        phase("empty_payload", {0: empty, 1: empty})
        plans = [case["plan"] for case in cases]
        if sorted(plan["generation"] for plan in plans) != list(
            range(1, len(cases) + 1)
        ):
            raise AssertionError("Missing, duplicate or reused buffer generation")
        if len(
            {
                (
                    plan["request"]["key"]["client_id"],
                    plan["request"]["key"]["session_epoch"],
                    plan["request"]["key"]["call_seq"],
                )
                for plan in plans
            }
        ) != len(cases):
            raise AssertionError("Duplicated call identity")
        for connection in clients:
            send_record(connection, {"op": "close"})
        for connection in clients:
            if receive_record(connection, launch.timeout_s)["kind"] != "finished":
                raise RuntimeError("Client shutdown did not complete")
        worker = receive_record(supervisor_pipes[2][0], launch.timeout_s)
        if worker["kind"] != "finished" or worker["completed_calls"] != len(cases):
            raise AssertionError("Worker completion count differs from returned calls")
        for child in children:
            child.join(20)
            if child.exitcode != 0:
                raise RuntimeError(
                    f"Validation child did not exit cleanly: {child.name}"
                )
        return {
            "cases": cases,
            "worker": worker,
            "independent_progress_passed": True,
            "buffer_generations_unique": True,
            "overlapping_call_pairs": overlaps,
        }
    finally:
        # Only these spawned children may be stopped. Never kill another job.
        for child in children:
            if child.pid is not None and child.is_alive():
                child.terminate()
        for child in children:
            if child.pid is not None:
                child.join(5)
                if child.is_alive():
                    child.kill()
                    child.join(5)
        for pair in (*data_pipes, *supervisor_pipes):
            for connection in pair:
                connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--gpus",
        nargs=3,
        type=int,
        required=True,
        help="Client 0, client 1, worker physical GPU IDs",
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 13, 26])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 257])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19])
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != 3 or min(args.gpus) < 0 or min(args.batches) <= 0:
        parser.error("Select three distinct GPUs and positive batches")
    if args.timeout <= 0 or any(
        not math.isfinite(value) or value < 0 for value in (args.atol, args.rtol)
    ):
        parser.error("Timeout must be positive and tolerances nonnegative")
    if args.output.exists():
        parser.error("Choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "model": str(args.model.resolve(strict=True)),
        "physical_gpus": args.gpus,
        "scope": (
            "Real NCCL and checkpoint component validation; "
            "no online A/KV service or performance claim"
        ),
        "input_kind": "Seeded synthetic BF16 activations with fresh native routing",
        "tolerances_declared_before_execution": {"atol": args.atol, "rtol": args.rtol},
        "timing_limitations": [
            "Input hashing and local reference computation are enabled for validation",
            "GPU transfer and compute phases are conservatively serialized",
            "Compute phase includes executor validation and output buffer copy",
            "Admitted queue time excludes unread control messages; "
            "client roundtrip includes them",
        ],
    }
    try:
        source_root = Path(__file__).resolve().parents[2]
        if Path(afd_plugin.__file__).resolve().parent != source_root / "afd_plugin":
            raise RuntimeError("Python imported another AFD tree; set PYTHONPATH")
        manifest = source_root / "source_manifest.json"
        if manifest.exists():
            report["source_manifest"] = json.loads(manifest.read_text())
            for name, digest in report["source_manifest"]["files"].items():
                if (
                    hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                    != digest
                ):
                    raise RuntimeError("Source snapshot changed since synchronization")
        with args.lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report["initial_gpu_state"] = check_idle(args.gpus)
            raw = json.loads((args.model / "config.json").read_text())
            directory = StaticDirectory(
                uuid.uuid4().hex,
                1,
                "worker-0",
                tuple(
                    ExpertPlacement(
                        layer,
                        raw["n_routed_experts"],
                        tuple(range(raw["n_routed_experts"])),
                    )
                    for layer in args.layers
                ),
                raw["hidden_size"],
                raw["num_experts_per_tok"],
                max(args.batches),
            )
            sockets = [socket.socket() for _ in range(2)]
            try:
                for reserved in sockets:
                    reserved.bind(("127.0.0.1", 0))
                ports = tuple(reserved.getsockname()[1] for reserved in sockets)
            finally:
                for reserved in sockets:
                    reserved.close()
            launch = Launch(
                str(args.model.resolve()),
                directory,
                ports,
                args.timeout,
                args.atol,
                args.rtol,
            )
            progress_path = args.output.with_suffix(".calls.jsonl")
            progress_path.touch(exist_ok=False)
            report["call_records"] = str(progress_path)
            report.update(
                run_validation(
                    launch,
                    args.gpus,
                    args.batches,
                    args.seeds,
                    progress_path,
                )
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
