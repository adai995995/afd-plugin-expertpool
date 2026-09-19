# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Single-host E service for independent vLLM engines; no HTTP or A/KV state.

Run with one visible CUDA device and an explicit private deployment JSON.
A launcher must supervise startup and terminate the process on a hung NCCL
operation. Startup connects all declared peers; execution has no peer barrier.
"""

import argparse
import json
import os
import stat
import tempfile
from contextlib import ExitStack
from multiprocessing.connection import Listener
from pathlib import Path

from afd_plugin.expert_pool.controller_service import connect_controller
from afd_plugin.expert_pool.deployment import PoolDeployment


def serve(
    deployment: PoolDeployment,
    *,
    worker_id: str | None = None,
    profile_dir: Path | None = None,
    profile_skip_calls: int = 512,
    profile_calls: int = 208,
) -> dict:
    # Keep module imports CPU-only so launchers can select the visible device
    # before importing Torch/vLLM in each spawned process.
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

    from afd_plugin.connectors.gpu.pool import PoolTransport
    from afd_plugin.expert_pool.checkpoint import DeepseekCheckpoint
    from afd_plugin.expert_pool.executor import ExpertExecutor
    from afd_plugin.expert_pool.profiling import WorkerProfiler
    from afd_plugin.expert_pool.worker import ExpertWorker, WorkerPeer

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    checkpoint = DeepseekCheckpoint(Path(deployment.model))
    directory = deployment.directory(worker_id)
    endpoints = deployment.worker_endpoints(directory.worker_id)
    config = EngineArgs(
        model=deployment.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=deployment.max_tokens,
        max_num_batched_tokens=deployment.max_tokens,
        kernel_config={"moe_backend": "triton"},
    ).create_engine_config()
    with ExitStack() as stack:
        listeners = []
        for endpoint in endpoints:
            parent = Path(endpoint.control_path).parent.stat()
            if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
                raise ValueError(
                    "Control sockets require an owned directory with mode 0700"
                )
            listeners.append(
                stack.enter_context(Listener(endpoint.control_path, family="AF_UNIX"))
            )
        rendezvous = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="pool-e-init-")
        )
        try:
            with set_current_vllm_config(config):
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    local_rank=0,
                    distributed_init_method=f"file://{rendezvous}/torch_init",
                    backend="nccl",
                )
                ensure_model_parallel_initialized(1, 1)
                init_workspace_manager(device)
                peers = []
                for listener, endpoint in zip(listeners, endpoints, strict=True):
                    control = listener.accept()
                    stack.callback(control.close)
                    transport = PoolTransport(
                        "127.0.0.1",
                        endpoint.nccl_port,
                        0,
                        device,
                        deployment.timeout_s,
                        reuse_events=deployment.execution.reuse_cuda_events,
                    )
                    stack.callback(transport.close)
                    peers.append(
                        WorkerPeer(
                            endpoint.client_id,
                            endpoint.session_epoch,
                            endpoint.domain,
                            control,
                            transport,
                        )
                    )
                executors = {
                    placement.layer_id: ExpertExecutor(
                        checkpoint,
                        placement,
                        device,
                        validate_values=deployment.execution.validate_worker_values,
                    )
                    for placement in directory.placements
                }
                profiler = (
                    WorkerProfiler(profile_dir, profile_skip_calls, profile_calls)
                    if profile_dir is not None
                    else None
                )
                controller = (
                    connect_controller(deployment, "worker", directory.worker_id)
                    if deployment.controller is not None
                    else None
                )
                if controller is not None:
                    stack.callback(controller.close)
                worker = ExpertWorker(
                    directory,
                    executors,
                    tuple(peers),
                    execution=deployment.execution,
                    profiler=profiler,
                    controller=controller,
                    demand_aware=deployment.demand_aware,
                )
                worker.run()
                return {
                    "worker_id": directory.worker_id,
                    "placement_version": directory.version,
                    "pid": os.getpid(),
                    "completed_calls": worker.completed_calls,
                    "client_calls": dict(worker.client_calls),
                    "layer_calls": dict(worker.layer_calls),
                    "resident_layers": sorted(executors),
                    "dispatch_mode": deployment.dispatch_mode,
                    "demand_aware": deployment.demand_aware,
                    "expert_assignments": worker.expert_assignments,
                    "resident_experts": {
                        str(layer): list(executor.placement.expert_ids)
                        for layer, executor in executors.items()
                    },
                    "resident_weight_bytes": sum(
                        e.weight_storage_bytes for e in executors.values()
                    ),
                    "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(
                        device
                    ),
                }
        finally:
            # ExitStack also handles partially established channels on startup
            # failure. PoolTransport.close and Connection.close are idempotent.
            reset_workspace_manager()
            destroy_model_parallel()
            destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument(
        "--worker-id", help="Required when the deployment has multiple E workers"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            serve(PoolDeployment.read(args.deployment), worker_id=args.worker_id)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
