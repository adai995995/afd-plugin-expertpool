#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Create a private, fixed-member deployment for independent A and E services.

This prepares metadata only. Start the controller, each E service, and each A
service in separate processes using the generated path. All members must be on
one host and use the same local checkpoint during this initial deployment.
"""

import argparse
import json
import os
import socket
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    ControllerConfig,
    ExecutionOptions,
    MAX_DEPLOYMENT_BYTES,
    PoolDeployment,
    WorkerPlacement,
)
from afd_plugin.expert_pool.protocol import MAX_RECEIVE_SLOTS


def _free_ports(count: int) -> tuple[int, ...]:
    reserved_sockets = []
    for _ in range(count):
        reserved = socket.socket()
        reserved.bind(("127.0.0.1", 0))
        reserved_sockets.append(reserved)
    try:
        return tuple(sock.getsockname()[1] for sock in reserved_sockets)
    finally:
        for sock in reserved_sockets:
            sock.close()


def build_deployment(
    model: Path,
    socket_dir: Path,
    client_count: int,
    worker_count: int,
    max_tokens: int,
    timeout_s: int,
    *,
    replicated_experts: tuple[int, ...] = (),
    split_assignments: bool = False,
) -> PoolDeployment:
    """Describe one shared E pool and distinct A clients without loading weights."""

    if (
        type(client_count) is not int
        or not 2 <= client_count <= MAX_RECEIVE_SLOTS
        or type(worker_count) is not int
        or worker_count < 1
    ):
        raise ValueError("Shared Pool needs 2–8 A clients and at least one E worker")
    model = model.resolve(strict=True)
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "deepseek_v2":
        raise ValueError("Initial shared Pool supports DeepSeek V2 checkpoints")
    expert_count = config["n_routed_experts"]
    if worker_count > expert_count:
        raise ValueError("Every E worker must own at least one routed Expert")
    if (
        not isinstance(replicated_experts, tuple)
        or any(
            type(expert) is not int or not 0 <= expert < expert_count
            for expert in replicated_experts
        )
        or len(set(replicated_experts)) != len(replicated_experts)
        or (replicated_experts and worker_count < 2)
        or (split_assignments and not replicated_experts)
    ):
        raise ValueError("Assignment splitting needs valid resident replicas")
    ports = iter(_free_ports(client_count * worker_count))
    clients = tuple(
        ClientEndpoint(
            f"client-{client_index}",
            1,
            f"domain-{client_index}",
            str(socket_dir / f"a{client_index}e{worker_index}.sock"),
            next(ports),
            f"worker-{worker_index}",
        )
        for worker_index in range(worker_count)
        for client_index in range(client_count)
    )
    workers = tuple(
        WorkerPlacement(
            f"worker-{worker_index}",
            expert_ids=tuple(
                sorted(
                    set(range(worker_index, expert_count, worker_count))
                    | set(replicated_experts)
                )
            ),
        )
        for worker_index in range(worker_count)
    )
    deployment = PoolDeployment(
        model=str(model),
        model_id=uuid.uuid4().hex,
        max_tokens=max_tokens,
        clients=clients,
        timeout_s=timeout_s,
        execution=ExecutionOptions(
            validate_client_values=False,
            validate_worker_values=False,
            reuse_cuda_events=True,
            defer_output_sync=True,
            warmup_before_ready=True,
        ),
        workers=workers,
        controller=ControllerConfig(str(socket_dir), "ready_first"),
        dispatch_mode="expert_partitioned",
        demand_aware=True,
        compact_output=True,
        receive_slots=client_count,
        pooled_admission=True,
        expert_replicated=bool(replicated_experts),
        split_assignments=split_assignments,
    )
    deployment.pool_directory()
    return deployment


def write_private_deployment(path: Path, deployment: PoolDeployment) -> None:
    """Publish the immutable startup manifest without a world-readable window."""

    payload = json.dumps(asdict(deployment), separators=(",", ":")).encode()
    if len(payload) > MAX_DEPLOYMENT_BYTES:
        raise ValueError("Deployment file exceeds its size limit")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--replicate-expert", action="append", type=int, default=[])
    parser.add_argument("--split-assignments", action="store_true")
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be an absolute path")
    socket_dir = Path(tempfile.mkdtemp(prefix="ep-", dir="/tmp"))
    try:
        deployment = build_deployment(
            args.model,
            socket_dir,
            args.clients,
            args.workers,
            args.max_tokens,
            args.timeout,
            replicated_experts=tuple(args.replicate_expert),
            split_assignments=args.split_assignments,
        )
        write_private_deployment(args.output, deployment)
    except BaseException:
        socket_dir.rmdir()
        raise
    print(
        json.dumps(
            {
                "deployment": str(args.output),
                "clients": deployment.client_ids,
                "workers": deployment.worker_ids,
                "socket_dir": str(socket_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
