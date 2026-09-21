# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit single-host deployment contract for independent vLLM engines.

The launcher binds one immutable local checkpoint to one model_id. A shared
architecture or tensor shape is never treated as evidence of equal weights.
Control sockets must live in a launcher-owned private directory. They carry
bounded JSON messages; activation tensors travel over the prebuilt NCCL pairs.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from afd_plugin.expert_pool.controller import CONTROLLER_POLICIES
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import CallKey
from afd_plugin.expert_pool.scheduler import StaticDirectory

MAX_DEPLOYMENT_BYTES = 65536
MAX_UNIX_PATH_BYTES = 100


@dataclass(frozen=True)
class ExecutionOptions:
    """Explicit execution checks; defaults preserve the validated prototype.

    Disabling value checks requires routes from the bound, trusted native
    router. Shapes, devices, control identities and buffer credits are always
    checked. These options never change routing, weights or the GEMM backend.
    """

    validate_client_values: bool = True
    validate_worker_values: bool = True
    reuse_cuda_events: bool = False
    defer_output_sync: bool = False

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("Execution options must be explicit booleans")


@dataclass(frozen=True)
class ClientEndpoint:
    client_id: str
    session_epoch: int
    domain: str
    control_path: str
    nccl_port: int
    worker_id: str = "worker-0"

    def __post_init__(self) -> None:
        CallKey(self.client_id, self.session_epoch, 0)
        if not isinstance(self.worker_id, str) or not 0 < len(self.worker_id) <= 128:
            raise ValueError("Invalid worker identity")
        if not isinstance(self.domain, str) or not 0 < len(self.domain) <= 128:
            raise ValueError("Invalid service domain")
        if (
            not Path(self.control_path).is_absolute()
            or len(self.control_path.encode()) > MAX_UNIX_PATH_BYTES
        ):
            raise ValueError("Control socket requires a short absolute local path")
        if type(self.nccl_port) is not int or not 1024 <= self.nccl_port <= 65535:
            raise ValueError("Invalid NCCL rendezvous port")


@dataclass(frozen=True)
class WorkerPlacement:
    """Resident layers and a shared expert subset; None means full coverage."""

    worker_id: str
    layer_ids: tuple[int, ...] | None = None
    expert_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.worker_id, str) or not 0 < len(self.worker_id) <= 128:
            raise ValueError("Invalid worker identity")
        if self.layer_ids is not None and (
            not isinstance(self.layer_ids, tuple)
            or not self.layer_ids
            or any(type(layer) is not int or layer < 0 for layer in self.layer_ids)
            or len(set(self.layer_ids)) != len(self.layer_ids)
        ):
            raise ValueError("Worker layers must be unique nonnegative integers")
        if self.expert_ids is not None and (
            not isinstance(self.expert_ids, tuple)
            or not self.expert_ids
            or any(type(expert) is not int or expert < 0 for expert in self.expert_ids)
            or len(set(self.expert_ids)) != len(self.expert_ids)
        ):
            raise ValueError("Worker experts must be unique nonnegative integers")


@dataclass(frozen=True)
class ControllerConfig:
    socket_dir: str
    scheduling_policy: str = "round_robin"

    def __post_init__(self) -> None:
        if self.scheduling_policy not in CONTROLLER_POLICIES:
            raise ValueError("Unknown controller scheduling policy")
        if (
            not isinstance(self.socket_dir, str)
            or not Path(self.socket_dir).is_absolute()
        ):
            raise ValueError("Controller requires an absolute private socket directory")


@dataclass(frozen=True)
class PoolDeployment:
    model: str
    model_id: str
    max_tokens: int
    clients: tuple[ClientEndpoint, ...]
    timeout_s: int = 120
    execution: ExecutionOptions = ExecutionOptions()
    workers: tuple[WorkerPlacement, ...] = (WorkerPlacement("worker-0"),)
    placement_version: int = 1
    controller: ControllerConfig | None = None
    dispatch_mode: str = "whole_layer"
    demand_aware: bool = False
    compact_output: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.execution, ExecutionOptions):
            raise ValueError("Deployment requires typed execution options")
        if not Path(self.model).is_absolute():
            raise ValueError("Deployment requires an absolute checkpoint path")
        if not isinstance(self.model_id, str) or not 0 < len(self.model_id) <= 256:
            raise ValueError("Invalid immutable checkpoint identity")
        if any(
            type(value) is not int or value <= 0
            for value in (self.max_tokens, self.timeout_s)
        ):
            raise ValueError("Deployment capacities must be positive integers")
        if not isinstance(self.clients, tuple) or not self.clients:
            raise ValueError("Deployment requires immutable client endpoints")
        if any(not isinstance(client, ClientEndpoint) for client in self.clients):
            raise ValueError("Invalid client endpoint")
        if (
            not isinstance(self.workers, tuple)
            or not self.workers
            or any(not isinstance(worker, WorkerPlacement) for worker in self.workers)
            or len({worker.worker_id for worker in self.workers}) != len(self.workers)
        ):
            raise ValueError("Deployment requires unique immutable worker placements")
        if type(self.placement_version) is not int or self.placement_version < 0:
            raise ValueError("Invalid placement version")
        if type(self.demand_aware) is not bool or (
            self.demand_aware and self.dispatch_mode != "expert_partitioned"
        ):
            raise ValueError("Expert demand requires explicit expert partitioning")
        if type(self.compact_output) is not bool or (
            self.compact_output
            and (not self.demand_aware or self.dispatch_mode != "expert_partitioned")
        ):
            raise ValueError("Compact output requires demand-aware expert partitioning")
        if not isinstance(self.dispatch_mode, str) or self.dispatch_mode not in {
            "whole_layer",
            "expert_partitioned",
        }:
            raise ValueError("Unknown expert dispatch mode")
        if self.dispatch_mode == "whole_layer":
            if any(worker.expert_ids is not None for worker in self.workers):
                raise ValueError("Whole-layer dispatch requires full worker experts")
        else:
            if (
                not isinstance(self.controller, ControllerConfig)
                or self.controller.scheduling_policy != "ready_first"
            ):
                raise ValueError(
                    "Expert partitioning requires a ready-first controller"
                )
            if any(worker.expert_ids is None for worker in self.workers):
                raise ValueError("Expert partitioning requires explicit worker experts")
        pairs = {(client.client_id, client.worker_id) for client in self.clients}
        expected = {
            (client_id, worker_id)
            for client_id in self.client_ids
            for worker_id in self.worker_ids
        }
        if pairs != expected or len(pairs) != len(self.clients):
            raise ValueError("Each client requires exactly one channel to every worker")
        for client_id in self.client_ids:
            identities = {
                (endpoint.session_epoch, endpoint.domain)
                for endpoint in self.client_endpoints(client_id)
            }
            if len(identities) != 1:
                raise ValueError(
                    "A client must keep the same session and service domain"
                )
        for field in ("control_path", "nccl_port"):
            values = [asdict(client)[field] for client in self.clients]
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate client {field}")
        if self.controller is not None:
            if not isinstance(self.controller, ControllerConfig):
                raise ValueError("Deployment requires typed controller configuration")
            paths = [
                self.controller_path(role, identity)
                for role, identities in (
                    ("client", self.client_ids),
                    ("worker", self.worker_ids),
                )
                for identity in identities
            ]
            if set(paths) & {c.control_path for c in self.clients}:
                raise ValueError("Controller and data bootstrap sockets must differ")

    @classmethod
    def read(cls, path: Path) -> "PoolDeployment":
        with path.open("rb") as handle:
            payload = handle.read(MAX_DEPLOYMENT_BYTES + 1)
        if len(payload) > MAX_DEPLOYMENT_BYTES:
            raise ValueError("Deployment file exceeds its size limit")
        raw = json.loads(payload)
        raw["clients"] = tuple(ClientEndpoint(**client) for client in raw["clients"])
        raw["execution"] = ExecutionOptions(**raw.get("execution", {}))
        if raw.get("controller") is not None:
            raw["controller"] = ControllerConfig(**raw["controller"])
        if "workers" in raw:
            raw["workers"] = tuple(
                WorkerPlacement(
                    **{
                        **worker,
                        "layer_ids": (
                            tuple(worker["layer_ids"])
                            if worker.get("layer_ids") is not None
                            else None
                        ),
                        "expert_ids": (
                            tuple(worker["expert_ids"])
                            if worker.get("expert_ids") is not None
                            else None
                        ),
                    }
                )
                for worker in raw["workers"]
            )
        return cls(**raw)

    def controller_path(self, role: str, identity: str) -> str:
        if self.controller is None or role not in {"client", "worker"}:
            raise ValueError("Controller role is not configured")
        identities = self.client_ids if role == "client" else self.worker_ids
        path = str(
            Path(self.controller.socket_dir)
            / f"{role}-{identities.index(identity)}.sock"
        )
        if len(path.encode()) > MAX_UNIX_PATH_BYTES:
            raise ValueError("Controller socket path is too long")
        return path

    @property
    def worker_ids(self) -> tuple[str, ...]:
        return tuple(sorted(worker.worker_id for worker in self.workers))

    @property
    def client_ids(self) -> tuple[str, ...]:
        return tuple(sorted({endpoint.client_id for endpoint in self.clients}))

    def client_endpoints(self, client_id: str) -> tuple[ClientEndpoint, ...]:
        endpoints = tuple(
            sorted(
                (e for e in self.clients if e.client_id == client_id),
                key=lambda e: e.worker_id,
            )
        )
        if not endpoints:
            raise ValueError("Client is absent from the deployment")
        return endpoints

    def worker_endpoints(self, worker_id: str) -> tuple[ClientEndpoint, ...]:
        if worker_id not in self.worker_ids:
            raise ValueError("Worker is absent from the deployment")
        return tuple(
            sorted(
                (e for e in self.clients if e.worker_id == worker_id),
                key=lambda e: e.client_id,
            )
        )

    def endpoint(self, client_id: str, worker_id: str | None = None) -> ClientEndpoint:
        endpoints = self.client_endpoints(client_id)
        if worker_id is None:
            if len(endpoints) != 1:
                raise ValueError("Select a worker for a multi-worker deployment")
            return endpoints[0]
        for endpoint in endpoints:
            if endpoint.worker_id == worker_id:
                return endpoint
        raise ValueError("Worker is absent from this client's deployment")

    def pool_directory(self) -> PoolDirectory:
        config = json.loads((Path(self.model) / "config.json").read_text())
        required = {
            layer
            for layer in range(
                config["first_k_dense_replace"], config["num_hidden_layers"]
            )
            if layer % config["moe_layer_freq"] == 0
        }
        directories = []
        covered: set[int] = set()
        for worker in sorted(self.workers, key=lambda worker: worker.worker_id):
            layers = required if worker.layer_ids is None else set(worker.layer_ids)
            if not layers <= required:
                raise ValueError("Placement includes a non-MoE or nonexistent layer")
            covered.update(layers)
            directories.append(
                StaticDirectory(
                    self.model_id,
                    self.placement_version,
                    worker.worker_id,
                    tuple(
                        ExpertPlacement(
                            layer,
                            config["n_routed_experts"],
                            (
                                worker.expert_ids
                                if worker.expert_ids is not None
                                else tuple(range(config["n_routed_experts"]))
                            ),
                        )
                        for layer in sorted(layers)
                    ),
                    config["hidden_size"],
                    config["num_experts_per_tok"],
                    self.max_tokens,
                    allow_partial_experts=self.dispatch_mode == "expert_partitioned",
                )
            )
        if covered != required:
            raise ValueError("Pool placement must cover every model MoE layer")
        return PoolDirectory(
            tuple(directories),
            expert_partitioned=self.dispatch_mode == "expert_partitioned",
        )

    def directory(self, worker_id: str | None = None) -> StaticDirectory:
        if worker_id is None:
            if len(self.workers) != 1:
                raise ValueError("Select a worker for a multi-worker deployment")
            worker_id = self.workers[0].worker_id
        for worker in self.pool_directory().workers:
            if worker.worker_id == worker_id:
                return worker
        raise ValueError("Worker is absent from the deployment")
