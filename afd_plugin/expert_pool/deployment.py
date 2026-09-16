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

from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import CallKey
from afd_plugin.expert_pool.scheduler import StaticDirectory

MAX_DEPLOYMENT_BYTES = 65536
MAX_UNIX_PATH_BYTES = 100


@dataclass(frozen=True)
class ClientEndpoint:
    client_id: str
    session_epoch: int
    domain: str
    control_path: str
    nccl_port: int

    def __post_init__(self) -> None:
        CallKey(self.client_id, self.session_epoch, 0)
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
class PoolDeployment:
    model: str
    model_id: str
    max_tokens: int
    clients: tuple[ClientEndpoint, ...]
    timeout_s: int = 120

    def __post_init__(self) -> None:
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
        for field in ("client_id", "control_path", "nccl_port"):
            values = [asdict(client)[field] for client in self.clients]
            if len(set(values)) != len(values):
                raise ValueError(f"Duplicate client {field}")

    @classmethod
    def read(cls, path: Path) -> "PoolDeployment":
        with path.open("rb") as handle:
            payload = handle.read(MAX_DEPLOYMENT_BYTES + 1)
        if len(payload) > MAX_DEPLOYMENT_BYTES:
            raise ValueError("Deployment file exceeds its size limit")
        raw = json.loads(payload)
        raw["clients"] = tuple(ClientEndpoint(**client) for client in raw["clients"])
        return cls(**raw)

    def endpoint(self, client_id: str) -> ClientEndpoint:
        for endpoint in self.clients:
            if endpoint.client_id == client_id:
                return endpoint
        raise ValueError("Client is absent from the deployment")

    def directory(self) -> StaticDirectory:
        config = json.loads((Path(self.model) / "config.json").read_text())
        return StaticDirectory(
            self.model_id,
            1,
            "worker-0",
            tuple(
                ExpertPlacement(
                    layer,
                    config["n_routed_experts"],
                    tuple(range(config["n_routed_experts"])),
                )
                for layer in range(
                    config["first_k_dense_replace"], config["num_hidden_layers"]
                )
                if layer % config["moe_layer_freq"] == 0
            ),
            config["hidden_size"],
            config["num_experts_per_tok"],
            self.max_tokens,
        )
