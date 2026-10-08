# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scoped, opt-in last-row MoE captures for isolated forced-prefix diagnosis.

No hooks are installed by normal inference. Captures retain owned CUDA clones
until the request drains; the final RPC copies them to CPU. These probes change
GPU work and must never be used for throughput or SLO claims. Request/tensor
records belong in private experiment storage, not in Git.
"""

from functools import partial
from types import MethodType

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.router.base_router import FusedMoERouter
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.v1.worker.gpu_worker import Worker

from afd_plugin.expert_pool.protocol import Message
from afd_plugin.expert_pool.replica_client import ReplicaPoolClient
from afd_plugin.model_executor.models.pool_deepseek_v2 import PoolRemoteMoE


class ProbeClient(ReplicaPoolClient):
    """Only the module-facing execute seam is wrapped; the owner stays intact."""

    def __init__(self, delegate: ReplicaPoolClient, probe: "MoEProbe") -> None:
        self.delegate = delegate
        self.probe = probe

    def execute(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, Message]:
        result, completion = self.delegate.execute(
            layer_id, hidden_states, topk_weights, topk_ids
        )
        self.probe.save(layer_id, "routed_output", result)
        return result, completion


class MoEProbe:
    def __init__(self, model: nn.Module) -> None:
        self.records: dict[int, dict[str, torch.Tensor | int]] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.routers: list[tuple[FusedMoERouter, MethodType]] = []
        self.clients: list[tuple[PoolRemoteMoE, ReplicaPoolClient]] = []
        try:
            for name, module in model.named_modules():
                if not isinstance(module, (PoolRemoteMoE, DeepseekV2MoE)):
                    continue
                layer = int(name.split(".")[-2])
                self.handles.append(
                    module.register_forward_pre_hook(partial(self.before, layer))
                )
                self.handles.append(
                    module.register_forward_hook(
                        partial(self.after, layer, "moe_output")
                    )
                )
                self.handles.append(
                    module.gate.register_forward_hook(
                        partial(self.after, layer, "router_logits")
                    )
                )
                if module.shared_experts is not None:
                    self.handles.append(
                        module.shared_experts.register_forward_hook(
                            partial(self.after, layer, "shared_output")
                        )
                    )
                router = (
                    module.router
                    if isinstance(module, PoolRemoteMoE)
                    else module.experts.router
                )
                self.capture_router(router, layer)
                if isinstance(module, PoolRemoteMoE):
                    if not isinstance(module.client, ReplicaPoolClient):
                        raise ValueError("Probe requires a bound replica Pool client")
                    self.clients.append((module, module.client))
                    module.client = ProbeClient(module.client, self)
            if not self.handles:
                raise ValueError("No supported MoE layers found")
        except BaseException:
            self.close()
            raise

    def save(self, layer: int, name: str, tensor: torch.Tensor) -> None:
        self.records[layer][name] = tensor.detach()[-1:].clone()

    def before(
        self, layer: int, module: nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        self.records[layer] = {"num_rows": inputs[0].shape[0]}
        self.save(layer, "hidden_states", inputs[0])

    def after(
        self,
        layer: int,
        name: str,
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor | tuple[torch.Tensor, None],
    ) -> None:
        self.save(layer, name, output[0] if isinstance(output, tuple) else output)

    def capture_router(self, router: FusedMoERouter, layer: int) -> None:
        original = router.select_experts
        probe = self

        # Diagnostic patch: observe exact native routing without copying the
        # backend selector. Signature matches vLLM 0.26.0. Delegation preserves
        # the operator; close() removes this wrapper after one isolated request.
        def select_experts(
            self: FusedMoERouter,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            topk_indices_dtype: torch.dtype | None = None,
            *,
            input_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # ### PATCH START: scoped diagnostic CUDA clones.
            weights, ids = original(
                hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids
            )
            probe.save(layer, "routing_weights", weights)
            probe.save(layer, "expert_ids", ids)
            return weights, ids
            # ### PATCH END

        self.routers.append((router, original))
        router.select_experts = MethodType(select_experts, router)

    def close(self) -> None:
        for module, client in self.clients:
            module.client = client
        for router, original in self.routers:
            router.select_experts = original
        for handle in self.handles:
            handle.remove()
        self.clients.clear()
        self.routers.clear()
        self.handles.clear()

    def snapshot(self) -> dict:
        return {
            "scope": "last-row-of-final-MoE-call-per-layer; isolated prefix",
            "layers": {
                str(layer): {
                    name: value.cpu().tolist()
                    if isinstance(value, torch.Tensor)
                    else value
                    for name, value in record.items()
                }
                for layer, record in self.records.items()
            },
        }


def begin_probe(worker: Worker) -> None:
    worker.expert_pool_probe = MoEProbe(worker.model_runner.get_model())


def end_probe(worker: Worker) -> dict:
    probe = worker.expert_pool_probe
    worker.expert_pool_probe = None
    try:
        return probe.snapshot()
    finally:
        probe.close()
