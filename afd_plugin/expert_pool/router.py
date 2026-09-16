# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A-side native routing for the initial DeepSeek-V2-Lite adapter."""

import os

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)

from afd_plugin.expert_pool.checkpoint import DeepseekCheckpoint


class DeepseekPoolRouter(nn.Module):
    """Own the gate and native Top-k selection, with no routed expert weights.

    Parameters match DeepseekV2MoE in vLLM 0.26.0 for the explicitly supported
    checkpoint. EPLB remapping is absent, so returned IDs are logical IDs.
    Router simulation would invalidate this boundary check and is rejected.
    """

    def __init__(
        self, checkpoint: DeepseekCheckpoint, layer_id: int, device: torch.device
    ) -> None:
        super().__init__()
        checkpoint.placement(layer_id, (0,))
        if os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", ""):
            raise ValueError("Expert Pool correctness requires actual model routing")
        config = checkpoint.config
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        with torch.device(device):
            self.gate = GateLinear(
                config.hidden_size,
                config.n_routed_experts,
                params_dtype=torch.bfloat16,
                prefix=f"pool.layers.{layer_id}.gate",
            )
        weight = checkpoint.tensor(f"model.layers.{layer_id}.mlp.gate.weight")
        if weight.shape != self.gate.weight.shape:
            raise ValueError("Checkpoint gate shape mismatch")
        with torch.no_grad():
            self.gate.weight.copy_(weight)
        self.gate.requires_grad_(False)
        self.router = create_fused_moe_router(
            top_k=config.num_experts_per_tok,
            global_num_experts=config.n_routed_experts,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func=config.scoring_func,
            routed_scaling_factor=config.routed_scaling_factor,
        )

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (routing weights, logical expert IDs), as native vLLM does."""
        if hidden_states.shape[0] == 0:
            return (
                torch.empty(
                    (0, self.top_k), dtype=torch.float32, device=hidden_states.device
                ),
                torch.empty(
                    (0, self.top_k), dtype=torch.int32, device=hidden_states.device
                ),
            )
        router_logits, _ = self.gate(hidden_states)
        return self.router.select_experts(
            hidden_states, router_logits, topk_indices_dtype=torch.int32
        )
