# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native Attention/KV and shared FFN, with routed MoE served by Expert Pool.

Pinned to vLLM 0.26.0: BF16 DeepSeek-V2-Lite, TP/PP/DP=1, eager execution.
Each process retains its native request scheduler and KV cache. No layer ever
constructs a local FusedMoE or allocates routed expert weights, even transiently.
"""

import os
from collections.abc import Iterable
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn
from transformers import DeepseekV2Config
from vllm.config import ParallelConfig, VllmConfig
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models import deepseek_v2 as native
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekAttention,
    DeepseekV2Attention,
    DeepseekV2MLAAttention,
    DeepseekV2MLP,
    PPMissingLayer,
    RMSNorm,
    VocabParallelEmbedding,
    current_platform,
    get_pp_group,
    make_empty_intermediate_tensors_factory,
    make_layers,
    rocm_aiter_ops,
)

from afd_plugin.expert_pool.checkpoint import DeepseekCheckpoint
from afd_plugin.expert_pool.client import PoolClient
from afd_plugin.expert_pool.deployment import PoolDeployment
from afd_plugin.expert_pool.replica_client import ReplicaPoolClient


def validate_pool_config(config: VllmConfig) -> None:
    parallel = config.parallel_config
    if version("vllm") != "0.26.0":
        raise ValueError("Pool model is pinned to vLLM 0.26.0")
    if (
        config.model_config.dtype != torch.bfloat16
        or not config.model_config.enforce_eager
        or config.compilation_config.mode != 0
        or config.quant_config is not None
        or config.lora_config is not None
        or config.speculative_config is not None
        or config.kv_transfer_config is not None
        or config.model_config.enable_sleep_mode
        or any(
            size != 1
            for size in (
                parallel.tensor_parallel_size,
                parallel.pipeline_parallel_size,
                parallel.data_parallel_size,
            )
        )
        or parallel.enable_eplb
        or parallel.enable_expert_parallel
        or parallel.use_sequence_parallel_moe
        or config.offload_config.uva.cpu_offload_gb != 0
        or config.offload_config.prefetch.offload_group_size != 0
        or config.additional_config.get("afd") is not None
    ):
        raise ValueError(
            "Pool requires eager BF16, TP/PP/DP=1, no AFD/EPLB/offload/LoRA"
        )
    if os.environ.get("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", ""):
        raise ValueError("Pool requires real checkpoint routing")
    settings = config.additional_config["expert_pool"]
    deployment = PoolDeployment.read(Path(settings["deployment"]))
    deployment.client_endpoints(settings["client_id"])
    deployment.pool_directory()
    if Path(config.model_config.model).resolve() != Path(deployment.model).resolve():
        raise ValueError("A and E must bind the same immutable local checkpoint")
    DeepseekCheckpoint(Path(deployment.model))
    if config.scheduler_config.max_num_batched_tokens > deployment.max_tokens:
        raise ValueError("A batch capacity exceeds the published E buffer capacity")


class PoolRemoteMoE(nn.Module):
    def __init__(
        self,
        config: DeepseekV2Config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ) -> None:
        super().__init__()
        self.layer_id = int(prefix.split(".")[-2])
        self.client: PoolClient | ReplicaPoolClient | None = None
        self.completed_calls = 0
        self.token_rows = 0
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            params_dtype=torch.bfloat16,
            prefix=f"{prefix}.gate",
        )
        self.shared_experts = (
            DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                hidden_act=config.hidden_act,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
            if config.n_shared_experts is not None
            else None
        )
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.client is None:
            raise RuntimeError("Bind PoolClient before native profiling or inference")
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        router_logits, _ = self.gate(hidden_states)
        weights, ids = self.router.select_experts(
            hidden_states, router_logits, topk_indices_dtype=torch.int32
        )
        routed, _ = self.client.execute(self.layer_id, hidden_states, weights, ids)
        self.completed_calls += 1
        self.token_rows += hidden_states.shape[0]
        if self.shared_experts is not None:
            return routed + self.shared_experts(hidden_states)
        return routed


class PoolDeepseekV2DecoderLayer(native.DeepseekV2DecoderLayer):
    # Native v0.26.0 has no MoE factory seam. Copy only construction to avoid
    # allocating E on A; inherit the entire native Attention/residual forward.
    # Signature matches upstream. Remove this copy when a factory is available.
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        # ### PATCH START: construct native Attention without routed E allocation.
        nn.Module.__init__(self)
        # ### PATCH END

        if config is None:
            config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        # verify MLA attention specific fields
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )

        self.use_mha = use_mha

        if use_mha:
            attn_cls = DeepseekAttention
        elif model_config.use_mla:
            attn_cls = DeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention
        is_moe_layer = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        # TODO(wentao): enable SP MoE with PP after the PP boundary logic can safely
        # send/receive sequence-parallel hidden_states across stages.
        self.use_sequence_parallel_moe = (
            parallel_config.use_sequence_parallel_moe
            and parallel_config.pipeline_parallel_size == 1
            and is_moe_layer
        )
        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
        )

        if is_moe_layer:
            # ### PATCH START: routed work leaves this independent A/KV engine.
            self.mlp = PoolRemoteMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                # aiter applies routed_scaling_factor internally
                apply_routed_scale_to_output=not rocm_aiter_ops.is_fused_moe_enabled(),
            )
            # ### PATCH END
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)


class PoolDeepseekV2Model(native.DeepseekV2Model):
    # Native v0.26.0 has no Decoder factory. Copy construction, change only the
    # factory, and retain native embedding, forward and weight loader methods.
    # Eager is required: the inherited compile wrapper bypasses compilation.
    # Signature matches upstream; remove when Decoder injection is supported.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: no native decoder construction or compile wrapping.
        nn.Module.__init__(self)
        validate_pool_config(vllm_config)
        self.do_not_compile = True
        # ### PATCH END

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                self.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            # ### PATCH START: use the A-only decoder factory.
            lambda prefix: PoolDeepseekV2DecoderLayer(
                vllm_config=vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            ),
            # ### PATCH END
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.hidden_size
        )

        self.aux_hidden_state_layers = tuple[int, ...]()

        # Needed by load_weights
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        self.use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )


class PoolDeepseekV2ForCausalLM(native.DeepseekV2ForCausalLM):
    model_cls = PoolDeepseekV2Model

    # The native introspector expects local FusedMoE parameters for EPLB. Pool
    # owns none on A, and explicitly rejects EPLB. Signature matches upstream.
    def set_moe_parameters(self):
        # ### PATCH START: advertise zero local routed expert storage.
        self.moe_mlp_layers = []
        self.moe_layers = []
        self.num_moe_layers = sum(
            isinstance(layer.mlp, PoolRemoteMoE) for layer in self.model.layers
        )
        self.num_expert_groups = 1
        self.num_logical_experts = self.config.n_routed_experts
        self.num_physical_experts = 0
        self.num_local_physical_experts = 0
        self.num_routed_experts = self.config.n_routed_experts
        self.num_shared_experts = self.config.n_shared_experts
        self.num_redundant_experts = 0
        # ### PATCH END

    # Native weight loading already handles packed Attention and shared/dense
    # FFN weights. Filter routed weights before delegating this large loader.
    # Signature matches upstream; remove when role filtering is upstreamed.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: routed weights belong exclusively to the E service.
        return super().load_weights(
            (name, weight) for name, weight in weights if ".mlp.experts." not in name
        )
        # ### PATCH END

    def bind_pool_client(self, client: PoolClient | ReplicaPoolClient) -> None:
        expected = {p.layer_id for p in client.directory.placements}
        layers = {
            layer.layer_idx: layer.mlp
            for layer in self.model.layers
            if isinstance(layer.mlp, PoolRemoteMoE)
        }
        if set(layers) != expected:
            raise ValueError("A model layers differ from resident E coverage")
        for layer in layers.values():
            if layer.client is not None:
                raise RuntimeError("Pool client is already bound")
            layer.client = client

    def pool_status(self) -> dict:
        return {
            "routed_parameter_bytes": sum(
                p.numel() * p.element_size()
                for name, p in self.named_parameters()
                if ".mlp.experts." in name
            ),
            "all_parameter_bytes": sum(
                p.numel() * p.element_size() for p in self.parameters()
            ),
            "layers": {
                str(layer.layer_idx): {
                    "calls": layer.mlp.completed_calls,
                    "token_rows": layer.mlp.token_rows,
                }
                for layer in self.model.layers
                if isinstance(layer.mlp, PoolRemoteMoE)
            },
        }
