# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Read only selected tensors from the existing local BF16 checkpoint."""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import DeepseekV2Config

from afd_plugin.expert_pool.placement import ExpertPlacement


class DeepseekCheckpoint:
    """The initial adapter supports DeepSeek-V2-Lite's unquantized MoE.

    No Hub access, remote code, weight download, or full-model construction.
    A path identifies this local checkpoint only; this is not a cross-model
    weight-equivalence scheme.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path.resolve(strict=True)
        raw_config = json.loads((self.model_path / "config.json").read_text())
        if (
            raw_config["model_type"] != "deepseek_v2"
            or raw_config.get("quantization_config") is not None
            or raw_config.get("torch_dtype", raw_config.get("dtype")) != "bfloat16"
            or raw_config["hidden_act"] != "silu"
            or raw_config["topk_method"] != "greedy"
            or raw_config["scoring_func"] != "softmax"
            or raw_config["n_group"] != 1
            or raw_config["topk_group"] != 1
            or raw_config["routed_scaling_factor"] != 1.0
            or raw_config.get("moe_router_dtype") not in (None, "bfloat16")
        ):
            raise ValueError(
                "Initial Pool adapter requires BF16 DeepSeek V2 with greedy "
                "softmax routing, one group, SiLU and routed scale 1.0"
            )
        self.config = DeepseekV2Config.from_dict(raw_config)
        index = json.loads(
            (self.model_path / "model.safetensors.index.json").read_text()
        )
        self.weight_map: dict[str, str] = index["weight_map"]

    def placement(self, layer_id: int, expert_ids: tuple[int, ...]) -> ExpertPlacement:
        config = self.config
        if (
            not config.first_k_dense_replace <= layer_id < config.num_hidden_layers
            or layer_id % config.moe_layer_freq != 0
        ):
            raise ValueError(f"Layer {layer_id} is not a routed MoE layer")
        return ExpertPlacement(layer_id, config.n_routed_experts, expert_ids)

    def iter_tensors(self, names: Sequence[str]) -> Iterator[tuple[str, torch.Tensor]]:
        """Stream requested CPU tensors; never materialize an entire shard."""
        shards: dict[Path, list[str]] = {}
        for name in names:
            shard = (self.model_path / self.weight_map[name]).resolve(strict=True)
            if not shard.is_relative_to(self.model_path):
                raise ValueError("Checkpoint shard escapes the model directory")
            shards.setdefault(shard, []).append(name)
        for shard, tensor_names in shards.items():
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for name in tensor_names:
                    tensor = handle.get_tensor(name)
                    if tensor.dtype != torch.bfloat16:
                        raise ValueError(f"Expected original BF16 tensor: {name}")
                    yield name, tensor

    def tensor(self, name: str) -> torch.Tensor:
        return next(self.iter_tensors([name]))[1]

    def expert_tensor_names(self, placement: ExpertPlacement) -> tuple[str, ...]:
        self.placement(placement.layer_id, placement.expert_ids)
        if placement.num_experts != self.config.n_routed_experts:
            raise ValueError("Placement does not match this checkpoint")
        prefix = f"model.layers.{placement.layer_id}.mlp.experts"
        return tuple(
            f"{prefix}.{expert_id}.{projection}.weight"
            for expert_id in placement.expert_ids
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
