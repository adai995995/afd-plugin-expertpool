# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A local E executor consuming externally selected logical routes."""

from importlib.metadata import version

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

from afd_plugin.expert_pool.checkpoint import DeepseekCheckpoint
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.slot_backend import fused_expert_slots


class ExpertExecutor(nn.Module):
    """One layer, one GPU, BF16 Triton experts, with actual subset allocation.

    No Router, shared branch, Attention, KV, transport, or request scheduler.
    ``forward`` returns only this worker's routed contribution. A caller must
    arrange exactly one owner for every selected (token, top-k slot), combine
    the contributions, and add the A-side shared branch once.

    Value checks synchronize with the host and default to enabled. Disabling
    them is only valid for trusted routes produced by the bound native router.
    """

    def __init__(
        self,
        checkpoint: DeepseekCheckpoint,
        placement: ExpertPlacement,
        device: torch.device,
        *,
        validate_values: bool = True,
    ) -> None:
        super().__init__()
        if version("vllm") != "0.26.0":
            raise RuntimeError("Expert Pool executor is pinned to vLLM 0.26.0")
        if device.type != "cuda":
            raise ValueError("The initial Expert Pool executor requires CUDA")
        names = checkpoint.expert_tensor_names(placement)
        self.placement = placement
        self.validate_values = validate_values
        self.checkpoint_path = checkpoint.model_path
        self.hidden_size = checkpoint.config.hidden_size
        self.intermediate_size = checkpoint.config.moe_intermediate_size
        self.top_k = checkpoint.config.num_experts_per_tok
        num_local = len(placement.expert_ids)
        self.register_buffer(
            "w13",
            torch.empty(
                (num_local, 2 * self.intermediate_size, self.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        self.register_buffer(
            "w2",
            torch.empty(
                (num_local, self.hidden_size, self.intermediate_size),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        # Always provide a map, including for full coverage. The functional
        # Triton kernel then zeros absent route slots before the final sum.
        self.register_buffer(
            "expert_map",
            torch.tensor(placement.global_to_local(), dtype=torch.int32, device=device),
        )
        slots = {expert_id: slot for slot, expert_id in enumerate(placement.expert_ids)}
        for name, weight in checkpoint.iter_tensors(names):
            parts = name.split(".")
            slot = slots[int(parts[-3])]
            projection = parts[-2]
            if projection == "gate_proj":
                destination = self.w13[slot, : self.intermediate_size]
            elif projection == "up_proj":
                destination = self.w13[slot, self.intermediate_size :]
            else:
                destination = self.w2[slot]
            if destination.shape != weight.shape:
                raise ValueError(f"Checkpoint expert shape mismatch: {name}")
            destination.copy_(weight)

    @property
    def weight_storage_bytes(self) -> int:
        """Allocated tensor storage, excluding mapping/buffers/CUDA allocator cache."""
        return self.w13.untyped_storage().nbytes() + self.w2.untyped_storage().nbytes()

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        """Validate metadata, with optional explicit CUDA value auditing."""
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != self.hidden_size
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
        ):
            raise ValueError(
                "Expected contiguous [tokens, hidden_size] BF16 activations"
            )
        expected_shape = (hidden_states.shape[0], self.top_k)
        if (
            topk_weights.shape != expected_shape
            or topk_ids.shape != expected_shape
            or topk_weights.dtype != torch.float32
            or topk_ids.dtype != torch.int32
            or not topk_weights.is_contiguous()
            or not topk_ids.is_contiguous()
        ):
            raise ValueError(
                "Expected contiguous FP32 weights and INT32 logical Top-k IDs"
            )
        if any(
            tensor.device != self.w13.device
            for tensor in (hidden_states, topk_weights, topk_ids)
        ):
            raise ValueError(
                "Inputs and resident weights must be on the same CUDA device"
            )
        if self.validate_values:
            if bool(((topk_ids < 0) | (topk_ids >= self.placement.num_experts)).any()):
                raise ValueError("Router supplied an invalid logical expert ID")
            if not bool(torch.isfinite(topk_weights).all()) or bool(
                (topk_weights < 0).any()
            ):
                raise ValueError("Routing weights must be finite and nonnegative")

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        assignment_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute resident routes; an optional mask selects this physical copy.

        ``assignment_mask`` addresses token/top-k slots, never changes their
        logical IDs or weights, and must not assign an absent expert here.
        The caller must check global coverage/uniqueness across all workers.
        Unassigned slots use Triton's internal -1 skip sentinel.
        """
        self._validate_inputs(hidden_states, topk_weights, topk_ids)
        if assignment_mask is not None:
            if (
                assignment_mask.shape != topk_ids.shape
                or assignment_mask.dtype != torch.bool
                or assignment_mask.device != self.w13.device
            ):
                raise ValueError("Assignment mask must be a matching CUDA bool tensor")
            if bool((assignment_mask & (self.expert_map[topk_ids.long()] < 0)).any()):
                raise ValueError(
                    "Dispatch assigned an expert that is not resident here"
                )
            topk_ids = topk_ids.masked_fill(~assignment_mask, -1)
        if hidden_states.shape[0] == 0:
            return torch.zeros_like(hidden_states)
        return fused_experts(
            hidden_states,
            self.w13,
            self.w2,
            topk_weights,
            topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=self.placement.num_experts,
            expert_map=self.expert_map,
        )

    @torch.inference_mode()
    def forward_slots(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return weighted BF16 [tokens, top_k, hidden] resident contributions.

        Static disjoint ownership replaces a per-token assignment mask. Missing
        local experts produce zero slots; the A performs the final reduction
        once after collecting the unique owner for every logical route slot.
        """
        self._validate_inputs(hidden_states, topk_weights, topk_ids)
        return fused_expert_slots(
            hidden_states,
            self.w13,
            self.w2,
            topk_weights,
            topk_ids,
            global_num_experts=self.placement.num_experts,
            expert_map=self.expert_map,
        )
