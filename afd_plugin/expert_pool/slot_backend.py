# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: N806
# Retain the upstream matrix dimension names in the copied function below.
"""Unreduced routed contributions from the pinned vLLM 0.26.0 Triton path.

The copied functional implementation preserves the upstream GEMMs and their
configuration. Only its final reduction is removed so separate Expert workers
can return the original top-k slots for one reduction on the Attention side.
Remove this copy when the pinned upstream functional API exposes those slots;
on a vLLM upgrade, recopy the function and reapply the marked differences.
"""

import functools
from importlib.metadata import version

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _get_config_quant_dtype,
    _prepare_expert_assignment,
    dispatch_fused_moe_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.triton_utils import tl

if version("vllm") != "0.26.0":
    raise RuntimeError("Expert Pool slot backend is pinned to vLLM 0.26.0")


def fused_expert_slots(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int,
    expert_map: torch.Tensor,
) -> torch.Tensor:
    """Return weighted BF16 ``[tokens, top_k, hidden]`` resident contributions.

    This entry point supports unquantized BF16 SiLU experts with full hidden
    dimensions. It performs no tensor-parallel reduction. Shapes, dtypes and
    layout are checked without reading CUDA values on the host. Logical route
    validity and unique ownership across workers remain the caller's contract.
    Missing local experts produce zero slots; routing IDs and weights are not
    changed. The caller must keep the original slot order until ``moe_sum``.
    """
    if (
        hidden_states.ndim != 2
        or hidden_states.dtype != torch.bfloat16
        or not hidden_states.is_cuda
        or not hidden_states.is_contiguous()
    ):
        raise ValueError("Expected contiguous CUDA BF16 hidden states")
    if (
        w1.ndim != 3
        or w2.ndim != 3
        or w1.shape[0] != w2.shape[0]
        or w1.shape[1] != 2 * w2.shape[2]
        or w1.shape[2] != hidden_states.shape[1]
        or w2.shape[1] != hidden_states.shape[1]
        or w1.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
        or w1.stride(-1) != 1
        or w2.stride(-1) != 1
    ):
        raise ValueError("Expected matching BF16 SiLU expert weights")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape[0] != hidden_states.shape[0]
        or topk_ids.shape[1] == 0
        or topk_weights.shape != topk_ids.shape
        or topk_ids.dtype != torch.int32
        or topk_weights.dtype != torch.float32
        or not topk_ids.is_contiguous()
        or not topk_weights.is_contiguous()
    ):
        raise ValueError("Expected contiguous INT32 IDs and FP32 top-k weights")
    if (
        global_num_experts <= 0
        or not 0 < w1.shape[0] <= global_num_experts
        or expert_map.ndim != 1
        or expert_map.shape[0] != global_num_experts
        or expert_map.dtype != torch.int32
        or not expert_map.is_contiguous()
    ):
        raise ValueError("Expected a complete INT32 global-to-local expert map")
    if any(
        tensor.device != hidden_states.device
        for tensor in (w1, w2, topk_weights, topk_ids, expert_map)
    ):
        raise ValueError("Expert weights and routes must share the CUDA device")
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0, topk_ids.shape[1], hidden_states.shape[1]))
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
    )


# Upstream: vllm/model_executor/layers/fused_moe/fused_moe.py, vLLM 0.26.0.
# Patch reason: functional fused_experts exposes only the reduced BF16 output.
# Patch functionality: retain weighted top-k slots for a single remote merge.
# Signature matches upstream, with no added parameters. The return tensor is
# [tokens, top_k, hidden] instead of [tokens, hidden]; use the BF16 wrapper above.
# Upstream processes the supplied token batch once; no chunking is introduced.
def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_shape: list[int] | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if ocp_mx_scheme is not None:
        raise NotImplementedError(
            f"Using ocp_mx_scheme={ocp_mx_scheme} in functional fused_experts call is "
            "deprecated. Please use OCP_MXQuantizationEmulationTritonExperts."
        )

    # Convert string activation to enum for internal use
    activation_enum = MoEActivation.from_str(activation)

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.size(1) // 2 == w1.size(2), "Hidden size mismatch"
    else:
        assert hidden_states.size(1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
        )

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)

    M = num_tokens

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    # Note: for use_int8_w8a16 or use_int4_w4a16, the activations are
    # quantized prior to calling fused_experts.
    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # This needs separate memory since it's used concurrently with cache1
    activation_out_dim = mk.FusedMoEExpertsModular.adjust_N_for_activation(
        N, activation_enum
    )
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    # ### PATCH START: No reduced output buffer
    # Return the existing top-k workspace instead of allocating [tokens, hidden].
    # ### PATCH END: No reduced output buffer

    qhidden_states, a1q_scale = moe_kernel_quantize_input(
        A=hidden_states,
        A_scale=a1_scale,
        quant_dtype=quant_dtype,
        per_act_token_quant=per_channel_quant,
        block_shape=block_shape,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
        topk_ids,
        config,
        num_tokens,
        top_k_num,
        global_num_experts,
        expert_map,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        block_shape=block_shape,
        ignore_invalid_experts=True,
    )

    dispatch_fused_moe_kernel(
        qhidden_states,
        w1,
        intermediate_cache1,
        a1q_scale,
        w1_scale,
        w1_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        top_k_num,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        B_bias=w1_bias,
    )

    apply_moe_activation(
        activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N)
    )

    qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
        A=intermediate_cache2,
        A_scale=a2_scale,
        quant_dtype=quant_dtype,
        per_act_token_quant=per_channel_quant,
        block_shape=block_shape,
    )

    if expert_map is not None:
        intermediate_cache3.zero_()

    dispatch_fused_moe_kernel(
        qintermediate_cache2,
        w2,
        intermediate_cache3,
        a2q_scale,
        w2_scale,
        w2_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        B_bias=w2_bias,
    )

    # ### PATCH START: Preserve routed slots
    return intermediate_cache3
    # ### PATCH END: Preserve routed slots
