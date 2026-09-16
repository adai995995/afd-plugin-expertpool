# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Experimental Expert Pool with opt-in, independent vLLM Attention engines."""


def register_expert_pool() -> None:
    """Register only the Pool model; do not enable the legacy AFD patches.

    Call before constructing EngineArgs in each application process. The Pool
    worker repeats registration in its spawned process before loading the model.
    Lazy import keeps the control protocol usable without vLLM or CUDA installed.
    """
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "PoolDeepseekV2ForCausalLM",
        "afd_plugin.model_executor.models.pool_deepseek_v2:PoolDeepseekV2ForCausalLM",
    )
