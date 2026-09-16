# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bind a Pool client without replacing the native vLLM scheduler or runner."""

import os
import time
from multiprocessing.connection import Client
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker

from afd_plugin.connectors.gpu.pool import PoolTransport
from afd_plugin.expert_pool import register_expert_pool
from afd_plugin.expert_pool.client import PoolClient
from afd_plugin.expert_pool.deployment import PoolDeployment
from afd_plugin.model_executor.models.pool_deepseek_v2 import (
    PoolDeepseekV2ForCausalLM,
    validate_pool_config,
)

SOCKET_RETRY_INTERVAL_S = 0.05


class PoolAttentionWorker(Worker):
    pool_client: PoolClient | None = None

    # Register in the spawned worker before the native loader resolves the
    # custom architecture. The large native device/runner setup is unchanged;
    # delegation is intentional. Signature matches vLLM 0.26.0.
    def init_device(self):
        # ### PATCH START: opt-in Pool model and supported runtime contract.
        register_expert_pool()
        validate_pool_config(self.vllm_config)
        if self.use_v2_model_runner:
            raise ValueError("Pool integration currently requires the V1 model runner")
        # ### PATCH END
        super().init_device()

    # Connect after native loading but before profiling performs a forward.
    # No runner substitution or global monkeypatch is needed. Delegation keeps
    # the native loading/memory-pool lifecycle intact. Signature matches upstream.
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        # ### PATCH START: all A/E weights must come from the bound checkpoint.
        if load_dummy_weights or self.load_config.load_format == "dummy":
            raise ValueError("Pool does not support dummy or randomized weights")
        # ### PATCH END
        super().load_model(load_dummy_weights=load_dummy_weights)
        # ### PATCH START: attach one independent client to all local MoE layers.
        model = self.model_runner.get_model()
        if not isinstance(model, PoolDeepseekV2ForCausalLM):
            raise ValueError("PoolAttentionWorker requires the Pool model architecture")
        settings = self.vllm_config.additional_config["expert_pool"]
        deployment = PoolDeployment.read(Path(settings["deployment"]))
        endpoint = deployment.endpoint(settings["client_id"])
        deadline = time.monotonic() + deployment.timeout_s
        while True:
            try:
                control = Client(endpoint.control_path, family="AF_UNIX")
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "E service control socket did not become ready"
                    ) from None
                time.sleep(SOCKET_RETRY_INTERVAL_S)
        try:
            transport = PoolTransport(
                "127.0.0.1", endpoint.nccl_port, 1, self.device, deployment.timeout_s
            )
        except BaseException:
            control.close()
            raise
        self.pool_client = PoolClient(
            endpoint.client_id,
            endpoint.session_epoch,
            deployment.directory(),
            control,
            transport,
            deployment.timeout_s,
        )
        model.bind_pool_client(self.pool_client)
        # ### PATCH END

    def pool_status(self) -> dict:
        model = self.model_runner.get_model()
        if not isinstance(model, PoolDeepseekV2ForCausalLM):
            raise RuntimeError("Pool model was not loaded")
        return {
            "pid": os.getpid(),
            "client_id": self.pool_client.client_id if self.pool_client else None,
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            **model.pool_status(),
        }

    def close_pool(self) -> None:
        """Call after draining requests, before terminating the native engine."""
        if self.pool_client is not None:
            self.pool_client.close()

    # Close the external service channel while CUDA is alive, then use native
    # cleanup. Delegation avoids copying unrelated teardown. Signature unchanged.
    def shutdown(self) -> None:
        try:
            # ### PATCH START: release this engine's E channel exactly once.
            self.close_pool()
            # ### PATCH END
        finally:
            super().shutdown()
