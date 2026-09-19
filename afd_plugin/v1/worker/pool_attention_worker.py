# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bind a Pool client without replacing the native vLLM scheduler or runner."""

import os
import time
from contextlib import ExitStack
from multiprocessing.connection import Client
from pathlib import Path

import torch
from vllm.v1.worker.gpu_worker import Worker

from afd_plugin.connectors.gpu.pool import PoolTransport
from afd_plugin.expert_pool import register_expert_pool
from afd_plugin.expert_pool.client import PoolClient
from afd_plugin.expert_pool.controlled_client import ControlledPoolClient
from afd_plugin.expert_pool.controller_service import connect_controller
from afd_plugin.expert_pool.deployment import PoolDeployment
from afd_plugin.expert_pool.replica_client import ReplicaPoolClient
from afd_plugin.model_executor.models.pool_deepseek_v2 import (
    PoolDeepseekV2ForCausalLM,
    validate_pool_config,
)

SOCKET_RETRY_INTERVAL_S = 0.05


class PoolAttentionWorker(Worker):
    pool_client: ReplicaPoolClient | None = None

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
        directories = {
            worker.worker_id: worker for worker in deployment.pool_directory().workers
        }
        # Both sides connect in sorted identity order. Arbitrary input JSON
        # order must not create a cycle during pairwise NCCL initialization.
        with ExitStack() as startup:
            channels = []
            deadline = time.monotonic() + deployment.timeout_s
            for endpoint in deployment.client_endpoints(settings["client_id"]):
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
                startup.callback(control.close)
                transport = PoolTransport(
                    "127.0.0.1",
                    endpoint.nccl_port,
                    1,
                    self.device,
                    deployment.timeout_s,
                    reuse_events=deployment.execution.reuse_cuda_events,
                )
                startup.callback(transport.close)
                channels.append(
                    PoolClient(
                        endpoint.client_id,
                        endpoint.session_epoch,
                        directories[endpoint.worker_id],
                        control,
                        transport,
                        deployment.timeout_s,
                        validate_values=deployment.execution.validate_client_values,
                    )
                )
            if deployment.controller is not None:
                controller = connect_controller(
                    deployment, "client", settings["client_id"]
                )
                startup.callback(controller.close)
                client = ControlledPoolClient(
                    tuple(channels),
                    controller,
                    scheduling_policy=deployment.controller.scheduling_policy,
                )
            else:
                client = ReplicaPoolClient(
                    tuple(channels),
                    client_offset=deployment.client_ids.index(settings["client_id"]),
                )
            model.bind_pool_client(client)
            self.pool_client = client
            startup.pop_all()  # Ownership passes to close_pool after binding.
        # ### PATCH END

    def pool_status(self) -> dict:
        model = self.model_runner.get_model()
        if not isinstance(model, PoolDeepseekV2ForCausalLM):
            raise RuntimeError("Pool model was not loaded")
        return {
            "pid": os.getpid(),
            "client_id": self.pool_client.client_id if self.pool_client else None,
            "dispatch": (
                self.pool_client.dispatch_status() if self.pool_client else None
            ),
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "call_metrics": (
                self.pool_client.metrics.snapshot()
                if self.pool_client is not None and self.pool_client.metrics is not None
                else None
            ),
            **model.pool_status(),
        }

    def pool_set_metrics(self, enabled: bool = True) -> None:
        """Reset scalar aggregates after warmup and before submitting requests."""
        if self.pool_client is None:
            raise RuntimeError("Pool client was not connected")
        self.pool_client.set_metrics(enabled)

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
