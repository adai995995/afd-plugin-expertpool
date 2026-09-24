#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run one independent A/KV/Router HTTP service against a shared E pool.

The fixed-member Pool controller and E services must be launched separately.
This initial service intentionally exposes only a small local inference API;
its purpose is to prove two independent vLLM engines share resident Experts.
"""

import argparse
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from afd_plugin.expert_pool.deployment import PoolDeployment


def create_app(
    deployment_path: Path,
    client_id: str,
    *,
    max_model_len: int = 1024,
    kv_cache_bytes: int = 256 * 1024 * 1024,
):
    # Runtime imports follow CUDA_VISIBLE_DEVICES selection in main().
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    from afd_plugin.expert_pool import register_expert_pool

    deployment = PoolDeployment.read(deployment_path)
    if client_id not in deployment.client_ids or not deployment.pooled_admission:
        raise ValueError("A service must belong to one pooled-admission deployment")
    register_expert_pool()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=deployment.model,
                dtype="bfloat16",
                enforce_eager=True,
                seed=0,
                max_model_len=max_model_len,
                max_num_batched_tokens=deployment.max_tokens,
                max_num_seqs=8,
                kv_cache_memory_bytes=kv_cache_bytes,
                gpu_memory_utilization=0.4,
                enable_prefix_caching=False,
                enable_chunked_prefill=True,
                async_scheduling=False,
                kernel_config={"moe_backend": "triton"},
                disable_log_stats=True,
                worker_cls=(
                    "afd_plugin.v1.worker.pool_attention_worker.PoolAttentionWorker"
                ),
                hf_overrides={"architectures": ["PoolDeepseekV2ForCausalLM"]},
                additional_config={
                    "expert_pool": {
                        "deployment": str(deployment_path),
                        "client_id": client_id,
                    }
                },
            )
        )
        app.state.engine = engine
        try:
            yield
        finally:
            try:
                await engine.collective_rpc("close_pool")
            finally:
                engine.shutdown(timeout=10)

    app = FastAPI(lifespan=lifespan)

    class GenerateRequest(BaseModel):
        prompt: str = Field(min_length=1)
        max_tokens: int = Field(default=16, ge=1, le=128)

    @app.get("/health")
    async def health() -> dict:
        return {"ready": True, "client_id": client_id}

    @app.get("/pool/status")
    async def pool_status() -> dict:
        return {
            "client_id": client_id,
            "workers": await app.state.engine.collective_rpc("pool_status"),
        }

    @app.post("/generate")
    async def generate(request: GenerateRequest) -> dict:
        params = SamplingParams(
            temperature=0,
            max_tokens=request.max_tokens,
            ignore_eos=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        final = None
        async for result in app.state.engine.generate(
            request.prompt, params, uuid.uuid4().hex
        ):
            final = result
        if final is None or not final.finished:
            raise RuntimeError("The A engine did not finish inference")
        output = final.outputs[0]
        return {
            "client_id": client_id,
            "token_ids": list(output.token_ids),
            "text": output.text,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.gpu < 0 or not 1 <= args.port <= 65535:
        parser.error("GPU index and HTTP port must be valid")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import uvicorn

    uvicorn.run(
        create_app(args.deployment, args.client_id),
        host=args.host,
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
