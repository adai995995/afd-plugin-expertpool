#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Measure real captured Expert calls separately and together, on resident weights.

This is an isolated kernel calibration, not an online replay or a request SLO
result. Captured arrival dependencies cannot be assumed unchanged after a new
batching policy. CUDA-event intervals include host submission gaps; optional
profiling reports the sum of actual CUDA activity durations separately.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

from afd_plugin.expert_pool.deployment import PoolDeployment
from tools.expert_pool.benchmark_workload import percentile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 13, 26])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--profile-cuda", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 5 or not args.layers:
        parser.error("Calibration requires layers and at least five samples")
    # The caller selects an idle visible GPU before importing the runtime.
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

    from afd_plugin.expert_pool.checkpoint import DeepseekCheckpoint
    from afd_plugin.expert_pool.executor import ExpertExecutor

    deployment = PoolDeployment.read(args.deployment)
    if not deployment.local_full_pipeline:
        parser.error("Calibration currently requires full local E")
    captured = torch.load(args.inputs, map_location="cpu", weights_only=True)
    candidates = []
    for sample in captured["samples"]:
        offset = 0
        for member in sample["members"]:
            end = offset + member["num_tokens"]
            candidates.append(
                {
                    "layer_id": sample["layer_id"],
                    **member,
                    "inputs": tuple(t[offset:end] for t in sample["inputs"]),
                }
            )
            offset = end
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    checkpoint = DeepseekCheckpoint(Path(deployment.model))
    directory = deployment.directory()
    config = EngineArgs(
        model=deployment.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=deployment.max_tokens,
        max_num_batched_tokens=deployment.batching.max_tokens,
        kernel_config={"moe_backend": "triton"},
    ).create_engine_config()
    results = []

    def measure(operation):
        for _ in range(3):
            operation()
        torch.cuda.synchronize()
        stream_times, host_times = [], []
        for _ in range(args.repeats):
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            started = time.perf_counter_ns()
            operation()
            host_times.append((time.perf_counter_ns() - started) / 1e6)
            end.record()
            end.synchronize()
            stream_times.append(begin.elapsed_time(end))
        return {
            "stream_elapsed_ms_p50": percentile(stream_times, 0.5),
            "stream_elapsed_ms_p95": percentile(stream_times, 0.95),
            "host_submit_ms_p50": percentile(host_times, 0.5),
        }

    def cuda_activities(operation):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            operation()
            torch.cuda.synchronize()
        events = [e for e in profile.events() if e.device_type.name == "CUDA"]
        return {
            "activity_ms": sum(e.device_time_total for e in events) / 1000,
            "activity_count": len(events),
        }

    try:
        with (
            tempfile.TemporaryDirectory(prefix="pool-calibration-") as rendezvous,
            set_current_vllm_config(config),
            torch.inference_mode(),
        ):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{rendezvous}/init",
                backend="nccl",
            )
            ensure_model_parallel_initialized(1, 1)
            init_workspace_manager(device)
            for layer in args.layers:
                calls = [c for c in candidates if c["layer_id"] == layer]
                pair = next(
                    (
                        (a, b)
                        for i, a in enumerate(calls)
                        for b in calls[i + 1 :]
                        if a["client_id"] != b["client_id"]
                    ),
                    None,
                )
                if pair is None:
                    raise RuntimeError(
                        "Capture lacks two distinct A clients at the requested layer"
                    )
                executor = ExpertExecutor(
                    checkpoint,
                    next(p for p in directory.placements if p.layer_id == layer),
                    device,
                    validate_values=False,
                )
                left, right = (
                    tuple(t.contiguous().to(device) for t in c["inputs"]) for c in pair
                )
                combined = tuple(
                    torch.cat((a, b), dim=0) for a, b in zip(left, right, strict=True)
                )
                rows = [left[0].shape[0], right[0].shape[0]]

                def merge(combined=combined, left=left, right=right, rows=rows):
                    for destination, a, b in zip(combined, left, right, strict=True):
                        destination[: rows[0]].copy_(a)
                        destination[rows[0] :].copy_(b)

                def separate(executor=executor, left=left, right=right):
                    return torch.cat((executor(*left), executor(*right)), dim=0)

                def merged(executor=executor, combined=combined):
                    return executor(*combined)

                reference = separate().clone()
                actual = merged().clone()
                if not torch.allclose(reference, actual, atol=0.02, rtol=0.02):
                    raise AssertionError(
                        "Captured route output exceeds the declared tolerance"
                    )
                merged_result = measure(merged)
                merge_result = measure(merge)

                # Output concatenation is needed only to compare values, not
                # for two separate E executions. Time their original kernels.
                def separate_kernels(executor=executor, left=left, right=right):
                    executor(*left)
                    executor(*right)

                separate_result = measure(separate_kernels)
                counts = [
                    torch.bincount(
                        t[2].flatten().long(),
                        minlength=checkpoint.config.n_routed_experts,
                    )
                    .cpu()
                    .tolist()
                    for t in (left, right)
                ]
                record = {
                    "layer_id": layer,
                    "members": [
                        {k: v for k, v in c.items() if k != "inputs"} for c in pair
                    ],
                    "rows": rows,
                    "expert_token_counts": counts,
                    "max_abs_error": float(
                        (reference.float() - actual.float()).abs().max()
                    ),
                    "separate": separate_result,
                    "merged_kernel": merged_result,
                    "input_merge": merge_result,
                    "predicted_stream_saving_ms": separate_result[
                        "stream_elapsed_ms_p50"
                    ]
                    - merged_result["stream_elapsed_ms_p50"]
                    - merge_result["stream_elapsed_ms_p50"],
                }
                if args.profile_cuda:
                    record["cuda_activity"] = {
                        "separate": cuda_activities(separate_kernels),
                        "merged": cuda_activities(merged),
                        "input_merge": cuda_activities(merge),
                    }
                results.append(record)
                del executor
    finally:
        reset_workspace_manager()
        destroy_model_parallel()
        destroy_distributed_environment()
    report = {
        "scope": "isolated real-input kernel calibration; excludes A/network/queueing",
        "tensor_atol": 0.02,
        "tensor_rtol": 0.02,
        "repeats": args.repeats,
        "timing_scope": (
            "eager CUDA-event intervals include host submission gaps; "
            "profiler CUDA activity sum is separate"
        ),
        "cases": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(results),
                "max_abs_error": max(c["max_abs_error"] for c in results),
                "stream_savings_ms": [c["predicted_stream_saving_ms"] for c in results],
            }
        )
    )


if __name__ == "__main__":
    main()
