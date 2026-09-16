#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real-checkpoint GPU correctness test, not online inference or a benchmark.

Use one idle GPU, VLLM_PLUGINS='', local weights and the synced PYTHONPATH.
The native reference uses unmodified vLLM 0.26.0 DeepseekV2MoE and its weight
loaders. Inputs are seeded synthetic hidden states; routing is computed afresh.
BF16 reduction changes are reported against predeclared tolerances, separately
from exact gate/route/weight checks. No timing from this script is a speedup.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import traceback
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

import afd_plugin
from afd_plugin.expert_pool.checkpoint import DeepseekCheckpoint
from afd_plugin.expert_pool.executor import ExpertExecutor
from afd_plugin.expert_pool.router import DeepseekPoolRouter

DEFAULT_ATOL = 0.01
DEFAULT_RTOL = 0.02
IDLE_MEMORY_LIMIT_MIB = 128


def compare(
    actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float
) -> dict[str, float | bool]:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
    delta = (actual.float() - expected.float()).abs()
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs_error": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.mean()) if delta.numel() else 0.0,
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ),
        "within_tolerance": True,
    }


def load_native(
    checkpoint: DeepseekCheckpoint, layer_id: int, config: VllmConfig
) -> DeepseekV2MoE:
    prefix = f"model.layers.{layer_id}.mlp"
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        native = DeepseekV2MoE(
            checkpoint.config, config.parallel_config, prefix=prefix
        ).eval()
    native.requires_grad_(False)
    native.gate.weight.copy_(checkpoint.tensor(f"{prefix}.gate.weight"))
    placement = checkpoint.placement(
        layer_id, tuple(range(checkpoint.config.n_routed_experts))
    )
    list(
        native.experts.load_weights(
            (name.removeprefix(f"{prefix}.experts."), tensor)
            for name, tensor in checkpoint.iter_tensors(
                checkpoint.expert_tensor_names(placement)
            )
        )
    )
    routed = native.experts.routed_experts
    routed.quant_method.process_weights_after_loading(routed)
    if native.shared_experts is not None:
        shared = native.shared_experts
        for shard, projection in enumerate(("gate_proj", "up_proj")):
            shared.gate_up_proj.weight_loader(
                shared.gate_up_proj.weight,
                checkpoint.tensor(f"{prefix}.shared_experts.{projection}.weight"),
                shard,
            )
        shared.down_proj.weight_loader(
            shared.down_proj.weight,
            checkpoint.tensor(f"{prefix}.shared_experts.down_proj.weight"),
        )
    return native


def expect_rejection(call: Callable[[], torch.Tensor], description: str) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f"Invalid request accepted: {description}")


@torch.inference_mode()
def validate_layer(
    checkpoint: DeepseekCheckpoint,
    config: VllmConfig,
    layer_id: int,
    batches: list[int],
    seeds: list[int],
    atol: float,
    rtol: float,
) -> dict:
    native = load_native(checkpoint, layer_id, config)
    router = DeepseekPoolRouter(checkpoint, layer_id, torch.device("cuda", 0))
    total_experts = checkpoint.config.n_routed_experts
    placements = [
        checkpoint.placement(layer_id, tuple(range(total_experts))),
        checkpoint.placement(layer_id, tuple(reversed(range(0, total_experts, 2)))),
        checkpoint.placement(layer_id, tuple(reversed(range(1, total_experts, 2)))),
    ]
    full, even, odd = [
        ExpertExecutor(checkpoint, placement, torch.device("cuda", 0))
        for placement in placements
    ]
    reference_weights = native.experts.routed_experts
    compare(router.gate.weight, native.gate.weight, 0, 0)
    compare(full.w13, reference_weights.w13_weight, 0, 0)
    compare(full.w2, reference_weights.w2_weight, 0, 0)
    for executor in (even, odd):
        indices = torch.tensor(executor.placement.expert_ids, device="cuda")
        compare(executor.w13, reference_weights.w13_weight[indices], 0, 0)
        compare(executor.w2, reference_weights.w2_weight[indices], 0, 0)
    expected_bytes = 3 * total_experts * full.hidden_size * full.intermediate_size * 2
    if full.weight_storage_bytes != expected_bytes:
        raise AssertionError("Full weight storage does not match BF16 tensor sizes")
    if even.weight_storage_bytes + odd.weight_storage_bytes != expected_bytes:
        raise AssertionError("Expert subsets retain unexpected weight storage")
    cases = []
    for seed in seeds:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        for batch in batches:
            x = torch.randn(
                batch,
                full.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            original_input = x.clone()
            topk_weights, topk_ids = router(x)
            logits, _ = native.gate(x)
            native_weights, native_ids = native.experts.router.select_experts(
                x, logits, topk_indices_dtype=torch.int32
            )
            compare(topk_ids, native_ids, 0, 0)
            compare(topk_weights, native_weights, 0, 0)
            saved_ids, saved_weights = topk_ids.clone(), topk_weights.clone()
            with set_forward_context(None, config, num_tokens=batch):
                native_output = native(x)
                native_routed = reference_weights.forward_modular(
                    x, native_weights, native_ids
                )
            shared = (
                native.shared_experts(x)
                if native.shared_experts is not None
                else torch.zeros_like(x)
            )
            full_output = full(x, topk_weights, topk_ids)
            # FP32 addition avoids an extra intermediate BF16 accumulation.
            # Each worker's returned BF16 partial is already rounded: exact
            # equality is therefore reported, not assumed, after partitioning.
            split_output = (
                even(x, topk_weights, topk_ids).float()
                + odd(x, topk_weights, topk_ids).float()
            ).to(x.dtype)
            # Full + even placements overlap. Assign each original route slot
            # exactly once, including splitting an equivalent expert by row.
            even_assignment = (topk_ids % 2 == 0) & (
                torch.arange(batch, device="cuda")[:, None] % 2 == 0
            )
            full_assignment = ~even_assignment
            ownership = full_assignment.int() + even_assignment.int()
            if not bool((ownership == 1).all()):
                raise AssertionError("A route has duplicate or missing ownership")
            overlap_output = (
                full(x, topk_weights, topk_ids, full_assignment).float()
                + even(x, topk_weights, topk_ids, even_assignment).float()
            ).to(x.dtype)
            no_assignment = full(
                x, topk_weights, topk_ids, torch.zeros_like(topk_ids, dtype=torch.bool)
            )
            compare(no_assignment, torch.zeros_like(x), 0, 0)
            result = {
                "seed": seed,
                "tokens": batch,
                "native_topk_ids_exact": True,
                "native_topk_weights_exact": True,
                "unique_route_ownership": True,
                "full_routed": compare(full_output, native_routed, atol, rtol),
                "split_routed": compare(split_output, native_routed, atol, rtol),
                "overlap_routed": compare(overlap_output, native_routed, atol, rtol),
                "full_moe": compare(shared + full_output, native_output, atol, rtol),
                "split_moe": compare(shared + split_output, native_output, atol, rtol),
                "overlap_moe": compare(
                    shared + overlap_output, native_output, atol, rtol
                ),
            }
            compare(x, original_input, 0, 0)
            compare(topk_ids, saved_ids, 0, 0)
            compare(topk_weights, saved_weights, 0, 0)
            cases.append(result)
            print(json.dumps({"layer": layer_id, **result}), flush=True)

    empty = torch.empty(0, full.hidden_size, dtype=torch.bfloat16, device="cuda")
    empty_weights, empty_ids = router(empty)
    compare(full(empty, empty_weights, empty_ids), empty, 0, 0)
    invalid_ids = topk_ids.clone()
    invalid_ids[0, 0] = total_experts
    expect_rejection(lambda: full(x, topk_weights, invalid_ids), "out-of-range ID")
    invalid_ids[0, 0] = -1
    expect_rejection(
        lambda: full(x, topk_weights, invalid_ids), "external skip sentinel"
    )
    invalid_weights = topk_weights.clone()
    invalid_weights[0, 0] = float("nan")
    expect_rejection(lambda: full(x, invalid_weights, topk_ids), "NaN routing weight")
    expect_rejection(
        lambda: full(x, topk_weights, topk_ids.long()), "wrong route dtype"
    )
    forced_odd = torch.ones_like(topk_ids)
    compare(even(x, topk_weights, forced_odd), torch.zeros_like(x), 0, 0)
    expect_rejection(
        lambda: even(
            x, topk_weights, forced_odd, torch.ones_like(topk_ids, dtype=torch.bool)
        ),
        "assignment to a nonresident expert",
    )
    return {
        "layer_id": layer_id,
        "native_weight_loaders_match_exactly": True,
        "native_backend": str(reference_weights.quant_method.unquantized_backend),
        "weight_storage_bytes": {
            "full": full.weight_storage_bytes,
            "even": even.weight_storage_bytes,
            "odd": odd.weight_storage_bytes,
        },
        "empty_input_and_unassigned_output_exact": True,
        "nonresident_only_output_exact_zero": True,
        "invalid_payload_checks_passed": True,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 13, 26])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 257])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(batch <= 0 for batch in args.batches):
        parser.error("All batches must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        parser.error("Set CUDA_VISIBLE_DEVICES to exactly --physical-gpu")
    if os.environ.get("VLLM_PLUGINS") != "":
        parser.error("Set VLLM_PLUGINS='' to keep the native reference unmodified")
    if version("vllm") != "0.26.0":
        parser.error("Requires vLLM 0.26.0")
    if args.output.exists():
        parser.error("Output already exists; choose a new run path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[2]
    if Path(afd_plugin.__file__).resolve().parent != source_root / "afd_plugin":
        raise RuntimeError("Python imported a different AFD tree; fix PYTHONPATH")
    manifest_path = source_root / "source_manifest.json"
    report = {
        "status": "running",
        "host": socket.gethostname(),
        "model": str(args.model.resolve()),
        "vllm_version": version("vllm"),
        "torch_version": torch.__version__,
        "afd_import_path": afd_plugin.__file__,
        "source_manifest": json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else None,
        "input_kind": (
            "seeded synthetic BF16 hidden states; "
            "real checkpoint and fresh native routing"
        ),
        "scope": (
            "single-GPU per-layer correctness; "
            "no communication, online service or performance claim"
        ),
        "tolerances_declared_before_execution": {"atol": args.atol, "rtol": args.rtol},
        "layers": [],
    }
    try:
        if report["source_manifest"] is not None:
            for name, expected in report["source_manifest"]["files"].items():
                if (
                    hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise RuntimeError(f"Synced source changed since manifest: {name}")
        with args.lock_path.open("a+") as lock, tempfile.TemporaryDirectory() as temp:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gpu_state = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={args.physical_gpu}",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            memory, utilization = [int(value.strip()) for value in gpu_state.split(",")]
            processes = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={args.physical_gpu}",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()
            foreign_pids = [
                int(pid.strip())
                for pid in processes
                if pid.strip().isdigit() and int(pid.strip()) != os.getpid()
            ]
            if memory > IDLE_MEMORY_LIMIT_MIB or utilization != 0 or foreign_pids:
                raise RuntimeError(f"Selected GPU is busy: {gpu_state}")
            torch.cuda.set_device(0)
            checkpoint = DeepseekCheckpoint(args.model)
            config = EngineArgs(
                model=str(checkpoint.model_path),
                dtype="bfloat16",
                enforce_eager=True,
                max_model_len=4096,
                max_num_batched_tokens=max(4096, *args.batches),
                kernel_config={"moe_backend": "triton"},
            ).create_engine_config()
            try:
                with set_current_vllm_config(config):
                    init_distributed_environment(
                        world_size=1,
                        rank=0,
                        local_rank=0,
                        distributed_init_method=f"file://{temp}/torch_init",
                        backend="nccl",
                    )
                    ensure_model_parallel_initialized(1, 1)
                    init_workspace_manager(torch.device("cuda", 0))
                    for layer_id in args.layers:
                        report["layers"].append(
                            validate_layer(
                                checkpoint,
                                config,
                                layer_id,
                                args.batches,
                                args.seeds,
                                args.atol,
                                args.rtol,
                            )
                        )
                        config.compilation_config.static_forward_context.clear()
                        gc.collect()
                        torch.cuda.empty_cache()
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                report["gpu_name"] = torch.cuda.get_device_name(0)
                report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
                report["status"] = "passed"
            finally:
                reset_workspace_manager()
                destroy_model_parallel()
                destroy_distributed_environment()
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"RESULT {report['status']}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
