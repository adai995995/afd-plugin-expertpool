#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real-GPU compact-output correctness and helper-cost validation, without weights.

Reference indexing and explicit CUDA synchronization belong only to this test.
The production helper has neither a route-to-host copy nor a host synchronization.
Recorded CUDA-event intervals follow warmup and describe helper invocation costs,
including launch gaps; they are not service throughput or performance gains.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import statistics
import tempfile
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import afd_plugin
from tools.expert_pool.validate_service import BusyGPUError, check_idle

if TYPE_CHECKING:
    import torch

    from afd_plugin.expert_pool.compact_output import CompactOutputWorkspace

BATCHES = (0, 1, 16, 173, 1025)
TOP_K_VALUES = (1, 6, 8)
HIDDEN_SIZES = (257, 2048)
NUM_EXPERTS = 17
SLOT_TAG_BASE = 128
DESTINATION_GUARD = -17.0
PAYLOAD_GUARD = -23.0
STREAM_DELAY_CYCLES = 10_000_000
STREAM_VALUES = (7.0, 13.0, 19.0, 29.0)


def assert_bits(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    import torch

    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or not torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    ):
        raise AssertionError(f"Bitwise mismatch: {label}")


def check_stream_reuse(workspace: CompactOutputWorkspace) -> dict:
    import torch

    device = workspace.device
    tokens, top_k = 173, 6
    assignments = tokens * top_k
    ids = torch.zeros((tokens, top_k), dtype=torch.int32, device=device)
    ownership = torch.ones(NUM_EXPERTS, dtype=torch.bool, device=device)
    sources = [
        torch.full(
            (tokens, top_k, workspace.hidden_size),
            value,
            dtype=torch.bfloat16,
            device=device,
        )
        for value in STREAM_VALUES
    ]
    destinations = [torch.empty_like(source) for source in sources]
    # Warm these exact shapes so JIT compilation cannot serialize the race test.
    payload = workspace.receive_buffer(assignments)
    payload.copy_(sources[0].flatten(0, 1))
    workspace.scatter(
        payload, ids, ownership, destinations[0], num_assignments=assignments
    )
    torch.cuda.synchronize(device)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    streams = tuple(torch.cuda.Stream(device=device) for _ in range(2))
    for stream in streams:
        stream.wait_event(ready)
    # No host synchronization occurs between these alternating streams. A delay
    # keeps the first scatter pending while the next receive wants its buffer.
    for index, (source, destination) in enumerate(
        zip(sources, destinations, strict=True)
    ):
        with torch.cuda.stream(streams[index % len(streams)]):
            payload = workspace.receive_buffer(assignments)
            payload.copy_(source.flatten(0, 1))
            torch.cuda._sleep(STREAM_DELAY_CYCLES)
            workspace.scatter(
                payload, ids, ownership, destination, num_assignments=assignments
            )
    streams[(len(sources) - 1) % len(streams)].synchronize()
    for index, (actual, expected) in enumerate(zip(destinations, sources, strict=True)):
        assert_bits(actual, expected, f"alternating-stream destination {index}")

    # Also switch streams while pack's scratch and output writes are pending.
    # The earlier payload has no external consumer; only the final view is read.
    workspace.pack(sources[0], ids, ownership, num_assignments=assignments)
    torch.cuda.synchronize(device)
    with torch.cuda.stream(streams[0]):
        torch.cuda._sleep(STREAM_DELAY_CYCLES)
        workspace.pack(sources[0], ids, ownership, num_assignments=assignments)
    with torch.cuda.stream(streams[1]):
        packed = workspace.pack(
            sources[-1], ids, ownership, num_assignments=assignments
        )
    streams[1].synchronize()
    assert_bits(packed, sources[-1].flatten(0, 1), "alternating-stream pack")
    return {
        "alternating_scatter_calls": len(sources),
        "host_sync_between_reuses": False,
        "delayed_scatter_then_cross_stream_overwrite_passed": True,
        "cross_stream_pack_reuse_passed": True,
    }


def check_rejections(workspace: CompactOutputWorkspace) -> dict:
    import torch

    ids = torch.zeros((1, 1), dtype=torch.int32, device=workspace.device)
    ownership = torch.ones(NUM_EXPERTS, dtype=torch.bool, device=workspace.device)
    source = torch.zeros(
        (1, 1, workspace.hidden_size), dtype=torch.bfloat16, device=workspace.device
    )
    alias = workspace.receive_buffer(1).view_as(source)
    checks = (
        lambda: workspace.receive_buffer(-1),
        lambda: workspace.receive_buffer(workspace.max_assignments + 1),
        lambda: workspace.pack(source, ids, ownership, num_assignments=2),
        lambda: workspace.pack(source, ids.long(), ownership, num_assignments=1),
        lambda: workspace.pack(source, ids, ownership.float(), num_assignments=1),
        lambda: workspace.pack(source.float(), ids, ownership, num_assignments=1),
        lambda: workspace.pack(alias, ids, ownership, num_assignments=1),
        lambda: workspace.scatter(
            alias.flatten(0, 1), ids, ownership, alias, num_assignments=1
        ),
    )
    for check in checks:
        try:
            check()
        except ValueError:
            continue
        raise AssertionError("Invalid compact-output metadata or alias accepted")
    return {"metadata_and_alias_rejections": len(checks)}


def validate() -> dict:
    # Imports occur after the parent selects the physical GPU and checks idleness.
    import torch
    from vllm import _custom_ops as ops

    from afd_plugin.expert_pool.compact_output import CompactOutputWorkspace

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    maximum = max(BATCHES) * max(TOP_K_VALUES)
    experts = torch.arange(NUM_EXPERTS, device=device)
    ownerships = {
        "empty": torch.zeros(NUM_EXPERTS, dtype=torch.bool, device=device),
        "all": torch.ones(NUM_EXPERTS, dtype=torch.bool, device=device),
        "first_last": (experts == 0) | (experts == NUM_EXPERTS - 1),
        "alternating": experts % 2 == 0,
        "sparse": experts == NUM_EXPERTS // 2,
    }
    partitions = tuple(experts % 3 == index for index in range(3))
    sender = {
        size: CompactOutputWorkspace(maximum, size, device) for size in HIDDEN_SIZES
    }
    receiver = {
        size: CompactOutputWorkspace(maximum, size, device) for size in HIDDEN_SIZES
    }
    results, restored_results = [], []
    for hidden_size in HIDDEN_SIZES:
        send, receive = sender[hidden_size], receiver[hidden_size]
        for top_k in TOP_K_VALUES:
            for tokens in BATCHES:
                assignments = tokens * top_k
                generator = torch.Generator(device=device).manual_seed(tokens + top_k)
                ids = torch.randint(
                    NUM_EXPERTS,
                    (tokens, top_k),
                    dtype=torch.int32,
                    device=device,
                    generator=generator,
                )
                if assignments:
                    ids.view(-1)[0] = 0
                if assignments > 1:
                    ids.view(-1)[-1] = NUM_EXPERTS - 1
                if assignments > 2:
                    ids.view(-1)[assignments // 2] = NUM_EXPERTS // 2
                slots = torch.randn(
                    (tokens, top_k, hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                # Two exactly representable BF16 columns identify every slot,
                # including across prefix blocks; remaining values are random.
                tags = torch.arange(assignments, device=device)
                flat = slots.flatten(0, 1)
                flat[:, 0] = (tags % SLOT_TAG_BASE).to(torch.bfloat16)
                flat[:, 1] = (tags // SLOT_TAG_BASE).to(torch.bfloat16)
                original_slots, original_ids = slots.clone(), ids.clone()
                for name, ownership in ownerships.items():
                    original_ownership = ownership.clone()
                    # Dynamic indexing and CPU count inspection are reference
                    # operations confined to this validation tool.
                    selected = ownership[ids.long()].view(-1)
                    count = int(selected.sum())
                    expected = flat[selected]
                    destination = torch.full_like(slots, DESTINATION_GUARD)
                    expected_destination = destination.clone()
                    expected_destination.flatten(0, 1)[selected] = expected
                    send.receive_buffer(maximum).fill_(PAYLOAD_GUARD)
                    packed = send.pack(slots, ids, ownership, num_assignments=count)
                    buffer = receive.receive_buffer(count)
                    buffer.copy_(packed)
                    receive.scatter(
                        buffer, ids, ownership, destination, num_assignments=count
                    )
                    # This is correctness warmup, not a measured service run.
                    assert_bits(packed, expected, "packed order and values")
                    assert_bits(destination, expected_destination, "owned-only scatter")
                    tail = send.receive_buffer(maximum)[count:]
                    assert_bits(
                        tail, torch.full_like(tail, PAYLOAD_GUARD), "packed tail guard"
                    )
                    pack_ms = scatter_ms = 0.0
                    if count:
                        begin, end = (
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        )
                        begin.record()
                        packed = send.pack(slots, ids, ownership, num_assignments=count)
                        end.record()
                        end.synchronize()
                        pack_ms = begin.elapsed_time(end)
                        buffer = receive.receive_buffer(count)
                        buffer.copy_(packed)
                        begin.record()
                        receive.scatter(
                            buffer, ids, ownership, destination, num_assignments=count
                        )
                        end.record()
                        end.synchronize()
                        scatter_ms = begin.elapsed_time(end)
                    assert_bits(slots, original_slots, "input slots unchanged")
                    assert_bits(ids, original_ids, "input routes unchanged")
                    assert_bits(ownership, original_ownership, "ownership unchanged")
                    results.append(
                        {
                            "tokens": tokens,
                            "top_k": top_k,
                            "hidden_size": hidden_size,
                            "ownership": name,
                            "assignments": count,
                            "pack_bitwise_equal": True,
                            "scatter_bitwise_equal": True,
                            "unowned_slots_and_buffer_tail_unchanged": True,
                            "warm_pack_gpu_ms": pack_ms,
                            "warm_scatter_gpu_ms": scatter_ms,
                        }
                    )

                restored = torch.zeros_like(slots)
                for ownership in partitions:
                    count = int(ownership[ids.long()].sum())
                    packed = send.pack(slots, ids, ownership, num_assignments=count)
                    buffer = receive.receive_buffer(count)
                    buffer.copy_(packed)
                    receive.scatter(
                        buffer, ids, ownership, restored, num_assignments=count
                    )
                assert_bits(restored, slots, "multi-owner original slot restoration")
                actual = torch.empty(
                    (tokens, hidden_size), dtype=torch.bfloat16, device=device
                )
                expected = torch.empty_like(actual)
                if tokens:
                    ops.moe_sum(restored, actual)
                    ops.moe_sum(slots, expected)
                assert_bits(actual, expected, "one native moe_sum after restoration")
                restored_results.append(
                    {
                        "tokens": tokens,
                        "top_k": top_k,
                        "hidden_size": hidden_size,
                        "owners": len(partitions),
                        "restored_slots_bitwise_equal": True,
                        "native_moe_sum_bitwise_equal": True,
                        "native_moe_sum_launched": tokens > 0,
                    }
                )
    streams = check_stream_reuse(receiver[max(HIDDEN_SIZES)])
    rejections = check_rejections(sender[max(HIDDEN_SIZES)])
    timings = {}
    for key in ("warm_pack_gpu_ms", "warm_scatter_gpu_ms"):
        values = [case[key] for case in results if case["assignments"]]
        timings[key] = {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }
    return {
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
        "cases": results,
        "multi_owner_cases": restored_results,
        "stream_reuse": streams,
        "rejection_checks": rejections,
        "warm_helper_cuda_event_intervals_ms": timings,
        "maximum_flattened_slots": maximum,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path)
    args = parser.parse_args()
    if args.gpu < 0 or args.output.exists():
        parser.error("Select a nonnegative physical GPU and a new private output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = (
        args.lock_path
        or Path(tempfile.gettempdir()) / f"expert-pool-compact-{args.gpu}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "physical_gpu": args.gpu,
        "scope": (
            "Real GPU helper correctness without model weights, NCCL, "
            "or an online service"
        ),
        "timing_limitations": (
            "CUDA event intervals follow warmup and explicit validation "
            "synchronization; "
            "they include helper kernels and launch gaps, not a throughput benefit"
        ),
        "input_kind": (
            "Seeded synthetic BF16 values with exact per-slot tags and random routing"
        ),
    }
    try:
        source_root = Path(__file__).resolve().parents[2]
        if Path(afd_plugin.__file__).resolve().parent != source_root / "afd_plugin":
            raise RuntimeError("Python imported a different AFD source tree")
        manifest = source_root / "source_manifest.json"
        if manifest.exists():
            report["source_manifest"] = json.loads(manifest.read_text())
            for name, expected in report["source_manifest"]["files"].items():
                if (
                    hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise RuntimeError("Synced source changed since its manifest")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report["initial_gpu_state"] = check_idle([args.gpu])
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            os.environ["VLLM_PLUGINS"] = ""
            report.update(validate())
            report["status"] = "passed"
    except BusyGPUError as error:
        report["status"] = "blocked_preflight"
        report["error"] = str(error)
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(report.get("cases", [])),
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    return (
        0
        if report["status"] == "passed"
        else 2
        if report["status"] == "blocked_preflight"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
