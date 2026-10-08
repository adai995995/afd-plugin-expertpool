#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Open-loop, two-domain HTTP benchmark for independent A services.

Run the driver near the A services. All arrival and token timestamps come from
one monotonic clock; no remote host clocks or private deployment data are
needed in the saved report.
"""

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen

from tools.expert_pool.benchmark_workload import (
    arrival_offsets,
    mixed_lengths,
    summarize,
)

PROMPT_STEMS = (
    "Explain how rivers affect a city. ",
    "Describe why forests support wildlife. ",
    "Compare rail and road transport. ",
    "Explain how weather changes farming. ",
    "Describe how people save energy. ",
    "Compare coastal and mountain climates. ",
    "Explain how public libraries help learning. ",
    "Describe the role of ports in trade. ",
)
PROMPT_FILL = "Give a concrete example and explain the reasoning in plain language. "
START_DELAY_NS = 1_000_000_000


def pool_control(endpoint: str, action: str, timeout_s: float) -> dict:
    """Use HTTP only outside the timed request window; omit URLs from errors."""
    if action == "status":
        request = Request(endpoint + "/pool/status", method="GET")
    else:
        request = Request(
            endpoint + "/pool/metrics",
            data=json.dumps({"enabled": action == "enable"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            if response.status != 200:
                raise RuntimeError("Unexpected HTTP status")
            result = json.load(response)
    except Exception as exception:
        raise RuntimeError(
            f"Pool metrics {action} failed ({type(exception).__name__})"
        ) from None
    if not isinstance(result, dict):
        raise RuntimeError(f"Pool metrics {action} returned an invalid response")
    return result


def compact_pool_status(status: dict) -> list[dict]:
    """Keep timing and small dispatch counters, not full deployment status."""
    workers = status.get("workers")
    if status.get("native_reference") or not isinstance(workers, list) or not workers:
        raise RuntimeError("Pool metrics require a pooled A service")
    result = []
    for worker_index, worker in enumerate(workers):
        if not isinstance(worker, dict) or not isinstance(
            worker.get("call_metrics"), dict
        ):
            raise RuntimeError("Pooled A worker metrics were not enabled")
        dispatch = worker.get("dispatch")
        if not isinstance(dispatch, dict):
            raise RuntimeError("Pooled A worker dispatch status is unavailable")
        dispatch_workers = dispatch.get("workers", {})
        if not isinstance(dispatch_workers, dict):
            raise RuntimeError("Pooled A dispatch worker status is invalid")
        small_dispatch = {
            name: dispatch[name]
            for name in (
                "policy",
                "direct_dispatch",
                "packed_input",
                "input_transfer",
                "output_transfer",
                "packed_tasks",
                "dense_tasks",
                "controller_roundtrips",
            )
            if name in dispatch
        }
        small_dispatch["workers"] = [
            {
                "worker_index": index,
                "calls": details.get("calls"),
                "call_metrics": details.get("call_metrics"),
                **(
                    {"output_transfer": details["output_transfer"]}
                    if "output_transfer" in details
                    else {}
                ),
            }
            for index, (_, details) in enumerate(sorted(dispatch_workers.items()))
        ]
        small_worker = {
            "worker_index": worker_index,
            "call_metrics": worker["call_metrics"],
            "dispatch": small_dispatch,
        }
        layers = worker.get("layers", {})
        if not isinstance(layers, dict):
            raise RuntimeError("Pooled A router layer status is invalid")
        small_worker["router_layers"] = [
            {
                "layer_id": layer_id,
                "router_metrics": details["router_metrics"],
                "shared_expert_overlap": details.get("shared_expert_overlap"),
                "shared_metrics": details.get("shared_metrics"),
                "shared_timing_pending": details.get("shared_timing_pending"),
                "shared_timing_dropped": details.get("shared_timing_dropped"),
                **(
                    {"router_gpu_unready": details["router_gpu_unready"]}
                    if "router_gpu_unready" in details
                    else {}
                ),
            }
            for layer_id, details in sorted(
                layers.items(), key=lambda item: int(item[0])
            )
            if isinstance(details, dict) and "router_metrics" in details
        ]
        result.append(small_worker)
    return result


def fetch(
    endpoint: str,
    prompt: str,
    output_tokens: int,
    timeout_s: float,
    *,
    request_id: str,
    domain: int,
    replica: int,
    planned_ns: int,
    length_class: int,
    capture_output_tokens: bool = False,
    capture_logprobs: bool = False,
) -> dict:
    submitted_ns = time.perf_counter_ns()
    tokens: list[int] = []
    probabilities: list[list[dict]] = []
    chunks: list[tuple[int, int]] = []
    finished = False
    status = "completed"
    error = None
    deadline_ns = planned_ns + round(timeout_s * 1e9)
    try:
        body = {"prompt": prompt, "max_tokens": output_tokens}
        if capture_logprobs:
            body["return_logprobs"] = True
        payload = json.dumps(body).encode()
        request = Request(
            endpoint + "/generate-stream",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout_s) as response:
            if response.status != 200:
                raise RuntimeError("HTTP generation failed")
            for line in response:
                received_ns = time.perf_counter_ns()
                if received_ns > deadline_ns:
                    raise TimeoutError("Request exceeded its total deadline")
                part = json.loads(line)
                delta = part["token_ids"]
                if delta:
                    chunks.append((received_ns, len(delta)))
                    tokens.extend(delta)
                    if capture_logprobs:
                        steps = part["top_logprobs"]
                        if len(steps) != len(delta):
                            raise RuntimeError("Missing diagnostic token probabilities")
                        probabilities.extend(steps)
                if part["finished"]:
                    finished = True
                    break
        if not finished or len(tokens) != output_tokens:
            status = "incomplete"
        elif any(count != 1 for _, count in chunks):
            # One arrival timestamp cannot recover the spacing of several tokens.
            status = "unmeasurable_tpot"
            error = "multi_token_stream_chunk"
    except Exception as exception:
        status = "failed"
        error = type(exception).__name__
    finished_ns = time.perf_counter_ns()
    record = {
        "request_id": request_id,
        "domain": domain,
        "replica": replica,
        "length_class": length_class,
        "input_chars": len(prompt),
        "output_tokens": len(tokens),
        "status": status,
        "error": error,
        "started_ns": planned_ns,
        "submitted_ns": submitted_ns,
        "finished_ns": finished_ns,
        "submission_lag_ms": (submitted_ns - planned_ns) / 1e6,
        "e2e_ms": (finished_ns - planned_ns) / 1e6,
        "stream_chunks": [(at - planned_ns, count) for at, count in chunks],
        "stream_chunk_count": len(chunks),
        "max_chunk_tokens": max((count for _, count in chunks), default=0),
    }
    if status == "completed":
        record.update(
            ttft_ms=(chunks[0][0] - planned_ns) / 1e6,
            tpot_ms=(chunks[-1][0] - chunks[0][0]) / 1e6 / (len(tokens) - 1),
            output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        )
    if capture_output_tokens or capture_logprobs:
        # Retain partial sequences too. Hashes alone cannot locate a first
        # divergence. These opt-in request artifacts must remain private.
        record["token_ids"] = tokens
    if capture_logprobs:
        record["top_logprobs"] = probabilities
    return record


async def run(args: argparse.Namespace, services: dict[int, list[str]]) -> dict:
    for domain, endpoints in services.items():
        for replica, endpoint in enumerate(endpoints):
            for warmup in range(args.warmup):
                for shape, repeats in enumerate(
                    (args.short_repeats, args.long_repeats)
                ):
                    prompt = (
                        PROMPT_STEMS[
                            (domain + replica + warmup + shape) % len(PROMPT_STEMS)
                        ]
                        + PROMPT_FILL * repeats
                    )
                    record = await asyncio.to_thread(
                        fetch,
                        endpoint,
                        prompt,
                        args.output_tokens,
                        args.timeout,
                        request_id=f"warmup-{domain}-{replica}-{warmup}-{shape}",
                        domain=domain,
                        replica=replica,
                        planned_ns=time.perf_counter_ns(),
                        length_class=repeats,
                    )
                    if record["status"] != "completed":
                        raise RuntimeError("Service warmup failed")

    collect_metrics = getattr(args, "collect_pool_metrics", False)
    if not collect_metrics:
        # Keep the original scheduling anchor in the default benchmark mode.
        start_ns = time.perf_counter_ns() + START_DELAY_NS
    trace = []
    for domain, endpoints in services.items():
        offsets = arrival_offsets(
            args.count_per_domain, args.rps / 2, args.arrival, args.seed + domain
        )
        lengths = mixed_lengths(
            args.count_per_domain,
            [args.short_repeats, args.long_repeats],
            args.seed + domain + 17,
        )
        for index, (offset, repeats) in enumerate(zip(offsets, lengths, strict=True)):
            replica = index % len(endpoints)
            prompt = (
                PROMPT_STEMS[(index + domain * 3) % len(PROMPT_STEMS)]
                + PROMPT_FILL * repeats
            )
            trace.append(
                {
                    "domain": domain,
                    "index": index,
                    "replica": replica,
                    "offset_ns": offset,
                    "length_class": repeats,
                    "prompt": prompt,
                    "endpoint": endpoints[replica],
                }
            )

    async def scheduled(item: dict) -> dict:
        planned_ns = start_ns + item["offset_ns"]
        await asyncio.sleep(max(0, (planned_ns - time.perf_counter_ns()) / 1e9))
        return await asyncio.to_thread(
            fetch,
            item["endpoint"],
            item["prompt"],
            args.output_tokens,
            args.timeout,
            request_id=f"d{item['domain']}-{item['index']}",
            domain=item["domain"],
            replica=item["replica"],
            planned_ns=planned_ns,
            length_class=item["length_class"],
            capture_output_tokens=getattr(args, "capture_output_tokens", False),
            capture_logprobs=getattr(args, "capture_logprobs", False),
        )

    service_keys = [
        (domain, replica, endpoint)
        for domain, endpoints in services.items()
        for replica, endpoint in enumerate(endpoints)
    ]
    enabled_endpoints: list[str] = []
    metrics_before: list[list[dict]] = []
    metrics_after: list[list[dict]] = []
    try:
        if collect_metrics:
            # Enable once per unique A service, after both warmup shapes have
            # completed. Status reads are outside the timed request window.
            for endpoint in dict.fromkeys(endpoint for _, _, endpoint in service_keys):
                await asyncio.to_thread(pool_control, endpoint, "enable", args.timeout)
                enabled_endpoints.append(endpoint)
            for _, _, endpoint in service_keys:
                status = await asyncio.to_thread(
                    pool_control, endpoint, "status", args.timeout
                )
                metrics_before.append(compact_pool_status(status))

        if collect_metrics:
            start_ns = time.perf_counter_ns() + START_DELAY_NS
        records = await asyncio.gather(*(scheduled(item) for item in trace))

        if collect_metrics:
            for _, _, endpoint in service_keys:
                status = await asyncio.to_thread(
                    pool_control, endpoint, "status", args.timeout
                )
                metrics_after.append(compact_pool_status(status))
    finally:
        # Always turn off the optional critical-path instrumentation, even if
        # a generation request or status read fails.
        disable_errors = await asyncio.gather(
            *(
                asyncio.to_thread(pool_control, endpoint, "disable", args.timeout)
                for endpoint in reversed(enabled_endpoints)
            ),
            return_exceptions=True,
        )
        if any(isinstance(error, BaseException) for error in disable_errors):
            raise RuntimeError("Could not disable Pool metrics on every service")

    window = (start_ns, max(record["finished_ns"] for record in records))
    slos = {
        domain: (args.ttft_slo_ms[domain], args.tpot_slo_ms[domain])
        for domain in services
    }
    trace_identity = [
        (item["domain"], item["index"], item["offset_ns"], item["prompt"])
        for item in trace
    ]
    result = {
        "kind": "expert-pool-http-open-loop-v1",
        "trace_sha256": hashlib.sha256(
            json.dumps(trace_identity, separators=(",", ":")).encode()
        ).hexdigest(),
        "configuration": {
            "total_rps": args.rps,
            "arrival": args.arrival,
            "count_per_domain": args.count_per_domain,
            "output_tokens": args.output_tokens,
            "seed": args.seed,
            "short_repeats": args.short_repeats,
            "long_repeats": args.long_repeats,
            "replicas_per_domain": {str(d): len(s) for d, s in services.items()},
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
        },
        "summary": summarize(records, window=window, slos=slos),
        "by_domain": {
            str(domain): summarize(
                [record for record in records if record["domain"] == domain],
                window=window,
                slos=slos,
            )
            for domain in services
        },
        "records": records,
    }
    if collect_metrics:
        result["pool_metrics"] = {
            "services": [
                {
                    "domain": domain,
                    "replica": replica,
                    "before": before,
                    "after": after,
                }
                for (domain, replica, _), before, after in zip(
                    service_keys, metrics_before, metrics_after, strict=True
                )
            ]
        }
        result["configuration"]["collect_pool_metrics"] = True
    if getattr(args, "capture_output_tokens", False) or getattr(
        args, "capture_logprobs", False
    ):
        result["configuration"].update(
            capture_output_tokens=True,
            capture_logprobs=getattr(args, "capture_logprobs", False),
            diagnostic_timing=bool(getattr(args, "capture_logprobs", False)),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", action="append", required=True)
    parser.add_argument("--rps", type=float, required=True)
    parser.add_argument(
        "--arrival", choices=("steady", "burst", "poisson"), default="steady"
    )
    parser.add_argument("--count-per-domain", type=int, default=12)
    parser.add_argument("--output-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--short-repeats", type=int, default=6)
    parser.add_argument("--long-repeats", type=int, default=18)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--capture-output-tokens",
        action="store_true",
        help="Save full generated sequences in the private benchmark report",
    )
    parser.add_argument(
        "--capture-logprobs",
        action="store_true",
        help="Diagnostic only: request top-two logprobs and retain output tokens",
    )
    parser.add_argument(
        "--collect-pool-metrics",
        action="store_true",
        help="Reset and collect pooled A worker timings for only the measured trace",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--ttft-slo-ms", type=float, nargs=2, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not math.isfinite(args.rps)
        or args.rps <= 0
        or args.count_per_domain < 2
        or args.output_tokens < 2
        or args.output_tokens > 128
        or args.warmup < 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or any(
            not math.isfinite(slo) or slo <= 0
            for slo in (*args.ttft_slo_ms, *args.tpot_slo_ms)
        )
    ):
        parser.error("Invalid offered load, output size, deadline or SLO")
    services: dict[int, list[str]] = {0: [], 1: []}
    for item in args.service:
        try:
            domain, endpoint = item.split("=", 1)
            domain_id = int(domain)
            if domain_id not in services or not endpoint.startswith("http://"):
                raise ValueError
            services[domain_id].append(endpoint.rstrip("/"))
        except ValueError:
            parser.error("Services must use 0=http://host:port or 1=http://host:port")
    if not all(services.values()):
        parser.error("Both service domains need at least one endpoint")
    result = asyncio.run(run(args, services))
    args.output.write_text(json.dumps(result, separators=(",", ":")))
    print(
        json.dumps(
            {
                "passed": result["summary"]["failed"] == 0,
                "trace_sha256": result["trace_sha256"],
                "summary": result["summary"],
            }
        )
    )


if __name__ == "__main__":
    main()
