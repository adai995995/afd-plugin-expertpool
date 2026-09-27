#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exercise independent HTTP A services, then verify drained shared batch records.

The supervisor starts/stops processes separately. ``run`` takes call-sequence
windows around actual requests; ``verify`` consumes the controller audit and
E shutdown reports. Warmup cannot count as evidence of cross-client batching.
Native reference services use the same checkpoint and generation settings.
This is a functional check, not a throughput or SLO benchmark.
"""

import argparse
import json
import math
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_LOGPROB_ATOL = 0.05
DEFAULT_OUTPUT_TOKENS = 8
HEALTH_POLL_S = 0.2
STAGGER_S = 0.075


def endpoints(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        client, separator, url = value.partition("=")
        parsed = urlsplit(url)
        if (
            not separator
            or not client
            or client in result
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Service endpoints must be unique CLIENT=http(s)://host:port"
            )
        result[client] = url.rstrip("/")
    return result


def http_json(url: str, timeout_s: float, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout_s) as response:
        return json.load(response)


def wait_ready(services: dict[str, str], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    pending = dict(services)
    while pending:
        for client, url in tuple(pending.items()):
            try:
                health = http_json(url + "/health", min(5, timeout_s))
                if health["ready"] and health["client_id"] == client:
                    del pending[client]
            except (OSError, ValueError, KeyError):
                pass
        if time.monotonic() >= deadline:
            raise TimeoutError("A services did not become ready: " + ",".join(pending))
        if pending:
            time.sleep(HEALTH_POLL_S)


def statuses(services: dict[str, str], timeout_s: float) -> dict[str, dict]:
    result = {}
    for client, url in services.items():
        status = http_json(url + "/pool/status", timeout_s)
        if status["client_id"] != client or len(status["workers"]) != 1:
            raise AssertionError(
                "Validation expects one independent TP=1 A per service"
            )
        worker = status["workers"][0]
        if worker["client_id"] != client or worker["routed_parameter_bytes"] != 0:
            raise AssertionError(
                "Routed Expert weights must reside in the shared E pool"
            )
        result[client] = worker
    return result


def request_window(before: dict, after: dict) -> dict:
    start = before["dispatch"]["next_call_seq"]
    end = after["dispatch"]["next_call_seq"]
    if end < start:
        raise AssertionError("A call sequence went backwards")
    old = before["dispatch"]["demand"]["expert_assignments"]
    new = after["dispatch"]["demand"]["expert_assignments"]
    assignments = {}
    for layer, counts in new.items():
        previous = old.get(layer, [0] * len(counts))
        if len(previous) != len(counts):
            raise AssertionError("Logical Expert layout changed during validation")
        delta = {
            str(e): n - previous[e] for e, n in enumerate(counts) if n != previous[e]
        }
        if any(count < 0 for count in delta.values()):
            raise AssertionError("Expert counters went backwards")
        if delta:
            assignments[layer] = delta
    return {"start": start, "end": end, "expert_assignments": assignments}


def compare_output(candidate: dict, reference: dict, atol: float) -> dict:
    if candidate["client_id"] != reference["client_id"]:
        raise AssertionError("Reference output belongs to a different A")
    same_tokens = candidate["token_ids"] == reference["token_ids"]
    candidate_probs, reference_probs = (
        candidate["token_logprobs"],
        reference["token_logprobs"],
    )
    if (
        not candidate["token_ids"]
        or len(candidate_probs) != len(candidate["token_ids"])
        or len(reference_probs) != len(reference["token_ids"])
        or any(not math.isfinite(p) for p in (*candidate_probs, *reference_probs))
    ):
        raise AssertionError("Invalid generated-token numerical evidence")
    error = (
        max(abs(a - b) for a, b in zip(candidate_probs, reference_probs, strict=True))
        if same_tokens
        else None
    )
    return {
        "same_tokens": same_tokens,
        "logprob_max_abs": error,
        "passed": same_tokens and error <= atol,
    }


def workload(clients: tuple[str, ...]) -> tuple[tuple[str, dict, dict], ...]:
    prompts = {
        client: (
            "Name the capital and currency of France."
            if i % 2 == 0
            else "Name the capital and currency of Japan."
        )
        for i, client in enumerate(clients)
    }
    mixed = dict(prompts)
    mixed[clients[-1]] = (
        "Read this passage: "
        + "A river carries water from mountains to the sea. " * 12
        + "Describe the passage in one sentence."
    )
    return (
        ("simultaneous", prompts, {}),
        ("mixed_lengths", mixed, {}),
        (
            "staggered",
            prompts,
            {client: i * STAGGER_S for i, client in enumerate(clients)},
        ),
        ("single_first", {clients[0]: prompts[clients[0]]}, {}),
        ("single_last", {clients[-1]: prompts[clients[-1]]}, {}),
    )


def generate(
    services: dict[str, str],
    prompts: dict[str, str],
    delays: dict[str, float],
    output_tokens: int,
    timeout_s: float,
) -> dict[str, dict]:
    # Synchronize HTTP submission only. A engines still advance independently;
    # there is no layer barrier or shared Attention/KV scheduler.
    launch = threading.Barrier(len(prompts))

    def call(client: str) -> dict:
        launch.wait(timeout=timeout_s)
        time.sleep(delays.get(client, 0))
        response = http_json(
            services[client] + "/generate",
            timeout_s,
            {
                "prompt": prompts[client],
                "max_tokens": output_tokens,
                "return_logprobs": True,
            },
        )
        if (
            response["client_id"] != client
            or len(response["token_ids"]) != output_tokens
        ):
            raise AssertionError("Response identity or generation length is wrong")
        return response

    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = {client: executor.submit(call, client) for client in prompts}
        return {client: future.result() for client, future in futures.items()}


def run_services(
    services: dict[str, str],
    references: dict[str, str],
    *,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    timeout_s: float = 600,
    logprob_atol: float = DEFAULT_LOGPROB_ATOL,
) -> dict:
    if len(services) < 2 or set(services) != set(references):
        raise ValueError(
            "At least two A services and matching native references are required"
        )
    if len(set(services.values())) != len(services) or set(services.values()) & set(
        references.values()
    ):
        raise ValueError("A services and native references must use distinct endpoints")
    wait_ready(services, timeout_s)
    wait_ready(references, timeout_s)
    for mapping, native in ((services, False), (references, True)):
        for url in mapping.values():
            if http_json(url + "/health", timeout_s)["native_reference"] != native:
                raise AssertionError(
                    "Pool and native reference service roles are wrong"
                )
    # Native engine initialization does not cover every live route-mask shape.
    # Warm the declared request shapes before taking sequence-window baselines.
    warmup_requests = 0
    for _, prompts, _ in workload(tuple(services))[:2]:
        for client, prompt in prompts.items():
            for mapping in (services, references):
                generate(mapping, {client: prompt}, {}, output_tokens, timeout_s)
            warmup_requests += 1
    initial = statuses(services, timeout_s)
    cases = []
    for name, prompts, delays in workload(tuple(services)):
        before = statuses(services, timeout_s)
        responses = generate(services, prompts, delays, output_tokens, timeout_s)
        after = statuses(services, timeout_s)
        native = generate(references, prompts, delays, output_tokens, timeout_s)
        comparisons = {
            client: compare_output(responses[client], native[client], logprob_atol)
            for client in prompts
        }
        cases.append(
            {
                "name": name,
                "active_clients": list(prompts),
                "responses": responses,
                "native_responses": native,
                "comparisons": comparisons,
                "windows": {
                    client: request_window(before[client], after[client])
                    for client in services
                },
            }
        )
        print(json.dumps({"case": name, "comparisons": comparisons}), flush=True)
    return {
        "kind": "shared-pool-functional-requests",
        "logprob_atol": logprob_atol,
        "warmup_pool_requests": warmup_requests,
        "initial_status": initial,
        "cases": cases,
        "numerical_passed": all(
            c["passed"] for case in cases for c in case["comparisons"].values()
        ),
    }


def verify_records(
    requests: dict, audit: dict, controller: dict, workers: list[dict]
) -> dict:
    if not requests["numerical_passed"] or not audit["enabled"]:
        raise AssertionError("Numerical comparisons and execution audit are required")
    if audit["dropped_batches"] or audit["pending_recorded_members"]:
        raise AssertionError("Truncated or incomplete audit cannot prove live batching")
    records = audit["records"]
    if (
        len(records) != audit["observed_batches"]
        or len(records) != audit["recorded_batches"]
    ):
        raise AssertionError("Audit batch totals disagree")
    clients = set(requests["initial_status"])
    if (
        controller["pending"]
        or controller["outstanding"]
        or controller["closed_clients"] != len(clients)
    ):
        raise AssertionError("Shared Pool did not drain")
    if len({(r["model_id"], r["placement_version"]) for r in records}) != 1:
        raise AssertionError("Audit mixes checkpoint identities or placement versions")
    actual: Counter = Counter()
    expected: Counter = Counter()
    covered: dict[tuple, set] = {}
    for index, case in enumerate(requests["cases"]):
        for client, window in case["windows"].items():
            covered[index, client] = set()
            if client in case["active_clients"] and window["start"] == window["end"]:
                raise AssertionError("Active A produced no MoE calls")
            if (
                client not in case["active_clients"]
                and window["start"] != window["end"]
            ):
                raise AssertionError("An idle A unexpectedly executed work")
            for layer, counts in window["expert_assignments"].items():
                expected.update(
                    {(index, client, str(layer), str(e)): n for e, n in counts.items()}
                )
    batches: Counter = Counter()
    rows: Counter = Counter()
    calls: Counter = Counter()
    histograms: dict[str, Counter] = {}
    sequences: dict[str, list[int]] = {}
    plan_ids = set()
    worker_calls = set()
    live_calls_by_worker: dict[str, Counter] = {}
    live_batches = cross_batches = coalesced_groups = singleton_batches = 0
    for record in records:
        worker = record["worker_id"]
        members = record["members"]
        if len({m["client_id"] for m in members}) != len(members):
            raise AssertionError("One execution repeats an A member")
        totals: Counter = Counter()
        live = []
        for member in members:
            key = (worker, member["client_id"], member["call_seq"])
            if (
                not member["completed"]
                or member["plan_id"] in plan_ids
                or key in worker_calls
            ):
                raise AssertionError("Incomplete or duplicate execution member")
            plan_ids.add(member["plan_id"])
            worker_calls.add(key)
            totals.update(member["expert_assignments"])
            client = member["client_id"]
            if client not in clients:
                raise AssertionError("Execution belongs to an undeclared A")
            matches = []
            for index, case in enumerate(requests["cases"]):
                window = case["windows"][client]
                if window["start"] <= member["call_seq"] < window["end"]:
                    matches.append(index)
            if len(matches) > 1:
                raise AssertionError("Request call windows overlap")
            if matches:
                index = matches[0]
                covered[index, client].add(member["call_seq"])
                actual.update(
                    {
                        (index, client, str(record["layer_id"]), str(e)): n
                        for e, n in member["expert_assignments"].items()
                    }
                )
                live.append(member)
                live_calls_by_worker.setdefault(worker, Counter())[client] += 1
        if dict(totals) != record["expert_assignments"]:
            raise AssertionError("Physical Expert batch differs from its members")
        if record["token_rows"] != sum(m["num_tokens"] for m in members):
            raise AssertionError("Execution row count disagrees with membership")
        batches[worker] += 1
        calls[worker] += len(members)
        rows[worker] += record["token_rows"]
        histograms.setdefault(worker, Counter())[str(len(members))] += 1
        sequences.setdefault(worker, []).append(record["batch_sequence"])
        if live:
            live_batches += 1
            singleton_batches += len(members) == 1
            if len(live) > 1:
                cross_batches += 1
                expert_sources: Counter = Counter()
                for member in live:
                    expert_sources.update(member["expert_assignments"].keys())
                coalesced_groups += sum(n > 1 for n in expert_sources.values())
    if actual != expected:
        raise AssertionError(
            "Live executed assignments disagree with A Router demand deltas"
        )
    for index, case in enumerate(requests["cases"]):
        for client, window in case["windows"].items():
            if covered[index, client] != set(range(window["start"], window["end"])):
                raise AssertionError(
                    "A live MoE call is missing from execution records"
                )
    if not cross_batches or not coalesced_groups:
        raise AssertionError("No live same-Expert cross-A batch was observed")
    if set(batches) != {worker["worker_id"] for worker in workers}:
        raise AssertionError("Worker shutdown evidence is incomplete")
    for report in workers:
        worker = report["worker_id"]
        stats = report["pipeline"]["batching"]
        if (
            sequences[worker] != list(range(1, batches[worker] + 1))
            or stats["executions"] != batches[worker]
            or stats["calls"] != calls[worker]
            or stats["token_rows"] != rows[worker]
            or {k: v for k, v in stats["calls_per_execution"].items() if v}
            != dict(histograms[worker])
            or report["completed_calls"] != calls[worker]
            or report["pipeline"]["active_slots"] != 0
            or controller["compute_batches_by_worker"][worker] != batches[worker]
        ):
            raise AssertionError(
                "Controller audit and actual E execution totals disagree"
            )
        if not all(
            live_calls_by_worker.get(worker, {}).get(client, 0) for client in clients
        ):
            raise AssertionError("Each E must demonstrate service to both A clients")
    return {
        "passed": True,
        "live_execution_batches": live_batches,
        "live_cross_client_batches": cross_batches,
        "live_coalesced_expert_groups": coalesced_groups,
        "live_singleton_batches": singleton_batches,
        "live_expert_assignments": sum(actual.values()),
        "live_calls_by_worker_client": live_calls_by_worker,
        "numerical_passed": True,
        "drained": True,
    }


def write_report(path: Path, report: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(report, output, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--service", action="append", required=True)
    run.add_argument("--reference-service", action="append", required=True)
    run.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--logprob-atol", type=float, default=DEFAULT_LOGPROB_ATOL)
    run.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--requests", type=Path, required=True)
    verify.add_argument("--batch-audit", type=Path, required=True)
    verify.add_argument("--controller", type=Path, required=True)
    verify.add_argument("--worker", type=Path, action="append", required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("--output must be a new absolute private path")
    if args.command == "run":
        if (
            not 1 <= args.output_tokens <= 128
            or not math.isfinite(args.timeout)
            or args.timeout <= 0
            or not math.isfinite(args.logprob_atol)
            or args.logprob_atol < 0
        ):
            parser.error("Invalid generation length, timeout or numerical tolerance")
        report = run_services(
            endpoints(args.service),
            endpoints(args.reference_service),
            output_tokens=args.output_tokens,
            timeout_s=args.timeout,
            logprob_atol=args.logprob_atol,
        )
    else:
        report = verify_records(
            json.loads(args.requests.read_text()),
            json.loads(args.batch_audit.read_text()),
            json.loads(args.controller.read_text()),
            [json.loads(path.read_text()) for path in args.worker],
        )
    write_report(args.output, report)
    passed = report.get("passed", report.get("numerical_passed", False))
    print(json.dumps({"passed": passed, "output": str(args.output)}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
