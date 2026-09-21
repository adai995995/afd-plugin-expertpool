# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Deterministic offered traffic and request-level SLO accounting, without GPUs."""

import math
import random


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("A percentile requires samples and a fraction in [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def domain_replicas(replica_count: int, repeat: int) -> tuple[tuple[int, ...], ...]:
    if type(replica_count) is not int or replica_count < 2:
        raise ValueError(
            "Two independent service domains require at least two replicas"
        )
    first = replica_count // 2 + int(replica_count % 2 and repeat % 2 == 0)
    return tuple(range(first)), tuple(range(first, replica_count))


def assignments(
    mode: str,
    concurrency: int,
    count: int,
    repeat: int,
    replica_count: int | None = None,
) -> list[dict]:
    """Closed-loop domain budget, with static round-robin routing within a domain."""
    if concurrency < 1 or count < concurrency or mode not in {"pool", "native"}:
        raise ValueError(
            "Need a known mode and enough requests for positive concurrency"
        )
    domains = domain_replicas(replica_count or (2 if mode == "pool" else 3), repeat)
    plans = []
    for domain, replicas in enumerate(domains):
        active = replicas[: min(len(replicas), concurrency)]
        for position, replica in enumerate(active):
            plans.append(
                {
                    "replica": replica,
                    "domain": domain,
                    "concurrency": concurrency // len(active)
                    + int(position < concurrency % len(active)),
                    "indices": list(range(position, count, len(active))),
                }
            )
    return plans


def arrival_offsets(
    count: int, domain_rps: float, pattern: str, seed: int
) -> list[int]:
    """Offsets in ns; finite open-loop trace independent of engine completion.

    Bursts contain four simultaneous arrivals, spaced to preserve target mean
    rate. Poisson rate is an expectation; observed finite trace rate is reported.
    """
    if count < 1 or not math.isfinite(domain_rps) or domain_rps <= 0:
        raise ValueError("Arrival count and rate must be positive")
    if pattern not in {"steady", "poisson", "burst"}:
        raise ValueError("Unknown arrival process")
    rng = random.Random(seed)
    now = 0.0
    offsets = []
    for index in range(count):
        if pattern == "poisson":
            now += rng.expovariate(domain_rps)
        elif pattern == "burst":
            now = (index // 4) * 4 / domain_rps
        else:
            now = index / domain_rps
        offsets.append(round(now * 1e9))
    return offsets


def mixed_lengths(count: int, lengths: list[int], seed: int) -> list[int]:
    """Balance the domain mix, then shuffle independently of replica routing.

    Alternating short/long by index would send every long request to the same
    native replica under two-way round-robin, manufacturing load imbalance.
    """
    if count < 1 or not lengths or any(n <= 0 for n in lengths):
        raise ValueError("Input mix requires positive lengths and request count")
    result = [lengths[i % len(lengths)] for i in range(count)]
    random.Random(seed).shuffle(result)
    return result


def open_loop_assignments(replica_count: int, count: int, repeat: int) -> list[dict]:
    # No concurrency gate: arrival tasks must not be refilled by completions.
    return [
        {
            "replica": replica,
            "domain": domain,
            "concurrency": 0,
            "indices": list(range(position, count, len(replicas))),
        }
        for domain, replicas in enumerate(domain_replicas(replica_count, repeat))
        for position, replica in enumerate(replicas)
    ]


def summarize(
    records: list[dict],
    *,
    window: tuple[int, int] | None = None,
    slos: dict[int, tuple[float, float]] | None = None,
) -> dict:
    if not records:
        raise ValueError("No offered requests")
    if slos is not None:
        for record in records:
            limits = slos[record["domain"]]
            if any(not math.isfinite(v) or v <= 0 for v in limits):
                raise ValueError("SLO limits must be finite and positive")
    if window is None:
        window = (
            min(r["started_ns"] for r in records),
            max(r["finished_ns"] for r in records),
        )
    if any(
        r["started_ns"] < window[0] or r["finished_ns"] > window[1] for r in records
    ):
        raise ValueError("Common observation window must contain every request")
    window_s = (window[1] - window[0]) / 1e9
    if window_s <= 0:
        raise ValueError("Nonpositive observation window")
    complete = [r for r in records if r.get("status", "completed") == "completed"]
    summary = {
        "requests": len(records),
        "completed": len(complete),
        "failed": len(records) - len(complete),
        "window_s": window_s,
        "offered_requests_per_observation_s": len(records) / window_s,
        "request_throughput_rps": len(complete) / window_s,
        "output_throughput_tps": sum(r["output_tokens"] for r in complete) / window_s,
    }
    for name in ("ttft_ms", "tpot_ms", "e2e_ms", "submission_lag_ms"):
        values = [r[name] for r in complete if name in r]
        summary[name] = (
            {
                "mean": sum(values) / len(values),
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
            }
            if values
            else None
        )
    if slos is not None:
        good = sum(
            r["ttft_ms"] <= slos[r["domain"]][0]
            and r["tpot_ms"] <= slos[r["domain"]][1]
            for r in complete
        )
        summary["slo"] = {
            "good_requests": good,
            "misses": len(records) - good,
            "attainment": good / len(records),
            "goodput_rps": good / window_s,
        }
    return summary


def verify_cost_accounting(controller: dict, workers: list[dict]) -> dict:
    """Reconcile one observation per physical execution after complete drain."""
    if controller["active_cost_tickets"]:
        raise AssertionError("Execution cost tickets did not drain")
    checks = {}
    for worker in workers:
        worker_id = worker["worker_id"]
        totals = controller["execution_cost_by_worker"][worker_id]["totals"]
        actual = worker["pipeline"]["batching"]
        for key, expected in (
            ("executions", actual["executions"]),
            ("child_calls", actual["calls"]),
            ("token_rows", actual["token_rows"]),
        ):
            if totals[key] != expected:
                raise AssertionError(
                    f"Execution cost accounting mismatch: {worker_id}/{key}"
                )
        for name, key in (
            ("compute_phase", "compute_phase_gpu_ms_total"),
            ("input_merge", "input_merge_phase_gpu_ms_total"),
        ):
            if not math.isclose(
                totals["sum_ms"].get(name, 0), actual[key], rel_tol=1e-6, abs_tol=1e-5
            ):
                raise AssertionError("Controller and worker phase costs disagree")
        checks[worker_id] = {
            key: totals[key] for key in ("executions", "child_calls", "token_rows")
        }
    return checks
