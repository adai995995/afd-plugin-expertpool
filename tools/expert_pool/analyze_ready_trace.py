#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Summarize E-local readiness and physical execution costs from live audit.

Nearest same-layer arrivals describe a measured opportunity, not a guarantee
that a peer would be available under another policy. Changing waits changes
subsequent A arrivals. No request-level SLO or compute-bound claim is inferred.
"""

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path

from tools.expert_pool.benchmark_workload import percentile


def analyze(audit: dict, windows: dict[str, tuple[int, int]]) -> dict:
    if audit["dropped_batches"] or audit["pending_recorded_members"]:
        raise ValueError("Readiness analysis requires a complete drained capture")
    selected = []
    arrivals: dict[tuple[int, str], list[int]] = defaultdict(list)
    costs = {"compute_phase_cost_gpu_ms": 0.0, "input_merge_phase_cost_gpu_ms": 0.0}
    executions, cross_a = 0, 0
    for batch in audit["records"]:
        members = batch["members"]
        if not all(
            m["client_id"] in windows
            and windows[m["client_id"]][0] <= m["call_seq"] < windows[m["client_id"]][1]
            for m in members
        ):
            continue
        if not all(
            m["completed"] and "input_ready_ns" in m and "metrics" in m for m in members
        ):
            raise ValueError(
                "Audit lacks live readiness timestamps or completion metrics"
            )
        executions += 1
        cross_a += len({m["client_id"] for m in members}) > 1
        for member in members:
            selected.append((batch["layer_id"], member))
            arrivals[batch["layer_id"], member["client_id"]].append(
                member["input_ready_ns"]
            )
            for name in costs:
                costs[name] += member["metrics"][name]
    if not selected:
        raise ValueError("Declared live windows contain no complete executions")
    for times in arrivals.values():
        times.sort()
    gaps, queues = [], []
    for layer, member in selected:
        ready = member["input_ready_ns"]
        queues.append((member["compute_submitted_ns"] - ready) / 1000)
        peers = []
        for (peer_layer, client), times in arrivals.items():
            if peer_layer != layer or client == member["client_id"]:
                continue
            index = bisect.bisect_left(times, ready)
            peers.extend(
                abs(at - ready) / 1000 for at in times[max(0, index - 1) : index + 1]
            )
        if peers:
            gaps.append(min(peers))

    def summary(values):
        return (
            {
                "samples": len(values),
                "p50_us": percentile(values, 0.5),
                "p95_us": percentile(values, 0.95),
            }
            if values
            else None
        )

    return {
        "scope": (
            "measured E-local readiness; future arrival dependencies not simulated"
        ),
        "calls": len(selected),
        "executions": executions,
        "cross_a_executions": cross_a,
        "cuda_event_interval_costs_ms": costs,
        "cost_scope": (
            "once per execution; CUDA-event intervals include host enqueue gaps"
        ),
        "ready_to_submit": summary(queues),
        "nearest_other_a_same_layer_ready": summary(gaps),
        "calls_with_peer_gap_at_most_us": {
            str(limit): sum(gap <= limit for gap in gaps)
            for limit in (0, 25, 50, 100, 200, 500, 1000)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker_report", type=Path)
    parser.add_argument(
        "--client-window",
        action="append",
        required=True,
        help="client=first_sequence:exclusive_end",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = {}
    for item in args.client_window:
        client, bounds = item.split("=", 1)
        first, end = map(int, bounds.split(":"))
        if client in windows or not 0 <= first < end:
            parser.error("Invalid or duplicated live sequence window")
        windows[client] = first, end
    result = analyze(
        json.loads(args.worker_report.read_text())["pipeline"]["batch_audit"], windows
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"scope", "cost_scope"}}
        )
    )


if __name__ == "__main__":
    main()
