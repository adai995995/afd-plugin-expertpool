# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Offline residency and exact ordinal coverage checks for audited E copies.

Counts alone cannot distinguish disjoint slices from duplicated Token slots.
Group sibling E executions by their original A call, then check that each
logical Expert's ranges partition [0, Router demand) exactly once.
"""

from collections import Counter, defaultdict


def verify_replica_assignments(
    requests: dict, records: list[dict], workers: list[dict]
) -> dict:
    resident = {w["worker_id"]: w["resident_experts"] for w in workers}
    if len(resident) != len(workers):
        raise AssertionError("Duplicate physical worker shutdown reports")
    parents = defaultdict(list)
    for record in records:
        for member in record["members"]:
            key = (member["client_id"], member["session_epoch"], member["call_seq"])
            parents[key].append((record, member))
    live_split_calls = live_split_assignments = live_whole_replica_calls = 0
    whole_locations = defaultdict(set)
    all_locations = defaultdict(set)
    for key, children in parents.items():
        first_record, first = children[0]
        layer = str(first_record["layer_id"])
        demand = first["expert_demand"]
        if (
            any(
                type(first[k]) is not int or first[k] <= 0
                for k in ("num_tokens", "top_k")
            )
            or not demand
            or any(
                not e.isdecimal() or str(int(e)) != e or type(n) is not int or n <= 0
                for e, n in demand.items()
            )
            or sum(demand.values()) != first["num_tokens"] * first["top_k"]
        ):
            raise AssertionError("Invalid original A call demand in audit")
        windows = [
            case["windows"][key[0]]
            for case in requests["cases"]
            if key[0] in case["windows"]
        ]
        live = any(w["start"] <= key[2] < w["end"] for w in windows)
        by_expert = defaultdict(list)
        seen_workers = set()
        for record, member in children:
            worker = record["worker_id"]
            if (
                worker in seen_workers
                or worker not in resident
                or any(
                    record[k] != first_record[k]
                    for k in ("model_id", "placement_version", "layer_id")
                )
                or any(
                    member[k] != first[k]
                    for k in ("num_tokens", "top_k", "expert_demand")
                )
            ):
                raise AssertionError(
                    "Sibling E copies disagree on their original A call"
                )
            seen_workers.add(worker)
            ranges = member["assignment_ranges"]
            counts = member["expert_assignments"]
            if (
                not counts
                or set(ranges) != set(counts)
                or not set(counts) <= set(demand)
            ):
                raise AssertionError(
                    "Slice metadata differs from admitted Expert counts"
                )
            for expert, count in counts.items():
                item = ranges[expert]
                if int(expert) not in resident[worker].get(layer, []):
                    raise AssertionError("Execution selected a nonresident Expert copy")
                if (
                    type(count) is not int
                    or count <= 0
                    or set(item) != {"start", "count"}
                    or type(item["start"]) is not int
                    or item["start"] < 0
                    or type(item["count"]) is not int
                    or item["count"] != count
                    or item["start"] + count > demand[expert]
                ):
                    raise AssertionError("Invalid physical Expert assignment range")
                by_expert[expert].append((item["start"], count, worker))
        if set(by_expert) != set(demand):
            raise AssertionError("An original A call lost a logical Expert")
        for expert, ranges in by_expert.items():
            end = 0
            for start, count, _ in sorted(ranges):
                if start != end:
                    raise AssertionError(
                        "Expert assignment slices overlap or leave a gap"
                    )
                end += count
            if end != demand[expert]:
                raise AssertionError(
                    "Expert assignment slices do not cover Router demand"
                )
            if not live:
                continue
            logical = (layer, expert)
            all_locations[logical].update(worker for _, _, worker in ranges)
            if len(ranges) > 1:
                live_split_calls += 1
                live_split_assignments += end
            elif (
                sum(
                    int(expert) in layers.get(layer, []) for layers in resident.values()
                )
                > 1
            ):
                live_whole_replica_calls += 1
                whole_locations[logical].add(ranges[0][2])
    split_coalesced_groups = 0
    for record in records:
        sources = Counter()
        for member in record["members"]:
            client, seq = member["client_id"], member["call_seq"]
            if not any(
                case["windows"][client]["start"] <= seq < case["windows"][client]["end"]
                for case in requests["cases"]
                if client in case["windows"]
            ):
                continue
            sources.update(
                expert
                for expert, n in member["expert_assignments"].items()
                if n < member["expert_demand"][expert]
            )
        split_coalesced_groups += sum(n > 1 for n in sources.values())
    return {
        "live_whole_replica_expert_calls": live_whole_replica_calls,
        "live_whole_experts_using_multiple_replicas": sum(
            len(locations) > 1 for locations in whole_locations.values()
        ),
        "live_experts_using_multiple_replicas": sum(
            len(locations) > 1 for locations in all_locations.values()
        ),
        "live_split_expert_calls": live_split_calls,
        "live_split_assignments": live_split_assignments,
        "live_split_coalesced_expert_groups": split_coalesced_groups,
        "assignment_ranges_verified": True,
        "resident_locations_verified": True,
    }
