# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""End-of-run reconciliation for direct A/E execution, outside timed requests."""


def verify_direct_pipeline(
    workers: list[dict], clients: list[dict], receive_slots: int
) -> dict:
    client_ids = sorted(item["client_id"] for item in clients)
    if len(set(client_ids)) != len(client_ids) or len(client_ids) != receive_slots:
        raise AssertionError("Direct client coverage differs from slot ownership")
    worker_ids = {item["worker_id"] for item in workers}
    if not worker_ids or len(worker_ids) != len(workers):
        raise AssertionError("Direct worker reports are missing or duplicated")
    total_calls = 0
    peak = 0
    for client in clients:
        dispatch = client["dispatch"]
        if (
            dispatch.get("direct_dispatch") is not True
            or dispatch["pending_feedback"] != 0
            or dispatch["controller_roundtrips"] != 0
            or set(dispatch["workers"]) != worker_ids
            or dispatch["feedback_messages"]
            != sum(worker["calls"] for worker in dispatch["workers"].values())
        ):
            raise AssertionError("Direct client results or accounting did not drain")
    for worker in workers:
        pipeline = worker["pipeline"]
        counts = {
            c["client_id"]: c["dispatch"]["workers"][worker["worker_id"]]["calls"]
            for c in clients
        }
        ordered = [counts[client] for client in client_ids]
        batching = pipeline["batching"]
        histogram = {int(k): v for k, v in batching["calls_per_execution"].items()}
        if (
            pipeline.get("direct_dispatch") is not True
            or not pipeline["enabled"]
            or pipeline["receive_slots"] != receive_slots
            or pipeline["compute_lanes"] != 1
            or pipeline["active_slots"] != 0
            or pipeline["direct_active"] != 0
            or pipeline["direct_pending"] != 0
            or pipeline["direct_closed_clients"] != len(clients)
            or pipeline["direct_slot_clients"] != client_ids
            or not 0 <= pipeline["peak_occupied_slots"] <= receive_slots
            or pipeline["slot_generations"] != ordered
            or pipeline["slot_completed_calls"] != ordered
            or pipeline["direct_completed_by_client"] != counts
            or worker["client_calls"] != counts
            or worker["completed_calls"] != sum(ordered)
            or batching["calls"] != sum(ordered)
            or sum(histogram.values()) != batching["executions"]
            or sum(k * v for k, v in histogram.items()) != batching["calls"]
        ):
            raise AssertionError("Direct GPU slots, batches and completions disagree")
        total_calls += sum(ordered)
        peak = max(peak, pipeline["peak_occupied_slots"])
    return {
        "worker_child_calls": total_calls,
        "receive_slots": receive_slots,
        "peak_occupied_slots": peak,
        "controller_roundtrips": 0,
        "all_slots_drained": True,
        "all_results_and_feedback_reconciled": True,
    }
