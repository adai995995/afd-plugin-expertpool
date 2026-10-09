# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in bounded execution membership records from existing CPU metadata.

No tensor values, Router re-execution, GPU synchronization or new control
messages are needed. The supervisor writes records after orderly shutdown.
Call sequence windows taken after startup separate live requests from warmup.
"""

from afd_plugin.expert_pool.protocol import BatchExecution, BatchSlot, ExecutionPlan

DEFAULT_AUDIT_RECORDS = 8192
MAX_AUDIT_RECORDS = 65536
AUDIT_VERSION = 3


class BatchAudit:
    def __init__(self, max_records: int = DEFAULT_AUDIT_RECORDS) -> None:
        if type(max_records) is not int or not 1 <= max_records <= MAX_AUDIT_RECORDS:
            raise ValueError("Invalid batch audit capacity")
        self.max_records = max_records
        self.records: list[dict] = []
        self.pending: dict[int, dict] = {}
        self.observed = 0
        self.dropped = 0

    def record_batch(
        self, batch: BatchExecution, plans: tuple[ExecutionPlan, ...]
    ) -> None:
        if (
            not plans
            or tuple(BatchSlot.from_plan(plan) for plan in plans) != batch.members
            or any(plan.worker_id != batch.worker_id for plan in plans)
        ):
            raise ValueError("Audit membership disagrees with admitted execution")
        self.observed += 1
        if len(self.records) == self.max_records:
            self.dropped += 1
            return
        members = []
        totals: dict[str, int] = {}
        row_begin = 0
        for plan in plans:
            counts = {str(e): plan.assignments_for(e) for e in plan.expert_ids}
            ranges = (
                {
                    str(item.expert_id): {"start": item.start, "count": item.count}
                    for item in plan.assignment_slices
                }
                if plan.assignment_slices
                else {expert: {"start": 0, "count": n} for expert, n in counts.items()}
            )
            member = {
                "client_id": plan.request.key.client_id,
                "session_epoch": plan.request.key.session_epoch,
                "call_seq": plan.request.key.call_seq,
                "plan_id": plan.plan_id,
                "slot_id": plan.slot_id,
                "generation": plan.generation,
                "num_tokens": plan.request.num_tokens,
                "top_k": plan.request.top_k,
                "expert_demand": None
                if plan.request.demand is None
                else {str(e): n for e, n in enumerate(plan.request.demand.counts) if n},
                "expert_assignments": counts,
                "assignment_ranges": ranges,
                "completed": False,
            }
            actual_rows = plan.input_rows or plan.request.num_tokens
            if plan.request.partial_reduction:
                member["packed_token_rows"] = actual_rows
                member["output_layout"] = "fp32-partial-token-hidden"
            if plan.request.demand is None or plan.request.partial_reduction:
                member["input_row_range"] = [
                    row_begin,
                    row_begin + actual_rows,
                ]
            row_begin += actual_rows
            members.append(member)
            self.pending[plan.plan_id] = member
            for expert, count in counts.items():
                totals[expert] = totals.get(expert, 0) + count
        self.records.append(
            {
                "worker_id": batch.worker_id,
                "batch_sequence": batch.sequence,
                "model_id": plans[0].request.model_id,
                "placement_version": plans[0].request.placement_version,
                "layer_id": plans[0].request.layer_id,
                "token_rows": sum(
                    plan.input_rows or plan.request.num_tokens for plan in plans
                ),
                "expert_assignments": totals,
                "members": members,
            }
        )

    def record_ready_timing(
        self,
        plan: ExecutionPlan,
        admitted_ns: int,
        ready_ns: int,
        compute_submitted_ns: int,
    ) -> None:
        """All timestamps use this E's host clock, observed after CUDA readiness.

        These are dependency-aware arrivals under the measured policy. An
        offline regrouping is an opportunity estimate, not future online SLOs.
        """
        if not 0 <= admitted_ns <= ready_ns <= compute_submitted_ns:
            raise ValueError("Execution cannot precede admission or input readiness")
        member = self.pending.get(plan.plan_id)
        if member is not None:
            member.update(
                admitted_ns=admitted_ns,
                input_ready_ns=ready_ns,
                compute_submitted_ns=compute_submitted_ns,
            )

    def record_done(
        self, plan: ExecutionPlan, metrics: dict[str, float] | None = None
    ) -> None:
        member = self.pending.pop(plan.plan_id, None)
        if member is not None:
            member["completed"] = True
            if metrics is not None:
                member["metrics"] = dict(metrics)

    def summary(self) -> dict:
        return {
            "version": AUDIT_VERSION,
            "enabled": True,
            "capacity": self.max_records,
            "observed_batches": self.observed,
            "recorded_batches": len(self.records),
            "dropped_batches": self.dropped,
            "pending_recorded_members": len(self.pending),
        }

    def snapshot(self) -> dict:
        return {**self.summary(), "records": self.records}
