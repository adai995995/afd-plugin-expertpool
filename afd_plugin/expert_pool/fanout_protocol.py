# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded reply demultiplexing for one parent with static expert owners."""

from afd_plugin.expert_pool.demand import DispatchPlan
from afd_plugin.expert_pool.protocol import (
    MAX_RECEIVE_SLOTS,
    CallRequest,
    ExecutionPlan,
    Message,
)

FANOUT_REPLY_KINDS = ("grant", "output_ready", "done")


class FanoutReplies:
    def __init__(
        self,
        request: CallRequest,
        owners: tuple[str, ...],
        *,
        dispatch_plan: DispatchPlan | None = None,
        receive_slots: int = 1,
    ) -> None:
        if not isinstance(request, CallRequest):
            raise ValueError("Fan-out replies require a parent request")
        if (
            not isinstance(owners, tuple)
            or not owners
            or any(
                not isinstance(owner, str) or not 0 < len(owner) <= 128
                for owner in owners
            )
            or len(set(owners)) != len(owners)
        ):
            raise ValueError("Fan-out replies require unique worker identities")
        self.request = request
        if (
            type(receive_slots) is not int
            or not 1 <= receive_slots <= MAX_RECEIVE_SLOTS
        ):
            raise ValueError("Invalid receive slot capacity")
        self.receive_slots = receive_slots
        if (request.demand is not None) != (dispatch_plan is not None):
            raise ValueError("Demand replies require the expected dispatch plan")
        if dispatch_plan is not None and (
            dispatch_plan.request != request or dispatch_plan.owners != owners
        ):
            raise ValueError("Dispatch plan and fan-out reply owners disagree")
        self.dispatch_plan = dispatch_plan
        self.owners = owners
        self.plans: dict[str, ExecutionPlan] = {}
        self.next_phase = dict.fromkeys(owners, 0)
        self.messages: dict[tuple[str, str], Message] = {}

    def accept(self, message: Message) -> None:
        if message.kind == "error":
            raise RuntimeError("Expert fan-out failed; terminate the deployment")
        if message.kind not in FANOUT_REPLY_KINDS or message.plan is None:
            raise RuntimeError("Unexpected fan-out reply")
        plan = message.plan
        if (
            plan.request != self.request
            or plan.worker_id not in self.next_phase
            or plan.slot_id >= self.receive_slots
        ):
            raise RuntimeError("Mismatched parent, worker or buffer slot")
        worker_id = plan.worker_id
        index = self.next_phase[worker_id]
        if (
            index == len(FANOUT_REPLY_KINDS)
            or message.kind != FANOUT_REPLY_KINDS[index]
        ):
            raise RuntimeError("Duplicate or out-of-order fan-out reply")
        if index == 0:
            if self.dispatch_plan is not None:
                task = self.dispatch_plan.task_for(worker_id)
                if (
                    plan.expert_ids,
                    plan.num_assignments,
                    plan.assignment_slices,
                ) != (
                    task.expert_ids,
                    task.num_assignments,
                    task.assignment_slices,
                ):
                    raise RuntimeError("Granted task disagrees with expert demand")
            if any(
                previous.plan_id == plan.plan_id for previous in self.plans.values()
            ):
                raise RuntimeError("Fan-out child plan identities must be unique")
            self.plans[worker_id] = plan
        elif self.plans[worker_id] != plan:
            raise RuntimeError("Mismatched fan-out child plan or buffer generation")
        self.messages[(message.kind, worker_id)] = message
        self.next_phase[worker_id] += 1

    def take(self, kind: str, worker_id: str) -> Message | None:
        if kind not in FANOUT_REPLY_KINDS or worker_id not in self.next_phase:
            raise ValueError("Unknown fan-out reply target")
        return self.messages.pop((kind, worker_id), None)
