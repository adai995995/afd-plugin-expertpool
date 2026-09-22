# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded JSON control messages for a trusted, local Expert Pool launch.

GPU tensors never travel through this channel. Session identity, the plan and
the buffer generation must match before any transfer or buffer reuse.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Connection

MAX_CONTROL_BYTES = 65536
MAX_DEMAND_EXPERTS = 4096
MAX_RECEIVE_SLOTS = 8
MESSAGE_KINDS = frozenset(
    {
        "submit",
        "grant",
        "output_ready",
        "done",
        "error",
        "close",
        "closed",
        "ready",
        "input_ready",
        "executing",
        "batch_executing",
        "dispatch",
        "status",
        "snapshot",
        "empty_done",
        "execute",
        "result",
    }
)


@dataclass(frozen=True)
class CallKey:
    client_id: str
    session_epoch: int
    call_seq: int

    def __post_init__(self) -> None:
        if not isinstance(self.client_id, str) or not 0 < len(self.client_id) <= 128:
            raise ValueError("Invalid client identity")
        if any(
            type(value) is not int or value < 0
            for value in (self.session_epoch, self.call_seq)
        ):
            raise ValueError("Session epoch and sequence must be nonnegative integers")


@dataclass(frozen=True)
class ExpertDemand:
    """CPU metadata counting routed top-k assignments, not unique token rows."""

    counts: tuple[int, ...]
    ready_ns: int
    version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.counts, tuple)
            or not 0 < len(self.counts) <= MAX_DEMAND_EXPERTS
            or any(type(count) is not int or count < 0 for count in self.counts)
        ):
            raise ValueError("Demand requires bounded nonnegative integer counts")
        if type(self.ready_ns) is not int or self.ready_ns < 0:
            raise ValueError("Demand readiness must be a nonnegative timestamp")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported expert demand version")


@dataclass(frozen=True)
class CallRequest:
    key: CallKey
    model_id: str
    placement_version: int
    layer_id: int
    num_tokens: int
    hidden_size: int
    top_k: int
    demand: ExpertDemand | None = None
    compact_output: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key, CallKey):
            raise ValueError("Missing call identity")
        if not isinstance(self.model_id, str) or not 0 < len(self.model_id) <= 256:
            raise ValueError("Invalid checkpoint identity")
        for value in (self.placement_version, self.layer_id, self.num_tokens):
            if type(value) is not int or value < 0:
                raise ValueError("Invalid placement, layer or token count")
        for value in (self.hidden_size, self.top_k):
            if type(value) is not int or value <= 0:
                raise ValueError("Invalid tensor shape")
        if type(self.compact_output) is not bool or (
            self.compact_output and self.demand is None
        ):
            raise ValueError("Compact output requires explicit expert demand")
        if self.demand is not None:
            if not isinstance(self.demand, ExpertDemand):
                raise ValueError("Request requires typed expert demand")
            if sum(self.demand.counts) != self.num_tokens * self.top_k:
                raise ValueError("Demand must count every routed top-k assignment")


@dataclass(frozen=True)
class ExpertTask:
    worker_id: str
    expert_ids: tuple[int, ...]
    num_assignments: int

    def __post_init__(self) -> None:
        if not isinstance(self.worker_id, str) or not 0 < len(self.worker_id) <= 128:
            raise ValueError("Invalid expert task worker")
        if (
            not isinstance(self.expert_ids, tuple)
            or not self.expert_ids
            or any(type(expert) is not int or expert < 0 for expert in self.expert_ids)
            or len(set(self.expert_ids)) != len(self.expert_ids)
        ):
            raise ValueError("Expert tasks require unique nonnegative expert IDs")
        if type(self.num_assignments) is not int or self.num_assignments <= 0:
            raise ValueError("Expert tasks require positive assignment counts")


@dataclass(frozen=True)
class DispatchPlan:
    request: CallRequest
    tasks: tuple[ExpertTask, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, CallRequest) or self.request.demand is None:
            raise ValueError("Dispatch requires a request with expert demand")
        if not isinstance(self.tasks, tuple) or any(
            not isinstance(task, ExpertTask) for task in self.tasks
        ):
            raise ValueError("Dispatch requires immutable expert tasks")
        if len({task.worker_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Each worker must receive at most one expert task")
        counts = self.request.demand.counts
        assigned: set[int] = set()
        for task in self.tasks:
            if any(expert >= len(counts) for expert in task.expert_ids):
                raise ValueError("Task expert is outside the demand range")
            if assigned.intersection(task.expert_ids):
                raise ValueError("Dispatch repeats a demanded expert")
            if any(counts[expert] == 0 for expert in task.expert_ids):
                raise ValueError("Dispatch includes an expert without demand")
            if task.num_assignments != sum(
                counts[expert] for expert in task.expert_ids
            ):
                raise ValueError("Task count disagrees with expert demand")
            assigned.update(task.expert_ids)
        if assigned != {expert for expert, count in enumerate(counts) if count > 0}:
            raise ValueError("Dispatch must cover every demanded expert exactly once")
        if sum(task.num_assignments for task in self.tasks) != (
            self.request.num_tokens * self.request.top_k
        ):
            raise ValueError("Dispatch assignment accounting is incomplete")

    @property
    def owners(self) -> tuple[str, ...]:
        return tuple(sorted(task.worker_id for task in self.tasks))

    def task_for(self, worker_id: str) -> ExpertTask:
        for task in self.tasks:
            if task.worker_id == worker_id:
                return task
        raise ValueError("Worker has no demanded expert task")


@dataclass(frozen=True)
class ExecutionPlan:
    """Bind output layout to the request and compact row count to assignments.

    When request.compact_output is true, num_assignments is the exact number
    of returned weighted rows. There is no separate unchecked transfer length.
    """

    request: CallRequest
    worker_id: str
    plan_id: int
    slot_id: int
    generation: int
    expert_ids: tuple[int, ...] = ()
    num_assignments: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request, CallRequest)
            or not isinstance(self.worker_id, str)
            or not 0 < len(self.worker_id) <= 128
        ):
            raise ValueError("Invalid execution target")
        if any(
            type(value) is not int or value < 0
            for value in (self.plan_id, self.slot_id, self.generation)
        ):
            raise ValueError("Invalid plan or buffer identity")
        if not isinstance(self.expert_ids, tuple):
            raise ValueError("Plan expert IDs must be immutable")
        if self.num_assignments is None:
            if self.expert_ids or self.request.demand is not None:
                raise ValueError("Legacy plans cannot carry expert demand")
            return
        if (
            type(self.num_assignments) is not int
            or self.num_assignments <= 0
            or self.request.demand is None
            or not self.expert_ids
            or any(
                type(expert) is not int
                or not 0 <= expert < len(self.request.demand.counts)
                for expert in self.expert_ids
            )
            or len(set(self.expert_ids)) != len(self.expert_ids)
        ):
            raise ValueError("Invalid demand execution plan")
        if any(self.request.demand.counts[expert] == 0 for expert in self.expert_ids):
            raise ValueError("Plan cannot dispatch experts without demand")
        if self.num_assignments != sum(
            self.request.demand.counts[expert] for expert in self.expert_ids
        ):
            raise ValueError("Plan assignment count disagrees with expert demand")


@dataclass(frozen=True)
class BatchSlot:
    """Reference an already admitted immutable plan without repeating demand."""

    plan_id: int
    slot_id: int
    generation: int

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int or value < 0
                for value in (self.plan_id, self.slot_id, self.generation)
            )
            or self.slot_id >= MAX_RECEIVE_SLOTS
        ):
            raise ValueError("Invalid batch slot identity")

    @classmethod
    def from_plan(cls, plan: ExecutionPlan) -> "BatchSlot":
        return cls(plan.plan_id, plan.slot_id, plan.generation)


@dataclass(frozen=True)
class BatchExecution:
    worker_id: str
    sequence: int
    members: tuple[BatchSlot, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.worker_id, str)
            or not 0 < len(self.worker_id) <= 128
            or type(self.sequence) is not int
            or self.sequence <= 0
            or not isinstance(self.members, tuple)
            or not 1 <= len(self.members) <= MAX_RECEIVE_SLOTS
            or any(not isinstance(member, BatchSlot) for member in self.members)
            or len({member.slot_id for member in self.members}) != len(self.members)
            or len({member.plan_id for member in self.members}) != len(self.members)
        ):
            raise ValueError("Invalid compute batch identity")


@dataclass(frozen=True)
class Message:
    kind: str
    request: CallRequest | None = None
    plan: ExecutionPlan | None = None
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)
    batch: BatchExecution | None = None
    dispatch: DispatchPlan | None = None

    def __post_init__(self) -> None:
        if self.request is not None and not isinstance(self.request, CallRequest):
            raise ValueError("Invalid request object")
        if self.plan is not None and not isinstance(self.plan, ExecutionPlan):
            raise ValueError("Invalid plan object")
        if self.kind not in MESSAGE_KINDS:
            raise ValueError("Unknown control message")
        if self.kind == "dispatch":
            if (
                not isinstance(self.dispatch, DispatchPlan)
                or self.request is not None
                or self.plan is not None
            ):
                raise ValueError("Dispatch announcement requires a validated plan only")
        elif self.dispatch is not None:
            raise ValueError("Unexpected dispatch announcement")
        if self.kind == "batch_executing":
            if (
                not isinstance(self.batch, BatchExecution)
                or self.plan is not None
                or self.request is not None
            ):
                raise ValueError("Batch execution requires a batch identity only")
        elif self.batch is not None:
            raise ValueError("Unexpected batch identity")
        if not isinstance(self.detail, str) or len(self.detail) > 2048:
            raise ValueError("Invalid error detail")
        if self.kind == "submit" and (self.request is None or self.plan is not None):
            raise ValueError("Submit requires a request only")
        if self.kind == "empty_done" and (
            self.request is None
            or self.request.demand is None
            or self.request.num_tokens != 0
            or self.plan is not None
        ):
            raise ValueError("Empty completion requires a zero-token demand request")
        if self.kind in {
            "grant",
            "input_ready",
            "executing",
            "output_ready",
            "done",
            "execute",
            "result",
        } and (self.plan is None or self.request is not None):
            raise ValueError("Execution reply requires a plan only")
        if self.kind in {"close", "closed", "ready", "status", "snapshot"} and (
            self.request is not None or self.plan is not None
        ):
            raise ValueError("Close messages cannot carry a call")
        if self.kind == "error" and (
            (self.request is None) == (self.plan is None) or not self.detail
        ):
            raise ValueError("Error requires one call identity and a reason")
        if not isinstance(self.metrics, dict) or any(
            not isinstance(key, str)
            or type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            for key, value in self.metrics.items()
        ):
            raise ValueError("Invalid timing values")
        if not isinstance(self.digests, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for key, value in self.digests.items()
        ):
            raise ValueError("Invalid payload digest")


def encode_message(message: Message) -> bytes:
    payload = json.dumps(asdict(message), allow_nan=False).encode()
    if len(payload) > MAX_CONTROL_BYTES:
        raise ValueError("Control message exceeds size limit")
    return payload


def _decode_request(raw: dict) -> CallRequest:
    if not isinstance(raw, dict):
        raise ValueError("Malformed request object")
    request = dict(raw)
    request["key"] = CallKey(**request["key"])
    if request.get("demand") is not None:
        demand = request["demand"]
        if not isinstance(demand, dict) or not isinstance(demand.get("counts"), list):
            raise ValueError("Malformed demand object")
        request["demand"] = ExpertDemand(
            **{**demand, "counts": tuple(demand["counts"])}
        )
    return CallRequest(**request)


def decode_message(payload: bytes) -> Message:
    if len(payload) > MAX_CONTROL_BYTES:
        raise ValueError("Control message exceeds size limit")
    try:
        data = json.loads(payload)
        if set(data) != {
            "kind",
            "request",
            "plan",
            "detail",
            "metrics",
            "digests",
            "batch",
            "dispatch",
        }:
            raise ValueError("Unexpected message fields")
        if data["dispatch"] is not None:
            decision = data["dispatch"]
            if not isinstance(decision, dict) or not isinstance(
                decision["tasks"], list
            ):
                raise ValueError("Malformed replica dispatch")
            tasks = []
            for task in decision["tasks"]:
                if not isinstance(task, dict) or not isinstance(
                    task["expert_ids"], list
                ):
                    raise ValueError("Malformed replica task")
                tasks.append(
                    ExpertTask(**{**task, "expert_ids": tuple(task["expert_ids"])})
                )
            data["dispatch"] = DispatchPlan(
                **{
                    **decision,
                    "request": _decode_request(decision["request"]),
                    "tasks": tuple(tasks),
                }
            )
        if data["batch"] is not None:
            batch = data["batch"]
            if not isinstance(batch, dict) or not isinstance(batch["members"], list):
                raise ValueError("Malformed compute batch identity")
            data["batch"] = BatchExecution(
                **{**batch, "members": tuple(BatchSlot(**m) for m in batch["members"])}
            )
        if data["request"] is not None:
            data["request"] = _decode_request(data["request"])
        if data["plan"] is not None:
            plan = data["plan"]
            plan["request"] = _decode_request(plan["request"])
            if "expert_ids" in plan:
                if not isinstance(plan["expert_ids"], list):
                    raise ValueError("Malformed plan expert IDs")
                plan["expert_ids"] = tuple(plan["expert_ids"])
            data["plan"] = ExecutionPlan(**plan)
        return Message(**data)
    except (KeyError, TypeError, UnicodeError) as error:
        raise ValueError("Malformed control message") from error


def send_message(connection: Connection, message: Message) -> None:
    connection.send_bytes(encode_message(message))


def receive_message(connection: Connection, timeout_s: float) -> Message:
    if not connection.poll(timeout_s):
        raise TimeoutError("Expert Pool control reply timed out")
    return decode_message(connection.recv_bytes(MAX_CONTROL_BYTES))
