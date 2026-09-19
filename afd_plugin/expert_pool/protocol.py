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
        "status",
        "snapshot",
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
class CallRequest:
    key: CallKey
    model_id: str
    placement_version: int
    layer_id: int
    num_tokens: int
    hidden_size: int
    top_k: int

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


@dataclass(frozen=True)
class ExecutionPlan:
    request: CallRequest
    worker_id: str
    plan_id: int
    slot_id: int
    generation: int

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


@dataclass(frozen=True)
class Message:
    kind: str
    request: CallRequest | None = None
    plan: ExecutionPlan | None = None
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.request is not None and not isinstance(self.request, CallRequest):
            raise ValueError("Invalid request object")
        if self.plan is not None and not isinstance(self.plan, ExecutionPlan):
            raise ValueError("Invalid plan object")
        if self.kind not in MESSAGE_KINDS:
            raise ValueError("Unknown control message")
        if not isinstance(self.detail, str) or len(self.detail) > 2048:
            raise ValueError("Invalid error detail")
        if self.kind == "submit" and (self.request is None or self.plan is not None):
            raise ValueError("Submit requires a request only")
        if self.kind in {
            "grant",
            "input_ready",
            "executing",
            "output_ready",
            "done",
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


def decode_message(payload: bytes) -> Message:
    if len(payload) > MAX_CONTROL_BYTES:
        raise ValueError("Control message exceeds size limit")
    try:
        data = json.loads(payload)
        if set(data) != {"kind", "request", "plan", "detail", "metrics", "digests"}:
            raise ValueError("Unexpected message fields")
        if data["request"] is not None:
            request = data["request"]
            request["key"] = CallKey(**request["key"])
            data["request"] = CallRequest(**request)
        if data["plan"] is not None:
            plan = data["plan"]
            request = plan["request"]
            request["key"] = CallKey(**request["key"])
            plan["request"] = CallRequest(**request)
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
