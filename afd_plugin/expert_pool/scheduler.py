# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static, single-worker admission with per-domain round-robin dispatch.

One execution slot and one outstanding call per client deliberately bound the
first runtime. Call fairness is not a token quota or an end-to-end SLO promise.
"""

from collections import deque
from dataclasses import dataclass

from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import CallKey, CallRequest, ExecutionPlan


@dataclass(frozen=True)
class StaticDirectory:
    """Resident coverage on one worker for explicitly enabled layers.

    model_id binds an immutable checkpoint selected by the launcher. It is not
    inferred from model architecture and does not prove cross-checkpoint equality.
    Partial expert coverage requires a partition-aware controller and client.
    """

    model_id: str
    version: int
    worker_id: str
    placements: tuple[ExpertPlacement, ...]
    hidden_size: int
    top_k: int
    max_tokens: int
    allow_partial_experts: bool = False

    def __post_init__(self) -> None:
        if type(self.allow_partial_experts) is not bool:
            raise ValueError("Partial expert coverage must be an explicit boolean")
        if (
            not isinstance(self.model_id, str)
            or not 0 < len(self.model_id) <= 256
            or not isinstance(self.worker_id, str)
            or not 0 < len(self.worker_id) <= 128
            or type(self.version) is not int
            or self.version < 0
        ):
            raise ValueError("Invalid directory identity")
        if any(
            type(value) is not int or value <= 0
            for value in (self.hidden_size, self.top_k, self.max_tokens)
        ):
            raise ValueError("Invalid directory capacity")
        if not isinstance(self.placements, tuple) or not self.placements:
            raise ValueError("Directory requires immutable placements")
        layers = [placement.layer_id for placement in self.placements]
        if len(set(layers)) != len(layers):
            raise ValueError("Duplicate layer placement")
        for placement in self.placements:
            if not self.allow_partial_experts and set(placement.expert_ids) != set(
                range(placement.num_experts)
            ):
                raise ValueError("Single-worker runtime requires complete coverage")
            if self.top_k > placement.num_experts:
                raise ValueError("Top-k exceeds expert count")

    def validate(self, request: CallRequest) -> None:
        if (request.model_id, request.placement_version) != (
            self.model_id,
            self.version,
        ):
            raise ValueError("Checkpoint or placement version mismatch")
        if request.layer_id not in {p.layer_id for p in self.placements}:
            raise ValueError("Requested layer is not resident")
        if (request.hidden_size, request.top_k) != (self.hidden_size, self.top_k):
            raise ValueError("Payload shape does not match checkpoint")
        if request.num_tokens > self.max_tokens:
            raise ValueError("Call exceeds buffer capacity")


@dataclass(frozen=True)
class PendingCall:
    request: CallRequest
    enqueued_ns: int


class StaticScheduler:
    def __init__(
        self, directory: StaticDirectory, max_pending_per_domain: int = 4
    ) -> None:
        if directory.allow_partial_experts:
            raise ValueError("Static scheduler requires complete worker coverage")
        if type(max_pending_per_domain) is not int or max_pending_per_domain <= 0:
            raise ValueError("Domain queue capacity must be positive")
        self.directory = directory
        self.max_pending_per_domain = max_pending_per_domain
        self.sessions: dict[str, tuple[int, str]] = {}
        self.last_sequence: dict[str, int] = {}
        self.pending: dict[str, deque[PendingCall]] = {}
        self.domains: deque[str] = deque()
        self.outstanding: dict[str, CallKey] = {}
        self.active: ExecutionPlan | None = None
        self.active_enqueued_ns = 0
        self.generation = 0

    def register(self, client_id: str, epoch: int, domain: str) -> None:
        CallKey(client_id, epoch, 0)
        if not isinstance(domain, str) or not 0 < len(domain) <= 128:
            raise ValueError("Missing service domain")
        if client_id in self.outstanding:
            raise ValueError("Drain the previous session before replacing it")
        previous = self.sessions.get(client_id)
        if previous is not None and epoch <= previous[0]:
            raise ValueError("Session epoch must increase")
        if previous is not None and domain != previous[1]:
            raise ValueError(
                "Session restart cannot change its configured service domain"
            )
        self.sessions[client_id] = (epoch, domain)
        self.last_sequence[client_id] = -1
        if domain not in self.pending:
            self.pending[domain] = deque()
            self.domains.append(domain)

    def submit(self, request: CallRequest, now_ns: int) -> None:
        self.directory.validate(request)
        key = request.key
        if key.client_id not in self.sessions:
            raise ValueError("Unknown client")
        epoch, domain = self.sessions[key.client_id]
        if key.session_epoch != epoch:
            raise ValueError("Stale client session")
        if key.call_seq <= self.last_sequence[key.client_id]:
            raise ValueError("Repeated or stale call sequence")
        if key.client_id in self.outstanding:
            raise ValueError("Only one outstanding call per client is supported")
        if len(self.pending[domain]) >= self.max_pending_per_domain:
            raise ValueError("Domain queue capacity exceeded")
        self.last_sequence[key.client_id] = key.call_seq
        self.outstanding[key.client_id] = key
        self.pending[domain].append(PendingCall(request, now_ns))

    def grant(self) -> ExecutionPlan | None:
        if self.active is not None:
            return None
        for _ in range(len(self.domains)):
            domain = self.domains[0]
            self.domains.rotate(-1)
            if not self.pending[domain]:
                continue
            pending = self.pending[domain].popleft()
            self.generation += 1
            self.active = ExecutionPlan(
                pending.request,
                self.directory.worker_id,
                self.generation,
                0,
                self.generation,
            )
            self.active_enqueued_ns = pending.enqueued_ns
            return self.active
        return None

    def complete(self, plan: ExecutionPlan) -> None:
        """Caller must drain GPU operations before releasing this buffer credit."""
        if self.active is None or self.active != plan:
            raise ValueError("Stale, duplicate or mismatched completion")
        del self.outstanding[plan.request.key.client_id]
        self.active = None

    def cancel_queued(self, key: CallKey) -> None:
        if self.active is not None and self.active.request.key == key:
            raise ValueError("In-flight GPU work must drain before cancellation")
        if self.outstanding.get(key.client_id) != key:
            raise ValueError("No matching queued call")
        domain = self.sessions[key.client_id][1]
        self.pending[domain] = deque(
            pending for pending in self.pending[domain] if pending.request.key != key
        )
        del self.outstanding[key.client_id]
