# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU metadata selection for one bounded, same-layer Expert execution.

Readiness is observed after the input CUDA event completes. The oldest ready
call anchors selection, so later calls of a popular layer cannot bypass it.
The deadline bounds intentional collection, not compute queueing or host
scheduling jitter. No routes or activation values are read on the CPU here.
"""

from dataclasses import dataclass

from afd_plugin.expert_pool.protocol import MAX_RECEIVE_SLOTS, ExecutionPlan

MAX_BATCH_WAIT_US = 1_000_000
NS_PER_US = 1_000


@dataclass(frozen=True)
class BatchingOptions:
    max_calls: int = 1
    max_tokens: int = 0
    max_wait_us: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.max_calls) is not int
            or not 1 <= self.max_calls <= MAX_RECEIVE_SLOTS
            or type(self.max_tokens) is not int
            or self.max_tokens < 0
            or type(self.max_wait_us) is not int
            or not 0 <= self.max_wait_us <= MAX_BATCH_WAIT_US
            or (self.enabled and self.max_tokens == 0)
            or (not self.enabled and (self.max_tokens != 0 or self.max_wait_us != 0))
        ):
            raise ValueError("Invalid bounded batching options")

    @property
    def enabled(self) -> bool:
        return self.max_calls > 1

    def validate_capacity(self, receive_slots: int, call_max_tokens: int) -> None:
        if self.enabled and (
            self.max_calls > receive_slots
            or not call_max_tokens
            <= self.max_tokens
            <= self.max_calls * call_max_tokens
        ):
            raise ValueError(
                "Batch capacity must fit one maximum call and the configured slots"
            )


DISABLED_BATCHING = BatchingOptions()


@dataclass(frozen=True)
class ReadyCall:
    plan: ExecutionPlan
    ready_ns: int


@dataclass(frozen=True)
class BatchDecision:
    slot_ids: tuple[int, ...]
    num_tokens: int
    deadline_ns: int
    wait_ns: int


def compatibility_key(plan: ExecutionPlan) -> tuple:
    request = plan.request
    return (
        plan.worker_id,
        request.model_id,
        request.placement_version,
        request.layer_id,
        request.hidden_size,
        request.top_k,
        request.compact_output,
    )


def select_batch(
    ready: tuple[ReadyCall, ...],
    options: BatchingOptions,
    now_ns: int,
    *,
    can_grow: bool = True,
) -> BatchDecision | None:
    if not ready:
        return None
    ordered = sorted(
        ready,
        key=lambda call: (call.ready_ns if options.enabled else 0, call.plan.plan_id),
    )
    anchor = ordered[0]
    selected = [anchor.plan.slot_id]
    tokens = anchor.plan.request.num_tokens
    if not options.enabled:
        return BatchDecision(tuple(selected), tokens, anchor.ready_ns, 0)
    if tokens > options.max_tokens:
        raise ValueError("A ready call exceeds the batch capacity")
    key = compatibility_key(anchor.plan)
    for call in ordered[1:]:
        if len(selected) == options.max_calls:
            break
        if (
            compatibility_key(call.plan) == key
            and tokens + call.plan.request.num_tokens <= options.max_tokens
        ):
            selected.append(call.plan.slot_id)
            tokens += call.plan.request.num_tokens
    deadline = anchor.ready_ns + options.max_wait_us * NS_PER_US
    full = (
        len(selected) == options.max_calls
        or tokens == options.max_tokens
        or not can_grow
    )
    return BatchDecision(
        tuple(selected), tokens, deadline, 0 if full else max(0, deadline - now_ns)
    )
