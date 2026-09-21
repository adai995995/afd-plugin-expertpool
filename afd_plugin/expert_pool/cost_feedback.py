# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded, observed execution costs for actual worker batches, without CUDA.

The controller owns immutable execution membership. Workers report event times
on the existing completion message. These are phase intervals, not active GEMM
time or a latency predictor; host submission gaps remain inside the intervals.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass

from afd_plugin.expert_pool.batching import compatibility_key
from afd_plugin.expert_pool.protocol import ExecutionPlan

DEFAULT_COST_BUCKETS = 512


def upper_power_of_two(value: int) -> int:
    return 0 if value == 0 else 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class ExecutionShape:
    layer: int
    calls: int
    token_rows: int
    # Distribution matters for Grouped GEMM. Equal total assignments alone
    # cannot distinguish one hot expert from many small expert problems.
    assignments_per_expert: tuple[int, ...]

    @classmethod
    def from_plans(cls, plans: tuple[ExecutionPlan, ...]) -> "ExecutionShape":
        if (
            not plans
            or len({compatibility_key(p) for p in plans}) != 1
            or len({p.worker_id for p in plans}) != 1
            or len({p.request.key.client_id for p in plans}) != len(plans)
        ):
            raise ValueError("Execution cost requires one compatible worker batch")
        counts: dict[int, int] = {}
        for plan in plans:
            if plan.request.demand is None or plan.num_assignments is None:
                raise ValueError("Execution cost requires validated expert demand")
            selected = {e: plan.request.demand.counts[e] for e in plan.expert_ids}
            if sum(selected.values()) != plan.num_assignments:
                raise ValueError("Execution cost demand does not match dispatch")
            for expert, count in selected.items():
                counts[expert] = counts.get(expert, 0) + count
        return cls(
            plans[0].request.layer_id,
            len(plans),
            sum(p.request.num_tokens for p in plans),
            tuple(sorted(count for count in counts.values() if count)),
        )

    @property
    def bucket(self) -> tuple:
        return (
            self.layer,
            self.calls,
            upper_power_of_two(self.token_rows),
            tuple(upper_power_of_two(n) for n in self.assignments_per_expert),
        )


class ExecutionCostBook:
    """LRU shape buckets with lifetime totals; no tensors or per-call history."""

    def __init__(self, max_buckets: int = DEFAULT_COST_BUCKETS) -> None:
        if type(max_buckets) is not int or max_buckets < 1:
            raise ValueError("Cost bucket capacity must be a positive integer")
        self.max_buckets = max_buckets
        self.buckets: OrderedDict[tuple, dict] = OrderedDict()
        self.totals = self._empty()
        self.evictions = 0

    @staticmethod
    def _empty() -> dict:
        return {
            "executions": 0,
            "child_calls": 0,
            "token_rows": 0,
            "expert_assignments": 0,
            "sum_ms": {},
            "min_ms": {},
            "max_ms": {},
        }

    def record(
        self, shape: ExecutionShape, compute_ms: float, merge_ms: float, pack_ms: float
    ) -> None:
        timings = {
            "compute_phase": compute_ms,
            "input_merge": merge_ms,
            "output_pack": pack_ms,
        }
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in timings.values()
        ):
            raise ValueError("Execution costs must be finite nonnegative intervals")
        key = shape.bucket
        if key not in self.buckets:
            if len(self.buckets) == self.max_buckets:
                self.buckets.popitem(last=False)
                self.evictions += 1
            self.buckets[key] = self._empty()
        self.buckets.move_to_end(key)
        for aggregate in (self.totals, self.buckets[key]):
            aggregate["executions"] += 1
            aggregate["child_calls"] += shape.calls
            aggregate["token_rows"] += shape.token_rows
            aggregate["expert_assignments"] += sum(shape.assignments_per_expert)
            for name, value in timings.items():
                aggregate["sum_ms"][name] = aggregate["sum_ms"].get(name, 0.0) + value
                aggregate["min_ms"][name] = min(
                    aggregate["min_ms"].get(name, value), value
                )
                aggregate["max_ms"][name] = max(
                    aggregate["max_ms"].get(name, value), value
                )

    @staticmethod
    def _snapshot(aggregate: dict) -> dict:
        return {
            **{k: v for k, v in aggregate.items() if not isinstance(v, dict)},
            **{k: dict(v) for k, v in aggregate.items() if isinstance(v, dict)},
            "mean_ms": {
                key: value / aggregate["executions"]
                for key, value in aggregate["sum_ms"].items()
            },
        }

    def snapshot(self) -> dict:
        return {
            "capacity": self.max_buckets,
            "evictions": self.evictions,
            "scope": (
                "Lifetime including engine initialization and warmup; CUDA event "
                "phase intervals, not active-kernel time or a latency predictor"
            ),
            "bucket_policy": (
                "Layer, calls, power-of-two upper bounds on input rows and sorted "
                "positive per-expert assignments; expert identities pooled within "
                "a layer"
            ),
            "totals": self._snapshot(self.totals),
            "buckets": [
                {
                    "layer": key[0],
                    "calls": key[1],
                    "token_rows_upper": key[2],
                    "expert_assignments_upper": list(key[3]),
                    **self._snapshot(value),
                }
                for key, value in self.buckets.items()
            ],
        }
