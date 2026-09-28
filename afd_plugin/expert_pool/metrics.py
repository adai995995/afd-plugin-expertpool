# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Optional scalar aggregates; no per-call traces, payloads or tensor retention."""

import math


ROW_BUCKETS = ("zero", "one", "many")


class CallMetrics:
    def __init__(self) -> None:
        self.calls = 0
        self.token_rows = 0
        self.sums: dict[str, float] = {}
        self.maxima: dict[str, float] = {}
        self.batch_calls: dict[int, int] = {}
        self.layer_calls: dict[int, int] = {}
        self.row_bucket_calls = dict.fromkeys(ROW_BUCKETS, 0)
        self.row_bucket_sums: dict[str, dict[str, float]] = {
            bucket: {} for bucket in ROW_BUCKETS
        }
        self.row_bucket_maxima: dict[str, dict[str, float]] = {
            bucket: {} for bucket in ROW_BUCKETS
        }

    def record(self, layer: int, rows: int, metrics: dict[str, float]) -> None:
        # Shapes and layer IDs have already passed directory validation. State
        # scales with configured shapes/layers, never with completed call count.
        if any(not math.isfinite(value) or value < 0 for value in metrics.values()):
            raise ValueError("Timing metrics must be finite and nonnegative")
        self.calls += 1
        self.token_rows += rows
        self.batch_calls[rows] = self.batch_calls.get(rows, 0) + 1
        self.layer_calls[layer] = self.layer_calls.get(layer, 0) + 1
        bucket = "zero" if rows == 0 else "one" if rows == 1 else "many"
        self.row_bucket_calls[bucket] += 1
        bucket_sums = self.row_bucket_sums[bucket]
        bucket_maxima = self.row_bucket_maxima[bucket]
        for name, value in metrics.items():
            self.sums[name] = self.sums.get(name, 0.0) + value
            self.maxima[name] = max(self.maxima.get(name, 0.0), value)
            bucket_sums[name] = bucket_sums.get(name, 0.0) + value
            bucket_maxima[name] = max(bucket_maxima.get(name, 0.0), value)

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "token_rows": self.token_rows,
            "sum_ms": dict(self.sums),
            "max_ms": dict(self.maxima),
            "batch_calls": {str(k): v for k, v in sorted(self.batch_calls.items())},
            "layer_calls": {str(k): v for k, v in sorted(self.layer_calls.items())},
            "timings_by_rows": {
                bucket: {
                    "calls": self.row_bucket_calls[bucket],
                    "sum_ms": dict(self.row_bucket_sums[bucket]),
                    "max_ms": dict(self.row_bucket_maxima[bucket]),
                }
                for bucket in ROW_BUCKETS
            },
        }
