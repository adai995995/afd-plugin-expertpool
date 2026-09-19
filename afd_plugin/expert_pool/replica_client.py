# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Per-instance dispatch over preconnected resident whole-layer replicas.

This module makes CPU-only decisions. Payloads go directly from the caller to
one selected E worker. Calls are never retried on another replica after dispatch.
Each A remains independent; each E retains its own admission queue and buffers.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from afd_plugin.expert_pool.directory import PoolDirectory, ReplicaSelector
from afd_plugin.expert_pool.metrics import CallMetrics

if TYPE_CHECKING:
    import torch

    from afd_plugin.expert_pool.client import PoolClient
    from afd_plugin.expert_pool.protocol import Message


class ReplicaPoolClient:
    def __init__(
        self, channels: tuple[PoolClient, ...], *, client_offset: int = 0
    ) -> None:
        self._initialize_channels(channels)
        self.selector = ReplicaSelector(self.directory, client_offset)

    def _initialize_channels(
        self, channels: tuple[PoolClient, ...], *, expert_partitioned: bool = False
    ) -> None:
        """Bind fresh channels; subclasses supply a compatible dispatch policy."""
        if not channels:
            raise ValueError("Pool client requires worker channels")
        first = channels[0]
        if any(
            (channel.client_id, channel.session_epoch, channel.transport.device)
            != (first.client_id, first.session_epoch, first.transport.device)
            or channel.sequence != 0
            or channel.closed
            or channel.failed
            for channel in channels
        ):
            raise ValueError("Channels must bind one fresh A session on one GPU")
        self.directory = PoolDirectory(
            tuple(channel.directory for channel in channels),
            expert_partitioned=expert_partitioned,
        )
        self.channels = {channel.directory.worker_id: channel for channel in channels}
        self.client_id = first.client_id
        self.session_epoch = first.session_epoch
        self.sequence = 0
        self.lock = threading.Lock()
        self.closed = False
        self.failed = False
        self.metrics: CallMetrics | None = None
        self.worker_calls = dict.fromkeys(self.channels, 0)
        self.worker_layer_calls: dict[str, dict[int, int]] = {
            worker.worker_id: dict.fromkeys(
                (placement.layer_id for placement in worker.placements), 0
            )
            for worker in self.directory.workers
        }

    def execute(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, Message]:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("This A instance already has an outstanding operation")
        try:
            if self.closed or self.failed:
                raise RuntimeError("Pool client is closed or failed")
            worker_id = self.selector.select(layer_id)
            sequence = self.sequence
            self.sequence += 1
            output, completion = self.channels[worker_id].execute(
                layer_id, hidden_states, topk_weights, topk_ids, call_seq=sequence
            )
            self.worker_calls[worker_id] += 1
            self.worker_layer_calls[worker_id][layer_id] += 1
            if self.metrics is not None:
                self.metrics.record(
                    layer_id, hidden_states.shape[0], completion.metrics
                )
            return output, completion
        except BaseException:
            # The channel knows whether GPU work was issued. An uncertain
            # completion poisons the whole client; another copy is not a retry.
            self.failed = self.failed or any(c.failed for c in self.channels.values())
            raise
        finally:
            self.lock.release()

    def set_metrics(self, enabled: bool) -> None:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Drain the client before changing metrics")
        try:
            self.metrics = CallMetrics() if enabled else None
            for channel in self.channels.values():
                channel.set_metrics(enabled)
        finally:
            self.lock.release()

    def dispatch_status(self) -> dict:
        """Read bounded counters after draining; this is not live busy feedback."""
        return {
            "policy": "resident-layer-round-robin",
            "placement_version": self.directory.workers[0].version,
            "next_call_seq": self.sequence,
            "workers": {
                worker_id: {
                    "calls": self.worker_calls[worker_id],
                    "layer_calls": {
                        str(layer): count
                        for layer, count in self.worker_layer_calls[worker_id].items()
                    },
                    "call_metrics": (
                        channel.metrics.snapshot()
                        if channel.metrics is not None
                        else None
                    ),
                }
                for worker_id, channel in self.channels.items()
            },
        }

    def close(self) -> None:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Drain the outstanding call before closing")
        try:
            if self.closed:
                return
            first_error = None
            for channel in self.channels.values():
                try:
                    channel.close()
                except Exception as error:
                    if first_error is None:
                        first_error = error
            self.closed = True
            if first_error is not None:
                raise first_error
        finally:
            self.lock.release()
