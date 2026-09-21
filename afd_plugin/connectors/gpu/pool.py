# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Prebuilt Client/Worker NCCL P2P channels for the initial eager Pool.

Each pair initializes once. Runtime transfers involve only the granted pair.
The blocking API is retained. post() returns a bounded, single-flight handle;
its CUDA completion must be observed before the channel or buffers are reused.
"""

from importlib.metadata import version

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


class PoolTransfer:
    """One posted transfer, retaining its payload until actual CUDA completion."""

    def __init__(
        self,
        transport: "PoolTransport",
        tensors: tuple[torch.Tensor, ...],
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        finished: torch.cuda.Event,
    ) -> None:
        self.transport = transport
        self.tensors = tensors
        self.start = start
        self.end = end
        self.finished = finished
        self.elapsed_ms: float | None = None

    def query(self) -> bool:
        if self.transport.failed:
            raise RuntimeError("Failed transfer requires deployment teardown")
        return self.elapsed_ms is not None or self.finished.query()

    def finish(self) -> float:
        """Reclaim a completed transfer without waiting on the host."""
        if self.elapsed_ms is not None:
            return self.elapsed_ms
        if (
            self.transport.failed
            or self.transport.pending is not self
            or not self.finished.query()
        ):
            raise RuntimeError(
                "Transfer is not complete; retain its channel and buffers"
            )
        torch.cuda.current_stream(self.transport.device).wait_event(self.finished)
        self.elapsed_ms = self.start.elapsed_time(self.end)
        self.transport.pending = None
        self.tensors = ()
        return self.elapsed_ms

    def wait(self) -> float:
        """Compatibility path with the original explicit host wait."""
        if self.elapsed_ms is None:
            self.finished.synchronize()
        return self.finish()


class PoolTransport:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        device: torch.device,
        timeout_s: int = 60,
        *,
        reuse_events: bool = False,
    ) -> None:
        if version("vllm") != "0.26.0" or rank not in (0, 1):
            raise ValueError("Requires pinned vLLM and a two-rank channel")
        if device.type != "cuda" or device.index is None:
            raise ValueError("Transport requires an explicit CUDA device index")
        self.device = device
        self.peer = 1 - rank
        self.group = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=rank,
            world_size=2,
            store_timeout=timeout_s,
        )
        self.communicator = PyNcclCommunicator(self.group, device=device)
        if not self.communicator.available or self.communicator.disabled:
            raise RuntimeError("NCCL P2P is unavailable; refusing a no-op transport")
        self.stream = torch.cuda.Stream(device=device)
        self.finished = torch.cuda.Event()
        # One outstanding handle owns these events. finish() observes actual
        # CUDA completion before another post may reuse the same channel.
        self.events = self._events() if reuse_events else None
        self.closed = False
        self.pending: PoolTransfer | None = None
        self.failed = False

    @staticmethod
    def _events() -> tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]:
        return (
            torch.cuda.Event(),
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def transfer(self, tensors: tuple[torch.Tensor, ...], *, send: bool) -> float:
        return self.post(tensors, send=send).wait()

    def post(self, tensors: tuple[torch.Tensor, ...], *, send: bool) -> PoolTransfer:
        if self.closed or self.failed or self.pending is not None:
            raise RuntimeError("Transport is closed, failed or already has a transfer")
        if any(t.device != self.device or not t.is_contiguous() for t in tensors):
            raise ValueError("NCCL payload must be contiguous on the channel device")
        producer, start, end = self.events or self._events()
        producer.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(producer)
        transfer = PoolTransfer(self, tensors, start, end, self.finished)
        self.pending = transfer
        try:
            with torch.cuda.stream(self.stream):
                start.record()
                self.communicator.group_start()
                try:
                    for tensor in tensors:
                        tensor.record_stream(self.stream)
                        if tensor.numel() == 0:
                            continue
                        if send:
                            self.communicator.send(
                                tensor, self.peer, stream=self.stream
                            )
                        else:
                            self.communicator.recv(
                                tensor, self.peer, stream=self.stream
                            )
                finally:
                    self.communicator.group_end()
                end.record()
                self.finished.record()
        except BaseException:
            # Posting may have partially succeeded. No channel or buffer reuse
            # is safe; the supervisor must tear down the deployment.
            self.failed = True
            raise
        return transfer

    def close(self) -> None:
        if not self.closed:
            if self.failed or (self.pending is not None and not self.pending.query()):
                raise RuntimeError("Undrained NCCL work requires deployment teardown")
            if self.pending is not None:
                self.pending.finish()
            self.communicator.destroy()
            self.closed = True
