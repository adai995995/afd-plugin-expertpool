# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Prebuilt Client/Worker NCCL P2P channels for the initial eager Pool.

Each pair initializes once. Runtime transfers involve only the granted pair.
The first implementation synchronizes each transfer phase on the host, so it
does not claim communication/compute overlap or NCCL fault recovery.
"""

from importlib.metadata import version

import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup


class PoolTransport:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        device: torch.device,
        timeout_s: int = 60,
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
        self.closed = False

    def transfer(self, tensors: tuple[torch.Tensor, ...], *, send: bool) -> float:
        if self.closed:
            raise RuntimeError("Transport is closed")
        if any(t.device != self.device or not t.is_contiguous() for t in tensors):
            raise ValueError("NCCL payload must be contiguous on the channel device")
        producer = torch.cuda.Event()
        producer.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(producer)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with torch.cuda.stream(self.stream):
            start.record()
            self.communicator.group_start()
            try:
                for tensor in tensors:
                    if tensor.numel() == 0:
                        continue
                    if send:
                        self.communicator.send(tensor, self.peer, stream=self.stream)
                    else:
                        self.communicator.recv(tensor, self.peer, stream=self.stream)
            finally:
                self.communicator.group_end()
            end.record()
            self.finished.record()
        self.finished.synchronize()
        torch.cuda.current_stream(self.device).wait_event(self.finished)
        return start.elapsed_time(end)

    def close(self) -> None:
        if not self.closed:
            self.communicator.destroy()
            self.closed = True
