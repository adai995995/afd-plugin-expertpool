# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host-side transfer ownership tests; actual NCCL is checked on real GPUs."""

import importlib.util
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class Event:
    def __init__(self):
        self.complete = False
        self.synchronizations = 0
        self.elapsed = 2.5

    def record(self, *args):
        self.complete = False

    def query(self):
        return self.complete

    def synchronize(self):
        self.synchronizations += 1
        self.complete = True

    def elapsed_time(self, other):
        return self.elapsed


class Stream:
    def __init__(self):
        self.dependencies = []

    def wait_event(self, event):
        self.dependencies.append(event)


class Tensor:
    device = "device"

    def __init__(self, contiguous=True):
        self.contiguous = contiguous
        self.recorded = []

    def is_contiguous(self):
        return self.contiguous

    def record_stream(self, stream):
        self.recorded.append(stream)

    def numel(self):
        return 4


class Communicator:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.destroyed = False

    def group_start(self):
        self.calls.append("start")

    def group_end(self):
        self.calls.append("end")

    def send(self, tensor, peer, *, stream):
        self.calls.append("send")
        if self.fail:
            raise RuntimeError("injected post failure")

    def recv(self, tensor, peer, *, stream):
        self.calls.append("recv")

    def destroy(self):
        self.destroyed = True


def load_transport(stream):
    # Keep these GPU-library substitutes local to loading this module. No
    # fake kernel execution or numerical result is used in these CPU tests.
    torch = ModuleType("torch")
    torch.Tensor = Tensor
    torch.device = str
    torch.cuda = SimpleNamespace(
        Event=Event,
        Stream=Stream,
        current_stream=lambda device: stream,
        stream=lambda device_stream: nullcontext(),
    )
    pynccl = ModuleType("vllm.distributed.device_communicators.pynccl")
    pynccl.PyNcclCommunicator = Communicator
    utils = ModuleType("vllm.distributed.utils")
    utils.StatelessProcessGroup = SimpleNamespace
    path = Path(__file__).resolve().parents[3] / "afd_plugin/connectors/gpu/pool.py"
    spec = importlib.util.spec_from_file_location("_transfer_contract_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        "sys.modules",
        {
            "torch": torch,
            "vllm.distributed.device_communicators.pynccl": pynccl,
            "vllm.distributed.utils": utils,
        },
    ):
        spec.loader.exec_module(module)
    return module.PoolTransport


class AsyncTransferTests(unittest.TestCase):
    def setUp(self):
        self.caller = Stream()
        transport_type = load_transport(self.caller)
        self.transport = transport_type.__new__(transport_type)
        self.transport.device = "device"
        self.transport.peer = 1
        self.transport.stream = Stream()
        self.transport.events = (Event(), Event(), Event())
        self.transport.finished = Event()
        self.transport.communicator = Communicator()
        self.transport.closed = False
        self.transport.failed = False
        self.transport.pending = None

    def test_post_retains_payload_without_host_wait_or_reuse(self):
        tensor = Tensor()
        ticket = self.transport.post((tensor,), send=False)
        self.assertIs(self.transport.pending, ticket)
        self.assertEqual(ticket.tensors, (tensor,))
        self.assertEqual(tensor.recorded, [self.transport.stream])
        self.assertFalse(ticket.query())
        self.assertEqual(ticket.finished.synchronizations, 0)
        with self.assertRaises(RuntimeError):
            self.transport.post((Tensor(),), send=True)
        with self.assertRaises(RuntimeError):
            ticket.finish()
        self.assertIs(self.transport.pending, ticket)
        with self.assertRaises(RuntimeError):
            self.transport.close()
        self.assertFalse(self.transport.communicator.destroyed)

    def test_completion_releases_channel_and_old_handle_survives_event_reuse(self):
        first = self.transport.post((Tensor(),), send=False)
        first.finished.complete = True
        self.assertEqual(first.finish(), 2.5)
        self.assertIsNone(self.transport.pending)
        self.assertEqual(first.tensors, ())
        second = self.transport.post((Tensor(),), send=True)
        second.start.elapsed = 7.0
        self.assertFalse(second.query())
        self.assertTrue(first.query())
        self.assertEqual(first.finish(), 2.5)
        self.assertIs(self.transport.pending, second)
        second.finished.complete = True
        self.assertEqual(second.finish(), 7.0)
        self.transport.close()
        self.assertTrue(self.transport.communicator.destroyed)

    def test_partial_post_failure_poisoned_channel_retains_ownership(self):
        self.transport.communicator.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.transport.post((Tensor(),), send=True)
        self.assertTrue(self.transport.failed)
        ticket = self.transport.pending
        self.assertIsNotNone(ticket)
        ticket.finished.complete = True
        for operation in (ticket.query, ticket.finish, self.transport.close):
            with self.assertRaises(RuntimeError):
                operation()
        with self.assertRaises(RuntimeError):
            self.transport.post((Tensor(),), send=False)
        self.assertTrue(ticket.tensors)
        self.assertEqual(self.transport.communicator.calls, ["start", "send", "end"])

    def test_preflight_rejection_does_not_poison_channel(self):
        with self.assertRaises(ValueError):
            self.transport.post((Tensor(contiguous=False),), send=False)
        self.assertIsNone(self.transport.pending)
        self.assertFalse(self.transport.failed)
        self.assertEqual(self.transport.transfer((Tensor(),), send=True), 2.5)
        self.assertEqual(self.transport.finished.synchronizations, 1)
        self.assertIsNone(self.transport.pending)


if __name__ == "__main__":
    unittest.main()
