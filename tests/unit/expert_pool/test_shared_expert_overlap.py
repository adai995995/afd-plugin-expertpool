# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA correctness and stream ordering for the A-side shared branch."""

import unittest
from collections import deque
from copy import deepcopy
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
class SharedExpertOverlapTests(unittest.TestCase):
    def test_all_moe_layers_reuse_one_shared_stream(self):
        from afd_plugin.model_executor.models.pool_deepseek_v2 import (
            PoolDeepseekV2ForCausalLM,
            PoolRemoteMoE,
        )

        model = PoolDeepseekV2ForCausalLM.__new__(PoolDeepseekV2ForCausalLM)
        torch.nn.Module.__init__(model)
        model.probe_weight = torch.nn.Parameter(torch.empty(1, device="cuda"))
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList()
        for layer_id in (1, 2):
            layer = torch.nn.Module()
            layer.layer_idx = layer_id
            layer.mlp = PoolRemoteMoE.__new__(PoolRemoteMoE)
            torch.nn.Module.__init__(layer.mlp)
            layer.mlp.layer_id = layer_id
            layer.mlp.client = None
            layer.mlp.shared_stream = None
            model.model.layers.append(layer)

        client = SimpleNamespace(
            directory=SimpleNamespace(
                placements=[SimpleNamespace(layer_id=i) for i in (1, 2)]
            )
        )
        model.bind_pool_client(client, shared_expert_overlap=True)
        left, right = (layer.mlp for layer in model.model.layers)
        self.assertIsNotNone(left.shared_stream)
        self.assertIs(left.shared_stream, right.shared_stream)
        self.assertIs(left.client, client)
        self.assertIs(right.client, client)

    def test_shared_branch_matches_serial_result_on_an_independent_stream(self):
        from afd_plugin.expert_pool.metrics import CallMetrics
        from afd_plugin.model_executor.models.pool_deepseek_v2 import PoolRemoteMoE

        class Gate(torch.nn.Module):
            def forward(self, hidden):
                return hidden, None

        class Shared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(16, 16, bias=False).to(
                    device="cuda", dtype=torch.bfloat16
                )
                self.stream = None
                self.done = None

            def forward(self, hidden):
                self.stream = torch.cuda.current_stream(hidden.device)
                output = self.projection(hidden)
                if self.done is not None:
                    self.done.record(self.stream)
                return output

        class Router:
            def select_experts(self, hidden, logits, *, topk_indices_dtype):
                rows = hidden.shape[0]
                return (
                    torch.ones(rows, 1, device=hidden.device, dtype=hidden.dtype),
                    torch.zeros(
                        rows, 1, device=hidden.device, dtype=topk_indices_dtype
                    ),
                )

        class Client:
            def __init__(self):
                self.stream = None
                self.calls = 0

            def execute(self, layer_id, hidden, weights, ids):
                self.stream = torch.cuda.current_stream(hidden.device)
                self.calls += 1
                return hidden * 2, None

        def make_moe(shared, overlap):
            moe = PoolRemoteMoE.__new__(PoolRemoteMoE)
            torch.nn.Module.__init__(moe)
            moe.layer_id = 1
            moe.client = Client()
            moe.completed_calls = 0
            moe.token_rows = 0
            moe.router_metrics = None
            moe.router_timing_events = None
            moe.router_gpu_unready = 0
            moe.shared_expert_overlap = overlap
            moe.shared_stream = None
            moe.shared_metrics = CallMetrics()
            moe.pending_shared_timings = deque()
            moe.shared_timing_dropped = 0
            moe.gate = Gate()
            moe.router = Router()
            moe.shared_experts = shared
            return moe

        torch.manual_seed(7)
        hidden = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        shared = Shared()
        serial = make_moe(shared, False)
        overlap = make_moe(deepcopy(shared), True)
        serial_output = serial(hidden)
        overlap_output = overlap(hidden)
        torch.cuda.synchronize()

        torch.testing.assert_close(overlap_output, serial_output, atol=0, rtol=0)
        self.assertNotEqual(
            overlap.shared_experts.stream.cuda_stream,
            overlap.client.stream.cuda_stream,
        )
        self.assertEqual(serial.shared_experts.stream, serial.client.stream)
        self.assertEqual(overlap.client.calls, 1)
        self.assertEqual(overlap.shared_metrics_snapshot()["calls"], 1)
        self.assertEqual(serial.shared_metrics_snapshot()["calls"], 1)
        self.assertEqual(overlap.shared_timing_dropped, 0)

        empty = overlap(hidden[:0])
        self.assertEqual(empty.shape, hidden[:0].shape)
        self.assertEqual(overlap.client.calls, 1)

        # The side stream must make progress while the host is blocked waiting
        # for the remote E path, rather than merely enqueue work for later.
        entered = Event()
        release = Event()

        class BlockingClient(Client):
            def execute(self, layer_id, hidden, weights, ids):
                self.stream = torch.cuda.current_stream(hidden.device)
                self.calls += 1
                entered.set()
                if not release.wait(30):
                    raise TimeoutError("fake remote execute did not resume")
                return hidden * 2, None

        blocked_shared = Shared()
        blocked_shared.load_state_dict(shared.state_dict())
        blocked = make_moe(blocked_shared, True)
        blocked.client = BlockingClient()
        blocked.shared_experts.done = torch.cuda.Event()
        worker_result = {}
        torch.cuda.synchronize()

        def run_forward():
            try:
                worker_result["output"] = blocked(hidden)
            except Exception as error:
                worker_result["error"] = error

        worker = Thread(target=run_forward)
        worker.start()
        try:
            self.assertTrue(entered.wait(20), "remote execute was not entered")
            deadline = monotonic() + 20
            while not blocked.shared_experts.done.query() and monotonic() < deadline:
                sleep(0.01)
            self.assertTrue(
                blocked.shared_experts.done.query(),
                "shared GPU work did not complete during the remote wait",
            )
            self.assertTrue(worker.is_alive(), "remote execute returned too early")
        finally:
            release.set()
            worker.join(timeout=30)
        self.assertFalse(worker.is_alive())
        if "error" in worker_result:
            raise worker_result["error"]
        torch.cuda.synchronize()
        torch.testing.assert_close(
            worker_result["output"], serial_output, atol=0, rtol=0
        )


if __name__ == "__main__":
    unittest.main()
