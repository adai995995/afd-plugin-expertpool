# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Diagnostic wrappers must observe routing and restore all production seams."""

import unittest

import torch
from torch import nn

from afd_plugin.expert_pool.replica_client import ReplicaPoolClient
from afd_plugin.model_executor.models.pool_deepseek_v2 import PoolRemoteMoE
from tools.expert_pool.moe_probe import MoEProbe


class Router:
    def select_experts(
        self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None
    ):
        return torch.ones(hidden_states.shape[0], 1), torch.zeros(
            hidden_states.shape[0], 1, dtype=torch.int32
        )


class Gate(nn.Module):
    def forward(self, hidden):
        return hidden[:, :2], None


class Delegate(ReplicaPoolClient):
    def __init__(self):
        self.calls = 0

    def execute(self, layer_id, hidden_states, topk_weights, topk_ids):
        self.calls += 1
        return hidden_states * 2, None


class ProbeMoE(PoolRemoteMoE):
    def __init__(self):
        nn.Module.__init__(self)
        self.gate = Gate()
        self.router = Router()
        self.shared_experts = nn.Identity()
        self.client = Delegate()

    def forward(self, hidden_states):
        logits, _ = self.gate(hidden_states)
        weights, ids = self.router.select_experts(hidden_states, logits, torch.int32)
        routed, _ = self.client.execute(1, hidden_states, weights, ids)
        return routed + self.shared_experts(hidden_states)


class ProbeTests(unittest.TestCase):
    def model(self):
        return nn.ModuleDict(
            {
                "layers": nn.ModuleList(
                    [
                        nn.ModuleDict({"mlp": ProbeMoE()}),
                        nn.ModuleDict({"mlp": ProbeMoE()}),
                    ]
                )
            }
        )

    def test_capture_exact_routing_and_owned_return_then_restore(self):
        model = self.model()
        layer = model["layers"][1]["mlp"]
        original_client, original_router = layer.client, layer.router.select_experts
        probe = MoEProbe(model)
        hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        output = layer(hidden)
        self.assertTrue(torch.equal(output, hidden * 3))
        hidden.zero_()
        record = probe.snapshot()["layers"]["1"]
        self.assertEqual(record["num_rows"], 3)
        self.assertEqual(record["hidden_states"], [[8, 9, 10, 11]])
        self.assertEqual(record["routed_output"], [[16, 18, 20, 22]])
        self.assertEqual(record["expert_ids"], [[0]])
        probe.close()
        probe.close()
        self.assertIs(layer.client, original_client)
        self.assertEqual(layer.router.select_experts, original_router)
        self.assertFalse(layer._forward_hooks)
        self.assertFalse(layer._forward_pre_hooks)
        self.assertEqual(original_client.calls, 1)

    def test_partial_install_failure_restores_previous_hooks(self):
        model = self.model()
        first, second = (layer["mlp"] for layer in model["layers"])
        original_client, original_router = first.client, first.router.select_experts
        second.client = None
        with self.assertRaises(ValueError):
            MoEProbe(model)
        self.assertIs(first.client, original_client)
        self.assertEqual(first.router.select_experts, original_router)
        self.assertFalse(first._forward_hooks)
        self.assertFalse(second._forward_hooks)


if __name__ == "__main__":
    unittest.main()
