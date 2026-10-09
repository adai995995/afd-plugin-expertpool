# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Wire layout and deployment checks independent of CUDA or model weights."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from afd_plugin.expert_pool.batch_audit import BatchAudit
from afd_plugin.expert_pool.batching import BatchingOptions
from afd_plugin.expert_pool.deployment import PoolDeployment
from afd_plugin.expert_pool.protocol import (
    BatchExecution,
    BatchSlot,
    CallKey,
    CallRequest,
    ExecutionPlan,
    ExpertDemand,
    Message,
    decode_message,
    encode_message,
)
from tools.expert_pool.prepare_shared_pool import (
    build_deployment,
    with_tcp_endpoints,
    write_private_deployment,
)


class PartialContractTests(unittest.TestCase):
    def request(self):
        return CallRequest(
            CallKey("a", 1, 7),
            "checkpoint",
            1,
            1,
            3,
            32,
            2,
            ExpertDemand((2, 1, 2, 1), 0),
            True,
            True,
        )

    def test_wire_roundtrip_binds_fp32_partial_to_exact_input_rows(self):
        request = self.request()
        plan = ExecutionPlan(request, "e", 14, 0, 8, (0, 2), 4, 2)
        message = Message("execute", plan=plan)
        self.assertEqual(decode_message(encode_message(message)), message)
        for rows in (None, 0, 1, 4):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                replace(plan, input_rows=rows)

    def test_unnegotiated_output_layout_is_rejected(self):
        for changes in ({"compact_output": False}, {"partial_reduction": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.request(), **changes)

    def test_audit_records_packed_execution_rows_and_original_parent_shape(self):
        first = ExecutionPlan(self.request(), "e", 14, 0, 8, (0, 2), 4, 2)
        second = ExecutionPlan(
            replace(self.request(), key=CallKey("b", 1, 9)),
            "e",
            19,
            1,
            2,
            (0, 1, 2, 3),
            6,
            3,
        )
        batch = BatchExecution(
            "e", 1, tuple(BatchSlot.from_plan(p) for p in (first, second))
        )
        audit = BatchAudit()
        audit.record_batch(batch, (first, second))
        record = audit.snapshot()["records"][0]
        self.assertEqual(record["token_rows"], 5)
        self.assertEqual([m["num_tokens"] for m in record["members"]], [3, 3])
        self.assertEqual(
            [m["input_row_range"] for m in record["members"]], [[0, 2], [2, 5]]
        )

    def test_startup_builds_direct_pool_without_a_controller(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v2",
                        "n_routed_experts": 4,
                        "first_k_dense_replace": 1,
                        "num_hidden_layers": 3,
                        "moe_layer_freq": 1,
                        "hidden_size": 32,
                        "num_experts_per_tok": 2,
                    }
                )
            )
            deployment = build_deployment(
                root,
                root,
                2,
                2,
                16,
                30,
                replicated_experts=(0, 1, 2, 3),
                batching=BatchingOptions(2, 32, 0),
                partial_reduction=True,
            )
            self.assertIsNone(deployment.controller)
            self.assertFalse(deployment.pooled_admission)
            self.assertTrue(deployment.direct_dispatch)
            self.assertTrue(deployment.packed_input)
            self.assertTrue(deployment.execution.audit_batch_members)
            self.assertEqual(deployment.active_expert_replicas, 1)
            split = replace(deployment, active_expert_replicas=2)
            self.assertEqual(split.active_expert_replicas, 2)
            remote = with_tcp_endpoints(
                deployment,
                ("127.0.0.1", "127.0.0.2"),
                None,
                48000,
                48100,
            )
            self.assertIsNone(remote.controller)
            path = root / "deployment.json"
            write_private_deployment(path, remote)
            self.assertEqual(PoolDeployment.read(path), remote)
            old = json.loads(path.read_text())
            del old["active_expert_replicas"]
            path.write_text(json.dumps(old))
            self.assertEqual(PoolDeployment.read(path).active_expert_replicas, 1)
            for changes in (
                {"packed_input": False},
                {"direct_dispatch": False},
                {"partial_reduction": 1},
                {"active_expert_replicas": 0},
                {"active_expert_replicas": 3},
                {"active_expert_replicas": True},
                {"active_expert_replicas": 2, "partial_reduction": False},
                {"active_expert_replicas": 2, "expert_replicated": False},
                {"batching": BatchingOptions(2, 32, 100)},
                {
                    "execution": replace(
                        deployment.execution, validate_client_values=True
                    )
                },
            ):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    replace(deployment, **changes)
