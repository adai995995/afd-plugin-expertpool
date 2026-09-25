# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The standalone manifest must bind independent A clients to one E pool."""

import json
import stat
import tempfile
import unittest
from pathlib import Path

from afd_plugin.expert_pool.deployment import PoolDeployment
from tools.expert_pool.prepare_shared_pool import (
    build_deployment,
    with_tcp_endpoints,
    write_private_deployment,
)


class SharedPoolPrepareTests(unittest.TestCase):
    def test_two_a_clients_share_the_same_e_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v2",
                        "n_routed_experts": 4,
                        "first_k_dense_replace": 1,
                        "num_hidden_layers": 3,
                        "moe_layer_freq": 1,
                        "hidden_size": 8,
                        "num_experts_per_tok": 2,
                    }
                )
            )
            socket_dir = root / "sockets"
            socket_dir.mkdir(mode=0o700)
            deployment = build_deployment(model, socket_dir, 2, 2, 16, 30)
            self.assertTrue(deployment.pooled_admission)
            self.assertEqual(deployment.client_ids, ("client-0", "client-1"))
            self.assertEqual(deployment.worker_ids, ("worker-0", "worker-1"))
            self.assertEqual(len(deployment.clients), 4)
            self.assertEqual(
                deployment.directory("worker-0").placements[0].expert_ids,
                (0, 2),
            )
            self.assertEqual(
                deployment.directory("worker-1").placements[0].expert_ids,
                (1, 3),
            )
            for worker_id in deployment.worker_ids:
                self.assertEqual(
                    {
                        endpoint.client_id
                        for endpoint in deployment.worker_endpoints(worker_id)
                    },
                    set(deployment.client_ids),
                )
            path = root / "deployment.json"
            write_private_deployment(path, deployment)
            self.assertEqual(PoolDeployment.read(path), deployment)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                write_private_deployment(path, deployment)

    def test_refuses_non_expert_model_and_empty_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "other", "n_routed_experts": 2})
            )
            with self.assertRaises(ValueError):
                build_deployment(model, root, 2, 1, 16, 30)
            (model / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v2", "n_routed_experts": 2})
            )
            with self.assertRaises(ValueError):
                build_deployment(model, root, 2, 3, 16, 30)

    def test_fixed_duplicate_expert_enables_assignment_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v2",
                        "n_routed_experts": 4,
                        "first_k_dense_replace": 1,
                        "num_hidden_layers": 3,
                        "moe_layer_freq": 1,
                        "hidden_size": 8,
                        "num_experts_per_tok": 2,
                    }
                )
            )
            deployment = build_deployment(
                model, root, 2, 2, 16, 30,
                replicated_experts=(0,), split_assignments=True,
            )
            self.assertTrue(deployment.expert_replicated)
            self.assertTrue(deployment.split_assignments)
            self.assertEqual(
                deployment.pool_directory().locations(1, 0),
                (("worker-0", 0), ("worker-1", 0)),
            )
            for replicas in ((), (4,), (0, 0)):
                with self.subTest(replicas=replicas), self.assertRaises(ValueError):
                    build_deployment(
                        model, root, 2, 2, 16, 30,
                        replicated_experts=replicas, split_assignments=True,
                    )

    def test_cross_host_manifest_uses_authenticated_unique_tcp_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "deepseek_v2",
                        "n_routed_experts": 4,
                        "first_k_dense_replace": 1,
                        "num_hidden_layers": 3,
                        "moe_layer_freq": 1,
                        "hidden_size": 8,
                        "num_experts_per_tok": 2,
                    }
                )
            )
            base = build_deployment(model, root, 2, 2, 16, 30)
            deployment = with_tcp_endpoints(
                base, ("127.0.0.1", "127.0.0.2"), "127.0.0.3", 47000, 47100
            )
            self.assertEqual(len(deployment.control_authkey), 32)
            self.assertEqual(
                [
                    (e.control_address(), e.nccl_host, e.nccl_port)
                    for e in deployment.clients
                ],
                [
                    (("127.0.0.1", 47000), "127.0.0.1", 47100),
                    (("127.0.0.1", 47001), "127.0.0.1", 47101),
                    (("127.0.0.2", 47002), "127.0.0.2", 47102),
                    (("127.0.0.2", 47003), "127.0.0.2", 47103),
                ],
            )
            self.assertEqual(
                deployment.controller_address("client", "client-0"),
                (("127.0.0.3", 47004), "AF_INET"),
            )
            self.assertEqual(
                deployment.controller_address("worker", "worker-1"),
                (("127.0.0.3", 47007), "AF_INET"),
            )
            path = root / "tcp-deployment.json"
            write_private_deployment(path, deployment)
            self.assertEqual(PoolDeployment.read(path), deployment)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with self.assertRaises(ValueError):
                with_tcp_endpoints(base, ("127.0.0.1",), "127.0.0.3", 47000, 47100)
            with self.assertRaises(ValueError):
                with_tcp_endpoints(
                    base, ("127.0.0.1", "127.0.0.2"), "127.0.0.3", 47000, 47001
                )


if __name__ == "__main__":
    unittest.main()
