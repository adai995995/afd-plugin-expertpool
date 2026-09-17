# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU coverage, single-owner dispatch and failure containment contracts."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from afd_plugin.expert_pool.deployment import (
    ClientEndpoint,
    PoolDeployment,
    WorkerPlacement,
)
from afd_plugin.expert_pool.directory import PoolDirectory, ReplicaSelector
from afd_plugin.expert_pool.metrics import CallMetrics
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import CallKey, CallRequest, Message
from afd_plugin.expert_pool.replica_client import ReplicaPoolClient
from afd_plugin.expert_pool.scheduler import StaticDirectory, StaticScheduler


class RecordingChannel:
    """Replace CUDA transport only; use real admission and completion identities."""

    def __init__(self, directory: StaticDirectory) -> None:
        self.directory = directory
        self.client_id = "a0"
        self.session_epoch = 1
        self.transport = SimpleNamespace(device="test-device")
        self.sequence = 0
        self.failed = False
        self.closed = False
        self.metrics = None
        self.calls = []
        self.raise_after_dispatch = False
        self.raise_on_close = False
        self.scheduler = StaticScheduler(directory)
        self.scheduler.register(self.client_id, self.session_epoch, "domain-a")

    def execute(self, layer, hidden, weights, ids, *, call_seq):
        request = CallRequest(
            CallKey(self.client_id, self.session_epoch, call_seq),
            self.directory.model_id,
            self.directory.version,
            layer,
            hidden.shape[0],
            self.directory.hidden_size,
            self.directory.top_k,
        )
        self.scheduler.submit(request, 0)
        plan = self.scheduler.grant()
        self.calls.append((request, hidden, weights, ids))
        self.sequence = call_seq + 1
        if self.raise_after_dispatch:
            self.failed = True
            raise TimeoutError("Completion is uncertain")
        self.scheduler.complete(plan)
        completion = Message("done", plan=plan, metrics={"client_roundtrip_ms": 1.0})
        if self.metrics is not None:
            self.metrics.record(layer, hidden.shape[0], completion.metrics)
        return hidden, completion

    def set_metrics(self, enabled):
        self.metrics = CallMetrics() if enabled else None

    def close(self):
        self.closed = True
        if self.raise_on_close:
            raise RuntimeError("Close failed")


class MultiWorkerTests(unittest.TestCase):
    def directory(self) -> PoolDirectory:
        return PoolDirectory(
            (
                StaticDirectory(
                    "checkpoint",
                    3,
                    "e0",
                    (
                        ExpertPlacement(1, 4, (3, 1, 0, 2)),
                        ExpertPlacement(2, 4, (0, 1, 2, 3)),
                    ),
                    32,
                    2,
                    64,
                ),
                StaticDirectory(
                    "checkpoint",
                    3,
                    "e1",
                    (
                        ExpertPlacement(1, 4, (0, 1, 2, 3)),
                        ExpertPlacement(3, 4, (0, 1, 2, 3)),
                    ),
                    32,
                    2,
                    64,
                ),
            )
        )

    def deployment(self, model="/checkpoint") -> PoolDeployment:
        return PoolDeployment(
            model,
            "checkpoint",
            64,
            tuple(
                ClientEndpoint(
                    f"a{client}",
                    1,
                    f"domain-{client}",
                    f"/private/e{worker}-a{client}.sock",
                    24001 + worker * 2 + client,
                    f"e{worker}",
                )
                for worker in range(2)
                for client in range(2)
            ),
            workers=(WorkerPlacement("e0", (1, 2)), WorkerPlacement("e1", (1, 3))),
            placement_version=3,
        )

    def test_mapping_uses_physical_slots_and_selection_respects_residency(self):
        directory = self.directory()
        self.assertEqual(directory.locations(1, 3), (("e0", 0), ("e1", 3)))
        self.assertEqual(directory.locations(2, 3), (("e0", 3),))
        left, right = ReplicaSelector(directory), ReplicaSelector(directory, 1)
        for _ in range(20):
            self.assertNotEqual(left.select(1), right.select(1))
            self.assertEqual(left.select(2), "e0")
            self.assertEqual(left.select(3), "e1")
        for layer, expert in ((4, 0), (1, -1), (1, 4)):
            with self.assertRaises(ValueError):
                directory.locations(layer, expert)

    def test_mismatched_checkpoint_version_shape_and_replica_counts_rejected(self):
        first, second = self.directory().workers
        for change in (
            {"model_id": "different-weights"},
            {"version": 4},
            {"hidden_size": 64},
            {"top_k": 1},
            {"max_tokens": 128},
            {"worker_id": "e0"},
            {"placements": (ExpertPlacement(1, 3, (0, 1, 2)),)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                PoolDirectory((first, replace(second, **change)))

    def test_every_call_has_one_owner_and_global_sequence_without_payload_changes(self):
        channels = tuple(
            RecordingChannel(worker) for worker in self.directory().workers
        )
        client = ReplicaPoolClient(channels)
        hidden = SimpleNamespace(shape=(5, 32))
        weights, ids = [0.2, 0.8], [3, 0]
        client.set_metrics(True)
        for layer in (1, 2, 1, 3, 1, 1):
            output, completion = client.execute(layer, hidden, weights, ids)
            self.assertIs(output, hidden)
            self.assertEqual(completion.plan.request.layer_id, layer)
        calls = [call for channel in channels for call in channel.calls]
        self.assertEqual(sorted(call[0].key.call_seq for call in calls), list(range(6)))
        self.assertEqual(len(calls), 6)
        for _, actual_hidden, actual_weights, actual_ids in calls:
            self.assertIs(actual_hidden, hidden)
            self.assertIs(actual_weights, weights)
            self.assertIs(actual_ids, ids)
        self.assertEqual(client.metrics.calls, 6)
        self.assertEqual(client.metrics.token_rows, 30)
        self.assertEqual(sum(client.worker_calls.values()), 6)
        self.assertEqual(client.worker_layer_calls["e0"][1], 2)
        self.assertEqual(client.worker_layer_calls["e1"][1], 2)

    def test_uncertain_gpu_completion_poisoned_without_retry_on_another_copy(self):
        channels = tuple(
            RecordingChannel(worker) for worker in self.directory().workers
        )
        client = ReplicaPoolClient(channels)
        channels[0].raise_after_dispatch = True
        with self.assertRaises(TimeoutError):
            client.execute(1, SimpleNamespace(shape=(1, 32)), [], [])
        with self.assertRaises(RuntimeError):
            client.execute(1, SimpleNamespace(shape=(1, 32)), [], [])
        self.assertEqual(len(channels[0].calls), 1)
        self.assertEqual(len(channels[1].calls), 0)
        self.assertTrue(client.failed)

    def test_validation_failure_does_not_poison_or_consume_a_worker_credit(self):
        channels = tuple(
            RecordingChannel(worker) for worker in self.directory().workers
        )
        client = ReplicaPoolClient(channels)
        with self.assertRaises(ValueError):
            client.execute(4, SimpleNamespace(shape=(1, 32)), [], [])
        self.assertFalse(client.failed)
        self.assertEqual(sum(len(channel.calls) for channel in channels), 0)
        client.execute(1, SimpleNamespace(shape=(1, 32)), [], [])

    def test_close_attempts_every_channel_and_cannot_release_inflight_calls(self):
        channels = tuple(
            RecordingChannel(worker) for worker in self.directory().workers
        )
        client = ReplicaPoolClient(channels)
        client.lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                client.close()
            with self.assertRaises(RuntimeError):
                client.set_metrics(True)
        finally:
            client.lock.release()
        channels[0].raise_on_close = True
        with self.assertRaises(RuntimeError):
            client.close()
        self.assertTrue(all(channel.closed for channel in channels))
        client.close()
        with self.assertRaises(RuntimeError):
            client.execute(1, SimpleNamespace(shape=(1, 32)), [], [])

    def test_mixed_a_sessions_and_reused_channels_rejected(self):
        for field, value in (
            ("session_epoch", 2),
            ("client_id", "a1"),
            ("sequence", 1),
        ):
            channels = tuple(
                RecordingChannel(worker) for worker in self.directory().workers
            )
            setattr(channels[1], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                ReplicaPoolClient(channels)

    def test_deployment_roundtrip_coverage_and_order_independent_startup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "first_k_dense_replace": 1,
                        "num_hidden_layers": 4,
                        "moe_layer_freq": 1,
                        "n_routed_experts": 4,
                        "hidden_size": 32,
                        "num_experts_per_tok": 2,
                    }
                )
            )
            deployment = self.deployment(str(root))
            deployment = replace(
                deployment,
                clients=deployment.clients[::-1],
                workers=deployment.workers[::-1],
            )
            path = root / "deployment.json"
            path.write_text(json.dumps(asdict(deployment)))
            self.assertEqual(PoolDeployment.read(path), deployment)
            self.assertEqual(deployment.directory("e0").version, 3)
            self.assertEqual(
                [p.layer_id for p in deployment.pool_directory().placements], [1, 2, 3]
            )
            self.assertEqual(
                [e.worker_id for e in deployment.client_endpoints("a0")], ["e0", "e1"]
            )
            self.assertEqual(
                [e.client_id for e in deployment.worker_endpoints("e0")], ["a0", "a1"]
            )
            for method in (deployment.directory, lambda: deployment.endpoint("a0")):
                with self.assertRaises(ValueError):
                    method()
            for layers in ((0, 3), (4,), (1,)):
                with self.subTest(layers=layers), self.assertRaises(ValueError):
                    replace(
                        deployment,
                        workers=(
                            WorkerPlacement("e0", (1, 2)),
                            WorkerPlacement("e1", layers),
                        ),
                    ).pool_directory()

    def test_missing_duplicate_and_inconsistent_channels_are_rejected(self):
        deployment = self.deployment()
        for clients in (
            deployment.clients[:-1],
            deployment.clients + (deployment.clients[0],),
            (
                replace(deployment.clients[0], domain="wrong-domain"),
                *deployment.clients[1:],
            ),
            (replace(deployment.clients[0], session_epoch=2), *deployment.clients[1:]),
            (
                replace(deployment.clients[0], worker_id="absent"),
                *deployment.clients[1:],
            ),
            (
                replace(
                    deployment.clients[0], nccl_port=deployment.clients[1].nccl_port
                ),
                *deployment.clients[1:],
            ),
        ):
            with self.subTest(clients=clients), self.assertRaises(ValueError):
                replace(deployment, clients=clients)

    def test_legacy_deployment_still_resolves_single_worker(self):
        deployment = self.deployment()
        legacy = asdict(deployment)
        legacy.pop("workers")
        legacy.pop("placement_version")
        legacy["clients"] = legacy["clients"][:2]
        for endpoint in legacy["clients"]:
            endpoint.pop("worker_id")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "deployment.json"
            path.write_text(json.dumps(legacy))
            restored = PoolDeployment.read(path)
        self.assertEqual(restored.worker_ids, ("worker-0",))
        self.assertEqual(restored.endpoint("a0").worker_id, "worker-0")


if __name__ == "__main__":
    unittest.main()
