# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU checks of identities, admission, fairness and safe credit reuse."""

import json
import multiprocessing
import unittest
from dataclasses import replace

from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import (
    MAX_CONTROL_BYTES,
    CallKey,
    CallRequest,
    Message,
    decode_message,
    encode_message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory, StaticScheduler


class RuntimeProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = StaticDirectory(
            "checkpoint-a",
            1,
            "worker-a",
            (ExpertPlacement(1, 4, (3, 1, 0, 2)), ExpertPlacement(13, 4, (0, 1, 2, 3))),
            32,
            2,
            64,
        )
        self.scheduler = StaticScheduler(self.directory, max_pending_per_domain=2)
        self.scheduler.register("a", 1, "domain-a")
        self.scheduler.register("b", 1, "domain-b")

    def request(self, client: str = "a", sequence: int = 0) -> CallRequest:
        return CallRequest(
            CallKey(client, 1, sequence), "checkpoint-a", 1, 1, 16, 32, 2
        )

    def test_independent_clients_can_use_same_sequence_and_different_layers(
        self,
    ) -> None:
        first = self.request()
        second = replace(self.request("b"), layer_id=13)
        self.scheduler.submit(first, 10)
        self.scheduler.submit(second, 20)
        plan = self.scheduler.grant()
        self.assertEqual(plan.request, first)
        self.assertIsNone(self.scheduler.grant())
        self.scheduler.complete(plan)
        self.assertEqual(self.scheduler.grant().request, second)

    def test_idle_client_never_gates_progress(self) -> None:
        for sequence in range(4):
            self.scheduler.submit(self.request("a", sequence), sequence)
            plan = self.scheduler.grant()
            self.assertEqual(plan.request.key.client_id, "a")
            self.scheduler.complete(plan)

    def test_service_domains_alternate_even_with_more_clients_in_one_domain(
        self,
    ) -> None:
        self.scheduler.register("a2", 1, "domain-a")
        for client in ("a", "a2", "b"):
            self.scheduler.submit(self.request(client), 1)
        order = []
        for _ in range(3):
            plan = self.scheduler.grant()
            order.append(plan.request.key.client_id)
            self.scheduler.complete(plan)
        self.assertEqual(order, ["a", "b", "a2"])

    def test_stale_or_duplicate_completion_cannot_release_a_new_slot(self) -> None:
        self.scheduler.submit(self.request(), 1)
        first = self.scheduler.grant()
        self.scheduler.complete(first)
        self.scheduler.submit(self.request("b"), 2)
        second = self.scheduler.grant()
        self.assertEqual(first.slot_id, second.slot_id)
        self.assertGreater(second.generation, first.generation)
        for stale in (
            first,
            replace(second, generation=first.generation),
            replace(second, worker_id="other"),
        ):
            with self.assertRaises(ValueError):
                self.scheduler.complete(stale)
            self.assertEqual(self.scheduler.active, second)
        self.scheduler.complete(second)
        with self.assertRaises(ValueError):
            self.scheduler.complete(second)

    def test_session_restart_requires_drain_and_rejects_old_epoch(self) -> None:
        self.scheduler.submit(self.request(), 1)
        with self.assertRaises(ValueError):
            self.scheduler.register("a", 2, "domain-a")
        self.scheduler.cancel_queued(self.request().key)
        self.scheduler.register("a", 2, "domain-a")
        with self.assertRaises(ValueError):
            self.scheduler.submit(self.request(), 2)
        fresh = replace(self.request(), key=CallKey("a", 2, 0))
        self.scheduler.submit(fresh, 3)
        self.assertEqual(self.scheduler.grant().request, fresh)

    def test_cancel_does_not_release_in_flight_buffer(self) -> None:
        self.scheduler.submit(self.request(), 1)
        self.scheduler.submit(self.request("b"), 2)
        plan = self.scheduler.grant()
        with self.assertRaises(ValueError):
            self.scheduler.cancel_queued(plan.request.key)
        self.scheduler.cancel_queued(self.request("b").key)
        self.assertEqual(self.scheduler.active, plan)
        self.scheduler.complete(plan)
        self.assertIsNone(self.scheduler.grant())

    def test_client_and_domain_admission_bounds(self) -> None:
        self.scheduler.register("a2", 1, "domain-a")
        self.scheduler.register("a3", 1, "domain-a")
        self.scheduler.submit(self.request(), 1)
        with self.assertRaises(ValueError):
            self.scheduler.submit(self.request(sequence=1), 2)
        self.scheduler.submit(self.request("a2"), 3)
        with self.assertRaises(ValueError):
            self.scheduler.submit(self.request("a3"), 4)
        self.assertNotIn("a3", self.scheduler.outstanding)

    def test_completed_or_cancelled_sequences_cannot_be_replayed(self) -> None:
        self.scheduler.submit(self.request(), 1)
        self.scheduler.complete(self.scheduler.grant())
        with self.assertRaises(ValueError):
            self.scheduler.submit(self.request(), 2)
        self.scheduler.submit(self.request(sequence=1), 3)
        self.scheduler.cancel_queued(self.request(sequence=1).key)
        with self.assertRaises(ValueError):
            self.scheduler.submit(self.request(sequence=1), 4)

    def test_invalid_request_never_consumes_credit(self) -> None:
        for changes in (
            {"model_id": "other"},
            {"placement_version": 2},
            {"layer_id": 2},
            {"num_tokens": 65},
            {"hidden_size": 64},
            {"top_k": 3},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.scheduler.submit(replace(self.request(), **changes), 1)
            self.assertIsNone(self.scheduler.grant())
            self.assertFalse(self.scheduler.outstanding)
        self.scheduler.submit(self.request(), 2)
        self.assertIsNotNone(self.scheduler.grant())

    def test_directory_rejects_missing_coverage_and_duplicate_layers(self) -> None:
        for placements in (
            (ExpertPlacement(1, 4, (0, 1)),),
            (self.directory.placements[0], self.directory.placements[0]),
        ):
            with self.assertRaises(ValueError):
                replace(self.directory, placements=placements)

    def test_wire_roundtrip_preserves_exact_identity_and_generation(self) -> None:
        self.scheduler.submit(self.request(), 1)
        plan = self.scheduler.grant()
        for message in (
            Message("submit", request=self.request()),
            Message("grant", plan=plan),
            Message("output_ready", plan=plan),
            Message(
                "done",
                plan=plan,
                metrics={"compute_gpu_ms": 0.25},
                digests={"hidden": "a" * 64},
            ),
            Message("error", request=self.request(), detail="capacity"),
            Message("error", plan=plan, detail="invalid payload"),
            Message("close"),
            Message("closed"),
        ):
            with self.subTest(kind=message.kind):
                self.assertEqual(decode_message(encode_message(message)), message)

    def test_malformed_messages_fail_before_dispatch(self) -> None:
        for payload in (
            b"null",
            b"[]",
            b"{}",
            b"not-json",
            b"x" * (MAX_CONTROL_BYTES + 1),
        ):
            with self.subTest(payload=payload[:10]), self.assertRaises(ValueError):
                decode_message(payload)
        valid = json.loads(encode_message(Message("submit", request=self.request())))
        valid["request"]["num_tokens"] = True
        with self.assertRaises(ValueError):
            decode_message(json.dumps(valid).encode())
        with self.assertRaises(ValueError):
            Message("submit", request=self.request(), metrics={"bad": float("nan")})

    def test_real_control_pipe_is_bounded_and_times_out(self) -> None:
        left, right = multiprocessing.Pipe()
        try:
            message = Message("submit", request=self.request())
            send_message(left, message)
            self.assertEqual(receive_message(right, 1), message)
            with self.assertRaises(TimeoutError):
                receive_message(right, 0.01)
        finally:
            left.close()
            right.close()


if __name__ == "__main__":
    unittest.main()
