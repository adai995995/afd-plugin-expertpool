# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Exercise fan-out control routing over real bounded connection messages."""

import multiprocessing
import threading
import unittest

from afd_plugin.expert_pool.controller import (
    ControllerClientIdentity,
    directory_digest,
)
from afd_plugin.expert_pool.controller_service import ControllerRuntime
from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.fanout_controller import FanoutControllerLedger
from afd_plugin.expert_pool.fanout_protocol import FanoutReplies
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.protocol import (
    CallKey,
    CallRequest,
    Message,
    receive_message,
    send_message,
)
from afd_plugin.expert_pool.scheduler import StaticDirectory


class FanoutRuntimeTests(unittest.TestCase):
    def setUp(self):
        resident = PoolDirectory(
            tuple(
                StaticDirectory(
                    "checkpoint",
                    1,
                    f"e{i}",
                    (ExpertPlacement(1, 4, (2 * i, 2 * i + 1)),),
                    32,
                    2,
                    64,
                    allow_partial_experts=True,
                )
                for i in range(2)
            ),
            expert_partitioned=True,
        )
        self.book = FanoutControllerLedger(
            resident,
            tuple(
                ControllerClientIdentity(f"a{i}", 1, f"domain-{i}") for i in range(2)
            ),
        )
        self.pairs = {
            identity: multiprocessing.Pipe() for identity in ("a0", "a1", "e0", "e1")
        }
        self.runtime = ControllerRuntime(
            self.book,
            {identity: self.pairs[identity][0] for identity in ("a0", "a1")},
            {identity: self.pairs[identity][0] for identity in ("e0", "e1")},
        )
        self.outcome = []

        def run():
            try:
                self.outcome.append(self.runtime.run())
            except BaseException as error:
                self.outcome.append(error)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup_runtime)
        for worker_id, directory in self.book.directories.items():
            self.send(worker_id, Message("ready", detail=directory_digest(directory)))

    def cleanup_runtime(self):
        for pair in self.pairs.values():
            for connection in pair:
                connection.close()
        self.thread.join(2)

    def send(self, identity, message):
        send_message(self.pairs[identity][1], message)

    def receive(self, identity):
        return receive_message(self.pairs[identity][1], 2)

    def submit(self, client_id):
        # Empty transfers can complete before another worker acknowledges its
        # grant; this is a valid protocol case, not a GPU performance model.
        request = CallRequest(CallKey(client_id, 1, 0), "checkpoint", 1, 1, 0, 32, 2)
        self.send(client_id, Message("submit", request=request))
        return request

    def child_plans(self, request):
        plans = {}
        for worker_id in ("e0", "e1"):
            grant = self.receive(worker_id)
            self.assertEqual(grant.kind, "grant")
            self.assertEqual(grant.plan.request, request)
            self.assertEqual(grant.plan.worker_id, worker_id)
            plans[worker_id] = grant.plan
        return plans

    def finish_child(self, plan):
        for kind in ("grant", "input_ready", "executing", "output_ready", "done"):
            self.send(plan.worker_id, Message(kind, plan=plan))

    def receive_events(self, client_id, replies, count):
        for _ in range(count):
            replies.accept(self.receive(client_id))

    def test_two_clients_interleaved_children_drain_and_close_cleanly(self):
        first_request = self.submit("a0")
        first = self.child_plans(first_request)
        second_request = self.submit("a1")
        first_replies = FanoutReplies(first_request, ("e0", "e1"))

        # The second worker completes before the first worker acknowledges.
        self.finish_child(first["e1"])
        self.receive_events("a0", first_replies, 3)
        self.assertEqual(first_replies.next_phase, {"e0": 0, "e1": 3})
        self.assertEqual(self.book.workers["e1"].phase, "held")
        self.assertEqual(self.book.snapshot()["completed_parent_calls"], 0)
        self.assertEqual(self.book.snapshot()["completed_child_calls"], 1)
        self.assertEqual(set(self.book.outstanding), {"a0", "a1"})
        self.assertEqual(self.book.pending_count, 1)
        self.assertFalse(self.pairs["e0"][1].poll(0))
        self.assertFalse(self.pairs["e1"][1].poll(0))

        self.finish_child(first["e0"])
        self.receive_events("a0", first_replies, 3)
        for kind in ("grant", "output_ready", "done"):
            for worker_id in ("e0", "e1"):
                self.assertEqual(
                    first_replies.take(kind, worker_id).plan, first[worker_id]
                )
        second = self.child_plans(second_request)
        self.assertEqual([plan.generation for plan in second.values()], [2, 2])
        self.send("a0", Message("close"))
        self.assertEqual(self.receive("a0").kind, "closed")

        # Cross-worker event order differs from both publication and completion
        # order. Per-worker ordering remains strict on each connection.
        phases = (
            ("e0", "grant"),
            ("e1", "grant"),
            ("e0", "input_ready"),
            ("e0", "executing"),
            ("e0", "output_ready"),
            ("e1", "input_ready"),
            ("e1", "executing"),
            ("e1", "output_ready"),
            ("e1", "done"),
            ("e0", "done"),
        )
        for worker_id, kind in phases:
            self.send(worker_id, Message(kind, plan=second[worker_id]))
        second_replies = FanoutReplies(second_request, ("e0", "e1"))
        self.receive_events("a1", second_replies, 6)
        self.assertEqual(second_replies.next_phase, {"e0": 3, "e1": 3})
        self.send("a1", Message("close"))
        self.assertEqual(self.receive("a1").kind, "closed")
        for worker_id in ("e0", "e1"):
            self.assertEqual(self.receive(worker_id).kind, "close")
            self.send(worker_id, Message("closed"))
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())
        result = self.outcome[0]
        self.assertIsInstance(result, dict)
        self.assertEqual(result["completed_parent_calls"], 2)
        self.assertEqual(result["completed_child_calls"], 4)
        self.assertEqual(result["completed_by_client"], {"a0": 1, "a1": 1})
        self.assertEqual(result["closed_clients"], 2)
        for counter in (
            "pending",
            "outstanding",
            "active_parent_calls",
            "pending_child_plans",
        ):
            self.assertEqual(result[counter], 0)
        for worker in result["workers"].values():
            self.assertEqual(worker["phase"], "closed")
            self.assertIsNone(worker["active"])
            self.assertEqual(worker["completed_calls"], 2)
            self.assertEqual(worker["client_calls"], {"a0": 1, "a1": 1})
            self.assertEqual(worker["layer_calls"], {"1": 2})

    def test_early_client_close_fails_without_releasing_partial_parent(self):
        request = self.submit("a0")
        plans = self.child_plans(request)
        self.finish_child(plans["e1"])
        self.receive_events("a0", FanoutReplies(request, ("e0", "e1")), 3)
        self.send("a0", Message("close"))
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())
        self.assertIsInstance(self.outcome[0], ValueError)
        self.assertNotIn("a0", self.book.closed)
        self.assertEqual(self.book.outstanding["a0"], request.key)
        self.assertEqual(self.book.workers["e1"].phase, "held")
        self.assertTrue(
            all(self.book.workers[key].active == plan for key, plan in plans.items())
        )
        self.assertEqual(self.book.snapshot()["completed_parent_calls"], 0)

    def test_child_error_terminates_runtime_and_preserves_gang_and_pending_parent(self):
        request = self.submit("a0")
        plans = self.child_plans(request)
        self.submit("a1")
        for kind in ("grant", "input_ready", "executing"):
            self.send("e0", Message(kind, plan=plans["e0"]))
        self.assertEqual(self.receive("a0").kind, "grant")
        self.send("e0", Message("error", plan=plans["e0"], detail="rejected input"))
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())
        self.assertIsInstance(self.outcome[0], RuntimeError)
        self.assertEqual(set(self.book.outstanding), {"a0", "a1"})
        self.assertEqual(self.book.pending_count, 1)
        self.assertTrue(
            all(self.book.workers[key].active == plan for key, plan in plans.items())
        )
        self.assertEqual(self.book.workers["e0"].phase, "executing")
        self.assertEqual(self.book.snapshot()["completed_parent_calls"], 0)
        self.assertEqual(self.book.snapshot()["completed_child_calls"], 0)


if __name__ == "__main__":
    unittest.main()
