# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Control readiness, timeout fidelity and high-descriptor fallback contracts."""

import threading
import unittest
from multiprocessing import Pipe
from unittest.mock import patch

from afd_plugin.expert_pool.progress_wait import ProgressWaiter


class ProgressWaiterTests(unittest.TestCase):
    def setUp(self):
        self.reader, self.writer = Pipe(duplex=True)
        self.addCleanup(self.reader.close)
        self.addCleanup(self.writer.close)
        self.waiter = ProgressWaiter(self.reader)

    def test_idle_deadline_returns_without_inventing_readiness(self):
        self.assertFalse(self.waiter.wait(0.0002))
        self.assertFalse(self.waiter.wait(0))

    def test_ready_control_is_not_consumed_by_wait(self):
        self.writer.send_bytes(b"grant")
        self.assertTrue(self.waiter.wait(0.0002))
        self.assertTrue(self.waiter.wait(0))
        self.assertEqual(self.reader.recv_bytes(), b"grant")
        self.assertFalse(self.waiter.wait(0))

    def test_control_arriving_during_wait_wakes_worker(self):
        sender = threading.Thread(target=self.writer.send_bytes, args=(b"grant",))
        sender.start()
        try:
            self.assertTrue(self.waiter.wait(2.0))
            self.assertEqual(self.reader.recv_bytes(), b"grant")
        finally:
            sender.join(2.0)
        self.assertFalse(sender.is_alive())

    def test_peer_close_is_readable_so_protocol_can_handle_eof(self):
        self.writer.close()
        self.assertTrue(self.waiter.wait(0.5))
        with self.assertRaises(EOFError):
            self.reader.recv_bytes()

    def test_submillisecond_deadline_reaches_select_unchanged(self):
        # Assert the timeout contract, not a flaky wall-clock timing limit.
        with patch(
            "afd_plugin.expert_pool.progress_wait.select.select",
            return_value=([], [], []),
        ) as select_call:
            self.assertFalse(self.waiter.wait(0.0002))
        self.assertEqual(select_call.call_args.args[-1], 0.0002)

    def test_unsupported_descriptor_falls_back_once_and_preserves_messages(self):
        with patch(
            "afd_plugin.expert_pool.progress_wait.select.select",
            side_effect=ValueError("filedescriptor out of range in select()"),
        ) as select_call:
            self.assertFalse(self.waiter.wait(0))
            self.writer.send_bytes(b"done")
            self.assertTrue(self.waiter.wait(0.0002))
            self.assertEqual(self.reader.recv_bytes(), b"done")
        self.assertEqual(select_call.call_count, 1)
        self.assertEqual(self.waiter.backend, "poll")

    def test_closed_local_connection_is_an_error(self):
        self.reader.close()
        with self.assertRaises((ValueError, OSError)):
            self.waiter.wait(0.0002)


if __name__ == "__main__":
    unittest.main()
