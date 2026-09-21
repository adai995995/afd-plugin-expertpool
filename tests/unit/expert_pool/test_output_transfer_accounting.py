# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Cross-check compact payload reports without modeling GPU execution."""

import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from afd_plugin.expert_pool.directory import PoolDirectory
from afd_plugin.expert_pool.placement import ExpertPlacement
from afd_plugin.expert_pool.scheduler import StaticDirectory
from tools.expert_pool.validate_engines import main, verify_output_transfer


def reports(compact_output: bool, *, empty: bool = False):
    directory = PoolDirectory(
        tuple(
            StaticDirectory(
                "checkpoint",
                1,
                f"e{i}",
                (ExpertPlacement(1, 4, (2 * i, 2 * i + 1)),),
                8,
                2,
                64,
                allow_partial_experts=True,
            )
            for i in range(2)
        ),
        expert_partitioned=True,
    )

    def status(client, rows, counts, dense_rows):
        if empty:
            rows, counts, dense_rows = 0, [0, 0, 0, 0], (0, 0)
        channels = {}
        for i in range(2):
            dense = dense_rows[i] * 32
            actual = sum(counts[2 * i : 2 * i + 2]) * 16 if compact_output else dense
            channels[f"e{i}"] = {
                "output_transfer": {
                    "received_bytes": actual,
                    "dense_equivalent_bytes": dense,
                }
            }
        return {
            "client_id": client,
            "layers": {"1": {"token_rows": rows}},
            "dispatch": {
                "compact_output": compact_output,
                "demand_aware": True,
                "demand": {"expert_assignments": {"1": counts}},
                "workers": channels,
                "output_transfer": {
                    field: sum(
                        channel["output_transfer"][field]
                        for channel in channels.values()
                    )
                    for field in ("received_bytes", "dense_equivalent_bytes")
                },
            },
        }

    ready = [status(f"a{i}", 2, [1, 1, 1, 1], (2, 2)) for i in range(2)]
    # A0's next batch hits both workers; A1's next batch skips e0 entirely.
    after = [
        status("a0", 5, [4, 2, 2, 2], (5, 5)),
        status("a1", 6, [1, 1, 5, 5], (2, 6)),
    ]
    workers = [
        {
            "worker_id": f"e{i}",
            "compact_output": compact_output,
            "output_transfer": {
                target: sum(
                    item["dispatch"]["workers"][f"e{i}"]["output_transfer"][source]
                    for item in after
                )
                for target, source in (
                    ("sent_bytes", "received_bytes"),
                    ("dense_equivalent_bytes", "dense_equivalent_bytes"),
                )
            },
        }
        for i in range(2)
    ]
    return directory, ready, [{"status": [item]} for item in after], workers


class OutputTransferAccountingTests(unittest.TestCase):
    def test_compact_counts_unique_assignments_and_excludes_warmup_from_deltas(self):
        result = verify_output_transfer(*reports(True), compact_output=True)
        self.assertEqual(result["received_bytes"], 352)
        self.assertEqual(result["sent_bytes"], 352)
        self.assertEqual(result["dense_equivalent_bytes"], 576)
        self.assertEqual(result["avoided_output_bytes"], 224)
        self.assertEqual(result["request_received_bytes"], 224)
        self.assertEqual(result["request_dense_equivalent_bytes"], 320)
        self.assertEqual(result["by_worker"]["e0"]["received_bytes"], 128)
        self.assertEqual(result["by_worker"]["e1"]["received_bytes"], 224)

    def test_dense_mode_counts_only_selected_calls_without_claiming_savings(self):
        result = verify_output_transfer(*reports(False), compact_output=False)
        self.assertEqual(result["received_bytes"], 576)
        self.assertEqual(result["dense_equivalent_bytes"], 576)
        self.assertEqual(result["avoided_output_bytes"], 0)
        self.assertEqual(result["request_received_bytes"], 320)

    def test_all_empty_calls_transfer_no_output_bytes(self):
        for compact in (False, True):
            result = verify_output_transfer(
                *reports(compact, empty=True), compact_output=compact
            )
            self.assertEqual(result["received_bytes"], 0)
            self.assertEqual(result["sent_bytes"], 0)
            self.assertEqual(result["dense_equivalent_bytes"], 0)
            self.assertEqual(result["request_received_bytes"], 0)

    def test_matching_a_and_e_overtransmission_still_fails_assignment_accounting(self):
        directory, ready, closed, workers = reports(True)
        dispatch = closed[0]["status"][0]["dispatch"]
        dispatch["workers"]["e0"]["output_transfer"]["received_bytes"] += 16
        dispatch["output_transfer"]["received_bytes"] += 16
        workers[0]["output_transfer"]["sent_bytes"] += 16
        with self.assertRaisesRegex(AssertionError, "routed assignments"):
            verify_output_transfer(
                directory, ready, closed, workers, compact_output=True
            )

    def test_independent_e_report_must_match_received_bytes(self):
        directory, ready, closed, workers = reports(True)
        workers[0]["output_transfer"]["sent_bytes"] += 16
        with self.assertRaisesRegex(AssertionError, "E sent bytes"):
            verify_output_transfer(
                directory, ready, closed, workers, compact_output=True
            )

    def test_a_total_cannot_hide_missing_channel_bytes(self):
        directory, ready, closed, workers = reports(True)
        closed[0]["status"][0]["dispatch"]["output_transfer"]["received_bytes"] -= 16
        with self.assertRaisesRegex(AssertionError, "worker channels"):
            verify_output_transfer(
                directory, ready, closed, workers, compact_output=True
            )

    def test_dense_equivalent_cannot_include_more_than_all_candidate_rows(self):
        directory, ready, closed, workers = reports(True)
        dispatch = closed[1]["status"][0]["dispatch"]
        dispatch["workers"]["e0"]["output_transfer"]["dense_equivalent_bytes"] = 224
        with self.assertRaisesRegex(AssertionError, "routed assignments"):
            verify_output_transfer(
                directory, ready, closed, workers, compact_output=True
            )

    def test_mode_disagreement_and_noninteger_bytes_are_rejected(self):
        for location in ("a", "e", "count"):
            directory, ready, closed, workers = reports(True)
            if location == "a":
                closed[0]["status"][0]["dispatch"]["compact_output"] = False
            elif location == "e":
                workers[0]["compact_output"] = False
            else:
                closed[0]["status"][0]["dispatch"]["workers"]["e0"]["output_transfer"][
                    "received_bytes"
                ] = 96.0
            with self.subTest(location=location), self.assertRaises(AssertionError):
                verify_output_transfer(
                    directory, ready, closed, workers, compact_output=True
                )

    def test_regressed_counters_cannot_be_reported_as_negative_request_traffic(self):
        directory, ready, closed, workers = reports(True)
        with self.assertRaisesRegex(AssertionError, "regressed"):
            verify_output_transfer(
                directory,
                [item["status"][0] for item in closed],
                [{"status": [item]} for item in ready],
                workers,
                compact_output=True,
            )

    def test_compact_cli_requires_demand_before_any_model_or_gpu_access(self):
        args = [
            "validate_engines",
            "--model",
            "/checkpoint",
            "--gpus",
            "0",
            "1",
            "2",
            "3",
            "--output",
            "/private/report.json",
            "--lock-path",
            "/private/lock",
            "--compact-output",
        ]
        with (
            patch("sys.argv", args),
            redirect_stderr(io.StringIO()) as error,
            self.assertRaises(SystemExit) as stopped,
        ):
            main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("--compact-output requires --expert-demand", error.getvalue())


if __name__ == "__main__":
    unittest.main()
