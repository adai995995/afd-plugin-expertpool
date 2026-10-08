# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""First divergence must be located before comparing different generated paths."""

import io
import unittest
from copy import deepcopy
from unittest.mock import patch

from tools.expert_pool.benchmark_http import fetch
from tools.expert_pool.compare_outputs import (
    compare_reports,
    distribution,
    first_difference,
)


class OutputDiagnosticsTests(unittest.TestCase):
    def report(self) -> dict:
        return {
            "trace_sha256": "same-requests",
            "records": [
                {
                    "request_id": "d0-0",
                    "domain": 0,
                    "length_class": 6,
                    "status": "completed",
                    "token_ids": [10, 11, 12],
                }
            ],
        }

    def test_first_difference_handles_prefix_and_complete_identity(self):
        self.assertIsNone(first_difference([1, 2], [1, 2]))
        self.assertEqual(first_difference([1, 2, 3], [1, 4, 5]), 1)
        self.assertEqual(first_difference([1], [1, 2]), 1)
        self.assertEqual(first_difference([], [1]), 0)

    def test_compare_only_probabilities_with_the_same_generated_prefix(self):
        reference = self.report()
        candidate = deepcopy(reference)
        candidate["records"][0]["token_ids"] = [10, 21, 22]
        candidate["records"][0]["top_logprobs"] = [
            [
                {"token_id": token, "logprob": -0.3},
                {"token_id": token + 1, "logprob": -0.4},
            ]
            for token in (10, 21, 22)
        ]
        result = compare_reports(reference, candidate)
        difference = result["differences"][0]
        self.assertEqual(difference["first_difference_step"], 2)
        self.assertEqual(difference["common_generated_prefix"], [10])
        self.assertEqual(difference["reference_token"], 11)
        self.assertEqual(difference["candidate_token"], 21)
        self.assertAlmostEqual(
            difference["candidate_distribution"]["top_two_margin"], 0.1
        )
        self.assertFalse(result["all_sequences_identical"])

    def test_comparison_rejects_missing_tokens_and_request_coverage(self):
        for mutate in (
            lambda r: r.update(trace_sha256="different"),
            lambda r: r["records"].clear(),
            lambda r: r["records"].append(deepcopy(r["records"][0])),
            lambda r: r["records"][0].pop("token_ids"),
        ):
            candidate = deepcopy(self.report())
            mutate(candidate)
            with self.assertRaises(ValueError):
                compare_reports(self.report(), candidate)
        candidate = self.report()
        candidate["records"][0]["status"] = "failed"
        self.assertEqual(
            compare_reports(self.report(), candidate)["incomplete_requests"], ["d0-0"]
        )
        with self.assertRaises(ValueError):
            distribution([{"token_id": 1, "logprob": float("nan")}])

    def test_stream_capture_is_opt_in_and_keeps_partial_outputs(self):
        class Response(io.BytesIO):
            status = 200

        body = (
            b'{"token_ids":[10],"finished":false}\n{"token_ids":[11],"finished":true}\n'
        )
        kwargs = {
            "request_id": "case",
            "domain": 0,
            "replica": 0,
            "planned_ns": 0,
            "length_class": 1,
        }
        with (
            patch(
                "tools.expert_pool.benchmark_http.urlopen", return_value=Response(body)
            ),
            patch(
                "tools.expert_pool.benchmark_http.time.perf_counter_ns",
                side_effect=[1, 2, 3, 4],
            ),
        ):
            default = fetch("http://example.invalid", "test", 2, 120, **kwargs)
        self.assertEqual(default["status"], "completed")
        self.assertNotIn("token_ids", default)
        with (
            patch(
                "tools.expert_pool.benchmark_http.urlopen", return_value=Response(body)
            ),
            patch(
                "tools.expert_pool.benchmark_http.time.perf_counter_ns",
                side_effect=[1, 2, 3, 4],
            ),
        ):
            captured = fetch(
                "http://example.invalid",
                "test",
                3,
                120,
                capture_output_tokens=True,
                **kwargs,
            )
        self.assertEqual(captured["status"], "incomplete")
        self.assertEqual(captured["token_ids"], [10, 11])


if __name__ == "__main__":
    unittest.main()
