# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU deployment isolation and full-model correctness-oracle regression tests."""

import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from afd_plugin.expert_pool.deployment import (
    MAX_DEPLOYMENT_BYTES,
    ClientEndpoint,
    PoolDeployment,
)
from tools.expert_pool.validate_engines import GENERATED_TOKENS, compare_request


class EngineContractTests(unittest.TestCase):
    def deployment(self) -> PoolDeployment:
        return PoolDeployment(
            "/checkpoint",
            "immutable-checkpoint-test",
            256,
            (
                ClientEndpoint("a0", 1, "domain0", "/private/a0.sock", 24001),
                ClientEndpoint("a1", 1, "domain1", "/private/a1.sock", 24002),
            ),
        )

    def test_roundtrip_keeps_domains_and_endpoints(self) -> None:
        deployment = self.deployment()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "deployment.json"
            path.write_text(json.dumps(asdict(deployment)))
            self.assertEqual(PoolDeployment.read(path), deployment)
        self.assertEqual(deployment.endpoint("a1").domain, "domain1")
        with self.assertRaises(ValueError):
            deployment.endpoint("unregistered")

    def test_reject_ambiguous_physical_endpoints(self) -> None:
        deployment = self.deployment()
        for field in ("client_id", "control_path", "nccl_port"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                duplicate = replace(
                    deployment.clients[1],
                    **{field: asdict(deployment.clients[0])[field]},
                )
                replace(deployment, clients=(deployment.clients[0], duplicate))

    def test_reject_invalid_socket_and_capacity(self) -> None:
        endpoint = self.deployment().clients[0]
        for path in ("relative.sock", "/" + "x" * 101):
            with self.subTest(path=path), self.assertRaises(ValueError):
                replace(endpoint, control_path=path)
        with self.assertRaises(ValueError):
            replace(self.deployment(), max_tokens=0)
        with self.assertRaises(ValueError):
            replace(self.deployment(), timeout_s=-1)

    def test_bounded_config_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oversized.json"
            path.write_bytes(b" " * (MAX_DEPLOYMENT_BYTES + 1))
            with self.assertRaises(ValueError):
                PoolDeployment.read(path)

    def request(self) -> dict:
        return {
            "case": 0,
            "prompt_token_ids": [1, 2, 3],
            "prompt_logprobs": [None, -1.5, -0.25],
            "token_ids": [4] * GENERATED_TOKENS,
            "logprobs": [-0.5] * GENERATED_TOKENS,
        }

    def test_equal_requests_pass(self) -> None:
        result = compare_request(self.request(), self.request(), 0.05)
        self.assertTrue(result["passed"])
        self.assertEqual(result["teacher_forced_tokens"], 2)

    def test_matching_greedy_tokens_do_not_hide_probability_error(self) -> None:
        actual = deepcopy(self.request())
        actual["prompt_logprobs"][1] -= 0.1
        result = compare_request(self.request(), actual, 0.05)
        self.assertTrue(result["tokens_exact"])
        self.assertFalse(result["passed"])

    def test_changed_continuation_not_compared_as_same_prefix(self) -> None:
        actual = deepcopy(self.request())
        actual["token_ids"][0] = 5
        result = compare_request(self.request(), actual, 0.05)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["generated_logprob_max_abs"])

    def test_mismatched_prefix_and_nonfinite_probability_rejected(self) -> None:
        for key, value in (
            ("prompt_token_ids", [1, 2, 4]),
            ("prompt_logprobs", [None, float("nan"), -0.25]),
            ("logprobs", [float("nan")] * GENERATED_TOKENS),
        ):
            actual = deepcopy(self.request())
            actual[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                compare_request(self.request(), actual, 0.05)


if __name__ == "__main__":
    unittest.main()
