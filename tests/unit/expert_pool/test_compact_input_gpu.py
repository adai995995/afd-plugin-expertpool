# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GPU contract: selected rows and returned assignment order stay aligned."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CompactInputGpuTests(unittest.TestCase):
    def test_sparse_and_duplicate_routes_keep_original_slot_order(self):
        from afd_plugin.expert_pool.compact_input import CompactInputWorkspace
        from afd_plugin.expert_pool.compact_output import CompactOutputWorkspace

        device = torch.device("cuda", 0)
        routes = torch.tensor(
            [[0, 2, 1], [3, 5, 1], [4, 2, 0], [1, 3, 5], [2, 4, 3]],
            dtype=torch.int32,
            device=device,
        )
        weights = torch.arange(1, 16, dtype=torch.float32, device=device).reshape(5, 3)
        hidden = torch.arange(5 * 32, dtype=torch.bfloat16, device=device).reshape(
            5, 32
        )
        for selected in ((0,), (0, 2), (1,)):
            with self.subTest(selected=selected):
                ownership = torch.tensor(
                    [expert in selected for expert in range(6)],
                    dtype=torch.bool,
                    device=device,
                )
                expected_rows = torch.nonzero(ownership[routes.long()].any(1)).flatten()
                workspace = CompactInputWorkspace(8, 32, 3, device)
                num_assignments = int(ownership[routes.long()].sum())
                packed = workspace.pack(
                    hidden, weights, routes, ownership, num_rows=num_assignments
                )
                self.assertEqual(packed[0].shape[0], num_assignments)
                for actual, source in zip(
                    packed, (hidden, weights, routes), strict=True
                ):
                    self.assertTrue(
                        torch.equal(actual[: expected_rows.numel()], source[expected_rows])
                    )
                self.assertTrue(
                    torch.count_nonzero(packed[0][expected_rows.numel() :]) == 0
                )
                self.assertTrue(
                    torch.count_nonzero(packed[1][expected_rows.numel() :]) == 0
                )
                self.assertTrue(
                    torch.all(packed[2][expected_rows.numel() :] == -1)
                )
                replicated_mask = (packed[2] >= 0) & ownership[
                    packed[2].clamp_min(0).long()
                ]
                self.assertEqual(int(replicated_mask.sum()), num_assignments)

                # E packs assignments in token-major order. A restores them
                # using its original routes without an extra index transfer.
                local_slots = packed[0][:, None, :].expand(-1, 3, -1).clone()
                output = CompactOutputWorkspace(24, 32, device)
                payload = output.pack(
                    local_slots, packed[2], ownership, num_assignments=num_assignments
                ).clone()
                restored = torch.zeros(
                    (5, 3, 32), dtype=torch.bfloat16, device=device
                )
                output.scatter(
                    payload,
                    routes,
                    ownership,
                    restored,
                    num_assignments=num_assignments,
                )
                expected = torch.zeros_like(restored)
                expanded = hidden[:, None, :].expand(-1, 3, -1)
                expected[ownership[routes.long()]] = expanded[
                    ownership[routes.long()]
                ]
                self.assertTrue(torch.equal(restored, expected))


if __name__ == "__main__":
    unittest.main()
