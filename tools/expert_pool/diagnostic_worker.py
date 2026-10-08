# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit diagnostic worker RPCs; no callable/pickle fallback is required."""

from vllm.v1.worker.gpu_worker import Worker

from afd_plugin.v1.worker.pool_attention_worker import PoolAttentionWorker
from tools.expert_pool.moe_probe import begin_probe, end_probe


class ProbeMethods:
    def begin_moe_probe(self) -> None:
        begin_probe(self)

    def end_moe_probe(self) -> dict:
        return end_probe(self)


class DiagnosticNativeWorker(ProbeMethods, Worker):
    pass


class DiagnosticPoolWorker(ProbeMethods, PoolAttentionWorker):
    pass
