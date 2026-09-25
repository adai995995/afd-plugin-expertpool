# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone control relay; GPU payloads never enter this process."""

import argparse
import json
import os
import stat
import time
from contextlib import ExitStack
from multiprocessing.connection import Client, Connection, Listener, wait
from pathlib import Path

from afd_plugin.expert_pool.batching import BatchingOptions
from afd_plugin.expert_pool.controller import ControllerClientIdentity, ControllerLedger
from afd_plugin.expert_pool.deployment import PoolDeployment
from afd_plugin.expert_pool.fanout_controller import FanoutControllerLedger
from afd_plugin.expert_pool.protocol import (
    Message,
    enable_tcp_nodelay,
    receive_message,
    send_message,
)

CONNECT_RETRY_S = 0.05


def connect_controller(
    deployment: PoolDeployment, role: str, identity: str
) -> Connection:
    address, family = deployment.controller_address(role, identity)
    deadline = time.monotonic() + deployment.timeout_s
    while True:
        try:
            connection = Client(
                address,
                family=family,
                authkey=deployment.control_authkey if family == "AF_INET" else None,
            )
            return enable_tcp_nodelay(connection) if family == "AF_INET" else connection
        except (FileNotFoundError, ConnectionRefusedError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError("Controller did not become available") from None
            time.sleep(CONNECT_RETRY_S)


class ControllerRuntime:
    def __init__(
        self,
        ledger: ControllerLedger,
        clients: dict[str, Connection],
        workers: dict[str, Connection],
        *,
        pooled_admission: bool = False,
    ) -> None:
        if set(clients) != set(ledger.clients) or set(workers) != set(ledger.workers):
            raise ValueError("Controller endpoints do not match the ledger")
        if type(pooled_admission) is not bool or (
            pooled_admission and not isinstance(ledger, FanoutControllerLedger)
        ):
            raise ValueError("Pooled admission requires a fan-out ledger")
        self.ledger = ledger
        self.clients = clients
        self.workers = workers
        self.pooled_admission = pooled_admission
        self.event_count = 0
        self.event_cpu_ms = 0.0
        self.max_event_cpu_ms = 0.0

    def run(self) -> dict:
        endpoints = {c: ("client", key) for key, c in self.clients.items()}
        endpoints.update({c: ("worker", key) for key, c in self.workers.items()})
        stopping = False
        while endpoints:
            for connection in wait(list(endpoints)):
                role, identity = endpoints[connection]
                message = receive_message(connection, 0)
                started = time.perf_counter_ns()
                if role == "client":
                    if message.kind == "close":
                        self.ledger.close_client(identity)
                        send_message(connection, Message("closed"))
                        del endpoints[connection]
                    elif message.kind == "status":
                        if identity in self.ledger.outstanding:
                            raise ValueError(
                                "Drain this client before requesting status"
                            )
                        # Small fixed-size summary; full per-worker records are
                        # returned to the supervisor after orderly shutdown.
                        send_message(
                            connection,
                            Message(
                                "snapshot",
                                metrics={
                                    "pending": self.ledger.pending_count,
                                    "reserved": self.ledger.reserved_count,
                                    "completed": sum(
                                        w.completed
                                        for w in self.ledger.workers.values()
                                    ),
                                    "completed_parents": sum(
                                        self.ledger.completed_by_client.values()
                                    ),
                                    "busy_replica_bypasses": (
                                        self.ledger.busy_replica_bypasses
                                    ),
                                },
                            ),
                        )
                    elif message.kind == "submit" and message.request is not None:
                        try:
                            self.ledger.submit(identity, message.request, started)
                        except ValueError as error:
                            send_message(
                                connection,
                                Message(
                                    "error", request=message.request, detail=str(error)
                                ),
                            )
                        else:
                            if (
                                message.request.demand is not None
                                and message.request.num_tokens == 0
                            ):
                                send_message(
                                    connection,
                                    Message("empty_done", request=message.request),
                                )
                    else:
                        raise ValueError("Unexpected client control message")
                elif message.kind == "ready":
                    self.ledger.ready(
                        identity,
                        message.detail,
                        receive_slots=message.metrics.get("receive_slots", 1),
                        startup_complete=message.metrics.get("startup_complete", 0),
                        collect_cost_feedback=message.metrics.get(
                            "collect_cost_feedback", 0
                        ),
                        batching=BatchingOptions(
                            max_calls=message.metrics.get("batch_max_calls", 1),
                            max_tokens=message.metrics.get("batch_max_tokens", 0),
                            max_wait_us=message.metrics.get("batch_max_wait_us", 0),
                        ),
                    )
                elif message.kind == "batch_executing":
                    if not isinstance(self.ledger, FanoutControllerLedger):
                        raise ValueError("Batch execution requires partitioned control")
                    assert message.batch is not None
                    self.ledger.start_batch(identity, message.batch)
                elif message.kind == "closed":
                    if not stopping or self.ledger.workers[identity].active is not None:
                        raise ValueError("Worker closed before draining")
                    self.ledger.workers[identity].ready = False
                    self.ledger.workers[identity].phase = "closed"
                    del endpoints[connection]
                elif message.plan is not None:
                    if isinstance(self.ledger, FanoutControllerLedger):
                        self.ledger.progress(
                            identity, message.kind, message.plan, message.metrics
                        )
                    else:
                        self.ledger.progress(identity, message.kind, message.plan)
                    if message.kind in {"grant", "output_ready", "done", "error"} and not (
                        self.pooled_admission and message.kind == "grant"
                    ):
                        # The ledger processes completion before the A can
                        # receive done and submit its next layer.
                        send_message(
                            self.clients[message.plan.request.key.client_id], message
                        )
                else:
                    raise ValueError("Unexpected worker control message")
                elapsed = (time.perf_counter_ns() - started) / 1e6
                self.event_count += 1
                self.event_cpu_ms += elapsed
                self.max_event_cpu_ms = max(self.max_event_cpu_ms, elapsed)
            while (granted := self.ledger.grant(time.perf_counter_ns())) is not None:
                plan, queue_ms = granted
                if isinstance(self.ledger, FanoutControllerLedger):
                    while self.ledger.dispatch_notifications:
                        dispatch = self.ledger.dispatch_notifications.popleft()
                        send_message(
                            self.clients[dispatch.request.key.client_id],
                            Message("dispatch", dispatch=dispatch),
                        )
                if self.pooled_admission:
                    # The ledger already reserved this exact slot and expert
                    # subset. A can post its direct GPU send while E receives
                    # the same plan; E's later grant confirms progress only.
                    send_message(
                        self.clients[plan.request.key.client_id],
                        Message(
                            "grant",
                            plan=plan,
                            metrics={"controller_queue_ms": queue_ms},
                        ),
                    )
                send_message(
                    self.workers[plan.worker_id],
                    Message(
                        "grant", plan=plan, metrics={"controller_queue_ms": queue_ms}
                    ),
                )
            if not stopping and len(self.ledger.closed) == len(self.clients):
                if self.ledger.outstanding or self.ledger.pending_count:
                    raise RuntimeError("Controller closed with outstanding work")
                stopping = True
                for connection in self.workers.values():
                    send_message(connection, Message("close"))
        return {
            **self.ledger.snapshot(),
            "pooled_admission": self.pooled_admission,
            "control_events": self.event_count,
            "event_processing_ms_total": self.event_cpu_ms,
            "event_processing_ms_max": self.max_event_cpu_ms,
        }


def serve_controller(deployment: PoolDeployment) -> dict:
    if deployment.controller is None:
        raise ValueError("Controller is not configured")
    if not deployment.controller.tcp_host:
        parent = Path(deployment.controller.socket_dir).stat()
        if parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
            raise ValueError(
                "Controller sockets require an owned directory with mode 0700"
            )
    identities = tuple(
        ControllerClientIdentity(c[0].client_id, c[0].session_epoch, c[0].domain)
        for client_id in deployment.client_ids
        for c in (deployment.client_endpoints(client_id),)
    )
    if deployment.dispatch_mode == "expert_partitioned":
        ledger = FanoutControllerLedger(
            deployment.pool_directory(),
            identities,
            scheduling_policy=deployment.controller.scheduling_policy,
            demand_aware=deployment.demand_aware,
            compact_output=deployment.compact_output,
            receive_slots=deployment.receive_slots,
            batching=deployment.batching,
            collect_cost_feedback=deployment.execution.collect_cost_feedback,
            split_assignments=deployment.split_assignments,
        )
    else:
        ledger = ControllerLedger(
            deployment.pool_directory(),
            identities,
            scheduling_policy=deployment.controller.scheduling_policy,
        )
    ledger.requires_warmup = deployment.execution.warmup_before_ready
    with ExitStack() as stack:
        listeners = {
            (role, identity): stack.enter_context(
                Listener(
                    deployment.controller_address(role, identity)[0],
                    family=deployment.controller_address(role, identity)[1],
                    authkey=(
                        deployment.control_authkey
                        if deployment.controller.tcp_host
                        else None
                    ),
                )
            )
            for role, members in (
                ("client", deployment.client_ids),
                ("worker", deployment.worker_ids),
            )
            for identity in members
        }
        connections = {}
        for key, listener in listeners.items():
            connection = listener.accept()
            if deployment.controller.tcp_host:
                enable_tcp_nodelay(connection)
            stack.callback(connection.close)
            connections[key] = connection
        return ControllerRuntime(
            ledger,
            {key: connections[("client", key)] for key in deployment.client_ids},
            {key: connections[("worker", key)] for key in deployment.worker_ids},
            pooled_admission=deployment.pooled_admission,
        ).run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(serve_controller(PoolDeployment.read(args.deployment))), flush=True
    )


if __name__ == "__main__":
    main()
