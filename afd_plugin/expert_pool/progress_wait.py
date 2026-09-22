# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Socket waits that preserve sub-millisecond CUDA progress polling intervals."""

import select
from multiprocessing.connection import Connection, wait


class ProgressWaiter:
    """Wait for control traffic without rounding every GPU poll up to 1 ms.

    multiprocessing.connection.wait uses poll/epoll on Linux, which rounds a
    positive fractional-millisecond timeout up to a whole millisecond. select
    accepts the original timeout and still wakes early for control traffic.
    OS scheduling can delay either wait; this is not a hard realtime bound.

    Descriptors outside select's supported range retain the original poll
    backend. Falling back is remembered, so those workers do not repeatedly
    raise exceptions in their progress loop. Neither backend consumes messages.
    """

    def __init__(self, connection: Connection | tuple[Connection, ...]) -> None:
        self.connections = (
            connection if isinstance(connection, tuple) else (connection,)
        )
        self.backend = "select"

    def wait(self, timeout_s: float) -> bool:
        if self.backend == "select":
            try:
                return bool(select.select(self.connections, [], [], timeout_s)[0])
            except ValueError:
                # select rejects high-numbered descriptors. The poll backend
                # also validates closed/invalid connections, rather than
                # treating a failed wait as successful readiness.
                self.backend = "poll"
        return bool(wait(self.connections, timeout=timeout_s))
