# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A single layer's resident weight slots, independent of runtime routing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertPlacement:
    """Logical IDs in physical weight-slot order on one worker.

    This is a static local placement, not a global directory or scheduler.
    Multiple workers may contain the same logical expert. Dispatch must still
    assign each (token row, top-k slot) to exactly one resident copy.
    """

    layer_id: int
    num_experts: int
    expert_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.layer_id < 0 or self.num_experts <= 0:
            raise ValueError("Invalid layer ID or global expert count")
        if not isinstance(self.expert_ids, tuple) or not self.expert_ids:
            raise ValueError("expert_ids must be a nonempty immutable tuple")
        if len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("A local weight slot must have a unique logical expert")
        if any(
            type(expert_id) is not int or not 0 <= expert_id < self.num_experts
            for expert_id in self.expert_ids
        ):
            raise ValueError("Logical expert ID is outside the checkpoint range")

    def global_to_local(self) -> tuple[int, ...]:
        """Missing experts map to the pinned Triton kernel's -1 sentinel."""
        mapping = [-1] * self.num_experts
        for slot, expert_id in enumerate(self.expert_ids):
            mapping[expert_id] = slot
        return tuple(mapping)
