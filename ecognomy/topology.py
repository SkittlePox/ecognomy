"""Region graph: an arbitrary weighted directed graph.

Cycles are expected. Edge weight is *distance*, not a toll -- an agent crosses
it at its own mobility, so the same edge costs different agents different
amounts of forgone trading time. Capacity limits simultaneous occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Topology:
    """A directed region graph, stored as edge arrays."""

    n_regions: int
    src: np.ndarray  # (E,) int32
    dst: np.ndarray  # (E,) int32
    weight: np.ndarray  # (E,) float32, distance
    capacity: np.ndarray  # (E,) int32, max simultaneous occupants

    def __post_init__(self) -> None:
        e = len(self.src)
        for name in ("dst", "weight", "capacity"):
            if len(getattr(self, name)) != e:
                raise ValueError(f"topology array '{name}' has length {len(getattr(self, name))}, expected {e}")
        if e and (self.src.max() >= self.n_regions or self.dst.max() >= self.n_regions):
            raise ValueError("edge references a region outside [0, n_regions)")
        if np.any(self.weight <= 0):
            raise ValueError("edge weights must be positive distances")

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def out_edges(self, region: int) -> np.ndarray:
        """Indices of edges leaving `region`."""
        return np.flatnonzero(self.src == region)

    @classmethod
    def line(cls, n_regions: int = 3, weight: float = 3.0, capacity: int = 1_000_000) -> "Topology":
        """A line of regions with edges in both directions.

        The default world. With n_regions == 3 the middle region is
        structurally a broker: nothing moves end to end without crossing it.
        """
        src, dst = [], []
        for r in range(n_regions - 1):
            src += [r, r + 1]
            dst += [r + 1, r]
        return cls(
            n_regions=n_regions,
            src=np.array(src, dtype=np.int32),
            dst=np.array(dst, dtype=np.int32),
            weight=np.full(len(src), weight, dtype=np.float32),
            capacity=np.full(len(src), capacity, dtype=np.int32),
        )

    @classmethod
    def complete(cls, n_regions: int, weight: float = 3.0, capacity: int = 1_000_000) -> "Topology":
        """Every region reachable from every other in one hop."""
        src, dst = [], []
        for a in range(n_regions):
            for b in range(n_regions):
                if a != b:
                    src.append(a)
                    dst.append(b)
        return cls(
            n_regions=n_regions,
            src=np.array(src, dtype=np.int32),
            dst=np.array(dst, dtype=np.int32),
            weight=np.full(len(src), weight, dtype=np.float32),
            capacity=np.full(len(src), capacity, dtype=np.int32),
        )

    @classmethod
    def isolated(cls, n_regions: int = 3) -> "Topology":
        """No edges at all. A failure-mode topology: regions are autarkic."""
        empty_i = np.zeros(0, dtype=np.int32)
        return cls(n_regions, empty_i, empty_i, np.zeros(0, dtype=np.float32), empty_i)

    def with_capacity(self, edge: int, capacity: int) -> "Topology":
        """Copy with one edge's capacity changed. This is the chokepoint knob."""
        cap = self.capacity.copy()
        cap[edge] = capacity
        return Topology(self.n_regions, self.src, self.dst, self.weight, cap)
