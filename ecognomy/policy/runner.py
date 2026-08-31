"""Driving a world with a policy."""

from __future__ import annotations


def run(world, policy, ticks: int, metrics=None):
    """Drive a world for `ticks` steps, recording metrics."""
    from ecognomy.metrics import Metrics

    metrics = metrics or Metrics()
    for _ in range(ticks):
        world.step(policy.act(world, world.rng))
        metrics.record(world)
    return metrics
