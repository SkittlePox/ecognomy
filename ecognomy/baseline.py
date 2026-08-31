"""The autarky counterfactual.

Total welfare on its own says little — a world can post a high number simply
because production is cheap. What matters to a mechanism designer is how much of
it the *market* is responsible for. So every run is measured against the same
world, same seed, same policy, with the market switched off: agents may produce,
consume and move, but no offer can ever match.

The difference is the gains from trade. A run below its own autarky baseline is
one where trading is actively destroying value, which is a real and informative
outcome rather than a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecognomy.config import WorldConfig
from ecognomy.metrics import Metrics


@dataclass
class Comparison:
    """A run measured against its own no-market counterfactual."""

    welfare: float
    baseline_welfare: float
    consumption_utility: float
    baseline_consumption_utility: float
    per_agent: np.ndarray           # (N,) cumulative reward, with market
    baseline_per_agent: np.ndarray  # (N,) cumulative reward, without

    @property
    def gain(self) -> float:
        """Welfare attributable to the market."""
        return self.welfare - self.baseline_welfare

    @property
    def ratio(self) -> float:
        """Welfare as a multiple of autarky, or nan when that is meaningless.

        Only defined for a positive baseline. With both sides negative the
        quotient inverts: -0.96 / -1.04 reads as 0.92, suggesting a loss, when
        the run is in fact +0.08 ahead. `gain` is always the safe reading.
        """
        if self.baseline_welfare <= 1e-12:
            return float("nan")
        return self.welfare / self.baseline_welfare

    @property
    def verdict(self) -> str:
        if self.gain > 1e-6:
            return "trade is creating value"
        if self.gain < -1e-6:
            return "trade is destroying value"
        return "trade is doing nothing"

    def share_of_agents_helped(self) -> float:
        """Fraction of agents better off than their autarky counterpart.

        Total welfare can rise while most agents lose, so this is worth seeing
        beside the headline: a market that lifts the total by enriching two
        agents and impoverishing eighteen is a different world from one that
        lifts everybody.
        """
        return float((self.per_agent > self.baseline_per_agent).mean())


def run_baseline(config: WorldConfig, policy, ticks: int, scenario=None):
    """Run the same world with the market disabled. Returns (world, Metrics).

    Disabled by setting sight to zero, so no agent can see any counterparty. That
    leaves every other parameter untouched, which is what makes the difference
    attributable to the market rather than to anything else changing.
    """
    import copy

    from ecognomy.world import World

    config = copy.deepcopy(config)
    config.visibility.sight_mean = 0.0
    world = World(config, scenario=scenario)
    metrics = Metrics()
    for _ in range(ticks):
        world.step(policy.act(world, world.rng))
        metrics.record(world)
    return world, metrics


def compare(config: WorldConfig, policy, ticks: int, world=None, metrics=None,
            scenario=None) -> Comparison:
    """Measure a run against its own autarky counterfactual.

    Pass an already-finished `world`/`metrics` to avoid re-running the market
    arm; otherwise both arms are run here.
    """
    from ecognomy.policy import run
    from ecognomy.world import World

    if world is None:
        world = World(config, scenario=scenario)
        metrics = run(world, policy, ticks)

    base_world, base_metrics = run_baseline(config, policy, ticks, scenario=scenario)
    return Comparison(
        welfare=world.welfare,
        baseline_welfare=base_world.welfare,
        consumption_utility=float(world.cumulative_utility.sum()),
        baseline_consumption_utility=float(base_world.cumulative_utility.sum()),
        per_agent=world.cumulative_reward.copy(),
        baseline_per_agent=base_world.cumulative_reward.copy(),
    )
