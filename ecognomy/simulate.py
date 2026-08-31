"""Headless run entry point.

    python -m ecognomy.simulate --ticks 1000 --agents 20 --out runs/my-run
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ecognomy.config import WorldConfig
from ecognomy.policy import MyopicPolicy, RandomPolicy
from ecognomy.recorder import simulate
from ecognomy.scenarios import ALL as SCENARIOS, get as get_scenario
from ecognomy.topology import Topology


def main() -> None:
    p = argparse.ArgumentParser(description="Run the sandbox headless and record it.")
    p.add_argument("--ticks", type=int, default=None, help="default: recording.default_ticks (1000)")
    p.add_argument("--agents", type=int, default=20)
    p.add_argument("--regions", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sight-mean", type=float, default=3.0,
                   help="average number of a region's posted prices an agent sees (search friction)")
    p.add_argument("--shape-spread", type=float, default=None,
                   help="how differently agents are good at things; wider means more "
                        "comparative advantage and more reason to trade")
    p.add_argument("--concentration", type=float, default=None,
                   help="Dirichlet concentration on tastes; lower means sharper preferences")
    p.add_argument("--sight-spread", type=float, default=0.6,
                   help="dispersion in market access; 0 gives every agent the same K")
    p.add_argument("--anneal-ticks", type=int, default=0, help="0 disables the token anneal")
    p.add_argument("--policy", choices=("random", "myopic"), default="myopic",
                   help="random is the control; myopic is rational with no learned parameters")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default=None,
                   help="run a hand-built diagnostic world instead of a sampled one; "
                        "overrides --agents, --regions and --n-producible")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the autarky counterfactual (halves runtime)")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    policy = RandomPolicy() if args.policy == "random" else MyopicPolicy()

    scenario = get_scenario(args.scenario) if args.scenario else None
    if scenario is not None:
        cfg = scenario.config(seed=args.seed)
        print(f"scenario {scenario.name}: {scenario.description}")
        print(f"  tests {scenario.tests}; solvable by direct exchange: "
              f"{scenario.solvable_bilaterally}")
    else:
        cfg = WorldConfig(
            n_agents=args.agents, seed=args.seed, topology=Topology.line(args.regions)
        )
    if args.shape_spread is not None:
        cfg.production.shape_spread = args.shape_spread
    if args.concentration is not None:
        cfg.preference.dirichlet_concentration = args.concentration
    cfg.visibility.sight_mean = args.sight_mean
    cfg.visibility.sight_spread = args.sight_spread
    cfg.token.anneal_ticks = args.anneal_ticks
    out = Path(args.out) if args.out else Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")

    rec = simulate(cfg, policy, ticks=args.ticks, out=out, scenario=scenario,
                   baseline=not args.no_baseline)
    print(f"wrote {out}  ({args.policy})")
    for k, v in rec.metrics.summary().items():
        print(f"  {k:34s} {v}")

    c = rec.comparison
    if c is not None:
        ratio = "n/a" if c.ratio != c.ratio else f"x{c.ratio:.2f}"
        print()
        print("  welfare vs autarky — the designer's score")
        print(f"    with market      {c.welfare:12.3f}")
        print(f"    autarky baseline {c.baseline_welfare:12.3f}")
        print(f"    gain from trade  {c.gain:+12.3f}   {ratio}")
        print(f"    agents better off than autarky: {c.share_of_agents_helped():.0%}")
        print(f"    -> {c.verdict}")


if __name__ == "__main__":
    main()
