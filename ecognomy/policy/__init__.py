"""Policies.

A policy proposes actions; the world disposes. `world` never imports this
package, and a policy never mutates state.

The ladder, in order of what each rung needs to know:

    RandomPolicy   nothing                      -- the control
    MyopicPolicy   its own preferences + stock  -- rational, no learned parameters

Rungs above these are not built yet:

    2  adaptive shading      learns whether its own posted prices get taken
    3  acceptance model      P(taken | posted price, region), so shading has a
                             basis and offers can be selective
    4  depth-2 planning      the first rung that can hold a good it does not
                             consume, hence the first that can support indirect
                             exchange, money, or arbitrage
    5  per-counterparty priors  posteriors over each partner's theta and e

The jump that matters is 3 to 4. Everything below it prices goods by consumption
alone, so it will refuse any trade that leaves it merely indifferent -- which is
exactly what the `triangular` scenario is built to detect.
"""

from ecognomy.policy.myopic import MyopicPolicy
from ecognomy.policy.random_policy import RandomPolicy
from ecognomy.policy.runner import run

__all__ = ["RandomPolicy", "MyopicPolicy", "run"]
