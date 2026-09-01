"""Policies.

A policy proposes actions; the world disposes. `world` never imports this
package, and a policy never mutates state.

The ladder, in order of what each rung needs to know:

    RandomPolicy   nothing                      -- the control
    MyopicPolicy   its own preferences + holdings  -- rational, nothing learned

Rungs above these are not built yet:

    2  adaptive shading      widens or narrows its own spread by whether its
                             postings get taken. This became expressible only
                             with the rate matrix: under a price vector, raising
                             a good's price made the agent a tougher seller and a
                             keener buyer of it at the same time, so there was no
                             direction to shade in.
    3  acceptance model      P(taken | posted rate, region), so shading has a
                             basis and postings can be selective
    4  depth-2 planning      the first rung that can hold a good it does not
                             consume, hence the first that can support indirect
                             exchange, money, or arbitrage
    5  per-counterparty priors  posteriors over each partner's theta and e

The jump that matters is 3 to 4. Everything below it prices goods by consumption
alone, so it will refuse any trade that leaves it merely indifferent -- which is
exactly what the `triangular` scenario is built to detect.

Both built rungs post a reciprocal-consistent matrix and so quote no spread:
`MyopicPolicy` because it shades nothing, `RandomPolicy` because it is not
trying. The two-sided quote sits unused until rung 2, and `metrics.round_trip`
is where its arrival will show up.
"""

from ecognomy.policy.myopic import MyopicPolicy
from ecognomy.policy.random_policy import RandomPolicy
from ecognomy.policy.runner import run

__all__ = ["RandomPolicy", "MyopicPolicy", "run"]
