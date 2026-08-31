"""Reward.

    u_i(q) = sum_g  theta_ig * q_ig

An agent's reward is the amount of each good it consumed, multiplied by its
preference for that good. Nothing else.

This is deliberately linear, and the consequences are worth stating because they
are load-bearing rather than incidental:

  * **A posted price is exact at any quantity.** The value of a good never
    changes with how much you hold, so the rate an agent should demand for one
    unit is the rate it should demand for a thousand. Under the previous concave
    (CES) reward, a posted price was a *marginal* valuation and trades were for
    half a holding, so the mechanism routinely approved trades that left a
    participant worse off -- 14.8% of trade sides, destroying about a fifth of
    the gross gains. That failure mode is now structurally impossible.

  * **Gains from trade come only from differing preferences**, never from
    differing inventories. Two agents who value goods identically have nothing
    to gain by exchanging, however lopsided their holdings. That is correct for
    linear reward: with no taste for variety, an agent simply makes and eats
    whatever is worth most to it.

  * **Willingness to pay does not respond to scarcity.** Holding almost none of
    a good does not make an agent want it more. Regional price differences are
    therefore compositional -- they reflect which agents are standing where --
    rather than driven by local shortage.

Curvature was removed rather than set to zero: a reward function with a
substitution parameter permanently pinned to "linear" is an unused branch that
rots. The concave version is in the git history if it is ever wanted back.
"""

from __future__ import annotations

import numpy as np


def utility(consumed: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Per-agent reward from a consumption bundle.

    Args:
        consumed: (N, G) non-negative quantities consumed this tick.
        theta: (N, G) non-negative preference weights.

    Returns:
        (N,) reward.
    """
    q = np.maximum(np.asarray(consumed, dtype=np.float64), 0.0)
    return (np.asarray(theta, dtype=np.float64) * q).sum(axis=1).astype(np.float32)


def marginal_value(theta: np.ndarray) -> np.ndarray:
    """(N, G) value of one more unit of each good.

    Constant, and equal to the preference weight itself -- which is the whole
    point of a linear reward. It takes no arguments beyond `theta` because it
    depends on nothing else, and in particular not on what the agent holds.
    """
    return np.asarray(theta, dtype=np.float64)
