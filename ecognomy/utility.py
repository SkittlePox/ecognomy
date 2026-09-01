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


def honest_ask(theta: np.ndarray) -> np.ndarray:
    """(N, G, G) the reservation matrix that shades nothing.

    `ask[i, a, b] = theta[i, a] / theta[i, b]` -- give up a unit of `a` only for
    enough `b` to replace exactly the utility lost. Every round trip is exactly
    1.0, so an honest matrix carries no spread and cannot be money-pumped: it is
    the reciprocal-consistent surface the old price vector was confined to, and
    the origin any shading rung departs from.

    The two degenerate rows are the ones that matter:

      * `theta[b] == 0` -- a good the agent never consumes. It demands an
        infinite quantity, i.e. refuses, which is what keeps a worthless good
        from being accepted for a valuable one. This dominates, so 0/0 is a
        refusal rather than a giveaway.
      * `theta[a] == 0` -- a good the agent holds but does not want. It asks 0,
        giving the good away for any positive quantity of something it does want.

    Both follow from linearity rather than being policy choices, which is why
    they live here beside the reward and not in a policy.
    """
    t = np.asarray(theta, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ask = t[:, :, None] / t[:, None, :]
    ask[np.broadcast_to(t[:, None, :] <= 0.0, ask.shape)] = np.inf
    idx = np.arange(t.shape[1])
    ask[:, idx, idx] = np.inf
    return ask
