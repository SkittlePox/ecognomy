"""CES utility.

    u(q) = [ ( sum_g theta_g * q_g**rho ) ** (1/rho) ] ** alpha

The CES aggregate alone is homogeneous of degree 1 -- u(f*q) == f*u(q) -- so it
has constant returns to scale and no interior optimum in *how much* to consume.
`alpha` < 1 is a concave transform supplying diminishing returns to scale. It
leaves every substitution property untouched, because the marginal rate of
substitution is invariant under a monotone transform: alpha changes how much to
consume, rho changes what to consume, and the two do not interact.

`theta` sets relative desirability; `rho` sets substitutability. They are
independent: the marginal rate of substitution is

    MRS_ab = (theta_a/theta_b) * (q_a/q_b)**(rho-1)

so theta alone fixes the MRS at a balanced basket, while rho fixes how fast the
MRS moves as the basket tilts. At rho == 1 the MRS never moves and no interior
trade exists. See `docs/environment.md`.
"""

from __future__ import annotations

import numpy as np

# |rho| below this is treated as the Cobb-Douglas limit, where the closed form
# divides by zero.
COBB_DOUGLAS_TOL = 1e-4

# Floor on quantities in the complements regime. With rho < 0 a zero quantity
# sends its term to infinity; the limit of the whole expression is 0, which is
# economically right (a missing essential good is worthless) but arrives there
# through inf and would otherwise produce nan.
QTY_FLOOR = 1e-12


def ces_utility(
    q: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    alpha: np.ndarray | float | None = None,
) -> np.ndarray:
    """Per-agent CES utility of a consumption bundle.

    Args:
        q: (N, G) non-negative quantities consumed this tick.
        theta: (N, G) non-negative preference weights.
        rho: (N,) substitution parameters, each < 1.
        alpha: (N,) or scalar concavity in scale, each in (0, 1]. Defaults to 1,
            which is the raw CES aggregate and has constant returns to scale.

    Returns:
        (N,) utility.
    """
    q = np.maximum(np.asarray(q, dtype=np.float64), 0.0)
    theta = np.asarray(theta, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    weights = theta / np.maximum(theta.sum(axis=1, keepdims=True), QTY_FLOOR)

    # Cobb-Douglas limit: u = prod_g q_g ** w_g, computed in log space.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_q = np.log(np.maximum(q, QTY_FLOOR))
        cobb_douglas = np.exp((weights * log_q).sum(axis=1))
    # A true zero in any weighted good annihilates the product.
    annihilated = ((q <= 0.0) & (weights > 0.0)).any(axis=1)
    cobb_douglas = np.where(annihilated, 0.0, cobb_douglas)

    # General case. Guard the base so rho < 0 does not raise on zeros; the
    # `missing` mask below overrides the result where it matters.
    r = rho[:, None]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Exact zero for q == 0 when rho > 0. When rho < 0 the true limit is
        # inf, but the `missing` mask below sets those agents to 0 utility,
        # which is the correct limit of the whole expression.
        powered = np.where(q > 0.0, np.power(np.maximum(q, QTY_FLOOR), r), 0.0)
        total = (theta * powered).sum(axis=1)
        general = np.power(np.maximum(total, 0.0), 1.0 / np.where(np.abs(rho) < COBB_DOUGLAS_TOL, 1.0, rho))

    # Complements with a missing good: utility is exactly zero.
    missing = ((q <= 0.0) & (theta > 0.0)).any(axis=1) & (rho < 0.0)
    general = np.where(missing, 0.0, general)

    out = np.where(np.abs(rho) < COBB_DOUGLAS_TOL, cobb_douglas, general)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if alpha is not None:
        a = np.asarray(alpha, dtype=np.float64)
        out = np.power(np.maximum(out, 0.0), a)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def marginal_utility(
    q: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    alpha: np.ndarray | float | None = None,
    eps: float = 1e-4,
) -> np.ndarray:
    """(N, G) numerical marginal utility of each good at bundle `q`.

    Used by policies to price goods and by tests to check that MRS behaves as
    the spec claims. Numerical rather than closed-form so it stays correct if
    the utility form is swapped.
    """
    base = ces_utility(q, theta, rho, alpha)
    out = np.empty(q.shape, dtype=np.float32)
    for g in range(q.shape[1]):
        bumped = q.copy()
        bumped[:, g] += eps
        out[:, g] = (ces_utility(bumped, theta, rho, alpha) - base) / eps
    return out
