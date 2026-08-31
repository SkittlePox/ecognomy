"""Policy tests.

The myopic rung has no learned parameters, so its properties are checkable
directly: it should price honestly, consume at the closed-form optimum, produce
only where that pays, and beat the random control on reward.
"""

import numpy as np
import pytest

from ecognomy.config import VisibilityConfig, WorldConfig
from ecognomy.policy import MyopicPolicy, RandomPolicy, run
from ecognomy.topology import Topology
from ecognomy.utility import ces_utility, marginal_utility
from ecognomy.world import World


def _world(seed=1, n=20, **prod):
    cfg = WorldConfig(n_agents=n, seed=seed, topology=Topology.line(3))
    for k, v in prod.items():
        setattr(cfg.production, k, v)
    return World(cfg)


def test_myopic_beats_random_on_reward():
    """Rationality has to pay, or the scoring is wrong somewhere."""
    myopic = run(_world(), MyopicPolicy(), 400).summary()["mean_reward"]
    random = run(_world(), RandomPolicy(), 400).summary()["mean_reward"]
    assert myopic > random, f"myopic {myopic} vs random {random}"


def test_myopic_posts_its_true_marginal_valuation():
    """Rung 1 does not shade. Choosing how far to shade needs to know what
    rivals post, which is a later rung."""
    w = _world()
    a = MyopicPolicy().act(w, w.rng)
    expected = marginal_utility(w.inventory.astype(np.float64), w.theta, w.rho, w.alpha)
    assert np.allclose(a.price, np.maximum(expected, 0.0), atol=1e-5)


def test_myopic_consumption_fraction_matches_the_closed_form():
    """f = z/(1+z) with z = kappa**(1/(alpha-1)) maximises the consume-vs-hold
    score exactly, so no search over candidate fractions is needed."""
    w = _world()
    for kappa in (0.5, 1.0, 3.0):
        a = MyopicPolicy(kappa=kappa).act(w, w.rng)
        alpha = float(w.alpha[0])
        z = kappa ** (1.0 / (alpha - 1.0))
        expected = z / (1.0 + z)
        held = w.inventory[0] > 1e-6
        got = (a.consume[0][held] / w.inventory[0][held])
        assert np.allclose(got, expected, atol=1e-4), f"kappa={kappa}"


def test_consumption_fraction_is_a_genuine_interior_optimum():
    """Regression: the raw CES aggregate is homogeneous of degree 1, which makes
    the consume-vs-hold tradeoff linear in f, so the optimum is always a corner
    and kappa is a knife edge. alpha < 1 is what creates an interior answer."""
    theta = np.full((1, 3), 1 / 3)
    rho = np.array([0.5])
    x = np.array([[3.0, 2.0, 1.0]])
    kappa = 3.0

    def score(f, alpha):
        a = np.array([alpha])
        u_eat = ces_utility(x * f, theta, rho, a)[0]
        u_keep = ces_utility(x * (1 - f), theta, rho, a)[0]
        u_now = ces_utility(x, theta, rho, a)[0]
        return u_eat + kappa * (u_keep - u_now)

    grid = np.linspace(0.01, 0.99, 99)
    best = grid[np.argmax([score(f, 0.5) for f in grid])]
    z = kappa ** (1.0 / (0.5 - 1.0))
    assert best == pytest.approx(z / (1 + z), abs=0.02)
    # With alpha == 1 the score is monotone, so no interior optimum exists.
    flat = [score(f, 0.999) for f in grid]
    assert np.argmax(flat) in (0, len(grid) - 1)


def test_myopic_produces_only_where_it_pays():
    """Effort is linear within a tick, so the optimum is a corner: all of it on
    one good, and none at all when nothing clears the effort cost."""
    w = _world()
    a = MyopicPolicy().act(w, w.rng)
    per_agent = a.effort.sum(axis=1)
    assert ((per_agent == 0) | np.isclose(per_agent, 1.0)).all()
    active = a.effort > 0
    assert (active.sum(axis=1) <= 1).all(), "effort must go to a single good"


def test_myopic_emits_only_legal_actions():
    """Sanitising must be a no-op on a policy that knows the rules.

    Under the old discrete space this caught a real bug: consume actions arrived
    with a zero quantity, were rejected, and agents produced without ever eating.
    """
    w = _world()
    pol = MyopicPolicy()
    for _ in range(50):
        raw = pol.act(w, w.rng)
        clean = w.action_space.sanitize(w, raw)
        assert np.allclose(raw.consume, clean.consume, atol=1e-5)
        assert np.allclose(raw.effort, clean.effort, atol=1e-5)
        assert np.allclose(raw.max_trade, clean.max_trade, atol=1e-5)
        assert (raw.move == clean.move).all()
        w.step(raw)


def test_myopic_consumes_and_produces():
    w = _world()
    run(w, MyopicPolicy(), 300)
    assert w.goods_created > 0, "nothing was ever produced"
    assert w.goods_destroyed > 0, "nothing was ever consumed"


def test_sight_is_heterogeneous_by_default():
    """Each agent has a different K -- market access is a capability that varies
    across the population, which is what lets broker roles emerge."""
    w = _world(n=400)
    assert w.sight.min() >= 1
    assert len(np.unique(w.sight)) > 1
    uniform = World(WorldConfig(n_agents=400, seed=1,
                                visibility=VisibilityConfig(sight_mean=3.0, sight_spread=0.0)))
    assert len(np.unique(uniform.sight)) == 1


def test_sight_zero_disables_the_market():
    cfg = WorldConfig(n_agents=20, seed=1, visibility=VisibilityConfig(sight_mean=0.0))
    m = run(World(cfg), MyopicPolicy(), 200)
    assert m.summary()["total_trades"] == 0


def test_more_sight_does_not_reduce_trade():
    """Seeing more of the board should not make an agent trade less."""
    def trades(mean):
        cfg = WorldConfig(n_agents=20, seed=5, topology=Topology.line(3),
                          visibility=VisibilityConfig(sight_mean=mean, sight_spread=0.0))
        return run(World(cfg), MyopicPolicy(), 200).summary()["total_trades"]
    assert trades(12) >= trades(1)


def test_sparse_production_forces_dependence():
    """n_producible caps how many goods an agent can make at all. When everyone
    can make everything, autarky is optimal and trade is never necessary."""
    w = _world(n_producible=2)
    producible = (w.efficiency > 0).sum(axis=1)
    assert (producible <= 2).all()
    assert (producible >= 1).all(), "every agent must be able to make something"
