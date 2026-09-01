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
from ecognomy.metrics import arbitrage_depth, round_trip
from ecognomy.utility import honest_ask
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


def test_myopic_posts_its_true_preferences():
    """Rung 1 does not shade. Under a linear reward the honest rate for giving up
    `a` to get `b` is simply `theta_a / theta_b`, exact at any trade size."""
    w = _world()
    a = MyopicPolicy().act(w, w.rng)
    settled = w.region >= 0
    assert np.allclose(a.ask[settled], honest_ask(w.theta)[settled], atol=1e-6)


def test_myopic_posts_no_spread_and_cannot_be_pumped():
    """Rung 1 quotes both sides of every pair at the same rate, so its round trips
    are exactly 1.0 and no cycle takes anything off it. The two-sided quote the
    matrix makes expressible is real but unused until a shading rung uses it."""
    w = _world()
    a = MyopicPolicy().act(w, w.rng)
    settled = w.region >= 0
    trips = round_trip(a.ask[settled])
    assert np.allclose(trips[np.isfinite(trips)], 1.0, atol=1e-5)
    assert np.allclose(arbitrage_depth(a.ask[settled]), 1.0, atol=1e-6)


def test_random_postings_are_incoherent_and_that_is_the_point():
    """The control does not hang together, and the world does not protect it.
    Incoherence is legal and measured; a mechanism that refused to trade with an
    irrational agent would be doing its reasoning for it."""
    w = _world()
    a = RandomPolicy().act(w, w.rng)
    settled = w.region >= 0
    assert (arbitrage_depth(a.ask[settled]) < 1.0).mean() > 0.5


def test_myopic_offers_everything_and_eats_the_remainder():
    """Trade resolves before consumption, and any executed trade must raise both
    sides' posted value, so offering the lot can only improve the basket."""
    w = _world()
    a = MyopicPolicy().act(w, w.rng)
    settled = w.region >= 0
    assert np.allclose(a.max_trade[settled], w.inventory[settled], atol=1e-5)
    assert np.isinf(a.consume[settled]).all(), "consume should mean 'whatever is left'"


def test_infinite_consume_means_everything_not_nothing():
    """Regression: sanitising ran nan_to_num with posinf=0, which turned 'eat
    everything' into 'eat nothing' and drove population welfare negative."""
    w = _world()
    raw = MyopicPolicy().act(w, w.rng)
    clean = w.action_space.sanitize(w, raw)
    settled = w.region >= 0
    assert np.allclose(clean.consume[settled], w.inventory[settled], atol=1e-5)


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
    """Sanitising must not have to *correct* a policy that knows the rules.

    `consume` is exempt: the policy says +inf meaning "whatever is left after
    trading", and sanitising resolves that against inventory rather than fixing
    a mistake. Everything else must pass through untouched.
    """
    w = _world()
    pol = MyopicPolicy()
    for _ in range(50):
        raw = pol.act(w, w.rng)
        clean = w.action_space.sanitize(w, raw)
        assert np.allclose(raw.effort, clean.effort, atol=1e-5)
        assert np.allclose(raw.max_trade, clean.max_trade, atol=1e-5)
        assert (raw.move == clean.move).all()
        assert (clean.consume >= 0).all() and np.isfinite(clean.consume).all()
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


def test_every_agent_can_make_every_good():
    """Nothing caps which goods an agent may attempt.

    An agent spreads a fixed budget of effort over whichever goods it likes, so
    specialisation is a choice it makes rather than a restriction imposed on it.
    An earlier version capped each agent to its best two goods, which forced
    dependence and put a heavy thumb on the scale for trade happening at all.
    """
    w = _world()
    assert (w.efficiency > 0).all(), "no agent may be locked out of a good"
    assert w.efficiency.max(axis=1).mean() > w.efficiency.min(axis=1).mean(), \
        "efficiencies must still vary within an agent, or there is no specialisation"
