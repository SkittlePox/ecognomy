"""Environment tests.

Two kinds. Mechanics tests check the transition does what the spec says.
Failure-mode tests check the sandbox can *fail*: each asserts that a specific
degenerate configuration produces a dead world. If any of these ever passes with
a functioning economy, the implementation has assumed its answer and every
positive result is worthless.
"""

import numpy as np
import pytest

from ecognomy.actions import NO_MOVE, Actions
from ecognomy.config import PreferenceConfig, ProductionConfig, SinkConfig, VisibilityConfig, WorldConfig
from ecognomy.mechanism import BilateralMechanism
from ecognomy.policy import MyopicPolicy, RandomPolicy, run
from ecognomy.topology import Topology
from ecognomy.utility import marginal_value, utility
from ecognomy.world import IN_TRANSIT, World


# ------------------------------------------------------------------ reward

def test_reward_is_consumption_times_preference():
    theta = np.array([[0.5, 0.3, 0.2]])
    assert utility(np.array([[2.0, 0.0, 0.0]]), theta)[0] == pytest.approx(1.0)
    assert utility(np.array([[1.0, 1.0, 1.0]]), theta)[0] == pytest.approx(1.0)
    assert utility(np.zeros((1, 3)), theta)[0] == pytest.approx(0.0)


def test_reward_is_exactly_linear_in_quantity():
    """The property the whole design rests on: doubling a bundle doubles the
    reward, so a posted price is correct at any trade size."""
    theta = np.array([[0.4, 0.35, 0.25]])
    q = np.array([[1.3, 0.7, 2.1]])
    for k in (0.1, 2.0, 17.0):
        assert utility(q * k, theta)[0] == pytest.approx(k * utility(q, theta)[0], rel=1e-5)


def test_variety_is_worth_nothing_in_itself():
    """A concentrated bundle and a varied one of equal weighted value score the
    same. Gains from trade come from differing preferences, never from a taste
    for a mixed basket."""
    theta = np.array([[0.5, 0.5]])
    assert utility(np.array([[4.0, 0.0]]), theta)[0] == pytest.approx(
        utility(np.array([[2.0, 2.0]]), theta)[0])


def test_marginal_value_does_not_depend_on_holdings():
    """Scarcity does not raise willingness to pay. This is what makes a posted
    price exact, and equally what stops a chokepoint opening a price wedge
    through local shortage."""
    theta = np.array([[0.6, 0.4]])
    assert np.allclose(marginal_value(theta), theta)


# --------------------------------------------------------------- mechanics

def _two_in_a_region(w):
    same = np.flatnonzero(w.region == w.region[0])
    return int(same[0]), int(same[1])


def _post(w, i, j, price_i, price_j, qty=1.0):
    """Both agents post a price vector and offer `qty` of everything."""
    a = Actions.idle(w.n_agents, w.n_goods)
    a.price[i, :len(price_i)] = price_i
    a.price[j, :len(price_j)] = price_j
    a.max_trade[i] = qty
    a.max_trade[j] = qty
    return a


def test_goods_are_conserved_by_trade():
    """Exchange moves goods, it does not create or destroy them."""
    w = World(WorldConfig(seed=3, sink=SinkConfig(spoilage=(0.0,) * 5)))
    i, j = _two_in_a_region(w)
    # i values banana over apple; j the reverse, so each wants what the other has.
    a = _post(w, i, j, [1.0, 4.0, 1.0, 1.0, 1.0], [4.0, 1.0, 1.0, 1.0, 1.0])
    before = w.inventory.sum()
    w.step(a)
    assert w.last_trades, "opposed valuations should produce a trade"
    assert w.inventory.sum() == pytest.approx(before, rel=1e-5)


def test_execution_rate_is_the_geometric_mean():
    """Neither side's posted rate sets the price; the mean between them does."""
    w = World(WorldConfig(seed=3, sink=SinkConfig(spoilage=(0.0,) * 5)))
    i, j = _two_in_a_region(w)
    a = _post(w, i, j, [1.0, 2.0, 1.0, 1.0, 1.0], [4.0, 1.0, 1.0, 1.0, 1.0])
    w.step(a)
    tr = w.last_trades[0]
    r_i = a.price[i, tr.good_a] / a.price[i, tr.good_b]
    r_j = a.price[j, tr.good_a] / a.price[j, tr.good_b]
    assert tr.price == pytest.approx(float(np.sqrt(r_i * r_j)), rel=1e-4)
    assert r_i < tr.price < r_j, "execution must land strictly between the two rates"


def test_posting_larger_numbers_buys_no_advantage():
    """A price vector means the same thing at any positive scale.

    `du` is linear in the posting, so before postings were normalised an agent
    could multiply its whole vector by 1000, change nothing about the trade it
    was willing to make, and inflate its joint surplus 500-fold — which bought
    it the front of the greedy fill queue for nothing. Any learning rung would
    have found that immediately.
    """
    w = World(WorldConfig(seed=3, sink=SinkConfig(spoilage=(0.0,) * 5)))
    i, j = _two_in_a_region(w)
    base = [1.0, 4.0, 1.0, 1.0, 1.0]
    other = [4.0, 1.0, 1.0, 1.0, 1.0]

    results = []
    for scale in (1.0, 1000.0):
        w2 = World(WorldConfig(seed=3, sink=SinkConfig(spoilage=(0.0,) * 5)))
        a = _post(w2, i, j, [v * scale for v in base], other)
        w2.step(a)
        assert w2.last_trades, f"scale {scale} should still trade"
        tr = w2.last_trades[0]
        results.append((tr.good_a, tr.good_b, round(tr.price, 6), round(tr.qty_a, 6)))
    assert results[0] == results[1], f"scale changed the trade: {results}"


def test_identical_price_vectors_do_not_trade():
    """With no disagreement there is no gain, whatever the quantities on offer."""
    w = World(WorldConfig(seed=3))
    i, j = _two_in_a_region(w)
    same = [1.0, 2.0, 3.0, 1.0, 1.0]
    w.step(_post(w, i, j, same, same))
    assert w.last_trades == []


def test_a_worthless_good_is_never_accepted_for_a_valuable_one():
    """The artifact this guards against is subtle and broke two scenarios.

    Prices are floored before dividing, so a good priced at zero yields an
    enormous exchange rate. Scoring surplus with the floored price then lets
    `eps * rate` masquerade as gain. Surplus is scored with the true price, so a
    good genuinely valued at zero contributes exactly zero.
    """
    w = World(WorldConfig(seed=3, sink=SinkConfig(spoilage=(0.0,) * 5)))
    i, j = _two_in_a_region(w)
    # i wants only banana and holds only apple; j wants only apple and holds only
    # cherry. The single available exchange -- i's apple for j's cherry -- makes j
    # strictly better off and leaves i exactly indifferent, so it must not happen.
    w.inventory[i] = 0.0
    w.inventory[j] = 0.0
    w.inventory[i, 0] = 2.0   # apple
    w.inventory[j, 2] = 2.0   # cherry
    a = _post(w, i, j, [0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], qty=2.0)
    w.step(a)
    assert w.last_trades == [], "an indifferent side must not be traded into"


def test_transit_takes_distance_over_mobility_ticks():
    cfg = WorldConfig(seed=5, n_agents=4, topology=Topology.line(2, weight=4.0))
    w = World(cfg)
    w.mobility[:] = 1.0
    w.region[:] = 0
    a = Actions.idle(w.n_agents, w.n_goods)
    a.move[0] = int(w.topology.out_edges(0)[0])
    w.step(a)
    assert w.region[0] == IN_TRANSIT
    for _ in range(3):
        w.step(Actions.idle(w.n_agents, w.n_goods))
    assert w.region[0] == 1, "distance 4 at mobility 1 should arrive on the 4th tick"


def test_in_transit_agents_cannot_act():
    cfg = WorldConfig(seed=5, n_agents=4, topology=Topology.line(2, weight=10.0))
    w = World(cfg)
    w.mobility[:] = 1.0
    w.region[:] = 0
    a = Actions.idle(w.n_agents, w.n_goods)
    a.move[0] = int(w.topology.out_edges(0)[0])
    w.step(a)
    before = w.inventory[0].copy()
    b = Actions.idle(w.n_agents, w.n_goods)
    b.effort[0, 0] = 1.0
    w.step(b)
    assert w.last_production[0].sum() == 0.0, "no production while in transit"
    assert w.inventory[0, 0] <= before[0]


def test_chokepoint_capacity_refuses_traversal():
    topo = Topology.line(2, weight=3.0).with_capacity(0, 0)
    w = World(WorldConfig(seed=5, n_agents=6, topology=topo))
    w.region[:] = 0
    a = Actions.idle(w.n_agents, w.n_goods)
    a.move[:] = 0
    w.step(a)
    assert (w.region == 0).all(), "capacity 0 must refuse every traversal"


def test_anneal_drives_token_weight_to_target():
    cfg = WorldConfig(seed=7)
    cfg.token.anneal_ticks, cfg.token.anneal_end = 50, 0.0
    w = World(cfg)
    start = w.token_weight()
    for _ in range(60):
        w.step(Actions.idle(w.n_agents, w.n_goods))
    assert start > 0.01 and w.token_weight() == pytest.approx(0.0, abs=1e-6)


def test_identical_agents_end_up_identical():
    """Symmetry invariant, and the sharpest test of fairness in the transition.

    `mutual_gains` holds two interchangeable pairs. If anything in the tick
    depends on agent index rather than agent state, they diverge — which is
    exactly what happened: production looped over agents in order, each drawing
    from the stock the last one left, so once a resource ran short the low-index
    agents took all of it and the others were starved permanently.
    """
    from ecognomy.scenarios import get

    sc = get("mutual_gains")
    cfg = sc.config()
    cfg.visibility.sight_mean, cfg.visibility.sight_spread = 20.0, 0.0
    w = World(cfg, scenario=sc)
    run(w, MyopicPolicy(), 300)

    assert np.allclose(sc.theta[0], sc.theta[2]) and np.allclose(sc.theta[1], sc.theta[3])
    assert w.cumulative_reward[0] == pytest.approx(w.cumulative_reward[2], rel=1e-6)
    assert w.cumulative_reward[1] == pytest.approx(w.cumulative_reward[3], rel=1e-6)


def test_scarce_stock_is_shared_not_raced_for():
    """A shortfall is split pro rata, so no agent is served before another."""
    cfg = WorldConfig(seed=2, n_agents=6, sink=SinkConfig(spoilage=(0.0,) * 5))
    cfg.resource.stock_capacity, cfg.resource.regen_rate = 1.0, 0.0
    w = World(cfg)
    w.region[:] = 0
    w.efficiency[:] = 0.0
    w.efficiency[:, 0] = 2.0          # everyone wants the same scarce good, equally
    a = Actions.idle(w.n_agents, w.n_goods)
    a.effort[:, 0] = 1.0
    w.step(a)

    made = w.last_production[:, 0]
    assert made.sum() == pytest.approx(1.0, rel=1e-5), "cannot produce more than the stock"
    assert np.allclose(made, made[0]), f"stock was raced for, not shared: {made}"


def test_production_is_gated_by_regional_stock():
    cfg = WorldConfig(seed=8, n_agents=4)
    cfg.resource.stock_capacity, cfg.resource.regen_rate = 0.0, 0.0
    w = World(cfg)
    w.region[:] = 0
    a = Actions.idle(w.n_agents, w.n_goods)
    a.effort[:, 0] = 1.0
    w.step(a)
    assert w.last_production.sum() == 0.0


# ------------------------------------------------------- failure modes

def test_failure_no_comparative_advantage():
    """shape_spread == 0 gives every agent the same ranking over goods."""
    cfg = WorldConfig(seed=2, production=ProductionConfig(shape_spread=0.0, scale_spread=0.5))
    w = World(cfg)
    ranks = np.argsort(w.efficiency, axis=1)
    assert (ranks == ranks[0]).all(), "identical shape must remove comparative advantage"


def test_failure_identical_preferences_kill_gains_from_trade():
    """Under a linear reward, gains from trade come only from valuing goods
    differently. Two agents with the same preferences have nothing to gain from
    exchanging, however lopsided their holdings — so a population with uniform
    tastes is a dead world regardless of how production is arranged.
    """
    w = World(WorldConfig(seed=9, n_agents=8, sink=SinkConfig(spoilage=(0.0,) * 5)))
    w.theta[:] = np.full(w.n_goods, 1.0 / w.n_goods)  # everyone wants the same thing
    m = run(w, MyopicPolicy(), 200)
    assert m.summary()["total_trades"] == 0


def test_failure_no_meetings_means_no_trade():
    cfg = WorldConfig(seed=4, visibility=VisibilityConfig(sight_mean=0.0))
    m = run(World(cfg), RandomPolicy(), 200)
    assert m.summary()["total_trades"] == 0.0


def test_failure_isolated_regions_prevent_movement():
    w = World(WorldConfig(seed=4, n_agents=10, topology=Topology.isolated(3)))
    start = w.region.copy()
    run(w, RandomPolicy(), 100)
    assert (w.region == start).all(), "no edges means no agent may ever relocate"


def test_failure_no_drain_accumulates_without_bound():
    """A faucet with no sink only ever accumulates -- the classic homebrew failure."""
    cfg = WorldConfig(seed=6, sink=SinkConfig(spoilage=(0.0,) * 5))
    cfg.resource.regen_rate = 1.0
    w = World(cfg)
    a = Actions.idle(w.n_agents, w.n_goods)
    a.effort[:, 0] = 1.0
    stock = []
    for _ in range(50):
        w.step(a)
        stock.append(w.inventory.sum())
    assert stock[-1] > stock[0] * 5, "with no spoilage and no consumption, goods must pile up"


def test_random_play_does_not_produce_an_economy():
    """The double coincidence of wants must actually bite.

    Random offers should almost never match. If they do, matching is too
    permissive and any later 'emergence' would be an artifact of the mechanism.
    """
    w = World(WorldConfig(seed=1))
    m = run(w, RandomPolicy(), 300)
    trades_per_tick = m.summary()["total_trades"] / 300
    assert trades_per_tick < w.n_agents, f"random play traded too easily: {trades_per_tick}/tick"
