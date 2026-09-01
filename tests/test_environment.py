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
from ecognomy.metrics import arbitrage_depth, round_trip
from ecognomy.utility import honest_ask, marginal_value, utility
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


def _post(w, i, j, values_i, values_j, qty=1.0):
    """Both agents post the honest matrix implied by a valuation, offering `qty`.

    Honest postings carry no spread, so this is the subset of the action space
    the old price vector could reach -- which is what makes it the right helper
    for the tests that predate the matrix.
    """
    a = Actions.idle(w.n_agents, w.n_goods)
    theta = np.zeros((2, w.n_goods))
    theta[0, :len(values_i)] = values_i
    theta[1, :len(values_j)] = values_j
    ask = honest_ask(theta)
    a.ask[i], a.ask[j] = ask[0], ask[1]
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
    """Neither side's posted rate sets the price; the mean between them does.

    Stated without a numeraire: the giver's floor is its own ask, the receiver's
    ceiling is the reciprocal of the ask it posted the other way, and execution
    lands strictly between them.
    """
    w = World(WorldConfig(seed=3, sink=SinkConfig(spoilage=(0.0,) * 5)))
    i, j = _two_in_a_region(w)
    a = _post(w, i, j, [1.0, 2.0, 1.0, 1.0, 1.0], [4.0, 1.0, 1.0, 1.0, 1.0])
    w.step(a)
    tr = w.last_trades[0]
    floor = a.ask[i, tr.good_a, tr.good_b]
    ceiling = 1.0 / a.ask[j, tr.good_b, tr.good_a]
    assert tr.price == pytest.approx(float(np.sqrt(floor * ceiling)), rel=1e-4)
    assert floor < tr.price < ceiling, "execution must land strictly inside the interval"


def test_the_split_is_the_only_numeraire_free_one():
    """`r = lo**(1-k) * hi**k` must give the reciprocal rate when the same trade
    is written in the other good's units. Only k = 1/2 does, which is why the
    geometric mean is forced rather than chosen: every other split needs a
    nominated money good, and a barter economy has none.
    """
    lo, hi = 0.25, 4.0
    for k in (0.0, 0.25, 0.5, 0.75, 1.0):
        forward = lo ** (1 - k) * hi**k
        backward = (1 / hi) ** (1 - k) * (1 / lo) ** k
        consistent = forward * backward == pytest.approx(1.0, rel=1e-9)
        assert consistent == (k == 0.5), f"k={k} consistency should be {k == 0.5}"


def test_a_spread_is_expressible():
    """The whole point of the matrix, and the thing a price vector could not say.

    The agent will sell an apple for 2 bananas and buy one for 1.5, so a
    counterparty offering exactly 1.75 either way is refused in both directions.
    Under a vector, refusing at 1.75 one way *forces* acceptance at 1.75 the
    other, because the two rates were pinned to be reciprocal.
    """
    w = World(WorldConfig(seed=3, sink=SinkConfig(spoilage=(0.0,) * 5)))
    i, j = _two_in_a_region(w)
    apple, banana = 0, 1

    for direction in (1, -1):
        a = Actions.idle(w.n_agents, w.n_goods)
        a.ask[i, apple, banana] = 2.0     # sell an apple only for 2+ bananas
        a.ask[i, banana, apple] = 1 / 1.5  # buy an apple only at 1.5 or less
        a.ask[j, apple, banana] = 1.75 if direction > 0 else 1e9
        a.ask[j, banana, apple] = 1 / 1.75 if direction < 0 else 1e9
        a.max_trade[i] = a.max_trade[j] = 1.0
        w.step(a)
        assert w.last_trades == [], f"1.75 must be refused in direction {direction}"

    assert round_trip(a.ask[i])[apple, banana] == pytest.approx(2.0 / 1.5, rel=1e-5)


def test_buying_queue_priority_costs_terms_of_trade():
    """There is no free way to jump the fill queue.

    Depth is `1 / sqrt(ask_i * ask_j)` and the executed rate is
    `sqrt(ask_i / ask_j)`, so an agent that softens its ask to deepen the cross
    moves both by exactly the same square root. Priority is always paid for in
    the rate received, which is what makes escalation self-limiting rather than
    merely discouraged.
    """
    i, j = _two_in_a_region(World(WorldConfig(seed=3)))
    apple, banana = 0, 1

    seen = []
    for mine in (2.0, 1.0, 0.5):
        w = World(WorldConfig(seed=3, sink=SinkConfig(spoilage=(0.0,) * 5)))
        a = Actions.idle(w.n_agents, w.n_goods)
        a.ask[i, apple, banana] = mine
        a.ask[j, banana, apple] = 0.1
        a.max_trade[i] = a.max_trade[j] = 1.0
        w.step(a)
        assert w.last_trades, f"ask {mine} against 0.1 must cross"
        seen.append((1.0 / np.sqrt(mine * 0.1), w.last_trades[0].price))

    depths = [d for d, _ in seen]
    rates = [r for _, r in seen]
    assert depths == sorted(depths), "softening the ask must deepen the cross"
    assert rates == sorted(rates, reverse=True), "and must worsen the rate received"
    for depth, rate in seen:
        assert depth * rate == pytest.approx(1.0 / 0.1, rel=1e-4), "paid one for one"


def test_scale_cannot_be_posted_at_all():
    """The exploit the old `_normalised` guard existed for is now unsayable.

    A price vector meant the same thing multiplied by any constant, so an agent
    could post 1000x bigger numbers, change nothing it was willing to do, and
    inflate its ranked surplus enough to buy the front of the queue. A rate
    matrix has no free scale -- it is already a ratio -- so there is nothing
    left to normalise and nothing to exploit.
    """
    theta = np.array([[1.0, 4.0, 1.0, 1.0, 1.0]])
    assert np.allclose(honest_ask(theta), honest_ask(theta * 1000.0), equal_nan=True)


def test_identical_postings_do_not_trade():
    """With no disagreement there is no gain, whatever the quantities on offer."""
    w = World(WorldConfig(seed=3))
    i, j = _two_in_a_region(w)
    same = [1.0, 2.0, 3.0, 1.0, 1.0]
    w.step(_post(w, i, j, same, same))
    assert w.last_trades == []


def test_a_worthless_good_is_never_accepted_for_a_valuable_one():
    """The artifact this guards against is subtle and broke two scenarios.

    An agent that does not consume a good demands an infinite quantity of it,
    so no rate crosses and the swap cannot be reached at all. Under the price
    vector this needed a careful argument about flooring -- a good priced at
    zero yielded an enormous exchange rate, and scoring with the floored price
    let `eps * rate` masquerade as gain. A refusal states the same thing
    directly and leaves nothing to get wrong.
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


def test_random_play_does_not_produce_an_economy():
    """The double coincidence of wants must actually bite.

    Random offers should almost never match. If they do, matching is too
    permissive and any later 'emergence' would be an artifact of the mechanism.
    """
    w = World(WorldConfig(seed=1))
    m = run(w, RandomPolicy(), 300)
    trades_per_tick = m.summary()["total_trades"] / 300
    assert trades_per_tick < w.n_agents, f"random play traded too easily: {trades_per_tick}/tick"
