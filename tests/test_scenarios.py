"""Scenario tests.

Two layers. **Structure** tests check each scenario is the world it claims to be —
if `triangular` accidentally contained a bilateral opportunity it would stop
discriminating between rungs and would quietly pass for the wrong reason.
**Capability** tests record what each policy can and cannot do on them.

The capability tests are written to be *falsified by progress*: when a rung
capable of indirect exchange arrives, `test_no_policy_yet_solves_triangular`
should start failing, and that failure is the result being demonstrated.
"""

import numpy as np
import pytest

from ecognomy.policy import MyopicPolicy, RandomPolicy, run
from ecognomy.scenarios import ALL, get


@pytest.mark.parametrize("name", sorted(ALL))
def test_scenario_shapes_are_consistent(name):
    sc = get(name)
    n, g = sc.n_agents, sc.n_goods
    assert sc.theta.shape == (n, g)
    assert sc.efficiency.shape == (n, g)
    assert sc.inventory.shape == (n, g)
    assert sc.region.shape == (n,)
    assert len(sc.goods) == g
    assert (sc.theta >= 0).all() and (sc.efficiency >= 0).all()


@pytest.mark.parametrize("name", sorted(ALL))
def test_declared_solvability_matches_structure(name):
    """`solvable_bilaterally` must agree with the actual want/produce graph."""
    sc = get(name)
    assert sc.has_bilateral_double_coincidence() == sc.solvable_bilaterally


@pytest.mark.parametrize("name", sorted(ALL))
def test_scenario_applies_exactly(name):
    """Nothing may be left sampled — a scenario is fully specified."""
    sc = get(name)
    w = sc.world()
    assert np.allclose(w.theta, sc.theta)
    assert np.allclose(w.efficiency, sc.efficiency)
    assert np.allclose(w.inventory, sc.inventory)
    assert (w.region == sc.region).all()


def test_triangular_has_a_cycle_but_no_bilateral_pair():
    """The discriminator's defining property.

    Every agent produces something *somebody* wants, so a cycle exists and the
    world is not dead. But no pair can trade directly, so only a policy able to
    hold a good it does not consume can move anything at all.
    """
    sc = get("triangular")
    assert sc.has_supply_cycle()
    assert not sc.has_bilateral_double_coincidence()


def test_autarky_has_neither():
    """The negative control is genuinely inert."""
    sc = get("autarky")
    assert not sc.has_supply_cycle()
    assert not sc.has_bilateral_double_coincidence()


# ------------------------------------------------------------- capability

def test_myopic_trades_when_direct_exchange_suffices():
    """`mutual_gains` is the easiest possible trade; failing it means broken."""
    sc = get("mutual_gains")
    w = sc.world(visibility__sight_mean=20, visibility__sight_spread=0.0)
    s = run(w, MyopicPolicy(), 300).summary()
    assert s["total_trades"] > 0


def test_trading_pays_in_mutual_gains():
    """Trade must actually raise welfare, or the mechanism is moving goods the
    wrong way."""
    sc = get("mutual_gains")
    traded = run(sc.world(visibility__sight_mean=20, visibility__sight_spread=0.0),
                 MyopicPolicy(), 300).summary()
    idle = run(sc.world(visibility__sight_mean=0),
               MyopicPolicy(), 300).summary()
    assert traded["total_trades"] > 0 and idle["total_trades"] == 0
    assert traded["mean_reward"] > idle["mean_reward"]


def test_no_rational_policy_yet_solves_triangular():
    """The capability frontier, recorded as a test.

    No current policy can hold a good it does not consume, so none can close the
    three-cycle at any setting. **When a rung capable of indirect exchange lands,
    this test should fail** — that failure is the result.

    Scoped to rational policies deliberately. RandomPolicy does trade here, a
    handful of times, because it posts ratios without consulting its own
    preferences; stumbling into an exchange is not solving the scenario, which
    the companion test below pins down.
    """
    sc = get("triangular")
    for kappa in (0.5, 1.0, 3.0, 10.0):
        w = sc.world(visibility__sight_mean=20, visibility__sight_spread=0.0)
        s = run(w, MyopicPolicy(kappa=kappa), 300).summary()
        assert s["total_trades"] == 0, f"myopic traded at kappa={kappa}"


def test_triangular_is_traversable_but_not_by_a_myopic_agent():
    """The positive control for the discriminator.

    RandomPolicy posts prices unrelated to its preferences, so it accepts goods
    it does not consume and stumbles right around the cycle -- it earns real
    welfare here. That matters: it proves the three-cycle is *traversable*, so
    the myopic agent's exact zero is a missing capability rather than a world
    built to be dead.

    A myopic agent refuses every available exchange because each leaves it
    strictly indifferent, and will not even produce, since the only good it can
    make is worthless to it. Producing on spec pays only if someone will trade
    for it later, which this rung cannot represent.
    """
    sc = get("triangular")
    random = run(sc.world(visibility__sight_mean=20, visibility__sight_spread=0.0),
                 RandomPolicy(), 300).summary()
    myopic = run(sc.world(visibility__sight_mean=20, visibility__sight_spread=0.0),
                 MyopicPolicy(), 300).summary()

    assert random["total_trades"] > 0, "the cycle should be traversable at all"
    assert random["welfare"] > 0, "traversing it should produce real welfare"
    assert myopic["total_trades"] == 0
    assert myopic["welfare"] == pytest.approx(0.0, abs=1e-9)


def test_myopic_will_not_produce_for_exchange():
    """Triangular demands more than indirect exchange: production on spec.

    Agent 0 can only make apple, which it values at zero. Producing is worth
    doing only if someone will trade for it later, so a policy scoring goods by
    consumption alone will not even start the chain.
    """
    sc = get("triangular")
    w = sc.world(visibility__sight_mean=20, visibility__sight_spread=0.0)
    s = run(w, MyopicPolicy(), 300).summary()
    assert s["mean_fraction_producing"] == 0.0


def test_rational_policies_never_trade_in_autarky():
    """Everyone already makes exactly what they want; there is nothing to gain."""
    sc = get("autarky")
    for policy in (MyopicPolicy(), MyopicPolicy(kappa=1.0), MyopicPolicy(kappa=10.0)):
        w = sc.world(visibility__sight_mean=20, visibility__sight_spread=0.0)
        assert run(w, policy, 200).summary()["total_trades"] == 0


def test_autarky_trades_are_welfare_destroying():
    """The mechanism does not protect an agent from its own bad offers.

    RandomPolicy trades in autarky, and doing so leaves it worse off than the
    same policy with the market switched off. That is the correct behaviour:
    execution checks that both sides *posted* compatible ratios, not that either
    posted a sensible one. It is also why `autarky` is a useful control — a
    rational policy trading here would be a real bug.
    """
    sc = get("autarky")
    on = run(sc.world(visibility__sight_mean=20, visibility__sight_spread=0.0), RandomPolicy(), 300).summary()
    off = run(sc.world(visibility__sight_mean=0), RandomPolicy(), 300).summary()
    assert on["total_trades"] > 0
    assert on["mean_reward"] < off["mean_reward"]


def test_scenarios_conserve_goods_through_trade():
    sc = get("mutual_gains")
    w = sc.world(visibility__sight_mean=20, visibility__sight_spread=0.0)
    w.config.sink.spoilage = (0.0,) * sc.n_goods
    before = w.inventory.sum()
    run(w, MyopicPolicy(), 200)
    balance = before + w.goods_created - w.goods_destroyed
    assert w.inventory.sum() == pytest.approx(balance, rel=1e-3)
