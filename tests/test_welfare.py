"""Welfare accounting and the autarky counterfactual.

Total welfare is the designer's score, so the arithmetic behind it has to be
exact and the baseline has to be a genuine no-market run.
"""

import math

import numpy as np
import pytest

from ecognomy.baseline import Comparison, compare, run_baseline
from ecognomy.policy import MyopicPolicy, RandomPolicy, run
from ecognomy.scenarios import get
from ecognomy.world import World


def _cfg(scenario, **kw):
    cfg = scenario.config()
    cfg.visibility.sight_mean = kw.pop("sight", 20.0)
    cfg.visibility.sight_spread = 0.0
    return cfg


def test_welfare_is_the_sum_of_per_agent_rewards():
    sc = get("mutual_gains")
    w = World(_cfg(sc), scenario=sc)
    m = run(w, MyopicPolicy(), 200)
    assert w.welfare == pytest.approx(float(w.cumulative_reward.sum()))
    assert w.welfare == pytest.approx(m.series("reward_total").sum(), rel=1e-4)
    assert m.summary()["welfare"] == pytest.approx(w.welfare, rel=1e-4)


def test_consumption_utility_is_gross_of_costs():
    """Utility is the pleasure; welfare nets off the effort spent getting it."""
    sc = get("mutual_gains")
    w = World(_cfg(sc), scenario=sc)
    run(w, MyopicPolicy(), 200)
    assert w.cumulative_utility.sum() > w.cumulative_reward.sum()


def test_baseline_run_executes_no_trades():
    """The counterfactual must genuinely have no market."""
    sc = get("mutual_gains")
    base_world, base_metrics = run_baseline(_cfg(sc), MyopicPolicy(), 200,
                                            scenario=sc)
    assert base_metrics.summary()["total_trades"] == 0


def test_trade_beats_autarky_where_it_should():
    sc = get("mutual_gains")
    c = compare(_cfg(sc), MyopicPolicy(), 250, scenario=sc)
    assert c.gain > 0
    assert c.verdict == "trade is creating value"
    assert c.share_of_agents_helped() == 1.0


def test_no_gain_where_no_trade_is_possible():
    """Triangular and autarky must both come out exactly flat."""
    for name in ("triangular", "autarky"):
        sc = get(name)
        c = compare(_cfg(sc), MyopicPolicy(), 200, scenario=sc)
        assert c.gain == pytest.approx(0.0, abs=1e-9), name
        assert c.verdict == "trade is doing nothing", name


def test_ratio_is_undefined_rather_than_inverted_on_a_negative_baseline():
    """With both welfares negative the quotient reverses meaning.

    -0.96 / -1.04 is 0.92, which reads as a loss for a run that is actually
    ahead. `gain` is the safe reading and `ratio` must decline to answer.
    """
    c = Comparison(welfare=-0.96, baseline_welfare=-1.04,
                   consumption_utility=0.0, baseline_consumption_utility=0.0,
                   per_agent=np.zeros(2), baseline_per_agent=np.zeros(2))
    assert c.gain > 0
    assert math.isnan(c.ratio)
    assert c.verdict == "trade is creating value"


def test_share_of_agents_helped_catches_a_lopsided_market():
    """A total can rise while most agents lose; that has to be visible."""
    c = Comparison(welfare=100.0, baseline_welfare=50.0,
                   consumption_utility=0.0, baseline_consumption_utility=0.0,
                   per_agent=np.array([90.0, 4.0, 3.0, 3.0]),
                   baseline_per_agent=np.array([10.0, 13.0, 13.0, 14.0]))
    assert c.gain > 0
    assert c.share_of_agents_helped() == 0.25


def test_random_policy_can_fall_below_its_own_autarky():
    """Trading badly must be able to cost you. A world where the market can only
    help would be assuming its own answer."""
    sc = get("autarky")
    c = compare(_cfg(sc), RandomPolicy(), 300, scenario=sc)
    assert c.gain < 0
    assert c.verdict == "trade is destroying value"


def test_welfare_panel_builds_with_and_without_a_baseline(tmp_path):
    from ecognomy.recorder import RunData, simulate
    from ecognomy.viewer.panels import welfare

    sc = get("mutual_gains")
    for flag, name in ((True, "with"), (False, "without")):
        out = tmp_path / name
        simulate(_cfg(sc), MyopicPolicy(), ticks=60, out=out,
                 scenario=sc, baseline=flag)
        data = RunData(out)
        assert data.has("baseline_welfare") is flag
        assert welfare.build(data) is not None
