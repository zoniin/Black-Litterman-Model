import numpy as np
import pandas as pd
import pytest

from blacklitterman import (
    backtest,
    black_litterman_strategy,
    equal_weight,
    market_cap_weight,
    mean_variance,
    momentum_views,
)
from blacklitterman.backtest import Context, ViewSpec, performance_stats
from blacklitterman.data import market_cap_history

LOOKBACK = 60


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n, k = 400, 4
    drift = np.array([0.0008, 0.0003, -0.0002, 0.0004])
    rets = drift + rng.normal(0, 0.012, size=(n, k)) @ np.linalg.cholesky(0.5 * np.eye(k) + 0.5).T
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=list("ABCD"))


@pytest.fixture
def caps(prices) -> pd.DataFrame:
    return market_cap_history(prices, pd.Series([4.0, 3.0, 2.0, 1.0], index=prices.columns))


def fixed(weights: dict):
    return lambda ctx: pd.Series(weights)


def test_single_asset_tracks_asset(prices):
    res = backtest(prices, {"A": fixed({"A": 1.0})}, lookback=LOOKBACK, cost_bps=0)
    asset = prices["A"].pct_change().loc[res.returns.index]
    np.testing.assert_allclose(res.returns["A"], asset, atol=1e-14)


def test_cash_earns_risk_free(prices):
    rf = 0.05
    res = backtest(prices, {"cash": fixed({})}, lookback=LOOKBACK, risk_free_rate=rf)
    np.testing.assert_allclose(res.returns["cash"], (1 + rf) ** (1 / 252) - 1)
    assert res.stats.loc["cash", "CAGR"] == pytest.approx(rf)


def test_drift_between_rebalances_matches_manual(prices):
    w0 = {"A": 0.5, "B": 0.5}
    res = backtest(prices, {"ew": fixed(w0)}, lookback=LOOKBACK, cost_bps=0, rebalance="Y")
    # within the first holding period this is buy-and-hold of a 50/50 mix
    first_rebal = res.weights["ew"].index[0]
    second = res.weights["ew"].index[1]
    window = res.returns.loc[:second, "ew"]
    p = prices.loc[first_rebal:second, ["A", "B"]]
    bh = (p / p.iloc[0]) @ np.array([0.5, 0.5])
    np.testing.assert_allclose((1 + window).cumprod().to_numpy(), bh.iloc[1:].to_numpy(), rtol=1e-12)


def test_costs_charged_on_turnover(prices):
    gross = backtest(prices, {"ew": equal_weight()}, lookback=LOOKBACK, cost_bps=0)
    net = backtest(prices, {"ew": equal_weight()}, lookback=LOOKBACK, cost_bps=25)
    diff = (gross.returns["ew"] - net.returns["ew"]).sum()
    assert diff == pytest.approx(gross.turnover["ew"].sum() * 25e-4)
    assert gross.turnover["ew"].iloc[0] == pytest.approx(1.0)  # initial buy from cash


def test_rebalances_monthly_from_first_tradable_day(prices):
    res = backtest(prices, {"ew": equal_weight()}, lookback=LOOKBACK)
    reb = res.weights["ew"].index
    assert reb[0] == prices.index[LOOKBACK]  # returns start one day after prices
    assert reb[1:].to_period("M").is_unique
    assert res.returns.index[0] == prices.index[LOOKBACK + 1]


def test_no_lookahead(prices, caps):
    """Scrambling the future must not change any decision made before it."""
    strategies = {
        "mv": mean_variance(),
        "bl": black_litterman_strategy(momentum_views(lookback=40, skip=5, n_long=1, n_short=1)),
    }
    base = backtest(prices, strategies, market_caps=caps, lookback=LOOKBACK)
    cut = prices.index[250]
    shocked = prices.copy()
    shocked.loc[shocked.index > cut] *= np.linspace(1, 3, (shocked.index > cut).sum())[:, None] ** [1, -1, 2, 0.5]
    shocked_caps = caps.copy()
    shocked_caps.loc[shocked_caps.index > cut] *= 10.0
    alt = backtest(shocked, strategies, market_caps=shocked_caps, lookback=LOOKBACK)
    for name in strategies:
        before = base.weights[name].index <= cut
        np.testing.assert_allclose(base.weights[name][before], alt.weights[name][before], atol=1e-6)
        pd.testing.assert_series_equal(base.returns[name].loc[:cut], alt.returns[name].loc[:cut], atol=1e-10)


def test_context_window(prices):
    seen = []

    def spy(ctx: Context):
        seen.append(ctx)
        return pd.Series(0.25, index=ctx.assets)

    backtest(prices, {"spy": spy}, lookback=LOOKBACK)
    for ctx in seen:
        assert ctx.prices.index[-1] == ctx.date
        assert ctx.returns.index[-1] == ctx.date
        assert len(ctx.returns) == LOOKBACK


def test_bl_without_views_holds_market(prices, caps):
    res = backtest(
        prices,
        {"bl": black_litterman_strategy(), "mkt": market_cap_weight()},
        market_caps=caps,
        lookback=LOOKBACK,
    )
    np.testing.assert_allclose(res.weights["bl"], res.weights["mkt"], atol=1e-5)


def test_momentum_views_pick_winners():
    idx = pd.bdate_range("2020-01-01", periods=300)
    t = np.arange(300)
    prices = pd.DataFrame({"UP": 100 * 1.002**t, "FLAT": 100.0 + 0 * t, "DOWN": 100 * 0.998**t}, index=idx)
    ctx = Context(idx[-1], prices, prices.pct_change().iloc[-60:], None, 0.0)
    (view,) = momentum_views(n_long=1, n_short=1, premium=None, scale=1.0)(ctx)
    assert view.portfolio == {"UP": 1.0, "DOWN": -1.0}
    expected = (1.002**231 - 0.998**231) * 252 / 231
    assert view.q == pytest.approx(expected)
    (fixed_view,) = momentum_views(n_long=1, n_short=1, premium=0.04)(ctx)
    assert fixed_view.portfolio == view.portfolio and fixed_view.q == 0.04
    assert momentum_views()(Context(idx[10], prices.iloc[:11], prices.pct_change().iloc[1:11], None, 0.0)) == []


def test_custom_view_generator_moves_weights(prices, caps):
    def bullish_d(ctx):
        return [ViewSpec({"D": 1.0}, 0.40, confidence=0.9)]

    res = backtest(
        prices,
        {"bl": black_litterman_strategy(bullish_d), "mkt": market_cap_weight()},
        market_caps=caps,
        lookback=LOOKBACK,
    )
    assert (res.weights["bl"]["D"] > res.weights["mkt"]["D"] + 0.05).all()


def test_cap_strategies_require_caps(prices):
    with pytest.raises(ValueError, match="market_caps"):
        backtest(prices, {"mkt": market_cap_weight()}, lookback=LOOKBACK)


def test_too_little_history(prices):
    with pytest.raises(ValueError, match="lookback"):
        backtest(prices.iloc[:50], {"ew": equal_weight()}, lookback=LOOKBACK)


def test_performance_stats_known_series():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.DataFrame({"x": [0.10, -0.50, 0.20, 0.0]}, index=idx)
    s = performance_stats(r, frequency=4)
    assert s.loc["x", "max_drawdown"] == pytest.approx(-0.5)
    assert s.loc["x", "CAGR"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)


def test_market_cap_history():
    p = pd.DataFrame({"A": [50.0, 100.0], "B": [20.0, 10.0]})
    caps = market_cap_history(p, pd.Series({"A": 1000.0, "B": 300.0}))
    assert caps.loc[0].to_dict() == {"A": 500.0, "B": 600.0}
    with pytest.raises(ValueError, match="missing"):
        market_cap_history(p, pd.Series({"A": 1.0}))


def test_relative_to(prices):
    res = backtest(prices, {"A": fixed({"A": 1.0}), "B": fixed({"B": 1.0})}, lookback=LOOKBACK, cost_bps=0)
    rel = res.relative_to("A")
    active = res.returns["B"] - res.returns["A"]
    assert list(rel.index) == ["B"]
    assert rel.loc["B", "active_return"] == pytest.approx(active.mean() * 252)
    assert rel.loc["B", "information_ratio"] == pytest.approx(active.mean() / active.std() * np.sqrt(252))
