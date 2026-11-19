import numpy as np
import pandas as pd
import pytest

from blacklitterman import (
    efficient_frontier,
    efficient_return,
    max_quadratic_utility,
    max_sharpe,
    min_volatility,
    portfolio_performance,
)

WIDE = (-10.0, 10.0)  # effectively unconstrained


@pytest.fixture
def mu(cov) -> pd.Series:
    return pd.Series([0.07, 0.08, 0.06, 0.10, 0.03], index=cov.index)


def sharpe(w, mu, cov, rf=0.0):
    w = np.asarray(w)
    return (mu.to_numpy() @ w - rf) / np.sqrt(w @ cov.to_numpy() @ w)


def test_max_sharpe_matches_closed_form_tangency(mu, cov):
    rf = 0.02
    w = max_sharpe(mu, cov, risk_free_rate=rf, bounds=WIDE)
    ref = np.linalg.solve(cov.to_numpy(), mu.to_numpy() - rf)
    ref /= ref.sum()
    np.testing.assert_allclose(w, ref, atol=1e-5)


def test_max_sharpe_long_only_beats_random_portfolios(mu, cov):
    w = max_sharpe(mu, cov, risk_free_rate=0.02)
    assert (w >= 0).all() and w.sum() == pytest.approx(1.0)
    best = sharpe(w, mu, cov, 0.02)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        assert sharpe(rng.dirichlet(np.ones(5)), mu, cov, 0.02) <= best + 1e-6


def test_max_sharpe_respects_upper_bound(mu, cov):
    w = max_sharpe(mu, cov, bounds=(0.0, 0.3))
    assert w.max() <= 0.3 + 1e-6
    assert w.sum() == pytest.approx(1.0)


def test_max_sharpe_requires_positive_excess(mu, cov):
    with pytest.raises(ValueError, match="risk-free"):
        max_sharpe(mu, cov, risk_free_rate=0.5)


def test_min_volatility_matches_closed_form(cov):
    w = min_volatility(cov, bounds=WIDE)
    ones = np.ones(len(cov))
    ref = np.linalg.solve(cov.to_numpy(), ones)
    np.testing.assert_allclose(w, ref / ref.sum(), atol=1e-6)


def test_efficient_return_hits_target(mu, cov):
    w = efficient_return(mu, cov, target_return=0.08)
    assert mu @ w == pytest.approx(0.08, abs=1e-6)
    assert (w >= -1e-8).all()


def test_efficient_return_infeasible(mu, cov):
    with pytest.raises(RuntimeError):
        efficient_return(mu, cov, target_return=0.5)


def test_quadratic_utility_closed_form(mu, cov):
    # with sum(w)=1 and no bounds: w = S^-1 (mu - g 1)/delta, g chosen so sum = 1
    delta = 3.0
    w = max_quadratic_utility(mu, cov, delta, bounds=WIDE)
    S_inv = np.linalg.inv(cov.to_numpy())
    ones = np.ones(len(mu))
    g = (ones @ S_inv @ mu.to_numpy() - delta) / (ones @ S_inv @ ones)
    ref = S_inv @ (mu.to_numpy() - g) / delta
    np.testing.assert_allclose(w, ref, atol=1e-5)


def test_frontier_is_increasing_and_starts_at_min_vol(mu, cov):
    f = efficient_frontier(mu, cov, n_points=20)
    assert len(f) == 20
    assert np.all(np.diff(f["return"]) > -1e-9)
    assert np.all(np.diff(f["volatility"]) > -1e-6)
    w_min = min_volatility(cov)
    assert f["volatility"].iloc[0] == pytest.approx(np.sqrt(w_min @ cov @ w_min), rel=1e-4)
    assert f["return"].iloc[-1] == pytest.approx(mu.max(), abs=1e-6)
    np.testing.assert_allclose(f[list(cov.index)].sum(axis=1), 1.0, atol=1e-6)


def test_tangency_lies_on_frontier(mu, cov):
    f = efficient_frontier(mu, cov, n_points=200)
    w = max_sharpe(mu, cov)
    perf = portfolio_performance(w, mu, cov)
    frontier_vol = np.interp(perf["expected_return"], f["return"], f["volatility"])
    assert perf["volatility"] == pytest.approx(frontier_vol, rel=1e-3)


def test_series_in_series_out(mu, cov):
    w = max_sharpe(mu, cov)
    assert isinstance(w, pd.Series) and list(w.index) == list(cov.index)
    w_np = max_sharpe(mu.to_numpy(), cov.to_numpy())
    assert isinstance(w_np, np.ndarray)
    np.testing.assert_allclose(w, w_np, atol=1e-6)


def test_portfolio_performance(mu, cov):
    w = pd.Series(0.2, index=cov.index)
    p = portfolio_performance(w.iloc[::-1], mu, cov, risk_free_rate=0.01)  # aligned by label
    assert p["expected_return"] == pytest.approx(mu.mean())
    assert p["sharpe_ratio"] == pytest.approx((mu.mean() - 0.01) / p["volatility"])
    assert p["effective_n"] == pytest.approx(5.0)
