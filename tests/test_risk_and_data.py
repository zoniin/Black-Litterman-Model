import sys
import types

import numpy as np
import pandas as pd
import pytest

from blacklitterman import cov_to_corr, ledoit_wolf, returns_from_prices, sample_cov
from blacklitterman import data as bl_data


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rets = rng.multivariate_normal([0.0004, 0.0003, 0.0002], np.diag([1e-4, 2e-4, 3e-4]), size=500)
    idx = pd.bdate_range("2020-01-01", periods=500)
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=["A", "B", "C"])


def test_simple_returns(prices):
    r = returns_from_prices(prices)
    assert len(r) == len(prices) - 1
    np.testing.assert_allclose(r.iloc[0], prices.iloc[1] / prices.iloc[0] - 1)


def test_log_returns(prices):
    r = returns_from_prices(prices, log=True)
    np.testing.assert_allclose(r.iloc[0], np.log(prices.iloc[1] / prices.iloc[0]))


def test_sample_cov_annualised(prices):
    r = returns_from_prices(prices)
    np.testing.assert_allclose(sample_cov(r), r.cov() * 252)


def test_ledoit_wolf_matches_sklearn(prices):
    sk = pytest.importorskip("sklearn.covariance")
    r = returns_from_prices(prices)
    ours = ledoit_wolf(r, frequency=1)
    ref = sk.LedoitWolf().fit(r.to_numpy())
    np.testing.assert_allclose(ours, ref.covariance_, rtol=1e-10)
    assert ours.attrs["shrinkage"] == pytest.approx(ref.shrinkage_)


def test_ledoit_wolf_is_well_conditioned_when_p_exceeds_n():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.standard_normal((20, 40)) * 0.01)
    assert np.linalg.eigvalsh(ledoit_wolf(r).to_numpy()).min() > 0
    assert np.linalg.eigvalsh(sample_cov(r).to_numpy()).min() < 1e-12


def test_cov_to_corr(cov):
    c = cov_to_corr(cov)
    np.testing.assert_allclose(np.diag(c), 1.0)
    assert c.loc["US", "EU"] == pytest.approx(0.75)


def test_market_implied_risk_aversion():
    rng = np.random.default_rng(3)
    daily = rng.normal(0.08 / 252, 0.16 / np.sqrt(252), size=252 * 40)
    prices = pd.Series(100 * np.cumprod(1 + daily))
    r = prices.pct_change().dropna()
    expected = (r.mean() * 252 - 0.02) / (r.var() * 252)
    assert bl_data.market_implied_risk_aversion(prices, 0.02) == pytest.approx(expected)
    assert bl_data.market_implied_risk_aversion(prices.to_frame("SPY"), 0.02) == pytest.approx(expected)


class FakeYF(types.ModuleType):
    """Minimal stand-in for yfinance so the data layer is tested offline."""

    def __init__(self, frame, caps):
        super().__init__("yfinance")
        self.frame, self.caps, self.calls = frame, caps, []

    def download(self, tickers, **kw):
        self.calls.append(kw)
        cols = pd.MultiIndex.from_product([["Close", "Open"], tickers], names=["Price", "Ticker"])
        return pd.DataFrame(np.hstack([self.frame[tickers].to_numpy()] * 2), index=self.frame.index, columns=cols)

    def Ticker(self, t):
        cap = self.caps.get(t)
        return types.SimpleNamespace(fast_info={"marketCap": cap}, info={"marketCap": cap})


@pytest.fixture
def fake_yf(monkeypatch, prices):
    p = prices.copy()
    p.iloc[:5, 0] = np.nan  # late listing
    p.iloc[100, 1] = np.nan  # gap
    fake = FakeYF(p, {"A": 3e12, "B": 1e12, "C": None})
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return fake


def test_fetch_prices_adjusted_and_no_lookahead(fake_yf, prices):
    out = bl_data.fetch_prices(["B", "A"], period="2y")
    assert list(out.columns) == ["B", "A"]
    assert fake_yf.calls[0]["auto_adjust"] is True
    assert out.index[0] == prices.index[5]  # leading NaNs dropped, not back-filled
    assert out.loc[prices.index[100], "B"] == prices.iloc[99, 1]  # forward-filled
    assert not out.isna().any().any()


def test_fetch_market_caps(fake_yf):
    caps = bl_data.fetch_market_caps(["A", "B"])
    assert caps.to_dict() == {"A": 3e12, "B": 1e12}
    with pytest.raises(RuntimeError, match="C"):
        bl_data.fetch_market_caps(["A", "C"])
