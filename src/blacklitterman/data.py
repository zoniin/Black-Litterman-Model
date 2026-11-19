"""Market data helpers. ``yfinance`` is an optional dependency (``pip install .[data]``)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

__all__ = ["fetch_prices", "fetch_market_caps", "market_implied_risk_aversion"]


def _yf():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for data downloads: pip install 'black-litterman[data]'") from exc
    return yf


def fetch_prices(
    tickers: Sequence[str],
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
) -> pd.DataFrame:
    """Daily split- and dividend-adjusted close prices, one column per ticker.

    Gaps are forward-filled, never back-filled: back-filling would leak future
    prices into the past. Rows before every ticker has data are dropped.
    """
    yf = _yf()
    tickers = list(tickers)
    kwargs = {"start": start, "end": end} if start else {"period": period}
    raw = yf.download(tickers, auto_adjust=True, progress=False, **kwargs)
    if raw is None or raw.empty:
        raise RuntimeError(f"no price data returned for {tickers}")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing:
        raise RuntimeError(f"no price data for: {missing}")
    return close[tickers].ffill().dropna()


def fetch_market_caps(tickers: Sequence[str]) -> pd.Series:
    """Current market capitalisations from Yahoo Finance.

    Raises if any cap is unavailable. Silently substituting a placeholder would
    distort the equilibrium weights by orders of magnitude.
    """
    yf = _yf()
    caps, missing = {}, []
    for t in tickers:
        cap = None
        try:
            cap = yf.Ticker(t).fast_info["marketCap"]
        except Exception:
            pass
        if not cap:
            try:
                cap = yf.Ticker(t).info.get("marketCap")
            except Exception:
                cap = None
        if cap:
            caps[t] = float(cap)
        else:
            missing.append(t)
    if missing:
        raise RuntimeError(f"market cap unavailable for {missing}; pass market_caps explicitly")
    return pd.Series(caps, name="market_cap")


def market_implied_risk_aversion(
    market_prices: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.0,
    frequency: int = 252,
) -> float:
    """Implied market risk aversion ``delta = (E[R_m] - r_f) / Var(R_m)``.

    ``market_prices`` should be a broad market proxy (e.g. SPY or ACWI). The
    estimate is noisy, because historical mean returns are noisy, so treat it
    as a sanity check. Values of roughly 2-4 are typical.
    """
    if isinstance(market_prices, pd.DataFrame):
        if market_prices.shape[1] != 1:
            raise ValueError("pass a single market proxy series")
        market_prices = market_prices.iloc[:, 0]
    rets = market_prices.pct_change().dropna()
    excess = rets.mean() * frequency - risk_free_rate
    return float(excess / (rets.var() * frequency))
