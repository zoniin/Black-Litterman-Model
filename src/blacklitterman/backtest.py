"""Walk-forward backtesting of allocation strategies.

A *strategy* is any callable ``strategy(ctx) -> weights``. On each rebalance date
it sees a :class:`Context` with data up to and including that day's close only.
The new weights take effect from the next trading day and then drift with
prices until the next rebalance. Transaction costs are charged on turnover.
Any weight not invested (``1 - sum(w)``) earns the risk-free rate.

Built-in strategies cover the usual comparisons: market-cap, equal-weight,
historical-mean mean-variance, and Black-Litterman with pluggable views.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .model import BlackLitterman
from .optimize import Bounds, max_sharpe, min_volatility
from .risk import ledoit_wolf

__all__ = [
    "BacktestResult",
    "Context",
    "ViewSpec",
    "backtest",
    "black_litterman_strategy",
    "equal_weight",
    "market_cap_weight",
    "mean_variance",
    "momentum_views",
    "performance_stats",
]


# --------------------------------------------------------------------------- context
@dataclass(frozen=True)
class Context:
    """Everything a strategy may look at on a rebalance date (all data ≤ ``date``)."""

    date: pd.Timestamp
    prices: pd.DataFrame
    """Full price history up to and including ``date``."""
    returns: pd.DataFrame
    """Daily simple returns over the trailing ``lookback`` window."""
    market_caps: pd.Series | None
    """Market caps on ``date``, if a cap history was supplied."""
    risk_free_rate: float
    frequency: int = 252

    @property
    def assets(self) -> list:
        return list(self.prices.columns)

    def cov(self) -> pd.DataFrame:
        """Annualised Ledoit-Wolf covariance over the lookback window."""
        return ledoit_wolf(self.returns, self.frequency)


Strategy = Callable[[Context], pd.Series]


# ------------------------------------------------------------------------ strategies
def equal_weight() -> Strategy:
    """1/N, the benchmark that is notoriously hard to beat (DeMiguel et al., 2009)."""

    def strategy(ctx: Context) -> pd.Series:
        return pd.Series(1.0 / len(ctx.assets), index=ctx.assets)

    return strategy


def market_cap_weight() -> Strategy:
    """Hold the market portfolio, which is the Black-Litterman prior."""

    def strategy(ctx: Context) -> pd.Series:
        if ctx.market_caps is None:
            raise ValueError("market_cap_weight needs market_caps passed to backtest()")
        return ctx.market_caps / ctx.market_caps.sum()

    return strategy


def mean_variance(bounds: Bounds = (0.0, 1.0)) -> Strategy:
    """Max-Sharpe on *historical mean* returns. This is the classic, error-maximising
    baseline that Black-Litterman was designed to fix. Falls back to minimum
    variance when no asset's trailing mean beats the risk-free rate."""

    def strategy(ctx: Context) -> pd.Series:
        mu = ctx.returns.mean() * ctx.frequency
        cov = ctx.cov()
        if mu.max() <= ctx.risk_free_rate:
            return min_volatility(cov, bounds)
        return max_sharpe(mu, cov, ctx.risk_free_rate, bounds)

    return strategy


@dataclass(frozen=True)
class ViewSpec:
    """A view produced by a view generator: ``portfolio @ mu = q``."""

    portfolio: Mapping[Any, float]
    q: float
    confidence: float | None = None
    label: str | None = None


ViewGenerator = Callable[[Context], Sequence[ViewSpec]]


def black_litterman_strategy(
    views: ViewGenerator | None = None,
    *,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    bounds: Bounds = (0.0, 1.0),
) -> Strategy:
    """Black-Litterman posterior + bounded max-Sharpe.

    The prior is reverse-optimised from that day's market caps and Ledoit-Wolf
    covariance. ``views(ctx)`` returns the views for the day. With no views the
    strategy reproduces the market portfolio exactly, whenever the bounds allow
    it. Any outperformance is therefore down to the views.
    """

    def strategy(ctx: Context) -> pd.Series:
        if ctx.market_caps is None:
            raise ValueError("black_litterman_strategy needs market_caps passed to backtest()")
        bl = BlackLitterman(
            ctx.cov(),
            market_caps=ctx.market_caps,
            risk_aversion=risk_aversion,
            tau=tau,
            risk_free_rate=ctx.risk_free_rate,
        )
        for v in views(ctx) if views else ():
            bl.add_view(v.portfolio, v.q, confidence=v.confidence, label=v.label)
        return max_sharpe(bl.posterior_returns(), bl.posterior_cov(), ctx.risk_free_rate, bounds)

    return strategy


def momentum_views(
    lookback: int = 252,
    skip: int = 21,
    n_long: int = 3,
    n_short: int = 3,
    premium: float | None = 0.05,
    scale: float = 0.5,
    confidence: float = 0.25,
) -> ViewGenerator:
    """Cross-sectional momentum as a single Black-Litterman relative view.

    Rank assets by their ``lookback``-day return, skipping the most recent
    ``skip`` days (the standard 12-1 month momentum, Jegadeesh & Titman, 1993).
    The view says that an equal-weight basket of the ``n_long`` winners beats
    the ``n_short`` losers by ``q`` per year, held with ``confidence``.

    By default ``q`` is a fixed ``premium`` (5% a year, a conservative reading
    of the academic momentum premium). With ``premium=None``, ``q`` is
    ``scale`` times the basket's past annualised spread. That extrapolation
    tends to be far too aggressive, because past spreads between individual
    stocks are often 30%+ a year.
    """

    def generate(ctx: Context) -> list[ViewSpec]:
        p = ctx.prices
        if len(p) <= lookback:
            return []
        mom = p.iloc[-1 - skip] / p.iloc[-1 - lookback] - 1
        ranked = mom.sort_values(ascending=False)
        winners, losers = ranked.index[:n_long], ranked.index[-n_short:]
        if premium is not None:
            q = premium
        else:
            q = scale * (mom[winners].mean() - mom[losers].mean()) * 252 / (lookback - skip)
        portfolio = {**{a: 1.0 / n_long for a in winners}, **{a: -1.0 / n_short for a in losers}}
        return [ViewSpec(portfolio, q, confidence, label="momentum winners − losers")]

    return generate


# --------------------------------------------------------------------------- engine
@dataclass
class BacktestResult:
    returns: pd.DataFrame
    """Daily net returns, one column per strategy."""
    weights: dict[str, pd.DataFrame]
    """Target weights set on each rebalance date, per strategy."""
    turnover: pd.DataFrame
    """One-way turnover on each rebalance date, per strategy."""
    risk_free_rate: float = 0.0
    frequency: int = 252
    stats: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.stats = performance_stats(self.returns, self.risk_free_rate, self.frequency, self.turnover, self.weights)

    def relative_to(self, benchmark: str) -> pd.DataFrame:
        """Active return, tracking error and information ratio of each strategy
        against the ``benchmark`` column (annualised)."""
        active = self.returns.sub(self.returns[benchmark], axis=0).drop(columns=benchmark)
        te = active.std() * np.sqrt(self.frequency)
        ann = active.mean() * self.frequency
        return pd.DataFrame({"active_return": ann, "tracking_error": te, "information_ratio": ann / te})

    @property
    def equity(self) -> pd.DataFrame:
        """Growth of $1."""
        return (1 + self.returns).cumprod()

    @property
    def drawdown(self) -> pd.DataFrame:
        eq = self.equity
        return eq / eq.cummax() - 1


def _rebalance_dates(index: pd.DatetimeIndex, freq: str, first: int) -> pd.DatetimeIndex:
    eligible = pd.Series(index[first:], index=index[first:])
    return pd.DatetimeIndex(eligible.groupby(eligible.index.to_period(freq)).max().to_numpy())


def backtest(
    prices: pd.DataFrame,
    strategies: Mapping[str, Strategy],
    *,
    market_caps: pd.DataFrame | None = None,
    rebalance: str = "M",
    lookback: int = 504,
    cost_bps: float = 10.0,
    risk_free_rate: float = 0.0,
    frequency: int = 252,
) -> BacktestResult:
    """Run each strategy walk-forward over ``prices``.

    Parameters
    ----------
    prices
        Total-return (dividend-adjusted) prices, one column per asset.
    strategies
        ``{name: strategy}``. Each strategy maps a :class:`Context` to weights.
    market_caps
        Optional cap history aligned to ``prices`` (see
        :func:`blacklitterman.data.market_cap_history`), required by cap-based strategies.
    rebalance
        A pandas period alias: ``"M"`` for monthly, ``"Q"`` for quarterly, ``"W"`` for weekly.
    lookback
        Trailing window of daily returns given to strategies. Trading starts
        once this much history exists.
    cost_bps
        Cost per unit of one-way turnover, in basis points.
    """
    prices = prices.sort_index()
    rets = prices.pct_change().iloc[1:].fillna(0.0)
    if len(rets) <= lookback:
        raise ValueError(f"need more than lookback={lookback} days of returns, got {len(rets)}")
    if market_caps is not None:
        market_caps = market_caps.reindex(index=prices.index, columns=prices.columns)
    dates = rets.index
    # rebalance at each period end, plus once on the first day trading is possible
    rebal = {*_rebalance_dates(dates, rebalance, lookback - 1), dates[lookback - 1]}
    rf_daily = (1 + risk_free_rate) ** (1 / frequency) - 1
    R = rets.to_numpy()
    cost = cost_bps / 1e4

    out_rets, out_w, out_to = {}, {}, {}
    for name, strategy in strategies.items():
        w = np.zeros(prices.shape[1])  # start in cash
        daily, targets, turnovers = [], {}, {}
        pending_cost = 0.0
        for i in range(lookback, len(dates)):
            # trade at yesterday's close, if it was a rebalance day
            prev = dates[i - 1]
            if prev in rebal:
                ctx = Context(
                    date=prev,
                    prices=prices.loc[:prev],
                    returns=rets.iloc[i - lookback : i],
                    market_caps=None if market_caps is None else market_caps.loc[prev],
                    risk_free_rate=risk_free_rate,
                    frequency=frequency,
                )
                target = strategy(ctx).reindex(prices.columns).fillna(0.0).to_numpy(dtype=float)
                turnover = float(np.abs(target - w).sum())
                targets[prev], turnovers[prev] = target, turnover
                pending_cost = turnover * cost
                w = target
            cash = 1.0 - w.sum()
            gross = float(w @ R[i]) + cash * rf_daily
            daily.append(gross - pending_cost)
            pending_cost = 0.0
            # drift
            growth = 1.0 + gross
            w = w * (1.0 + R[i]) / growth if growth > 0 else w
        out_rets[name] = pd.Series(daily, index=dates[lookback:])
        out_w[name] = pd.DataFrame.from_dict(targets, orient="index", columns=prices.columns)
        out_to[name] = pd.Series(turnovers)

    return BacktestResult(
        returns=pd.DataFrame(out_rets),
        weights=out_w,
        turnover=pd.DataFrame(out_to),
        risk_free_rate=risk_free_rate,
        frequency=frequency,
    )


# ---------------------------------------------------------------------------- stats
def performance_stats(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    frequency: int = 252,
    turnover: pd.DataFrame | None = None,
    weights: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Annualised return, volatility, Sharpe, max drawdown and, when available,
    annual turnover and average effective number of holdings."""
    rows = {}
    for name, r in returns.items():
        years = len(r) / frequency
        eq = (1 + r).cumprod()
        cagr = eq.iloc[-1] ** (1 / years) - 1
        vol = r.std() * np.sqrt(frequency)
        excess = r - ((1 + risk_free_rate) ** (1 / frequency) - 1)
        row = {
            "CAGR": cagr,
            "volatility": vol,
            "sharpe": excess.mean() / r.std() * np.sqrt(frequency) if vol > 0 else np.nan,
            "max_drawdown": (eq / eq.cummax() - 1).min(),
        }
        if turnover is not None and name in turnover:
            row["turnover_pa"] = turnover[name].sum() / years
        if weights is not None and name in weights and len(weights[name]):
            w = weights[name].to_numpy()
            conc = np.sum(w**2, axis=1)
            row["effective_n"] = float(np.mean(1.0 / conc[conc > 0])) if (conc > 0).any() else np.nan
        rows[name] = row
    return pd.DataFrame(rows).T
