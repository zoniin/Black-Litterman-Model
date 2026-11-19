"""Mean-variance portfolio construction on top of any (mu, Sigma).

Every optimiser takes the expected returns and covariance as arrays or labelled
pandas objects and returns weights in the same form. ``bounds`` is a
``(lower, upper)`` pair of scalars or per-asset arrays. Use ``(0, 1)`` for
long-only portfolios (the default) and e.g. ``(-1, 1)`` to allow shorts.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

__all__ = [
    "max_sharpe",
    "min_volatility",
    "efficient_return",
    "max_quadratic_utility",
    "efficient_frontier",
    "portfolio_performance",
]

Bounds = tuple[float | np.ndarray, float | np.ndarray]


def _prepare(mu, cov):
    index = None
    if isinstance(mu, pd.Series):
        index = mu.index
        if isinstance(cov, pd.DataFrame):
            cov = cov.loc[index, index]
    elif isinstance(cov, pd.DataFrame):
        index = cov.index
    sigma = np.asarray(cov, dtype=float)
    mu = None if mu is None else np.asarray(mu, dtype=float)
    return mu, sigma, index


def _wrap(w: np.ndarray, index) -> np.ndarray | pd.Series:
    w = np.where(np.abs(w) < 1e-9, 0.0, w)
    return pd.Series(w, index=index, name="weight") if index is not None else w


def _solve(problem: cp.Problem, what: str) -> None:
    problem.solve()
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"{what}: solver returned status '{problem.status}'")


def _bound_constraints(w, scale, bounds: Bounds):
    lb, ub = bounds
    return [w >= np.asarray(lb) * scale, w <= np.asarray(ub) * scale]


def max_sharpe(mu, cov, risk_free_rate: float = 0.0, bounds: Bounds = (0.0, 1.0)):
    """Tangency portfolio: maximise ``(mu'w - r_f) / sqrt(w' Sigma w)``.

    Sharpe ratio is not concave, so the problem is solved through the
    Cornuejols-Tütüncü change of variables ``y = k*w``. That turns it into a
    convex QP: minimise ``y' Sigma y`` subject to ``(mu - r_f)'y = 1``,
    ``sum(y) = k`` and ``k*lb <= y <= k*ub``.
    """
    mu, sigma, index = _prepare(mu, cov)
    excess = mu - risk_free_rate
    if excess.max() <= 0:
        raise ValueError("max_sharpe needs at least one asset with return above the risk-free rate")
    y, k = cp.Variable(len(mu)), cp.Variable(nonneg=True)
    cons = [excess @ y == 1, cp.sum(y) == k, *_bound_constraints(y, k, bounds)]
    _solve(cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma))), cons), "max_sharpe")
    return _wrap(y.value / k.value, index)


def min_volatility(cov, bounds: Bounds = (0.0, 1.0)):
    """Global minimum-variance portfolio."""
    _, sigma, index = _prepare(None, cov)
    w = cp.Variable(sigma.shape[0])
    cons = [cp.sum(w) == 1, *_bound_constraints(w, 1, bounds)]
    _solve(cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), cons), "min_volatility")
    return _wrap(w.value, index)


def efficient_return(mu, cov, target_return: float, bounds: Bounds = (0.0, 1.0)):
    """Minimum-variance portfolio with expected return of at least ``target_return``."""
    mu, sigma, index = _prepare(mu, cov)
    w = cp.Variable(len(mu))
    cons = [cp.sum(w) == 1, mu @ w >= target_return, *_bound_constraints(w, 1, bounds)]
    _solve(cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), cons), "efficient_return")
    return _wrap(w.value, index)


def max_quadratic_utility(mu, cov, risk_aversion: float, bounds: Bounds = (0.0, 1.0)):
    """Maximise ``mu'w - (delta / 2) w' Sigma w`` with fully-invested weights."""
    mu, sigma, index = _prepare(mu, cov)
    w = cp.Variable(len(mu))
    obj = mu @ w - risk_aversion / 2 * cp.quad_form(w, cp.psd_wrap(sigma))
    cons = [cp.sum(w) == 1, *_bound_constraints(w, 1, bounds)]
    _solve(cp.Problem(cp.Maximize(obj), cons), "max_quadratic_utility")
    return _wrap(w.value, index)


def efficient_frontier(mu, cov, n_points: int = 50, bounds: Bounds = (0.0, 1.0)) -> pd.DataFrame:
    """Trace the efficient frontier from the minimum-variance portfolio up to the
    highest attainable return under ``bounds``.

    Returns a DataFrame with ``return`` and ``volatility`` columns plus one
    column of weights per asset. A single parametrised problem is reused for
    every point.
    """
    mu_arr, sigma, index = _prepare(mu, cov)
    n = len(mu_arr)
    names = list(index) if index is not None else [f"Asset {i}" for i in range(n)]

    w_min = np.asarray(min_volatility(sigma, bounds))
    lo = float(mu_arr @ w_min)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, *_bound_constraints(w, 1, bounds)]
    hi_prob = cp.Problem(cp.Maximize(mu_arr @ w), cons)
    _solve(hi_prob, "efficient_frontier (max return)")
    hi = float(hi_prob.value)

    target = cp.Parameter()
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), [*cons, mu_arr @ w >= target])
    rows = []
    for t in np.linspace(lo, hi, n_points):
        target.value = t
        prob.solve()
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            wv = w.value
            rows.append([float(mu_arr @ wv), float(np.sqrt(max(wv @ sigma @ wv, 0.0))), *wv])
    return pd.DataFrame(rows, columns=["return", "volatility", *names])


def portfolio_performance(weights, mu, cov, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Expected return, volatility, Sharpe ratio and effective number of holdings."""
    mu, sigma, index = _prepare(mu, cov)
    if isinstance(weights, pd.Series) and index is not None:
        weights = weights.reindex(index).fillna(0.0)
    w = np.asarray(weights, dtype=float)
    ret = float(mu @ w)
    vol = float(np.sqrt(w @ sigma @ w))
    return {
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": (ret - risk_free_rate) / vol if vol > 0 else np.nan,
        "effective_n": float(1.0 / np.sum(w**2)),
    }
