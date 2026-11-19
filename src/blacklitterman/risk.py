"""Return and covariance estimation from price histories."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["returns_from_prices", "sample_cov", "ledoit_wolf", "cov_to_corr"]


def returns_from_prices(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """Periodic returns from prices. These are simple returns by default.

    Black-Litterman works with arithmetic expected returns, so simple returns
    are the consistent choice. Pass ``log=True`` for log returns.
    """
    rets = np.log(prices).diff() if log else prices.pct_change(fill_method=None)
    return rets.dropna(how="all").dropna(axis=0, how="any")


def sample_cov(returns: pd.DataFrame, frequency: int = 252) -> pd.DataFrame:
    """Annualised sample covariance (unbiased, ``n - 1`` denominator)."""
    return returns.cov() * frequency


def ledoit_wolf(returns: pd.DataFrame, frequency: int = 252) -> pd.DataFrame:
    """Annualised Ledoit-Wolf (2004) covariance, shrunk towards a scaled identity.

    The shrinkage intensity is estimated analytically to minimise expected
    Frobenius loss. The result matches ``sklearn.covariance.LedoitWolf``
    without depending on scikit-learn.
    """
    X = returns.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    X2 = X**2
    # squared Frobenius distance of S to its target, and an estimate of
    # how noisy S itself is
    delta = np.sum((S - mu * np.eye(p)) ** 2)
    beta = min(np.sum(X2.T @ X2) / n - np.sum(S**2), n * delta) / n
    shrinkage = 0.0 if delta == 0 else beta / delta
    shrunk = (1 - shrinkage) * S + shrinkage * mu * np.eye(p)
    out = pd.DataFrame(shrunk * frequency, index=returns.columns, columns=returns.columns)
    out.attrs["shrinkage"] = shrinkage
    return out


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Covariance to correlation matrix."""
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)
