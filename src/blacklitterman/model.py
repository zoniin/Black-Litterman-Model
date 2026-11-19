"""The Black-Litterman model.

All returns are annualised arithmetic returns. The prior is the market-implied
(reverse-optimised) return vector

    Pi = r_f + delta * Sigma @ w_mkt

and views are blended in with the Bayesian update

    mu_BL     = Pi + tau*Sigma P' (P tau*Sigma P' + Omega)^-1 (Q - P Pi)
    Sigma_BL  = Sigma + tau*Sigma - tau*Sigma P' (P tau*Sigma P' + Omega)^-1 P tau*Sigma

This is algebraically identical to the "master formula"
``[(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 Pi + P' Omega^-1 Q]`` but it
never inverts ``Sigma`` or ``Omega``. That makes it better conditioned and lets
``Omega`` be singular, so a view held with 100% confidence is exact.

References
----------
- Black, F. & Litterman, R. (1992). Global Portfolio Optimization. *FAJ* 48(5).
- He, G. & Litterman, R. (1999). The Intuition Behind Black-Litterman Model Portfolios.
- Idzorek, T. (2005). A Step-by-Step Guide to the Black-Litterman Model.
- Walters, J. (2014). The Black-Litterman Model in Detail.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["BlackLitterman", "View", "market_implied_prior"]


def market_implied_prior(
    cov: pd.DataFrame | np.ndarray,
    market_weights: pd.Series | np.ndarray,
    risk_aversion: float,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    """Reverse-optimise equilibrium returns: ``r_f + delta * Sigma @ w``."""
    cov = np.asarray(cov, dtype=float)
    w = np.asarray(market_weights, dtype=float)
    return risk_free_rate + risk_aversion * cov @ w


@dataclass(frozen=True)
class View:
    """A single linear view ``p @ mu = q`` with uncertainty ``omega``.

    ``omega`` is left as ``None`` until the model resolves it from ``confidence``.
    """

    p: np.ndarray
    q: float
    confidence: float | None = None
    omega: float | None = None
    label: str = ""


@dataclass
class BlackLitterman:
    """Black-Litterman posterior returns from a covariance matrix and market weights.

    Parameters
    ----------
    cov
        Annualised covariance matrix. A labelled ``DataFrame`` lets views be
        written by ticker.
    market_caps
        Market capitalisations (or any weights) defining the equilibrium
        portfolio. Normalised internally. Defaults to equal weight.
    risk_aversion
        Market price of risk ``delta``. See
        :func:`blacklitterman.data.market_implied_risk_aversion`.
    tau
        Uncertainty of the prior relative to ``Sigma``. Common choices are
        0.025-0.05 or ``1 / n_observations``.
    risk_free_rate
        Added to the prior, so priors, views and posteriors are all *total*
        returns. Relative views are spreads and are unaffected.
    prior
        Override the equilibrium prior with your own return vector.
    assets
        Asset labels, used when ``cov`` is a bare array.

    Examples
    --------
    >>> bl = BlackLitterman(cov, market_caps=caps, risk_aversion=2.5)
    >>> bl.add_absolute_view("AAPL", 0.10, confidence=0.6)
    >>> bl.add_relative_view("MSFT", "AMZN", 0.02, confidence=0.4)
    >>> bl.posterior_returns()
    """

    cov: pd.DataFrame | np.ndarray
    market_caps: pd.Series | np.ndarray | Sequence[float] | None = None
    risk_aversion: float = 2.5
    tau: float = 0.05
    risk_free_rate: float = 0.0
    prior: pd.Series | np.ndarray | None = None
    assets: Sequence[Hashable] | None = None
    views: list[View] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.cov, pd.DataFrame):
            if self.assets is None:
                self.assets = list(self.cov.index)
            if list(self.cov.columns) != list(self.cov.index):
                raise ValueError("cov must have identical index and columns")
        sigma = np.asarray(self.cov, dtype=float)
        if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
            raise ValueError("cov must be a square matrix")
        if not np.allclose(sigma, sigma.T, atol=1e-10):
            raise ValueError("cov must be symmetric")
        if np.linalg.eigvalsh(sigma).min() < -1e-10:
            raise ValueError("cov must be positive semi-definite")
        if self.tau <= 0:
            raise ValueError("tau must be positive")

        n = sigma.shape[0]
        self.assets = list(self.assets) if self.assets is not None else [f"Asset {i}" for i in range(n)]
        if len(self.assets) != n:
            raise ValueError(f"expected {n} asset labels, got {len(self.assets)}")
        self._sigma = sigma
        self._index = {a: i for i, a in enumerate(self.assets)}

        if self.market_caps is None:
            w = np.full(n, 1.0 / n)
        else:
            caps = self._align(self.market_caps, "market_caps")
            if np.any(caps < 0) or caps.sum() <= 0:
                raise ValueError("market_caps must be non-negative with a positive sum")
            w = caps / caps.sum()
        self._w_mkt = w

        if self.prior is not None:
            self._pi = self._align(self.prior, "prior")
        else:
            self._pi = market_implied_prior(sigma, w, self.risk_aversion, self.risk_free_rate)

    # ------------------------------------------------------------------ helpers
    def _align(self, x, name: str) -> np.ndarray:
        if isinstance(x, pd.Series):
            missing = set(self.assets) - set(x.index)
            if missing:
                raise ValueError(f"{name} is missing assets: {sorted(map(str, missing))}")
            return x.reindex(self.assets).to_numpy(dtype=float)
        arr = np.asarray(x, dtype=float)
        if arr.shape != (len(self.assets),):
            raise ValueError(f"{name} must have length {len(self.assets)}")
        return arr

    def _loc(self, asset: Hashable | int) -> int:
        if asset in self._index:
            return self._index[asset]
        if isinstance(asset, (int, np.integer)) and 0 <= asset < len(self.assets):
            return int(asset)
        raise KeyError(f"unknown asset {asset!r}")

    def _series(self, x: np.ndarray, name: str) -> pd.Series:
        return pd.Series(x, index=self.assets, name=name)

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def market_weights(self) -> pd.Series:
        return self._series(self._w_mkt, "market_weight")

    @property
    def prior_returns(self) -> pd.Series:
        return self._series(self._pi, "prior")

    # -------------------------------------------------------------------- views
    def add_view(
        self,
        portfolio: Mapping[Hashable, float] | Sequence[float] | np.ndarray,
        q: float,
        *,
        confidence: float | None = None,
        omega: float | None = None,
        label: str | None = None,
    ) -> BlackLitterman:
        """Add a general view: "the portfolio ``portfolio`` will return ``q``".

        View uncertainty is set in one of three ways:

        - ``omega``: the variance of the view, set directly.
        - ``confidence`` in (0, 1]: Idzorek's intuitive confidence, using the
          closed form ``omega = (1 - c) / c * p tau Sigma p'`` (Walters 2014). The
          posterior then moves exactly a fraction ``c`` of the way from the prior
          to a 100%-confidence view, and ``c = 1`` makes the view exact.
        - neither: the He-Litterman default ``omega = p tau Sigma p'``, the same
          as ``confidence = 0.5``.

        Returns ``self``, so calls can be chained.
        """
        if isinstance(portfolio, Mapping):
            p = np.zeros(self.n_assets)
            for asset, weight in portfolio.items():
                p[self._loc(asset)] += weight
        else:
            p = np.asarray(portfolio, dtype=float)
            if p.shape != (self.n_assets,):
                raise ValueError(f"view vector must have length {self.n_assets}")
        if not np.any(p):
            raise ValueError("view portfolio must have at least one non-zero weight")
        if confidence is not None and omega is not None:
            raise ValueError("pass either confidence or omega, not both")
        if confidence is not None and not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if omega is not None and omega < 0:
            raise ValueError("omega must be non-negative")
        if label is None:
            label = " + ".join(f"{w:g}·{a}" for a, w in zip(self.assets, p, strict=False) if w)
        self.views.append(View(p=p, q=float(q), confidence=confidence, omega=omega, label=label))
        return self

    def add_absolute_view(self, asset: Hashable, q: float, **kw) -> BlackLitterman:
        """ "``asset`` will return ``q``." (a total return, e.g. 0.08 for 8%)"""
        return self.add_view({asset: 1.0}, q, label=kw.pop("label", f"{asset} = {q:.1%}"), **kw)

    def add_relative_view(self, long: Hashable, short: Hashable, q: float, **kw) -> BlackLitterman:
        """ "``long`` will outperform ``short`` by ``q``." """
        label = kw.pop("label", f"{long} − {short} = {q:+.1%}")
        return self.add_view({long: 1.0, short: -1.0}, q, label=label, **kw)

    def clear_views(self) -> BlackLitterman:
        self.views.clear()
        return self

    @property
    def P(self) -> np.ndarray:
        return np.array([v.p for v in self.views]).reshape(-1, self.n_assets)

    @property
    def Q(self) -> np.ndarray:
        return np.array([v.q for v in self.views])

    @property
    def omega(self) -> np.ndarray:
        """Diagonal view-uncertainty matrix, with each view's variance resolved."""
        ts = self.tau * self._sigma
        diag = []
        for v in self.views:
            if v.omega is not None:
                diag.append(v.omega)
            else:
                c = 0.5 if v.confidence is None else v.confidence
                diag.append((1.0 - c) / c * float(v.p @ ts @ v.p))
        return np.diag(diag)

    # ---------------------------------------------------------------- posterior
    def _posterior(self) -> tuple[np.ndarray, np.ndarray]:
        ts = self.tau * self._sigma
        if not self.views:
            return self._pi.copy(), self._sigma + ts
        P, Q, omega = self.P, self.Q, self.omega
        A = P @ ts @ P.T + omega
        try:
            # K = tau Sigma P' A^-1, computed with a solve, not an inverse
            K = np.linalg.solve(A, P @ ts).T
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "views are redundant or contradictory at 100% confidence: P tau Sigma P' + Omega is singular"
            ) from exc
        mu = self._pi + K @ (Q - P @ self._pi)
        M = ts - K @ P @ ts
        M = (M + M.T) / 2
        return mu, self._sigma + M

    def posterior_returns(self) -> pd.Series:
        """Posterior expected returns ``mu_BL``."""
        return self._series(self._posterior()[0], "posterior")

    def posterior_cov(self) -> pd.DataFrame:
        """Posterior covariance of returns, ``Sigma + M``."""
        return pd.DataFrame(self._posterior()[1], index=self.assets, columns=self.assets)

    def implied_weights(self) -> pd.Series:
        """Unconstrained optimal weights ``(delta Sigma_BL)^-1 (mu_BL - r_f)``.

        With no views these equal ``w_mkt / (1 + tau)`` (He & Litterman 1999):
        the investor holds the market, scaled down for estimation risk. Each view
        tilts them by a multiple of its own view portfolio and nothing else.
        """
        mu, cov = self._posterior()
        w = np.linalg.solve(self.risk_aversion * cov, mu - self.risk_free_rate)
        return self._series(w, "bl_weight")

    # ------------------------------------------------------------- diagnostics
    def summary(self) -> pd.DataFrame:
        """Per-asset table: market weight, prior, posterior, and view impact."""
        post = self.posterior_returns()
        return pd.DataFrame(
            {
                "market_weight": self.market_weights,
                "prior": self.prior_returns,
                "posterior": post,
                "view_impact": post - self.prior_returns,
                "bl_weight": self.implied_weights(),
            }
        )

    def view_summary(self) -> pd.DataFrame:
        """Per-view table: the view, what the prior and posterior imply, and the uncertainty.

        A view whose posterior value sits far from ``q`` was strongly overruled by
        the prior. That is often a sign that the confidence is too low or the view
        conflicts with the market.
        """
        if not self.views:
            return pd.DataFrame(columns=["q", "prior_implied", "posterior_implied", "confidence", "omega"])
        mu = self._posterior()[0]
        P = self.P
        return pd.DataFrame(
            {
                "q": self.Q,
                "prior_implied": P @ self._pi,
                "posterior_implied": P @ mu,
                "confidence": [v.confidence for v in self.views],
                "omega": np.diag(self.omega),
            },
            index=pd.Index([v.label for v in self.views], name="view"),
        )
