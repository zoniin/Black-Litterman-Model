"""Black-Litterman portfolio construction: equilibrium priors, investor views, and
mean-variance optimisation."""

from .backtest import (
    backtest,
    black_litterman_strategy,
    equal_weight,
    market_cap_weight,
    mean_variance,
    momentum_views,
)
from .model import BlackLitterman, View, market_implied_prior
from .optimize import (
    efficient_frontier,
    efficient_return,
    max_quadratic_utility,
    max_sharpe,
    min_volatility,
    portfolio_performance,
)
from .risk import cov_to_corr, ledoit_wolf, returns_from_prices, sample_cov

__version__ = "2.0.0"

__all__ = [
    "backtest",
    "black_litterman_strategy",
    "equal_weight",
    "market_cap_weight",
    "mean_variance",
    "momentum_views",
    "BlackLitterman",
    "View",
    "market_implied_prior",
    "max_sharpe",
    "min_volatility",
    "efficient_return",
    "max_quadratic_utility",
    "efficient_frontier",
    "portfolio_performance",
    "returns_from_prices",
    "sample_cov",
    "ledoit_wolf",
    "cov_to_corr",
]
