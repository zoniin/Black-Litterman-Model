"""
Black-Litterman Portfolio Optimization Package

A comprehensive implementation of the Black-Litterman model for portfolio optimization.
"""

from .black_litterman import BlackLittermanModel
from .data_utils import (
    fetch_price_data,
    calculate_returns,
    calculate_covariance_matrix,
    get_market_caps,
    prepare_bl_inputs,
    validate_inputs
)
from .visualization import (
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_return_attribution,
    plot_correlation_matrix,
    plot_weight_evolution,
    plot_risk_return_scatter,
    plot_views_impact,
    create_portfolio_dashboard
)

__version__ = "1.0.0"

__all__ = [
    'BlackLittermanModel',
    'fetch_price_data',
    'calculate_returns',
    'calculate_covariance_matrix',
    'get_market_caps',
    'prepare_bl_inputs',
    'validate_inputs',
    'plot_efficient_frontier',
    'plot_portfolio_weights',
    'plot_return_attribution',
    'plot_correlation_matrix',
    'plot_weight_evolution',
    'plot_risk_return_scatter',
    'plot_views_impact',
    'create_portfolio_dashboard'
]
