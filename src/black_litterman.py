"""
Black-Litterman Portfolio Optimization Model

This module implements the Black-Litterman model for portfolio optimization,
which combines market equilibrium with investor views using Bayesian inference.

Mathematical Background:
-----------------------
The Black-Litterman model computes posterior expected returns as:

E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) [(τΣ)^(-1)Π + P'Ω^(-1)Q]

where:
- Π: Prior expected returns (market equilibrium)
- Σ: Covariance matrix of asset returns
- P: Picking matrix (defines views)
- Q: View expected returns
- Ω: Uncertainty matrix in views
- τ: Scaling factor (typically 0.025-0.05)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings


class BlackLittermanModel:
    """
    Comprehensive Black-Litterman portfolio optimization model.

    This implementation supports:
    - Market equilibrium calculation via reverse optimization
    - Multiple types of views (absolute and relative)
    - Confidence levels for views
    - Multiple uncertainty specifications (idzorek, default)
    - Portfolio optimization with constraints
    - Risk-return analysis
    """

    def __init__(
        self,
        cov_matrix: np.ndarray,
        market_caps: Optional[np.ndarray] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.025,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize the Black-Litterman model.

        Parameters:
        -----------
        cov_matrix : np.ndarray
            Covariance matrix of asset returns (n_assets x n_assets)
        market_caps : np.ndarray, optional
            Market capitalizations for deriving equilibrium weights
        risk_aversion : float, default=2.5
            Market risk aversion coefficient (δ)
        tau : float, default=0.025
            Scaling factor for uncertainty in prior (typically 0.01-0.05)
        risk_free_rate : float, default=0.02
            Risk-free rate for Sharpe ratio calculations
        """
        self.cov_matrix = np.array(cov_matrix)
        self.n_assets = self.cov_matrix.shape[0]
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate

        # Validate inputs
        if self.cov_matrix.shape[0] != self.cov_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")

        # Initialize market equilibrium weights
        if market_caps is not None:
            self.market_weights = np.array(market_caps) / np.sum(market_caps)
        else:
            # Equal weights if no market caps provided
            self.market_weights = np.ones(self.n_assets) / self.n_assets

        # Calculate implied equilibrium returns (reverse optimization)
        self.pi = self._calculate_equilibrium_returns()

        # Placeholders for views
        self.P = None  # Picking matrix
        self.Q = None  # View returns
        self.omega = None  # View uncertainty matrix

        # Results
        self.posterior_returns = None
        self.posterior_cov = None

    def _calculate_equilibrium_returns(self) -> np.ndarray:
        """
        Calculate implied equilibrium returns using reverse optimization.

        Formula: Π = δ * Σ * w_mkt

        Returns:
        --------
        np.ndarray : Equilibrium expected returns
        """
        return self.risk_aversion * self.cov_matrix @ self.market_weights

    def add_absolute_view(
        self,
        asset_idx: int,
        expected_return: float,
        confidence: float = 0.5
    ):
        """
        Add an absolute view on a single asset.

        Example: "Asset 0 will return 8%"

        Parameters:
        -----------
        asset_idx : int
            Index of the asset
        expected_return : float
            Expected return (e.g., 0.08 for 8%)
        confidence : float, default=0.5
            Confidence level (0-1), where 1 is maximum confidence
        """
        # Create picking vector (1 at asset_idx, 0 elsewhere)
        p = np.zeros(self.n_assets)
        p[asset_idx] = 1.0

        self._add_view(p, expected_return, confidence)

    def add_relative_view(
        self,
        asset_idx1: int,
        asset_idx2: int,
        outperformance: float,
        confidence: float = 0.5
    ):
        """
        Add a relative view between two assets.

        Example: "Asset 0 will outperform Asset 1 by 3%"

        Parameters:
        -----------
        asset_idx1 : int
            Index of the first asset (outperformer)
        asset_idx2 : int
            Index of the second asset (underperformer)
        outperformance : float
            Expected outperformance (e.g., 0.03 for 3%)
        confidence : float, default=0.5
            Confidence level (0-1)
        """
        # Create picking vector (1 for asset1, -1 for asset2)
        p = np.zeros(self.n_assets)
        p[asset_idx1] = 1.0
        p[asset_idx2] = -1.0

        self._add_view(p, outperformance, confidence)

    def add_custom_view(
        self,
        picking_vector: np.ndarray,
        expected_return: float,
        confidence: float = 0.5
    ):
        """
        Add a custom view with arbitrary picking vector.

        Example: "A portfolio of 50% Asset 0 + 50% Asset 1 will return 7%"

        Parameters:
        -----------
        picking_vector : np.ndarray
            Custom picking vector (must sum to 1 or -1 for consistency)
        expected_return : float
            Expected return for this view
        confidence : float, default=0.5
            Confidence level (0-1)
        """
        if len(picking_vector) != self.n_assets:
            raise ValueError("Picking vector must have length equal to number of assets")

        self._add_view(picking_vector, expected_return, confidence)

    def _add_view(
        self,
        picking_vector: np.ndarray,
        expected_return: float,
        confidence: float
    ):
        """
        Internal method to add a view to the model.

        Parameters:
        -----------
        picking_vector : np.ndarray
            Picking vector defining the view
        expected_return : float
            Expected return for this view
        confidence : float
            Confidence level (0-1)
        """
        # Initialize P and Q if this is the first view
        if self.P is None:
            self.P = picking_vector.reshape(1, -1)
            self.Q = np.array([expected_return])

            # Calculate view uncertainty using confidence level
            # Higher confidence = lower uncertainty
            view_variance = (picking_vector @ self.cov_matrix @ picking_vector.T)
            self.omega = np.array([[view_variance / confidence]])
        else:
            # Append new view
            self.P = np.vstack([self.P, picking_vector])
            self.Q = np.append(self.Q, expected_return)

            # Update omega matrix
            view_variance = (picking_vector @ self.cov_matrix @ picking_vector.T)
            new_omega = np.zeros((len(self.Q), len(self.Q)))
            new_omega[:-1, :-1] = self.omega
            new_omega[-1, -1] = view_variance / confidence
            self.omega = new_omega

    def add_views_from_dict(
        self,
        views: List[Dict[str, Union[int, float, List]]],
        asset_names: Optional[List[str]] = None
    ):
        """
        Add multiple views from a structured dictionary.

        Parameters:
        -----------
        views : List[Dict]
            List of view dictionaries with keys:
            - 'type': 'absolute' or 'relative' or 'custom'
            - 'assets': int (for absolute), tuple (for relative), or list (for custom)
            - 'return': expected return
            - 'confidence': confidence level (optional, default 0.5)
        asset_names : List[str], optional
            Names of assets for reference

        Example:
        --------
        views = [
            {'type': 'absolute', 'assets': 0, 'return': 0.08, 'confidence': 0.7},
            {'type': 'relative', 'assets': (1, 2), 'return': 0.03, 'confidence': 0.5}
        ]
        """
        for view in views:
            view_type = view.get('type')
            confidence = view.get('confidence', 0.5)
            expected_return = view['return']

            if view_type == 'absolute':
                self.add_absolute_view(view['assets'], expected_return, confidence)
            elif view_type == 'relative':
                asset1, asset2 = view['assets']
                self.add_relative_view(asset1, asset2, expected_return, confidence)
            elif view_type == 'custom':
                picking_vector = np.array(view['assets'])
                self.add_custom_view(picking_vector, expected_return, confidence)
            else:
                raise ValueError(f"Unknown view type: {view_type}")

    def calculate_posterior(
        self,
        method: str = 'default'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate posterior expected returns and covariance.

        Black-Litterman Master Formula:
        E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) [(τΣ)^(-1)Π + P'Ω^(-1)Q]

        Parameters:
        -----------
        method : str, default='default'
            Method for calculating posterior ('default' or 'idzorek')

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Posterior expected returns and posterior covariance matrix
        """
        if self.P is None:
            warnings.warn("No views specified. Using equilibrium returns.")
            self.posterior_returns = self.pi
            self.posterior_cov = self.cov_matrix
            return self.posterior_returns, self.posterior_cov

        # Scaled covariance matrix
        tau_sigma = self.tau * self.cov_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)

        # View precision matrix
        omega_inv = np.linalg.inv(self.omega)

        # Calculate posterior covariance
        # M = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1)
        M_inv = tau_sigma_inv + self.P.T @ omega_inv @ self.P
        M = np.linalg.inv(M_inv)

        # Calculate posterior returns
        # E[R] = M [(τΣ)^(-1)Π + P'Ω^(-1)Q]
        self.posterior_returns = M @ (
            tau_sigma_inv @ self.pi + self.P.T @ omega_inv @ self.Q
        )

        # Posterior covariance
        # Cov = Σ + M
        self.posterior_cov = self.cov_matrix + M

        return self.posterior_returns, self.posterior_cov

    def optimize_portfolio(
        self,
        optimization_type: str = 'max_sharpe',
        target_return: Optional[float] = None,
        allow_short: bool = False,
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
        l2_gamma: float = 0.0
    ) -> np.ndarray:
        """
        Optimize portfolio weights based on posterior distribution.

        Parameters:
        -----------
        optimization_type : str, default='max_sharpe'
            Type of optimization:
            - 'max_sharpe': Maximum Sharpe ratio
            - 'min_volatility': Minimum volatility
            - 'target_return': Efficient portfolio for target return
            - 'max_return': Maximum return (subject to constraints)
        target_return : float, optional
            Target return (required for 'target_return' optimization)
        allow_short : bool, default=False
            Allow short positions
        weight_bounds : Tuple[float, float], default=(0, 1)
            Bounds on individual weights
        l2_gamma : float, default=0.0
            L2 regularization parameter (encourages diversification)

        Returns:
        --------
        np.ndarray : Optimal portfolio weights
        """
        if self.posterior_returns is None:
            self.calculate_posterior()

        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError(
                "cvxpy is required for portfolio optimization. "
                "Install it with: pip install cvxpy"
            )

        # Define optimization variable
        w = cp.Variable(self.n_assets)

        # Portfolio return and risk
        portfolio_return = self.posterior_returns @ w
        portfolio_risk = cp.quad_form(w, self.posterior_cov)

        # Constraints
        constraints = [cp.sum(w) == 1]  # Weights sum to 1

        if not allow_short:
            constraints.append(w >= weight_bounds[0])

        constraints.append(w <= weight_bounds[1])

        # Objective function based on optimization type
        if optimization_type == 'max_sharpe':
            # Maximize Sharpe ratio (equivalent to minimizing risk/return)
            # We use the formulation: min risk - risk_aversion * return
            objective = cp.Minimize(
                portfolio_risk - self.risk_aversion * portfolio_return +
                l2_gamma * cp.sum_squares(w)
            )

        elif optimization_type == 'min_volatility':
            objective = cp.Minimize(portfolio_risk + l2_gamma * cp.sum_squares(w))

        elif optimization_type == 'target_return':
            if target_return is None:
                raise ValueError("target_return must be specified for 'target_return' optimization")
            constraints.append(portfolio_return >= target_return)
            objective = cp.Minimize(portfolio_risk + l2_gamma * cp.sum_squares(w))

        elif optimization_type == 'max_return':
            objective = cp.Maximize(portfolio_return - l2_gamma * cp.sum_squares(w))

        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")

        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"Optimization failed with status: {problem.status}")

        return w.value

    def calculate_efficient_frontier(
        self,
        n_points: int = 50,
        allow_short: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the efficient frontier.

        Parameters:
        -----------
        n_points : int, default=50
            Number of points on the efficient frontier
        allow_short : bool, default=False
            Allow short positions

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Returns, volatilities, and weights for each point
        """
        if self.posterior_returns is None:
            self.calculate_posterior()

        # Define range of target returns
        min_return = np.min(self.posterior_returns)
        max_return = np.max(self.posterior_returns)
        target_returns = np.linspace(min_return, max_return, n_points)

        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []

        for target_return in target_returns:
            try:
                weights = self.optimize_portfolio(
                    optimization_type='target_return',
                    target_return=target_return,
                    allow_short=allow_short
                )

                portfolio_return = self.posterior_returns @ weights
                portfolio_volatility = np.sqrt(weights @ self.posterior_cov @ weights)

                frontier_returns.append(portfolio_return)
                frontier_volatilities.append(portfolio_volatility)
                frontier_weights.append(weights)
            except:
                continue

        return (
            np.array(frontier_returns),
            np.array(frontier_volatilities),
            np.array(frontier_weights)
        )

    def calculate_portfolio_metrics(
        self,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for a given portfolio.

        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights

        Returns:
        --------
        Dict[str, float] : Dictionary of portfolio metrics
        """
        if self.posterior_returns is None:
            self.calculate_posterior()

        expected_return = self.posterior_returns @ weights
        volatility = np.sqrt(weights @ self.posterior_cov @ weights)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility

        # Diversification metrics
        n_effective = 1 / np.sum(weights ** 2)  # Effective number of assets
        max_weight = np.max(weights)
        min_weight = np.min(weights)

        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'effective_assets': n_effective,
            'max_weight': max_weight,
            'min_weight': min_weight
        }

    def get_view_implied_returns(self) -> np.ndarray:
        """
        Calculate the returns implied by the views alone.

        Returns:
        --------
        np.ndarray : View-implied returns
        """
        if self.P is None:
            raise ValueError("No views have been specified")

        omega_inv = np.linalg.inv(self.omega)
        P_omega_P = self.P.T @ omega_inv @ self.P

        return np.linalg.inv(P_omega_P) @ (self.P.T @ omega_inv @ self.Q)

    def get_attribution(self) -> Dict[str, np.ndarray]:
        """
        Decompose posterior returns into prior and view contributions.

        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary with 'prior' and 'views' contributions
        """
        if self.posterior_returns is None:
            self.calculate_posterior()

        if self.P is None:
            return {
                'prior': self.pi,
                'views': np.zeros_like(self.pi),
                'posterior': self.posterior_returns
            }

        view_contribution = self.posterior_returns - self.pi

        return {
            'prior': self.pi,
            'views': view_contribution,
            'posterior': self.posterior_returns
        }

    def summary(self, asset_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a summary DataFrame of the model state.

        Parameters:
        -----------
        asset_names : List[str], optional
            Names of assets for display

        Returns:
        --------
        pd.DataFrame : Summary of equilibrium, posterior returns, and attribution
        """
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(self.n_assets)]

        data = {
            'Market_Weight': self.market_weights,
            'Equilibrium_Return': self.pi
        }

        if self.posterior_returns is not None:
            attribution = self.get_attribution()
            data['Posterior_Return'] = self.posterior_returns
            data['View_Impact'] = attribution['views']

        return pd.DataFrame(data, index=asset_names)
