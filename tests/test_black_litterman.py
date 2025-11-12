"""
Unit tests for Black-Litterman model implementation.
"""

import pytest
import numpy as np
import sys
sys.path.append('../src')

from src.black_litterman import BlackLittermanModel


class TestBlackLittermanModel:
    """Test suite for BlackLittermanModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # 3 assets
        n_assets = 3

        # Sample covariance matrix (must be positive semi-definite)
        cov_matrix = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.03],
            [0.02, 0.03, 0.16]
        ])

        # Market caps
        market_caps = np.array([1000, 500, 300])

        return {
            'cov_matrix': cov_matrix,
            'market_caps': market_caps,
            'n_assets': n_assets
        }

    def test_initialization(self, sample_data):
        """Test model initialization."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps'],
            risk_aversion=2.5,
            tau=0.025
        )

        assert bl_model.n_assets == sample_data['n_assets']
        assert bl_model.risk_aversion == 2.5
        assert bl_model.tau == 0.025
        assert bl_model.P is None
        assert bl_model.Q is None

    def test_market_weights(self, sample_data):
        """Test market weight calculation."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        # Weights should sum to 1
        assert np.isclose(bl_model.market_weights.sum(), 1.0)

        # Weights should be proportional to market caps
        expected_weights = sample_data['market_caps'] / sample_data['market_caps'].sum()
        np.testing.assert_array_almost_equal(bl_model.market_weights, expected_weights)

    def test_equilibrium_returns(self, sample_data):
        """Test equilibrium return calculation."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps'],
            risk_aversion=2.5
        )

        # Pi should equal delta * Sigma * w_mkt
        expected_pi = 2.5 * sample_data['cov_matrix'] @ bl_model.market_weights
        np.testing.assert_array_almost_equal(bl_model.pi, expected_pi)

    def test_add_absolute_view(self, sample_data):
        """Test adding absolute view."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        bl_model.add_absolute_view(
            asset_idx=0,
            expected_return=0.10,
            confidence=0.8
        )

        assert bl_model.P is not None
        assert bl_model.Q is not None
        assert bl_model.omega is not None

        # Check picking matrix
        assert bl_model.P.shape == (1, sample_data['n_assets'])
        assert bl_model.P[0, 0] == 1.0
        assert bl_model.P[0, 1] == 0.0
        assert bl_model.P[0, 2] == 0.0

        # Check view return
        assert bl_model.Q[0] == 0.10

    def test_add_relative_view(self, sample_data):
        """Test adding relative view."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        bl_model.add_relative_view(
            asset_idx1=0,
            asset_idx2=1,
            outperformance=0.05,
            confidence=0.6
        )

        # Check picking matrix
        assert bl_model.P[0, 0] == 1.0
        assert bl_model.P[0, 1] == -1.0
        assert bl_model.P[0, 2] == 0.0

        # Check view return
        assert bl_model.Q[0] == 0.05

    def test_multiple_views(self, sample_data):
        """Test adding multiple views."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        bl_model.add_absolute_view(0, 0.10, 0.8)
        bl_model.add_relative_view(1, 2, 0.03, 0.6)

        assert bl_model.P.shape == (2, sample_data['n_assets'])
        assert len(bl_model.Q) == 2
        assert bl_model.omega.shape == (2, 2)

    def test_posterior_calculation(self, sample_data):
        """Test posterior distribution calculation."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        bl_model.add_absolute_view(0, 0.10, 0.8)
        posterior_returns, posterior_cov = bl_model.calculate_posterior()

        # Check shapes
        assert posterior_returns.shape == (sample_data['n_assets'],)
        assert posterior_cov.shape == (sample_data['n_assets'], sample_data['n_assets'])

        # Posterior covariance should be positive semi-definite
        eigenvalues = np.linalg.eigvals(posterior_cov)
        assert np.all(eigenvalues >= -1e-8)

    def test_posterior_without_views(self, sample_data):
        """Test posterior calculation without views (should return equilibrium)."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        posterior_returns, posterior_cov = bl_model.calculate_posterior()

        # Without views, posterior should equal prior
        np.testing.assert_array_almost_equal(posterior_returns, bl_model.pi)
        np.testing.assert_array_almost_equal(posterior_cov, bl_model.cov_matrix)

    def test_optimize_portfolio(self, sample_data):
        """Test portfolio optimization."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        bl_model.add_absolute_view(0, 0.10, 0.8)
        bl_model.calculate_posterior()

        weights = bl_model.optimize_portfolio(
            optimization_type='max_sharpe',
            allow_short=False
        )

        # Check weights sum to 1
        assert np.isclose(weights.sum(), 1.0)

        # Check no short positions
        assert np.all(weights >= -1e-6)

    def test_optimize_min_volatility(self, sample_data):
        """Test minimum volatility optimization."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        bl_model.calculate_posterior()

        weights = bl_model.optimize_portfolio(
            optimization_type='min_volatility',
            allow_short=False
        )

        assert np.isclose(weights.sum(), 1.0)
        assert np.all(weights >= -1e-6)

    def test_calculate_portfolio_metrics(self, sample_data):
        """Test portfolio metrics calculation."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps'],
            risk_free_rate=0.02
        )

        bl_model.calculate_posterior()
        weights = np.array([0.5, 0.3, 0.2])

        metrics = bl_model.calculate_portfolio_metrics(weights)

        # Check all expected metrics are present
        assert 'expected_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'effective_assets' in metrics

        # Check values are reasonable
        assert metrics['volatility'] > 0
        assert 1 <= metrics['effective_assets'] <= 3

    def test_efficient_frontier(self, sample_data):
        """Test efficient frontier calculation."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        bl_model.calculate_posterior()

        returns, vols, weights = bl_model.calculate_efficient_frontier(
            n_points=10,
            allow_short=False
        )

        # Check shapes
        assert len(returns) > 0
        assert len(vols) > 0
        assert len(returns) == len(vols)

        # Volatilities should be positive
        assert np.all(vols > 0)

    def test_attribution(self, sample_data):
        """Test return attribution."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        bl_model.add_absolute_view(0, 0.10, 0.8)
        bl_model.calculate_posterior()

        attribution = bl_model.get_attribution()

        # Check all components present
        assert 'prior' in attribution
        assert 'views' in attribution
        assert 'posterior' in attribution

        # Prior + views should approximately equal posterior
        reconstructed = attribution['prior'] + attribution['views']
        np.testing.assert_array_almost_equal(reconstructed, attribution['posterior'])

    def test_invalid_covariance_matrix(self):
        """Test that invalid covariance matrix raises error."""
        # Non-square matrix
        with pytest.raises(ValueError):
            BlackLittermanModel(
                cov_matrix=np.array([[1, 2, 3], [4, 5, 6]])
            )

    def test_weight_bounds(self, sample_data):
        """Test portfolio optimization with weight bounds."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        bl_model.calculate_posterior()

        weights = bl_model.optimize_portfolio(
            optimization_type='max_sharpe',
            allow_short=False,
            weight_bounds=(0.0, 0.4)
        )

        # Check bounds are respected
        assert np.all(weights >= -1e-6)
        assert np.all(weights <= 0.4 + 1e-6)

    def test_custom_view(self, sample_data):
        """Test adding custom view."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        # Custom view: equal-weighted portfolio
        picking_vector = np.array([0.5, 0.5, 0.0])
        bl_model.add_custom_view(
            picking_vector=picking_vector,
            expected_return=0.08,
            confidence=0.7
        )

        np.testing.assert_array_almost_equal(bl_model.P[0], picking_vector)
        assert bl_model.Q[0] == 0.08

    def test_summary(self, sample_data):
        """Test summary DataFrame generation."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        bl_model.add_absolute_view(0, 0.10, 0.8)
        bl_model.calculate_posterior()

        summary = bl_model.summary(asset_names=['A', 'B', 'C'])

        # Check DataFrame structure
        assert len(summary) == 3
        assert 'Market_Weight' in summary.columns
        assert 'Equilibrium_Return' in summary.columns
        assert 'Posterior_Return' in summary.columns

    def test_views_from_dict(self, sample_data):
        """Test adding views from dictionary."""
        bl_model = BlackLittermanModel(
            cov_matrix=sample_data['cov_matrix'],
            market_caps=sample_data['market_caps']
        )

        views = [
            {'type': 'absolute', 'assets': 0, 'return': 0.10, 'confidence': 0.8},
            {'type': 'relative', 'assets': (1, 2), 'return': 0.03, 'confidence': 0.6}
        ]

        bl_model.add_views_from_dict(views)

        assert bl_model.P.shape == (2, 3)
        assert len(bl_model.Q) == 2
