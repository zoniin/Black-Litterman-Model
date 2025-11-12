"""
Unit tests for data utility functions.
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('../src')

from src.data_utils import (
    calculate_returns,
    calculate_covariance_matrix,
    calculate_expected_returns,
    estimate_risk_aversion,
    validate_inputs
)


class TestDataUtils:
    """Test suite for data utility functions."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'Asset_A': 100 * (1 + np.random.randn(100).cumsum() * 0.01),
            'Asset_B': 50 * (1 + np.random.randn(100).cumsum() * 0.01),
            'Asset_C': 200 * (1 + np.random.randn(100).cumsum() * 0.01)
        }, index=dates)

        # Ensure positive prices
        prices = prices.abs() + 10

        return prices

    def test_calculate_returns_log(self, sample_prices):
        """Test log return calculation."""
        returns = calculate_returns(sample_prices, method='log')

        # Check shape (one less row due to differencing)
        assert len(returns) == len(sample_prices) - 1
        assert returns.shape[1] == sample_prices.shape[1]

        # Check no NaN values
        assert not returns.isna().any().any()

    def test_calculate_returns_simple(self, sample_prices):
        """Test simple return calculation."""
        returns = calculate_returns(sample_prices, method='simple')

        assert len(returns) == len(sample_prices) - 1
        assert not returns.isna().any().any()

    def test_calculate_covariance_sample(self, sample_prices):
        """Test sample covariance calculation."""
        returns = calculate_returns(sample_prices)
        cov_matrix = calculate_covariance_matrix(returns, method='sample')

        # Check shape
        n_assets = returns.shape[1]
        assert cov_matrix.shape == (n_assets, n_assets)

        # Check symmetry
        np.testing.assert_array_almost_equal(cov_matrix, cov_matrix.T)

        # Check positive semi-definite
        eigenvalues = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvalues >= -1e-8)

    def test_calculate_covariance_shrinkage(self, sample_prices):
        """Test shrinkage covariance calculation."""
        returns = calculate_returns(sample_prices)
        cov_matrix = calculate_covariance_matrix(
            returns,
            method='shrinkage',
            shrinkage=0.2
        )

        # Check basic properties
        assert cov_matrix.shape == (3, 3)
        np.testing.assert_array_almost_equal(cov_matrix, cov_matrix.T)

    def test_calculate_expected_returns(self, sample_prices):
        """Test expected return calculation."""
        returns = calculate_returns(sample_prices)
        expected_returns = calculate_expected_returns(returns, method='mean')

        # Check shape
        assert len(expected_returns) == returns.shape[1]

        # Values should be reasonable (annualized)
        assert np.all(np.abs(expected_returns) < 10)  # Less than 1000% return

    def test_estimate_risk_aversion(self):
        """Test risk aversion estimation."""
        risk_aversion = estimate_risk_aversion(
            market_return=0.10,
            market_variance=0.04,
            risk_free_rate=0.02
        )

        # Should be positive
        assert risk_aversion > 0

        # Typical values are 1-5
        assert 0.5 < risk_aversion < 10

    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        cov_matrix = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])
        market_caps = np.array([1000, 500])

        # Should not raise exception
        assert validate_inputs(cov_matrix, market_caps)

    def test_validate_inputs_nonsquare(self):
        """Test validation catches non-square covariance matrix."""
        cov_matrix = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.03]
        ])

        with pytest.raises(ValueError, match="square"):
            validate_inputs(cov_matrix)

    def test_validate_inputs_nonsymmetric(self):
        """Test validation catches non-symmetric matrix."""
        cov_matrix = np.array([
            [0.04, 0.01],
            [0.02, 0.09]  # Not symmetric
        ])

        with pytest.raises(ValueError, match="symmetric"):
            validate_inputs(cov_matrix)

    def test_validate_inputs_negative_definite(self):
        """Test validation catches non-positive semi-definite matrix."""
        cov_matrix = np.array([
            [0.04, 0.10],
            [0.10, 0.01]  # Not positive semi-definite
        ])

        with pytest.raises(ValueError, match="positive semi-definite"):
            validate_inputs(cov_matrix)

    def test_validate_inputs_mismatched_caps(self):
        """Test validation catches mismatched market caps."""
        cov_matrix = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])
        market_caps = np.array([1000])  # Wrong length

        with pytest.raises(ValueError, match="Number of market caps"):
            validate_inputs(cov_matrix, market_caps)

    def test_validate_inputs_negative_caps(self):
        """Test validation catches negative market caps."""
        cov_matrix = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])
        market_caps = np.array([1000, -500])  # Negative cap

        with pytest.raises(ValueError, match="non-negative"):
            validate_inputs(cov_matrix, market_caps)
