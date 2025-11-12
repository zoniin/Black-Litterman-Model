"""
Data utilities for Black-Litterman portfolio model.

This module provides functions for:
- Fetching historical market data
- Calculating covariance matrices and returns
- Estimating market parameters
- Data preprocessing and validation
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import warnings
from datetime import datetime, timedelta


def fetch_price_data(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "2y"
) -> pd.DataFrame:
    """
    Fetch historical price data for given tickers.

    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    period : str, default='2y'
        Period to fetch if dates not specified (e.g., '1y', '2y', '5y')

    Returns:
    --------
    pd.DataFrame : DataFrame with adjusted close prices
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for fetching data. "
            "Install it with: pip install yfinance"
        )

    # Download data
    if start_date and end_date:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    else:
        data = yf.download(tickers, period=period, progress=False)

    # Extract adjusted close prices
    if len(tickers) == 1:
        prices = data['Adj Close'].to_frame()
        prices.columns = tickers
    else:
        prices = data['Adj Close']

    # Handle missing data
    prices = prices.fillna(method='ffill').fillna(method='bfill')

    return prices


def calculate_returns(
    prices: pd.DataFrame,
    method: str = 'log',
    frequency: str = 'daily'
) -> pd.DataFrame:
    """
    Calculate returns from price data.

    Parameters:
    -----------
    prices : pd.DataFrame
        DataFrame of prices
    method : str, default='log'
        'log' for log returns or 'simple' for simple returns
    frequency : str, default='daily'
        Frequency of data ('daily', 'weekly', 'monthly')

    Returns:
    --------
    pd.DataFrame : DataFrame of returns
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")

    # Drop NaN values
    returns = returns.dropna()

    return returns


def calculate_covariance_matrix(
    returns: pd.DataFrame,
    method: str = 'sample',
    shrinkage: Optional[float] = None,
    frequency: int = 252
) -> np.ndarray:
    """
    Calculate covariance matrix of returns.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of returns
    method : str, default='sample'
        Method for estimating covariance:
        - 'sample': Sample covariance
        - 'ledoit_wolf': Ledoit-Wolf shrinkage
        - 'shrinkage': Custom shrinkage to identity/diagonal
    shrinkage : float, optional
        Shrinkage parameter (0-1) for 'shrinkage' method
    frequency : int, default=252
        Number of periods per year for annualization (252 for daily data)

    Returns:
    --------
    np.ndarray : Annualized covariance matrix
    """
    if method == 'sample':
        cov_matrix = returns.cov().values * frequency

    elif method == 'ledoit_wolf':
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns).covariance_ * frequency
        except ImportError:
            warnings.warn(
                "scikit-learn not installed. Using sample covariance. "
                "Install with: pip install scikit-learn"
            )
            cov_matrix = returns.cov().values * frequency

    elif method == 'shrinkage':
        if shrinkage is None:
            shrinkage = 0.1

        sample_cov = returns.cov().values
        # Shrink to diagonal matrix (Ledoit-Wolf style)
        n = sample_cov.shape[0]
        diag_cov = np.diag(np.diag(sample_cov))

        cov_matrix = (
            shrinkage * diag_cov + (1 - shrinkage) * sample_cov
        ) * frequency

    else:
        raise ValueError(f"Unknown method: {method}")

    return cov_matrix


def calculate_expected_returns(
    returns: pd.DataFrame,
    method: str = 'mean',
    frequency: int = 252
) -> np.ndarray:
    """
    Calculate expected returns.

    Parameters:
    -----------
    returns : pd.DataFrame
        DataFrame of returns
    method : str, default='mean'
        Method for estimating expected returns:
        - 'mean': Historical mean
        - 'ewma': Exponentially weighted moving average
        - 'capm': Capital Asset Pricing Model (requires market return)
    frequency : int, default=252
        Number of periods per year for annualization

    Returns:
    --------
    np.ndarray : Annualized expected returns
    """
    if method == 'mean':
        expected_returns = returns.mean().values * frequency

    elif method == 'ewma':
        # Use exponentially weighted mean
        expected_returns = returns.ewm(span=60).mean().iloc[-1].values * frequency

    else:
        raise ValueError(f"Unknown method: {method}")

    return expected_returns


def get_market_caps(
    tickers: List[str]
) -> Dict[str, float]:
    """
    Fetch current market capitalizations for tickers.

    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols

    Returns:
    --------
    Dict[str, float] : Dictionary mapping tickers to market caps
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for fetching data. "
            "Install it with: pip install yfinance"
        )

    market_caps = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get('marketCap', None)

            if market_cap is None:
                warnings.warn(f"Market cap not available for {ticker}")
                market_cap = 1.0  # Default equal weight

            market_caps[ticker] = market_cap
        except Exception as e:
            warnings.warn(f"Error fetching market cap for {ticker}: {e}")
            market_caps[ticker] = 1.0

    return market_caps


def estimate_risk_aversion(
    market_return: float = 0.10,
    market_variance: float = 0.04,
    risk_free_rate: float = 0.02
) -> float:
    """
    Estimate market risk aversion coefficient.

    Formula: δ = (E[R_m] - R_f) / σ²_m

    Parameters:
    -----------
    market_return : float, default=0.10
        Expected market return (annualized)
    market_variance : float, default=0.04
        Market variance (annualized)
    risk_free_rate : float, default=0.02
        Risk-free rate (annualized)

    Returns:
    --------
    float : Risk aversion coefficient
    """
    return (market_return - risk_free_rate) / market_variance


def prepare_bl_inputs(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "2y",
    cov_method: str = 'ledoit_wolf',
    risk_free_rate: float = 0.02
) -> Dict:
    """
    Prepare all inputs needed for Black-Litterman model.

    This is a convenience function that fetches data and calculates
    all necessary parameters in one go.

    Parameters:
    -----------
    tickers : List[str]
        List of ticker symbols
    start_date : str, optional
        Start date for historical data
    end_date : str, optional
        End date for historical data
    period : str, default='2y'
        Period to fetch if dates not specified
    cov_method : str, default='ledoit_wolf'
        Method for covariance estimation
    risk_free_rate : float, default=0.02
        Risk-free rate

    Returns:
    --------
    Dict : Dictionary containing:
        - 'cov_matrix': Covariance matrix
        - 'market_caps': Market capitalization array
        - 'returns': Historical returns DataFrame
        - 'prices': Historical prices DataFrame
        - 'tickers': List of tickers
        - 'risk_aversion': Estimated risk aversion coefficient
    """
    print(f"Fetching data for {len(tickers)} assets...")

    # Fetch price data
    prices = fetch_price_data(tickers, start_date, end_date, period)

    # Calculate returns
    returns = calculate_returns(prices, method='log')

    # Calculate covariance matrix
    cov_matrix = calculate_covariance_matrix(returns, method=cov_method)

    # Get market caps
    market_caps_dict = get_market_caps(tickers)
    market_caps = np.array([market_caps_dict[ticker] for ticker in tickers])

    # Estimate risk aversion from market portfolio
    market_return = returns.mean().mean() * 252  # Average return
    market_variance = returns.var().mean() * 252  # Average variance
    risk_aversion = estimate_risk_aversion(
        market_return,
        market_variance,
        risk_free_rate
    )

    print(f"Data preparation complete!")
    print(f"  - Historical period: {returns.index[0]} to {returns.index[-1]}")
    print(f"  - Number of observations: {len(returns)}")
    print(f"  - Estimated risk aversion: {risk_aversion:.2f}")

    return {
        'cov_matrix': cov_matrix,
        'market_caps': market_caps,
        'returns': returns,
        'prices': prices,
        'tickers': tickers,
        'risk_aversion': risk_aversion,
        'risk_free_rate': risk_free_rate
    }


def validate_inputs(
    cov_matrix: np.ndarray,
    market_caps: Optional[np.ndarray] = None
) -> bool:
    """
    Validate inputs for Black-Litterman model.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix
    market_caps : np.ndarray, optional
        Market capitalizations

    Returns:
    --------
    bool : True if inputs are valid

    Raises:
    -------
    ValueError : If inputs are invalid
    """
    # Check covariance matrix is square
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square")

    # Check covariance matrix is positive semi-definite
    eigenvalues = np.linalg.eigvals(cov_matrix)
    if np.any(eigenvalues < -1e-8):
        raise ValueError("Covariance matrix must be positive semi-definite")

    # Check covariance matrix is symmetric
    if not np.allclose(cov_matrix, cov_matrix.T):
        raise ValueError("Covariance matrix must be symmetric")

    # Check market caps if provided
    if market_caps is not None:
        if len(market_caps) != cov_matrix.shape[0]:
            raise ValueError(
                "Number of market caps must match covariance matrix dimension"
            )
        if np.any(market_caps < 0):
            raise ValueError("Market caps must be non-negative")

    return True
