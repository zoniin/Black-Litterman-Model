# Black-Litterman Portfolio Optimization Model

A comprehensive, production-ready implementation of the Black-Litterman portfolio optimization model in Python. This package combines market equilibrium with investor views using Bayesian inference to generate robust, diversified portfolios.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Complete Black-Litterman Implementation**
  - Market equilibrium calculation via reverse optimization
  - Bayesian combination of prior and views
  - Multiple view types (absolute, relative, custom)
  - Confidence-based view weighting

- **Advanced Portfolio Optimization**
  - Maximum Sharpe ratio
  - Minimum volatility
  - Target return optimization
  - Custom constraints and regularization
  - Efficient frontier calculation

- **Data Integration**
  - Automatic data fetching via Yahoo Finance
  - Multiple covariance estimation methods (sample, Ledoit-Wolf, shrinkage)
  - Market capitalization-based equilibrium weights

- **Comprehensive Visualization**
  - Efficient frontier plots
  - Portfolio weight visualization
  - Return attribution analysis
  - Correlation heatmaps
  - Interactive dashboards

- **Production-Ready**
  - Extensive documentation
  - Unit tests
  - Type hints
  - Error handling
  - Validation utilities

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Mathematical Background](#mathematical-background)
- [Usage Guide](#usage-guide)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Black-Litterman-Model.git
cd Black-Litterman-Model

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

The package requires the following libraries:

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
yfinance>=0.2.0
cvxpy>=1.3.0
plotly>=5.14.0
```

## Quick Start

Here's a minimal example to get you started:

```python
from src.black_litterman import BlackLittermanModel
from src.data_utils import prepare_bl_inputs
from src.visualization import plot_efficient_frontier

# Define your asset universe
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']

# Fetch data and prepare inputs
data = prepare_bl_inputs(
    tickers=tickers,
    period='2y',
    risk_free_rate=0.03
)

# Initialize the Black-Litterman model
bl_model = BlackLittermanModel(
    cov_matrix=data['cov_matrix'],
    market_caps=data['market_caps'],
    risk_aversion=2.5,
    tau=0.025
)

# Add investor views
bl_model.add_absolute_view(
    asset_idx=0,  # Apple
    expected_return=0.12,  # 12% expected return
    confidence=0.8  # High confidence
)

# Calculate posterior distribution
posterior_returns, posterior_cov = bl_model.calculate_posterior()

# Optimize portfolio
optimal_weights = bl_model.optimize_portfolio(
    optimization_type='max_sharpe'
)

# Display results
print("Optimal Weights:", optimal_weights)
metrics = bl_model.calculate_portfolio_metrics(optimal_weights)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
```

## Mathematical Background

### The Black-Litterman Formula

The Black-Litterman model computes posterior expected returns by combining market equilibrium (prior) with investor views using Bayesian inference:

$$E[R] = [(\\tau\\Sigma)^{-1} + P'\\Omega^{-1}P]^{-1} [(\\tau\\Sigma)^{-1}\\Pi + P'\\Omega^{-1}Q]$$

where:

- **E[R]**: Posterior expected returns (what we want to find)
- **Π** (Pi): Prior expected returns from market equilibrium
- **Σ** (Sigma): Covariance matrix of asset returns
- **P**: Picking matrix that identifies assets in each view
- **Q**: Vector of view returns (expected returns from views)
- **Ω** (Omega): Diagonal matrix representing uncertainty in views
- **τ** (tau): Scalar parameter controlling uncertainty in prior (typically 0.01-0.05)

### Market Equilibrium (Prior)

The prior is derived from market equilibrium using reverse optimization (CAPM):

$$\\Pi = \\delta \\Sigma w_{mkt}$$

where:
- **δ** (delta): Risk aversion coefficient
- **w_mkt**: Market capitalization weights

### View Specification

Views are specified through the picking matrix **P** and view returns **Q**:

**Absolute View**: "Asset i will return q%"
```
P = [0, 0, ..., 1, ..., 0]  (1 at position i)
Q = [q]
```

**Relative View**: "Asset i will outperform asset j by q%"
```
P = [0, ..., 1, ..., -1, ..., 0]  (1 at i, -1 at j)
Q = [q]
```

### View Uncertainty

The uncertainty matrix **Ω** is typically diagonal:

$$\\Omega_{ii} = \\frac{P_i \\Sigma P_i'}{c_i}$$

where **c_i** is the confidence level (0 < c_i ≤ 1) for view i.

### Posterior Covariance

The posterior covariance matrix is:

$$Cov[R] = \\Sigma + [(\\tau\\Sigma)^{-1} + P'\\Omega^{-1}P]^{-1}$$

## Usage Guide

### 1. Data Preparation

```python
from src.data_utils import prepare_bl_inputs

# Automatic data preparation
data = prepare_bl_inputs(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    period='3y',  # Historical period
    cov_method='ledoit_wolf',  # Shrinkage estimator
    risk_free_rate=0.03
)
```

### 2. Model Initialization

```python
from src.black_litterman import BlackLittermanModel

bl_model = BlackLittermanModel(
    cov_matrix=data['cov_matrix'],
    market_caps=data['market_caps'],
    risk_aversion=2.5,  # Market risk aversion
    tau=0.025,  # Prior uncertainty
    risk_free_rate=0.03
)
```

### 3. Adding Views

#### Absolute Views

```python
# "Apple will return 12%"
bl_model.add_absolute_view(
    asset_idx=0,
    expected_return=0.12,
    confidence=0.8
)
```

#### Relative Views

```python
# "Microsoft will outperform Google by 3%"
bl_model.add_relative_view(
    asset_idx1=1,  # Microsoft
    asset_idx2=2,  # Google
    outperformance=0.03,
    confidence=0.6
)
```

#### Custom Views

```python
# "A portfolio of 50% AAPL + 50% MSFT will return 10%"
picking_vector = [0.5, 0.5, 0, 0, 0]
bl_model.add_custom_view(
    picking_vector=picking_vector,
    expected_return=0.10,
    confidence=0.7
)
```

#### Batch Views

```python
views = [
    {'type': 'absolute', 'assets': 0, 'return': 0.12, 'confidence': 0.8},
    {'type': 'relative', 'assets': (1, 2), 'return': 0.03, 'confidence': 0.6}
]
bl_model.add_views_from_dict(views)
```

### 4. Calculate Posterior

```python
posterior_returns, posterior_cov = bl_model.calculate_posterior()

# View summary
summary = bl_model.summary(asset_names=['AAPL', 'MSFT', 'GOOGL'])
print(summary)
```

### 5. Portfolio Optimization

```python
# Maximum Sharpe ratio
weights = bl_model.optimize_portfolio(
    optimization_type='max_sharpe',
    allow_short=False,
    weight_bounds=(0.0, 0.4),  # Max 40% per asset
    l2_gamma=0.1  # Regularization for diversification
)

# Minimum volatility
weights_min_vol = bl_model.optimize_portfolio(
    optimization_type='min_volatility'
)

# Target return
weights_target = bl_model.optimize_portfolio(
    optimization_type='target_return',
    target_return=0.10
)
```

### 6. Analysis and Visualization

```python
from src.visualization import (
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_return_attribution,
    create_portfolio_dashboard
)

# Efficient frontier
frontier_returns, frontier_vols, _ = bl_model.calculate_efficient_frontier(n_points=100)
plot_efficient_frontier(frontier_returns, frontier_vols, optimal_weights=weights)

# Portfolio weights
plot_portfolio_weights(weights, asset_names)

# Return attribution
plot_return_attribution(bl_model, asset_names)

# Comprehensive dashboard
create_portfolio_dashboard(bl_model, weights, asset_names)
```

### 7. Portfolio Metrics

```python
metrics = bl_model.calculate_portfolio_metrics(weights)

print(f"Expected Return: {metrics['expected_return']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Effective Assets: {metrics['effective_assets']:.2f}")
```

## Examples

### Example 1: Simple 5-Asset Portfolio

See `examples/simple_example.py` for a complete working example.

```bash
cd examples
python simple_example.py
```

### Example 2: Comprehensive Analysis

The Jupyter notebook `notebooks/comprehensive_example.ipynb` contains a detailed walkthrough including:

- Data preparation and exploration
- Multiple view specifications
- Sensitivity analysis
- Efficient frontier analysis
- Strategy comparison
- Full visualization suite

```bash
jupyter notebook notebooks/comprehensive_example.ipynb
```

## API Reference

### BlackLittermanModel

Main class for Black-Litterman portfolio optimization.

#### Initialization

```python
BlackLittermanModel(
    cov_matrix: np.ndarray,
    market_caps: Optional[np.ndarray] = None,
    risk_aversion: float = 2.5,
    tau: float = 0.025,
    risk_free_rate: float = 0.02
)
```

#### Key Methods

- `add_absolute_view(asset_idx, expected_return, confidence)`: Add absolute view
- `add_relative_view(asset_idx1, asset_idx2, outperformance, confidence)`: Add relative view
- `add_custom_view(picking_vector, expected_return, confidence)`: Add custom view
- `calculate_posterior()`: Compute posterior distribution
- `optimize_portfolio(optimization_type, **kwargs)`: Optimize portfolio weights
- `calculate_efficient_frontier(n_points, allow_short)`: Calculate efficient frontier
- `calculate_portfolio_metrics(weights)`: Compute portfolio metrics
- `summary(asset_names)`: Generate summary DataFrame

### Data Utilities

- `prepare_bl_inputs(tickers, start_date, end_date, period, cov_method, risk_free_rate)`: One-stop data preparation
- `fetch_price_data(tickers, start_date, end_date, period)`: Fetch historical prices
- `calculate_returns(prices, method, frequency)`: Calculate returns
- `calculate_covariance_matrix(returns, method, shrinkage, frequency)`: Estimate covariance
- `get_market_caps(tickers)`: Fetch market capitalizations

### Visualization Functions

- `plot_efficient_frontier(...)`: Plot efficient frontier
- `plot_portfolio_weights(...)`: Bar chart of portfolio weights
- `plot_return_attribution(...)`: Attribution analysis
- `plot_correlation_matrix(...)`: Correlation heatmap
- `create_portfolio_dashboard(...)`: Comprehensive dashboard

## Advanced Topics

### Custom Risk Aversion

Estimate market risk aversion from historical data:

```python
from src.data_utils import estimate_risk_aversion

delta = estimate_risk_aversion(
    market_return=0.10,
    market_variance=0.04,
    risk_free_rate=0.02
)
```

### Covariance Estimation Methods

```python
# Sample covariance
cov = calculate_covariance_matrix(returns, method='sample')

# Ledoit-Wolf shrinkage (recommended)
cov = calculate_covariance_matrix(returns, method='ledoit_wolf')

# Custom shrinkage
cov = calculate_covariance_matrix(returns, method='shrinkage', shrinkage=0.2)
```

### Portfolio Constraints

```python
# No short selling, max 30% per asset
weights = bl_model.optimize_portfolio(
    optimization_type='max_sharpe',
    allow_short=False,
    weight_bounds=(0.0, 0.3)
)

# L2 regularization for diversification
weights = bl_model.optimize_portfolio(
    optimization_type='max_sharpe',
    l2_gamma=0.5  # Higher = more diversification
)
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Project Structure

```
Black-Litterman-Model/
├── src/
│   ├── __init__.py
│   ├── black_litterman.py      # Core BL model
│   ├── data_utils.py           # Data fetching and processing
│   └── visualization.py        # Plotting functions
├── notebooks/
│   └── comprehensive_example.ipynb
├── examples/
│   └── simple_example.py
├── tests/
│   ├── test_black_litterman.py
│   ├── test_data_utils.py
│   └── test_visualization.py
├── requirements.txt
└── README.md
```

## Theoretical Foundation

The Black-Litterman model was developed by Fischer Black and Robert Litterman at Goldman Sachs in 1990-1992. It addresses key limitations of traditional mean-variance optimization:

1. **Input Sensitivity**: Small changes in expected returns lead to extreme portfolio changes
2. **Concentrated Portfolios**: Optimizer produces corner solutions with extreme weights
3. **Ignoring Market Equilibrium**: No reference to market consensus

### Key Advantages

- **Stable Portfolios**: Incorporates market equilibrium as anchor
- **Intuitive Views**: Express views in natural language (relative/absolute)
- **Confidence Levels**: Weight views by conviction
- **Diversification**: More balanced portfolios than MVO
- **Bayesian Framework**: Principled combination of prior and views

## References

1. Black, F., & Litterman, R. (1992). "Global Portfolio Optimization". Financial Analysts Journal, 48(5), 28-43.

2. He, G., & Litterman, R. (1999). "The Intuition Behind Black-Litterman Model Portfolios". Goldman Sachs Quantitative Resources Group.

3. Idzorek, T. (2005). "A Step-by-Step Guide to the Black-Litterman Model". Ibbotson Associates.

4. Walters, J. (2014). "The Black-Litterman Model in Detail". Available at SSRN.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Fischer Black and Robert Litterman for developing the model
- The quantitative finance community for various implementations and insights
- Contributors to the open-source libraries used in this project

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Disclaimer**: This software is for educational and research purposes only. It is not intended as financial advice. Always consult with qualified financial professionals before making investment decisions.
