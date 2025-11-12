"""
Simple Black-Litterman Example

This script demonstrates a basic workflow for using the Black-Litterman model.
"""

import sys
sys.path.append('../src')

import numpy as np
from black_litterman import BlackLittermanModel
from data_utils import prepare_bl_inputs
from visualization import plot_efficient_frontier, plot_portfolio_weights
import matplotlib.pyplot as plt


def main():
    """Run a simple Black-Litterman portfolio optimization."""

    print("="*60)
    print("BLACK-LITTERMAN PORTFOLIO OPTIMIZATION - Simple Example")
    print("="*60)

    # Define assets
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
    asset_names = ['Apple', 'Microsoft', 'Google', 'Amazon', 'JPMorgan']

    print(f"\nAssets: {', '.join(asset_names)}")

    # Prepare data
    print("\nFetching and preparing data...")
    data = prepare_bl_inputs(
        tickers=tickers,
        period='2y',
        cov_method='ledoit_wolf',
        risk_free_rate=0.03
    )

    # Initialize model
    print("\nInitializing Black-Litterman model...")
    bl_model = BlackLittermanModel(
        cov_matrix=data['cov_matrix'],
        market_caps=data['market_caps'],
        risk_aversion=data['risk_aversion'],
        tau=0.025,
        risk_free_rate=0.03
    )

    # Display equilibrium
    print("\nMarket Equilibrium Returns:")
    for name, ret in zip(asset_names, bl_model.pi):
        print(f"  {name:15s}: {ret*100:6.2f}%")

    # Add views
    print("\nAdding investor views...")

    # View 1: Apple will return 12%
    bl_model.add_absolute_view(
        asset_idx=0,  # Apple
        expected_return=0.12,
        confidence=0.8
    )
    print("  ✓ Apple expected to return 12% (high confidence)")

    # View 2: Microsoft will outperform Amazon by 2%
    bl_model.add_relative_view(
        asset_idx1=1,  # Microsoft
        asset_idx2=3,  # Amazon
        outperformance=0.02,
        confidence=0.6
    )
    print("  ✓ Microsoft to outperform Amazon by 2% (moderate confidence)")

    # Calculate posterior
    print("\nCalculating posterior distribution...")
    posterior_returns, posterior_cov = bl_model.calculate_posterior()

    print("\nPosterior Returns:")
    for name, ret in zip(asset_names, posterior_returns):
        print(f"  {name:15s}: {ret*100:6.2f}%")

    # Optimize portfolio
    print("\nOptimizing portfolio for maximum Sharpe ratio...")
    optimal_weights = bl_model.optimize_portfolio(
        optimization_type='max_sharpe',
        allow_short=False
    )

    # Display results
    print("\n" + "="*60)
    print("OPTIMAL PORTFOLIO")
    print("="*60)

    print("\nWeights:")
    for name, weight in zip(asset_names, optimal_weights):
        print(f"  {name:15s}: {weight*100:6.2f}%")

    # Calculate metrics
    metrics = bl_model.calculate_portfolio_metrics(optimal_weights)

    print("\nPortfolio Metrics:")
    print(f"  Expected Return:   {metrics['expected_return']*100:6.2f}%")
    print(f"  Volatility:        {metrics['volatility']*100:6.2f}%")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:6.3f}")
    print(f"  Effective Assets:  {metrics['effective_assets']:6.2f}")

    # Calculate efficient frontier
    print("\nCalculating efficient frontier...")
    frontier_returns, frontier_vols, frontier_weights = bl_model.calculate_efficient_frontier(
        n_points=50,
        allow_short=False
    )

    # Visualizations
    print("\nGenerating visualizations...")

    # Plot 1: Efficient frontier
    fig1 = plot_efficient_frontier(
        frontier_returns,
        frontier_vols,
        optimal_weights=optimal_weights,
        posterior_returns=posterior_returns,
        posterior_cov=posterior_cov,
        asset_names=asset_names,
        risk_free_rate=0.03,
        title="Black-Litterman Efficient Frontier"
    )
    plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: efficient_frontier.png")

    # Plot 2: Portfolio weights
    fig2 = plot_portfolio_weights(
        optimal_weights,
        asset_names=asset_names,
        title="Optimal Portfolio Weights"
    )
    plt.savefig('portfolio_weights.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: portfolio_weights.png")

    print("\n" + "="*60)
    print("Analysis complete! Check the output PNG files.")
    print("="*60)


if __name__ == "__main__":
    main()
