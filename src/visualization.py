"""
Visualization utilities for Black-Litterman portfolio model.

This module provides comprehensive visualization functions for:
- Efficient frontier plotting
- Portfolio weights visualization
- Return attribution analysis
- Risk-return scatter plots
- Correlation heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
import warnings


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_efficient_frontier(
    frontier_returns: np.ndarray,
    frontier_volatilities: np.ndarray,
    optimal_weights: Optional[np.ndarray] = None,
    posterior_returns: Optional[np.ndarray] = None,
    posterior_cov: Optional[np.ndarray] = None,
    asset_names: Optional[List[str]] = None,
    risk_free_rate: float = 0.02,
    title: str = "Efficient Frontier",
    show_assets: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot the efficient frontier with optional individual assets.

    Parameters:
    -----------
    frontier_returns : np.ndarray
        Array of returns on the efficient frontier
    frontier_volatilities : np.ndarray
        Array of volatilities on the efficient frontier
    optimal_weights : np.ndarray, optional
        Optimal portfolio weights to highlight
    posterior_returns : np.ndarray, optional
        Posterior expected returns for individual assets
    posterior_cov : np.ndarray, optional
        Posterior covariance matrix for individual assets
    asset_names : List[str], optional
        Names of assets
    risk_free_rate : float, default=0.02
        Risk-free rate for Sharpe ratio line
    title : str
        Plot title
    show_assets : bool, default=True
        Whether to show individual assets
    figsize : Tuple[int, int]
        Figure size

    Returns:
    --------
    plt.Figure : Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot efficient frontier
    ax.plot(
        frontier_volatilities,
        frontier_returns,
        'b-',
        linewidth=3,
        label='Efficient Frontier'
    )

    # Plot optimal portfolio if provided
    if optimal_weights is not None and posterior_returns is not None and posterior_cov is not None:
        optimal_return = posterior_returns @ optimal_weights
        optimal_volatility = np.sqrt(optimal_weights @ posterior_cov @ optimal_weights)
        optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility

        ax.scatter(
            optimal_volatility,
            optimal_return,
            color='red',
            marker='*',
            s=500,
            label=f'Optimal Portfolio (SR={optimal_sharpe:.2f})',
            zorder=5
        )

    # Plot individual assets if requested
    if show_assets and posterior_returns is not None and posterior_cov is not None:
        asset_volatilities = np.sqrt(np.diag(posterior_cov))

        ax.scatter(
            asset_volatilities,
            posterior_returns,
            color='green',
            marker='o',
            s=100,
            alpha=0.6,
            label='Individual Assets'
        )

        # Label assets
        if asset_names is not None:
            for i, name in enumerate(asset_names):
                ax.annotate(
                    name,
                    (asset_volatilities[i], posterior_returns[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9
                )

    # Add capital market line (if optimal portfolio exists)
    if optimal_weights is not None and posterior_returns is not None and posterior_cov is not None:
        optimal_return = posterior_returns @ optimal_weights
        optimal_volatility = np.sqrt(optimal_weights @ posterior_cov @ optimal_weights)

        # CML: E[R] = Rf + (E[R_opt] - Rf) / σ_opt * σ
        max_vol = np.max(frontier_volatilities) * 1.2
        cml_vols = np.linspace(0, max_vol, 100)
        cml_returns = risk_free_rate + (optimal_return - risk_free_rate) / optimal_volatility * cml_vols

        ax.plot(
            cml_vols,
            cml_returns,
            'r--',
            linewidth=2,
            alpha=0.5,
            label='Capital Market Line'
        )

    # Formatting
    ax.set_xlabel('Volatility (Standard Deviation)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format axes as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))

    plt.tight_layout()
    return fig


def plot_portfolio_weights(
    weights: np.ndarray,
    asset_names: Optional[List[str]] = None,
    title: str = "Portfolio Weights",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot portfolio weights as a bar chart.

    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights
    asset_names : List[str], optional
        Names of assets
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns:
    --------
    plt.Figure : Matplotlib figure object
    """
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(len(weights))]

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by weight
    sorted_indices = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_indices]
    sorted_names = [asset_names[i] for i in sorted_indices]

    # Color bars by positive/negative
    colors = ['green' if w > 0 else 'red' for w in sorted_weights]

    bars = ax.bar(range(len(sorted_weights)), sorted_weights, color=colors, alpha=0.7)

    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, sorted_weights)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{weight*100:.1f}%',
            ha='center',
            va='bottom' if height > 0 else 'top',
            fontsize=9
        )

    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    plt.tight_layout()
    return fig


def plot_return_attribution(
    bl_model,
    asset_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot return attribution showing prior, views, and posterior.

    Parameters:
    -----------
    bl_model : BlackLittermanModel
        Fitted Black-Litterman model
    asset_names : List[str], optional
        Names of assets
    figsize : Tuple[int, int]
        Figure size

    Returns:
    --------
    plt.Figure : Matplotlib figure object
    """
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(bl_model.n_assets)]

    attribution = bl_model.get_attribution()

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(asset_names))
    width = 0.25

    ax.bar(
        x - width,
        attribution['prior'],
        width,
        label='Prior (Equilibrium)',
        alpha=0.8,
        color='blue'
    )
    ax.bar(
        x,
        attribution['views'],
        width,
        label='View Impact',
        alpha=0.8,
        color='orange'
    )
    ax.bar(
        x + width,
        attribution['posterior'],
        width,
        label='Posterior',
        alpha=0.8,
        color='green'
    )

    ax.set_xlabel('Assets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return', fontsize=12, fontweight='bold')
    ax.set_title('Return Attribution Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(asset_names, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    cov_matrix: np.ndarray,
    asset_names: Optional[List[str]] = None,
    title: str = "Correlation Matrix",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot correlation matrix as a heatmap.

    Parameters:
    -----------
    cov_matrix : np.ndarray
        Covariance matrix
    asset_names : List[str], optional
        Names of assets
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns:
    --------
    plt.Figure : Matplotlib figure object
    """
    # Convert covariance to correlation
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(len(cov_matrix))]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=asset_names,
        yticklabels=asset_names,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_weight_evolution(
    frontier_weights: np.ndarray,
    frontier_returns: np.ndarray,
    asset_names: Optional[List[str]] = None,
    title: str = "Portfolio Weights Along Efficient Frontier",
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot how portfolio weights change along the efficient frontier.

    Parameters:
    -----------
    frontier_weights : np.ndarray
        Array of weights along frontier (n_points x n_assets)
    frontier_returns : np.ndarray
        Array of returns along frontier
    asset_names : List[str], optional
        Names of assets
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns:
    --------
    plt.Figure : Matplotlib figure object
    """
    n_assets = frontier_weights.shape[1]

    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(n_assets)]

    fig, ax = plt.subplots(figsize=figsize)

    # Create stacked area plot
    ax.stackplot(
        frontier_returns * 100,  # Convert to percentage
        frontier_weights.T * 100,  # Convert to percentage
        labels=asset_names,
        alpha=0.7
    )

    ax.set_xlabel('Expected Return (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Portfolio Weight (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return fig


def plot_risk_return_scatter(
    returns: np.ndarray,
    volatilities: np.ndarray,
    sharpe_ratios: np.ndarray,
    asset_names: Optional[List[str]] = None,
    title: str = "Risk-Return Analysis",
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot risk-return scatter with color-coded Sharpe ratios.

    Parameters:
    -----------
    returns : np.ndarray
        Expected returns
    volatilities : np.ndarray
        Volatilities (standard deviations)
    sharpe_ratios : np.ndarray
        Sharpe ratios
    asset_names : List[str], optional
        Names of assets
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns:
    --------
    plt.Figure : Matplotlib figure object
    """
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(len(returns))]

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        volatilities * 100,
        returns * 100,
        c=sharpe_ratios,
        s=200,
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )

    # Add labels
    for i, name in enumerate(asset_names):
        ax.annotate(
            name,
            (volatilities[i] * 100, returns[i] * 100),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10
        )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=11, fontweight='bold')

    ax.set_xlabel('Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_views_impact(
    prior_returns: np.ndarray,
    posterior_returns: np.ndarray,
    asset_names: Optional[List[str]] = None,
    title: str = "Impact of Views on Expected Returns",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot before/after comparison showing impact of views.

    Parameters:
    -----------
    prior_returns : np.ndarray
        Prior (equilibrium) expected returns
    posterior_returns : np.ndarray
        Posterior expected returns
    asset_names : List[str], optional
        Names of assets
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size

    Returns:
    --------
    plt.Figure : Matplotlib figure object
    """
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(len(prior_returns))]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(asset_names))
    width = 0.35

    ax.bar(
        x - width/2,
        prior_returns * 100,
        width,
        label='Prior (Equilibrium)',
        alpha=0.8,
        color='skyblue'
    )
    ax.bar(
        x + width/2,
        posterior_returns * 100,
        width,
        label='Posterior (After Views)',
        alpha=0.8,
        color='coral'
    )

    # Add arrows showing changes
    for i in range(len(asset_names)):
        if abs(posterior_returns[i] - prior_returns[i]) > 0.001:
            ax.annotate(
                '',
                xy=(i + width/2, posterior_returns[i] * 100),
                xytext=(i - width/2, prior_returns[i] * 100),
                arrowprops=dict(
                    arrowstyle='->',
                    color='green' if posterior_returns[i] > prior_returns[i] else 'red',
                    lw=2,
                    alpha=0.6
                )
            )

    ax.set_xlabel('Assets', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(asset_names, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    return fig


def create_portfolio_dashboard(
    bl_model,
    optimal_weights: np.ndarray,
    asset_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:
    """
    Create a comprehensive dashboard with multiple subplots.

    Parameters:
    -----------
    bl_model : BlackLittermanModel
        Fitted Black-Litterman model
    optimal_weights : np.ndarray
        Optimal portfolio weights
    asset_names : List[str], optional
        Names of assets
    figsize : Tuple[int, int]
        Figure size

    Returns:
    --------
    plt.Figure : Matplotlib figure object
    """
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(bl_model.n_assets)]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Portfolio weights
    ax1 = fig.add_subplot(gs[0, 0])
    sorted_indices = np.argsort(optimal_weights)[::-1]
    sorted_weights = optimal_weights[sorted_indices]
    sorted_names = [asset_names[i] for i in sorted_indices]
    colors = ['green' if w > 0 else 'red' for w in sorted_weights]
    ax1.bar(range(len(sorted_weights)), sorted_weights * 100, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(sorted_names)))
    ax1.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax1.set_ylabel('Weight (%)')
    ax1.set_title('Portfolio Weights', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Return attribution
    ax2 = fig.add_subplot(gs[0, 1])
    attribution = bl_model.get_attribution()
    x = np.arange(len(asset_names))
    width = 0.25
    ax2.bar(x - width, attribution['prior'] * 100, width, label='Prior', alpha=0.8)
    ax2.bar(x, attribution['views'] * 100, width, label='Views', alpha=0.8)
    ax2.bar(x + width, attribution['posterior'] * 100, width, label='Posterior', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(asset_names, rotation=45, ha='right')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Return Attribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Correlation matrix
    ax3 = fig.add_subplot(gs[1, :])
    std_devs = np.sqrt(np.diag(bl_model.cov_matrix))
    corr_matrix = bl_model.cov_matrix / np.outer(std_devs, std_devs)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=asset_names,
        yticklabels=asset_names,
        ax=ax3,
        cbar_kws={'label': 'Correlation'}
    )
    ax3.set_title('Correlation Matrix', fontweight='bold')

    # 4. Portfolio metrics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    metrics = bl_model.calculate_portfolio_metrics(optimal_weights)

    metrics_text = f"""
    Portfolio Metrics:

    Expected Return:        {metrics['expected_return']*100:.2f}%
    Volatility:             {metrics['volatility']*100:.2f}%
    Sharpe Ratio:           {metrics['sharpe_ratio']:.2f}
    Effective Assets:       {metrics['effective_assets']:.2f}
    Max Weight:             {metrics['max_weight']*100:.2f}%
    Min Weight:             {metrics['min_weight']*100:.2f}%
    """

    ax4.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Black-Litterman Portfolio Dashboard', fontsize=16, fontweight='bold', y=0.995)

    return fig
