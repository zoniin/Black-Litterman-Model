import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from blacklitterman import BlackLitterman, efficient_frontier, max_sharpe, plotting


@pytest.fixture
def bl(cov, caps):
    return BlackLitterman(cov, market_caps=caps).add_absolute_view("EM", 0.12, confidence=0.6)


def test_all_plots_render(bl):
    mu, pc = bl.posterior_returns(), bl.posterior_cov()
    w = max_sharpe(mu, pc)
    axes = [
        plotting.plot_returns(bl),
        plotting.plot_weights({"Market": bl.market_weights, "BL": w}),
        plotting.plot_efficient_frontier(efficient_frontier(mu, pc, 15), mu, pc, {"Max Sharpe": w}),
        plotting.plot_correlation(pc),
    ]
    for ax in axes:
        ax.figure.canvas.draw()
    assert "Absolute view" in [t.get_text() for t in axes[0].get_legend().get_texts()]
    plt.close("all")


def test_plot_backtest(cov, caps):
    import numpy as np
    import pandas as pd

    from blacklitterman import backtest, equal_weight

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=120)
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (120, 3)), axis=0), index=idx, columns=list("ABC"))
    res = backtest(prices, {"EW": equal_weight(), "A": lambda ctx: pd.Series({"A": 1.0})}, lookback=20)
    fig = plotting.plot_backtest(res)
    fig.canvas.draw()
    assert len(fig.axes) == 2
    plt.close("all")
