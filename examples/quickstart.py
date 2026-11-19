"""End-to-end Black-Litterman run on live Yahoo Finance data.

    pip install -e ".[plot,data]"
    python examples/quickstart.py --save docs/images

The views below are illustrative. They are not investment advice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from blacklitterman import (
    BlackLitterman,
    efficient_frontier,
    ledoit_wolf,
    max_sharpe,
    portfolio_performance,
    returns_from_prices,
)
from blacklitterman.data import fetch_market_caps, fetch_prices, market_implied_risk_aversion
from blacklitterman.plotting import plot_efficient_frontier, plot_returns, plot_weights

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "XOM", "JNJ"]
RISK_FREE = 0.04


def main(save: Path | None) -> None:
    prices = fetch_prices([*TICKERS, "SPY"], period="5y")
    returns = returns_from_prices(prices[TICKERS])
    cov = ledoit_wolf(returns)
    caps = fetch_market_caps(TICKERS)
    delta = market_implied_risk_aversion(prices["SPY"], RISK_FREE)
    print(f"{len(returns)} daily observations, {returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d}")
    print(f"Ledoit-Wolf shrinkage = {cov.attrs['shrinkage']:.3f}, implied risk aversion delta = {delta:.2f}\n")

    bl = BlackLitterman(cov, market_caps=caps, risk_aversion=delta, tau=0.05, risk_free_rate=RISK_FREE)
    bl.add_absolute_view("JPM", 0.14, confidence=0.6)
    bl.add_relative_view("XOM", "NVDA", 0.05, confidence=0.4)
    bl.add_view({"JNJ": 0.5, "GOOGL": 0.5}, 0.12, confidence=0.3, label="½JNJ + ½GOOGL = 12.0%")

    mu, post_cov = bl.posterior_returns(), bl.posterior_cov()
    w_bl = max_sharpe(mu, post_cov, risk_free_rate=RISK_FREE, bounds=(0.0, 0.35))
    w_mkt = bl.market_weights

    pd.options.display.width = 140
    pd.options.display.max_columns = None
    pd.options.display.float_format = "{:.2%}".format
    print(bl.summary().assign(max_sharpe=w_bl), "\n")
    pd.options.display.float_format = "{:.4f}".format
    print(bl.view_summary(), "\n")
    for name, w in {"Market": w_mkt, "BL max-Sharpe": w_bl}.items():
        perf = portfolio_performance(w, mu, post_cov, RISK_FREE)
        print(f"{name:>14}: " + ", ".join(f"{k}={v:.3f}" for k, v in perf.items()))

    frontier = efficient_frontier(mu, post_cov, n_points=60, bounds=(0.0, 0.35))
    plot_returns(bl)
    plot_weights(
        {"Market cap": w_mkt, "Black-Litterman max-Sharpe": w_bl}, title="Portfolio weights: market vs. Black-Litterman"
    )
    plot_efficient_frontier(
        frontier,
        mu,
        post_cov,
        {"BL max-Sharpe": w_bl, "Market": w_mkt},
        RISK_FREE,
        title="Efficient frontier (≤35% per name)",
    )
    figs = [plt.figure(n) for n in plt.get_fignums()]
    if save:
        save.mkdir(parents=True, exist_ok=True)
        for fig, name in zip(figs, ["returns", "weights", "frontier"], strict=True):
            fig.savefig(save / f"{name}.png", dpi=160)
        print(f"\nfigures saved to {save}/")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", type=Path, help="directory to write PNGs instead of showing them")
    main(parser.parse_args().save)
