"""Walk-forward backtest: does Black-Litterman with systematic momentum views help?

    pip install -e ".[plot,data]"
    python examples/backtest_momentum.py --save docs/images

Universe: 20 US large caps from across sectors that have traded since 2008.
Choosing today's survivors adds survivorship bias. That bias is shared by every
strategy, so the comparison between strategies is fairer than the absolute
numbers. Historical market caps use a constant-share-count proxy (see
``blacklitterman.data.market_cap_history``).

Every month each strategy rebalances, using only data up to that day's close.
It trades at that close, pays 10 bp of one-way turnover, and drifts until the
next month end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from blacklitterman import (
    backtest,
    black_litterman_strategy,
    equal_weight,
    market_cap_weight,
    mean_variance,
    momentum_views,
)
from blacklitterman.data import fetch_market_caps, fetch_prices, market_cap_history
from blacklitterman.plotting import PALETTE, plot_backtest

UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "INTC", "CSCO",  # tech
    "JPM", "BAC", "WFC",                              # financials
    "XOM", "CVX",                                     # energy
    "JNJ", "PFE", "MRK",                              # health care
    "PG", "KO", "WMT", "MCD", "HD",                   # consumer
    "CAT",                                            # industrials
]  # fmt: skip
START = "2008-01-01"
RF = 0.02
BOUNDS = (0.0, 0.30)


def main(save: Path | None) -> None:
    prices = fetch_prices(UNIVERSE, start=START)
    price_only = fetch_prices(UNIVERSE, start=START, adjust_dividends=False).reindex(prices.index).ffill()
    caps = market_cap_history(price_only, fetch_market_caps(UNIVERSE))

    strategies = {
        "Market cap": market_cap_weight(),
        "Equal weight": equal_weight(),
        "Mean-variance (historical μ)": mean_variance(BOUNDS),
        "Black-Litterman + momentum": black_litterman_strategy(momentum_views(confidence=0.25), bounds=BOUNDS),
    }
    # the naive calibration, kept for comparison: q = half the past spread
    naive = {
        "BL + momentum, q = ½ past spread": black_litterman_strategy(
            momentum_views(premium=None, scale=0.5, confidence=0.25), bounds=BOUNDS
        )
    }
    res = backtest(prices, strategies, market_caps=caps, rebalance="M", lookback=504, cost_bps=10, risk_free_rate=RF)

    pd.options.display.width = 140
    print(f"{res.returns.index[0]:%Y-%m-%d} to {res.returns.index[-1]:%Y-%m-%d}, monthly rebalance, 10 bp costs\n")
    print(res.stats.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nvs. market cap:")
    print(res.relative_to("Market cap").to_string(float_format=lambda x: f"{x:.3f}"))

    # sensitivity: how much should we trust the momentum signal?
    sweep = {
        f"confidence {c:.0%}": black_litterman_strategy(momentum_views(confidence=c), bounds=BOUNDS)
        for c in (0.05, 0.10, 0.25, 0.50, 0.75)
    }
    sens = backtest(
        prices,
        {"Market cap": market_cap_weight(), **sweep, **naive},
        market_caps=caps,
        lookback=504,
        cost_bps=10,
        risk_free_rate=RF,
    )
    print("\nBlack-Litterman momentum (q = 5%), by view confidence, plus the naive calibration:\n")
    sens_tbl = sens.stats.join(sens.relative_to("Market cap") if "Market cap" in sens.returns else pd.DataFrame())
    print(sens_tbl.to_string(float_format=lambda x: f"{x:.3f}"))

    fig = plot_backtest(
        res,
        colors={
            "Market cap": PALETTE["neutral"],
            "Equal weight": PALETTE["aqua"],
            "Mean-variance (historical μ)": PALETTE["orange"],
            "Black-Litterman + momentum": PALETTE["blue"],
        },
        title="20 US large caps, monthly rebalance, 10 bp costs",
    )
    # confidence acts as a risk budget: tracking error vs. market rises smoothly with it
    conf = [float(n.split()[1].rstrip("%")) / 100 for n in sweep]
    te = sens_tbl.loc[list(sweep), "tracking_error"].to_numpy()
    fig2, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    ax.plot(conf, te, color=PALETTE["blue"], linewidth=2, marker="o", markersize=6)
    for c, t, to in zip(conf, te, sens_tbl.loc[list(sweep), "turnover_pa"], strict=True):
        ax.annotate(f"turnover {to:.1f}×/yr", (c, t), xytext=(6, -12), textcoords="offset points", fontsize=8)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Momentum view confidence")
    ax.set_ylabel("Tracking error vs. market")
    ax.set_xlim(0, 0.85)
    ax.set_ylim(0, None)
    ax.set_title("View confidence is a risk budget", loc="left", fontweight="bold")
    ax.grid(color=PALETTE["grid"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if save:
        save.mkdir(parents=True, exist_ok=True)
        fig.savefig(save / "backtest.png", dpi=160)
        fig2.savefig(save / "confidence_tracking_error.png", dpi=160, facecolor=PALETTE["surface"])
        res.stats.join(res.relative_to("Market cap")).to_csv(save.parent / "backtest_stats.csv", float_format="%.4f")
        sens_tbl.to_csv(save.parent / "backtest_confidence_sweep.csv", float_format="%.4f")
        print(f"\nsaved figure and stats under {save.parent}/")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", type=Path, help="directory to write the PNG to instead of showing it")
    main(parser.parse_args().save)
