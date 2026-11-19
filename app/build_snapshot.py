"""Rebuild the demo's bundled market snapshot (``app/snapshot.json``).

    python app/build_snapshot.py

The snapshot holds only derived statistics: an annualised Ledoit-Wolf
covariance, market caps and the implied risk aversion. That lets the hosted
demo start instantly without calling Yahoo Finance, which often rate-limits
cloud hosts.
"""

from __future__ import annotations

import json
from pathlib import Path

from blacklitterman import ledoit_wolf, returns_from_prices
from blacklitterman.data import fetch_market_caps, fetch_prices, market_implied_risk_aversion

TICKERS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "JPM": "JPMorgan",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble",
    "WMT": "Walmart",
}
RISK_FREE = 0.04
OUT = Path(__file__).with_name("snapshot.json")


def main() -> None:
    tickers = list(TICKERS)
    prices = fetch_prices([*tickers, "SPY"], period="5y")
    returns = returns_from_prices(prices[tickers])
    cov = ledoit_wolf(returns)
    snapshot = {
        "as_of": f"{returns.index[-1]:%Y-%m-%d}",
        "window": f"{returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d}",
        "tickers": tickers,
        "names": TICKERS,
        "cov": cov.round(8).values.tolist(),
        "market_caps": fetch_market_caps(tickers).round(0).to_dict(),
        "risk_aversion": round(market_implied_risk_aversion(prices["SPY"], RISK_FREE), 4),
        "risk_free_rate": RISK_FREE,
    }
    OUT.write_text(json.dumps(snapshot, indent=1) + "\n")
    print(f"wrote {OUT} (as of {snapshot['as_of']}, delta={snapshot['risk_aversion']})")


if __name__ == "__main__":
    main()
