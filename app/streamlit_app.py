"""Interactive Black-Litterman demo.

pip install -e ".[app]"
streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from blacklitterman import (
    BlackLitterman,
    efficient_frontier,
    ledoit_wolf,
    max_sharpe,
    portfolio_performance,
    returns_from_prices,
)

SNAPSHOT = Path(__file__).with_name("snapshot.json")
REPO = "https://github.com/zoniin/Black-Litterman-Model"
GREY, BLUE, ORANGE, AQUA = "#9a9890", "#2a78d6", "#eb6834", "#1baf7a"
DEFAULT_VIEWS = pd.DataFrame(
    [
        {"Type": "Absolute", "Asset": "JPM", "vs.": None, "Return (%)": 14.0, "Confidence (%)": 60},
        {"Type": "Relative", "Asset": "XOM", "vs.": "NVDA", "Return (%)": 5.0, "Confidence (%)": 40},
    ]
)


# ----------------------------------------------------------------------------- data
@st.cache_data
def load_snapshot() -> dict:
    s = json.loads(SNAPSHOT.read_text())
    tickers = s["tickers"]
    return {
        "label": f"Snapshot, 10 US large caps (as of {s['as_of']})",
        "cov": pd.DataFrame(s["cov"], index=tickers, columns=tickers),
        "caps": pd.Series(s["market_caps"]).reindex(tickers),
        "risk_aversion": s["risk_aversion"],
        "risk_free_rate": s["risk_free_rate"],
        "names": s["names"],
        "window": s["window"],
    }


@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance…")
def load_live(tickers: tuple[str, ...], risk_free_rate: float) -> dict:
    from blacklitterman.data import fetch_market_caps, fetch_prices, market_implied_risk_aversion

    prices = fetch_prices([*tickers, "SPY"], period="5y")
    rets = returns_from_prices(prices[list(tickers)])
    return {
        "label": f"Live, {len(tickers)} tickers (to {rets.index[-1]:%Y-%m-%d})",
        "cov": ledoit_wolf(rets),
        "caps": fetch_market_caps(tickers),
        "risk_aversion": round(market_implied_risk_aversion(prices["SPY"], risk_free_rate), 2),
        "risk_free_rate": risk_free_rate,
        "names": {t: t for t in tickers},
        "window": f"{rets.index[0]:%Y-%m-%d} to {rets.index[-1]:%Y-%m-%d}",
    }


# ---------------------------------------------------------------------------- views
def parse_views(table: pd.DataFrame, assets: list[str]) -> tuple[list[dict], list[str]]:
    """Turn the editor's rows into view specs, skipping blank rows and reporting problems."""
    views, problems = [], []
    for i, row in table.reset_index(drop=True).iterrows():
        n = i + 1
        asset, other, kind = row.get("Asset"), row.get("vs."), row.get("Type")
        ret, conf = row.get("Return (%)"), row.get("Confidence (%)")
        if pd.isna(asset) and pd.isna(ret):
            continue  # empty row
        if pd.isna(asset) or asset not in assets:
            problems.append(f"View {n}: pick an asset.")
            continue
        if pd.isna(ret):
            problems.append(f"View {n}: enter a return.")
            continue
        conf = 50.0 if pd.isna(conf) else float(conf)
        if not 0 < conf <= 100:
            problems.append(f"View {n}: confidence must be between 1 and 100%.")
            continue
        if kind == "Relative":
            if pd.isna(other) or other not in assets or other == asset:
                problems.append(f"View {n}: a relative view needs a second, different asset in 'vs.'.")
                continue
            views.append({"long": asset, "short": other, "q": ret / 100, "confidence": conf / 100})
        else:
            views.append({"long": asset, "short": None, "q": ret / 100, "confidence": conf / 100})
    return views, problems


def build_model(data: dict, views: list[dict], tau: float, delta: float, rf: float) -> BlackLitterman:
    bl = BlackLitterman(data["cov"], market_caps=data["caps"], risk_aversion=delta, tau=tau, risk_free_rate=rf)
    for v in views:
        if v["short"] is None:
            bl.add_absolute_view(v["long"], v["q"], confidence=v["confidence"])
        else:
            bl.add_relative_view(v["long"], v["short"], v["q"], confidence=v["confidence"])
    return bl


# --------------------------------------------------------------------------- charts
def _layout(fig: go.Figure, yfmt: str = ".0%", **kw) -> go.Figure:
    settings = {
        "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        "hovermode": "x unified",
        "bargap": 0.25,
        "bargroupgap": 0.08,
        "height": 420,
    }
    fig.update_layout(**{**settings, **kw})
    fig.update_yaxes(tickformat=yfmt, zeroline=True)
    return fig


def returns_chart(bl: BlackLitterman) -> go.Figure:
    assets = bl.assets
    fig = go.Figure()
    fig.add_bar(x=assets, y=bl.prior_returns, name="Prior (equilibrium)", marker_color=GREY, hovertemplate="%{y:.2%}")
    fig.add_bar(
        x=assets,
        y=bl.posterior_returns(),
        name="Posterior (Black-Litterman)",
        marker_color=BLUE,
        hovertemplate="%{y:.2%}",
    )
    absolute = [(bl.assets[int(np.flatnonzero(v.p)[0])], v.q) for v in bl.views if np.count_nonzero(v.p) == 1]
    if absolute:
        fig.add_scatter(
            x=[a for a, _ in absolute],
            y=[q for _, q in absolute],
            mode="markers",
            name="Your absolute view",
            marker={"symbol": "diamond", "size": 12, "color": ORANGE, "line": {"width": 2, "color": "white"}},
            hovertemplate="view %{y:.2%}",
        )
    return _layout(fig, barmode="group")


def weights_chart(market: pd.Series, bl_w: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=market.index, y=market, name="Market cap", marker_color=GREY, hovertemplate="%{y:.1%}")
    fig.add_bar(x=bl_w.index, y=bl_w, name="Black-Litterman max-Sharpe", marker_color=BLUE, hovertemplate="%{y:.1%}")
    return _layout(fig, barmode="group")


def frontier_chart(frontier, mu, cov, portfolios: dict, rf: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=frontier["volatility"],
        y=frontier["return"],
        mode="lines",
        name="Efficient frontier",
        line={"color": BLUE, "width": 3},
        hovertemplate="vol %{x:.1%}<br>return %{y:.1%}<extra></extra>",
    )
    vols = np.sqrt(np.diag(cov.loc[mu.index, mu.index]))
    fig.add_scatter(
        x=vols,
        y=mu,
        mode="markers+text",
        text=list(mu.index),
        textposition="top right",
        name="Assets",
        marker={"color": GREY, "size": 9},
        hovertemplate="%{text}<br>vol %{x:.1%}<br>return %{y:.1%}<extra></extra>",
    )
    styles = {"Black-Litterman": ("star", ORANGE, 18), "Market": ("circle", AQUA, 12)}
    for name, w in portfolios.items():
        p = portfolio_performance(w, mu, cov, rf)
        symbol, color, size = styles[name]
        fig.add_scatter(
            x=[p["volatility"]],
            y=[p["expected_return"]],
            mode="markers",
            name=f"{name} (Sharpe {p['sharpe_ratio']:.2f})",
            marker={"symbol": symbol, "color": color, "size": size, "line": {"width": 2, "color": "white"}},
            hovertemplate=f"{name}<br>vol %{{x:.1%}}<br>return %{{y:.1%}}<extra></extra>",
        )
    fig = _layout(fig, hovermode="closest")
    fig.update_xaxes(tickformat=".0%", title="Volatility", rangemode="tozero")
    fig.update_yaxes(title="Expected return")
    return fig


# ------------------------------------------------------------------------------ app
def main() -> None:
    st.set_page_config(page_title="Black-Litterman playground", page_icon="📈", layout="wide")
    st.title("Black-Litterman playground")
    st.markdown(
        "Start from the portfolio the market already holds, add your own views on returns, and see how far they "
        "move the optimal portfolio. Edit the views table below. "
        f"[Source & write-up]({REPO})"
    )

    # sidebar
    with st.sidebar:
        st.header("Data")
        source = st.radio("Market data", ["Bundled snapshot", "Live (Yahoo Finance)"], label_visibility="collapsed")
        data = load_snapshot()
        if source.startswith("Live"):
            raw = st.text_input("Tickers (comma-separated)", ", ".join(data["cov"].index))
            tickers = tuple(dict.fromkeys(t.strip().upper() for t in raw.split(",") if t.strip()))
            if len(tickers) < 2:
                st.warning("Enter at least two tickers.")
            else:
                try:
                    data = load_live(tickers, 0.04)
                except Exception as exc:  # network / rate limit / bad ticker
                    st.error(f"Live download failed, so the snapshot is used instead. ({exc})")
        st.caption(f"{data['label']}. Covariance: Ledoit-Wolf on daily returns, {data['window']}.")

        st.header("Model")
        delta = st.number_input(
            "Risk aversion δ",
            0.5,
            10.0,
            float(data["risk_aversion"]),
            0.1,
            help="Implied from SPY's excess return over its variance. It sets the equilibrium prior.",
        )
        tau = st.slider(
            "τ (prior uncertainty)",
            0.01,
            0.20,
            0.05,
            0.01,
            help="How uncertain the prior is relative to Σ. Larger τ lets views move returns further.",
        )
        rf = st.number_input("Risk-free rate", 0.0, 0.10, float(data["risk_free_rate"]), 0.005, format="%.3f")

        st.header("Portfolio")
        max_w = st.slider("Max weight per asset", 0.10, 1.00, 0.35, 0.05, format="%.2f")
        shorts = st.checkbox("Allow short positions", help=f"Weights may go down to −{max_w:.0%}.")
    assets = list(data["cov"].index)
    bounds = (-max_w if shorts else 0.0, max_w)

    # views
    st.subheader("Your views")
    st.caption(
        "**Absolute:** *Asset* returns *Return* per year. **Relative:** *Asset* beats *vs.* by *Return* per year. "
        "**Confidence** is how far the view may move the forecast: at 50% it goes halfway from the market's "
        "implied return to yours."
    )
    defaults = DEFAULT_VIEWS[
        DEFAULT_VIEWS["Asset"].isin(assets) & (DEFAULT_VIEWS["vs."].isna() | DEFAULT_VIEWS["vs."].isin(assets))
    ]
    table = st.data_editor(
        defaults,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"views-{','.join(assets)}",
        column_config={
            "Type": st.column_config.SelectboxColumn(
                options=["Absolute", "Relative"], required=True, default="Absolute"
            ),
            "Asset": st.column_config.SelectboxColumn(options=assets, required=True),
            "vs.": st.column_config.SelectboxColumn(options=assets, help="Only for relative views"),
            "Return (%)": st.column_config.NumberColumn(format="%.1f", min_value=-100.0, max_value=200.0),
            "Confidence (%)": st.column_config.NumberColumn(format="%d", min_value=1, max_value=100, default=50),
        },
    )
    views, problems = parse_views(table, assets)
    for p in problems:
        st.warning(p)

    # model
    try:
        bl = build_model(data, views, tau, delta, rf)
        mu, cov = bl.posterior_returns(), bl.posterior_cov()
        w_bl = max_sharpe(mu, cov, rf, bounds)
    except Exception as exc:
        st.error(f"Could not solve this configuration: {exc}")
        st.stop()
    w_mkt = bl.market_weights
    perf_bl, perf_mkt = portfolio_performance(w_bl, mu, cov, rf), portfolio_performance(w_mkt, mu, cov, rf)
    active = (w_bl - w_mkt).to_numpy()
    tracking_error = float(np.sqrt(active @ data["cov"].to_numpy() @ active))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Expected return",
        f"{perf_bl['expected_return']:.1%}",
        f"{perf_bl['expected_return'] - perf_mkt['expected_return']:+.1%} vs market",
    )
    c2.metric(
        "Volatility",
        f"{perf_bl['volatility']:.1%}",
        f"{perf_bl['volatility'] - perf_mkt['volatility']:+.1%} vs market",
        delta_color="inverse",
    )
    c3.metric(
        "Sharpe ratio",
        f"{perf_bl['sharpe_ratio']:.2f}",
        f"{perf_bl['sharpe_ratio'] - perf_mkt['sharpe_ratio']:+.2f} vs market",
    )
    c4.metric("Tracking error vs market", f"{tracking_error:.1%}", help="Ex-ante, using the sample covariance.")
    if not views:
        st.info("With no views, Black-Litterman holds the market portfolio. Add a view above to tilt it.")

    tab_w, tab_r, tab_f, tab_d, tab_m = st.tabs(
        ["Weights", "Expected returns", "Efficient frontier", "View diagnostics", "How it works"]
    )
    with tab_w:
        st.plotly_chart(weights_chart(w_mkt, w_bl), use_container_width=True)
        st.caption(
            f"Long-{'short' if shorts else 'only'} max-Sharpe portfolio on the posterior, each weight between "
            f"{bounds[0]:.0%} and {bounds[1]:.0%}. Assets you have no view on stay close to their market weight."
        )
    with tab_r:
        st.plotly_chart(returns_chart(bl), use_container_width=True)
        st.caption(
            "Correlated assets move too: a view on one stock shifts the forecasts of the stocks it co-moves with."
        )
    with tab_f:
        frontier = efficient_frontier(mu, cov, 40, bounds)
        st.plotly_chart(
            frontier_chart(frontier, mu, cov, {"Black-Litterman": w_bl, "Market": w_mkt}, rf), use_container_width=True
        )
    with tab_d:
        st.markdown("**Per asset**")
        summary = bl.summary().assign(max_sharpe_weight=w_bl)
        summary.columns = [
            "Market weight",
            "Prior return",
            "Posterior return",
            "View impact",
            "Unconstrained BL weight",
            "Max-Sharpe weight",
        ]
        st.dataframe(summary.style.format("{:.2%}"), use_container_width=True)
        if views:
            st.markdown(
                "**Per view.** If *Posterior implied* is far from *View (q)*, the market prior pushed back hard on that view."
            )
            vs = bl.view_summary()
            vs.columns = ["View (q)", "Prior implied", "Posterior implied", "Confidence", "Ω (view variance)"]
            pct = {c: "{:.2%}" for c in vs.columns[:3]}
            st.dataframe(
                vs.style.format({**pct, "Confidence": "{:.0%}", "Ω (view variance)": "{:.5f}"}),
                use_container_width=True,
            )
    with tab_m:
        st.markdown(
            r"""
**1. Equilibrium prior.** If the market-cap portfolio $w_{mkt}$ is optimal for risk aversion $\delta$, expected returns must be
$\Pi = r_f + \delta \Sigma w_{mkt}$. This is *reverse optimisation*: instead of estimating noisy historical means, back them out of prices.

**2. Views.** Each row above is a linear statement $p_k^\top \mu = q_k$ with uncertainty $\Omega_{kk} = \frac{1-c_k}{c_k} p_k \tau\Sigma p_k^\top$,
where $c_k$ is your confidence (Idzorek's method, in Walters' closed form).

**3. Bayesian update.**
$$\mu_{BL} = \Pi + \tau\Sigma P^\top (P\tau\Sigma P^\top + \Omega)^{-1}(Q - P\Pi)$$

**4. Optimise.** The posterior goes into a max-Sharpe optimiser, which is convex through the Cornuejols-Tütüncü transform, with your weight bounds.

Plain mean-variance on historical means swings to extreme portfolios. Black-Litterman moves away from the market only where you
have an opinion, and only as far as your confidence allows. [Full write-up, tests and backtest →]("""
            + REPO
            + ")"
        )


if __name__ == "__main__":
    main()
