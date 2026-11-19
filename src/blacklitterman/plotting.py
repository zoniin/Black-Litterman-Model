"""Matplotlib charts for Black-Litterman results.

Every function draws onto ``ax`` if given (otherwise it makes a new figure) and
returns the ``Axes``. Colours follow one fixed scheme: neutral grey for the
market/prior, then blue, orange and aqua for the other series.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from .model import BlackLitterman
from .risk import cov_to_corr

__all__ = [
    "PALETTE",
    "plot_returns",
    "plot_weights",
    "plot_efficient_frontier",
    "plot_correlation",
]

PALETTE = {
    "neutral": "#9a9890",
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e4e3df",
    "surface": "#fcfcfb",
}
SERIES = [PALETTE["neutral"], PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"]]


def _style(ax, title: str | None = None, grid_axis: str = "both", title_pad: float = 30) -> None:
    ax.set_facecolor(PALETTE["surface"])
    ax.figure.set_facecolor(PALETTE["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9, length=0)
    ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=PALETTE["text"], pad=title_pad)


def _legend(ax, above: bool = False, **kw):
    if above:
        kw = {"loc": "lower left", "bbox_to_anchor": (0, 1.0), "borderaxespad": 0.2, **kw}
    return ax.legend(frameon=False, fontsize=9, labelcolor=PALETTE["text"], handlelength=1.2, **kw)


def _ax(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def _grouped_bars(ax, frame: pd.DataFrame, colors) -> None:
    n_groups, n_series = frame.shape
    x = np.arange(n_groups)
    width = 0.8 / n_series
    for j, (name, col) in enumerate(frame.items()):
        ax.bar(
            x + (j - (n_series - 1) / 2) * width,
            col.to_numpy(),
            width=width * 0.92,
            color=colors[j % len(colors)],
            label=name,
            edgecolor=PALETTE["surface"],
            linewidth=1,
        )
    ax.set_xticks(x, [str(i) for i in frame.index])
    ax.axhline(0, color=PALETTE["muted"], linewidth=0.8)


def plot_returns(bl: BlackLitterman, ax=None, title: str = "Expected returns: prior vs. posterior"):
    """Prior (equilibrium) and posterior returns per asset, with absolute views marked."""
    ax = _ax(ax, (9, 4.5))
    frame = pd.DataFrame(
        {"Prior (equilibrium)": bl.prior_returns, "Posterior (Black-Litterman)": bl.posterior_returns()}
    )
    _grouped_bars(ax, frame, SERIES)
    # mark absolute views on top of their asset
    x = np.arange(bl.n_assets)
    shown = False
    for v in bl.views:
        nz = np.flatnonzero(v.p)
        if len(nz) == 1 and v.p[nz[0]] == 1:
            ax.scatter(
                x[nz[0]] + 0.2,
                v.q,
                marker="D",
                s=48,
                color=PALETTE["orange"],
                edgecolor=PALETTE["surface"],
                linewidth=1.5,
                zorder=5,
                label=None if shown else "Absolute view",
            )
            shown = True
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _style(ax, title, grid_axis="y")
    _legend(ax, above=True, ncols=3)
    return ax


def plot_weights(portfolios: Mapping[str, pd.Series], ax=None, title: str = "Portfolio weights"):
    """Side-by-side weights of several portfolios, e.g. market vs. Black-Litterman."""
    ax = _ax(ax, (9, 4.5))
    frame = pd.DataFrame(dict(portfolios)).fillna(0.0)
    _grouped_bars(ax, frame, SERIES)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _style(ax, title, grid_axis="y")
    _legend(ax, above=True, ncols=len(frame.columns))
    return ax


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    mu: pd.Series,
    cov: pd.DataFrame,
    portfolios: Mapping[str, pd.Series] | None = None,
    risk_free_rate: float = 0.0,
    ax=None,
    title: str = "Efficient frontier (posterior)",
):
    """Efficient frontier from :func:`~blacklitterman.optimize.efficient_frontier`,
    with individual assets and any named portfolios marked."""
    ax = _ax(ax, (8, 5.5))
    ax.plot(frontier["volatility"], frontier["return"], color=PALETTE["blue"], linewidth=2, label="Efficient frontier")

    vols = np.sqrt(np.diag(cov.loc[mu.index, mu.index]))
    ax.scatter(vols, mu, s=36, color=PALETTE["neutral"], zorder=3, label="Individual assets")
    for name, xv, yv in zip(mu.index, vols, mu, strict=False):
        ax.annotate(str(name), (xv, yv), xytext=(6, 2), textcoords="offset points", fontsize=8, color=PALETTE["muted"])

    markers = ["*", "o", "s", "^"]
    colors = [PALETTE["orange"], PALETTE["aqua"], PALETTE["text"], PALETTE["blue"]]
    for i, (name, w) in enumerate((portfolios or {}).items()):
        w = w.reindex(mu.index).fillna(0.0).to_numpy()
        r = float(mu.to_numpy() @ w)
        s = float(np.sqrt(w @ cov.loc[mu.index, mu.index].to_numpy() @ w))
        sr = (r - risk_free_rate) / s
        ax.scatter(
            s,
            r,
            marker=markers[i % 4],
            s=220 if markers[i % 4] == "*" else 70,
            color=colors[i % 4],
            edgecolor=PALETTE["surface"],
            linewidth=1.5,
            zorder=6,
            label=f"{name} (Sharpe {sr:.2f})",
        )
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Volatility", color=PALETTE["muted"], fontsize=9)
    ax.set_ylabel("Expected return", color=PALETTE["muted"], fontsize=9)
    _style(ax, title, title_pad=12)
    _legend(ax, loc="lower right")
    return ax


def plot_correlation(cov: pd.DataFrame, ax=None, title: str = "Correlation"):
    """Correlation heatmap, on a diverging scale centred on zero."""
    ax = _ax(ax, (6.5, 5.5))
    corr = cov_to_corr(cov)
    cmap = LinearSegmentedColormap.from_list("bl_div", [PALETTE["blue"], "#f1f0ed", PALETTE["orange"]])
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    labels = [str(c) for c in corr.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for (i, j), v in np.ndenumerate(corr.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=PALETTE["text"])
    _style(ax, title)
    ax.grid(False)
    ax.figure.colorbar(im, ax=ax, shrink=0.8).outline.set_visible(False)
    return ax
