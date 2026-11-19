"""Smoke and behaviour tests for the Streamlit demo (runs offline on the bundled snapshot)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"
ASSETS = ["AAPL", "MSFT", "JPM", "XOM"]


@pytest.fixture(scope="module")
def app_module():
    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_app(**widgets) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    for key, value in widgets.items():
        getattr(at, key)[0].set_value(value)
    return at.run() if widgets else at


def test_default_run_has_no_errors():
    at = run_app()
    assert not at.exception
    assert not at.error
    labels = [m.label for m in at.metric]
    assert labels == ["Expected return", "Volatility", "Sharpe ratio", "Tracking error vs market"]
    assert len(at.tabs) == 5


def test_shorts_and_bounds_rerun_cleanly():
    at = run_app(checkbox=True)
    assert not at.exception and not at.error
    at.slider[1].set_value(0.2).run()  # max weight
    assert not at.exception and not at.error


def test_snapshot_is_valid(app_module):
    data = app_module.load_snapshot()
    cov = data["cov"].to_numpy()
    assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > 0
    assert list(data["caps"].index) == list(data["cov"].index)
    assert 0.5 < data["risk_aversion"] < 10


def test_parse_views(app_module):
    table = pd.DataFrame(
        [
            {"Type": "Absolute", "Asset": "JPM", "vs.": None, "Return (%)": 12.0, "Confidence (%)": 60},
            {"Type": "Relative", "Asset": "XOM", "vs.": "AAPL", "Return (%)": 3.0, "Confidence (%)": None},
            {"Type": "Relative", "Asset": "XOM", "vs.": "XOM", "Return (%)": 3.0, "Confidence (%)": 50},
            {"Type": "Absolute", "Asset": "MSFT", "vs.": None, "Return (%)": None, "Confidence (%)": 50},
            {"Type": "Absolute", "Asset": None, "vs.": None, "Return (%)": None, "Confidence (%)": None},
        ]
    )
    views, problems = app_module.parse_views(table, ASSETS)
    assert views == [
        {"long": "JPM", "short": None, "q": 0.12, "confidence": 0.6},
        {"long": "XOM", "short": "AAPL", "q": 0.03, "confidence": 0.5},
    ]
    assert len(problems) == 2
    assert "View 3" in problems[0] and "View 4" in problems[1]


def test_build_model_matches_library(app_module):
    data = app_module.load_snapshot()
    views = [{"long": "JPM", "short": None, "q": 0.14, "confidence": 0.6}]
    bl = app_module.build_model(data, views, tau=0.05, delta=2.5, rf=0.04)
    assert bl.view_summary()["posterior_implied"].iloc[0] == pytest.approx(
        bl.prior_returns["JPM"] + 0.6 * (0.14 - bl.prior_returns["JPM"]), rel=0.2
    )
    assert len(bl.views) == 1
