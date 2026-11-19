import numpy as np
import pandas as pd
import pytest

ASSETS = ["US", "EU", "JP", "EM", "BOND"]


@pytest.fixture
def cov() -> pd.DataFrame:
    vols = np.array([0.16, 0.18, 0.17, 0.24, 0.06])
    corr = np.array(
        [
            [1.00, 0.75, 0.55, 0.65, 0.10],
            [0.75, 1.00, 0.55, 0.65, 0.15],
            [0.55, 0.55, 1.00, 0.55, 0.10],
            [0.65, 0.65, 0.55, 1.00, 0.05],
            [0.10, 0.15, 0.10, 0.05, 1.00],
        ]
    )
    return pd.DataFrame(np.outer(vols, vols) * corr, index=ASSETS, columns=ASSETS)


@pytest.fixture
def caps() -> pd.Series:
    return pd.Series([45.0, 15.0, 6.0, 10.0, 24.0], index=ASSETS)
