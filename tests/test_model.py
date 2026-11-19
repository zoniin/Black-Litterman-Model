import numpy as np
import pandas as pd
import pytest

from blacklitterman import BlackLitterman, market_implied_prior


def make(cov, caps, **kw):
    kw.setdefault("risk_aversion", 2.5)
    kw.setdefault("tau", 0.05)
    return BlackLitterman(cov, market_caps=caps, **kw)


def master_formula(bl):
    """Textbook form: [(tS)^-1 + P'O^-1 P]^-1 [(tS)^-1 Pi + P'O^-1 Q]."""
    ts_inv = np.linalg.inv(bl.tau * bl.cov.to_numpy())
    o_inv = np.linalg.inv(bl.omega)
    M = np.linalg.inv(ts_inv + bl.P.T @ o_inv @ bl.P)
    return M @ (ts_inv @ bl.prior_returns.to_numpy() + bl.P.T @ o_inv @ bl.Q), M


class TestPrior:
    def test_reverse_optimisation(self, cov, caps):
        bl = make(cov, caps, risk_free_rate=0.03)
        w = caps / caps.sum()
        np.testing.assert_allclose(bl.prior_returns, 0.03 + 2.5 * cov.to_numpy() @ w)
        np.testing.assert_allclose(bl.market_weights.sum(), 1.0)

    def test_market_caps_aligned_by_label(self, cov, caps):
        shuffled = caps.iloc[::-1]
        np.testing.assert_allclose(make(cov, shuffled).market_weights, make(cov, caps).market_weights)

    def test_equal_weight_default(self, cov):
        bl = BlackLitterman(cov)
        np.testing.assert_allclose(bl.market_weights, 0.2)

    def test_custom_prior(self, cov):
        prior = pd.Series(0.05, index=cov.index)
        bl = BlackLitterman(cov, prior=prior)
        np.testing.assert_allclose(bl.prior_returns, 0.05)

    def test_helper(self, cov, caps):
        np.testing.assert_allclose(
            market_implied_prior(cov, caps / caps.sum(), 3.0), 3.0 * cov.to_numpy() @ (caps / caps.sum())
        )


class TestPosterior:
    def test_no_views_returns_prior(self, cov, caps):
        bl = make(cov, caps)
        pd.testing.assert_series_equal(bl.posterior_returns(), bl.prior_returns, check_names=False)
        np.testing.assert_allclose(bl.posterior_cov(), cov * (1 + bl.tau))

    def test_no_views_weights_are_scaled_market(self, cov, caps):
        # He & Litterman (1999): w* = w_mkt / (1 + tau)
        bl = make(cov, caps, risk_free_rate=0.02)
        np.testing.assert_allclose(bl.implied_weights(), bl.market_weights / (1 + bl.tau))

    def test_matches_master_formula(self, cov, caps):
        bl = make(cov, caps)
        bl.add_absolute_view("JP", 0.09, confidence=0.4)
        bl.add_relative_view("US", "EU", 0.03, confidence=0.7)
        bl.add_view({"EM": 0.5, "JP": 0.5, "BOND": -1.0}, 0.05)
        mu_ref, M_ref = master_formula(bl)
        np.testing.assert_allclose(bl.posterior_returns(), mu_ref, rtol=1e-10)
        np.testing.assert_allclose(bl.posterior_cov(), cov.to_numpy() + M_ref, rtol=1e-10)

    def test_full_confidence_view_is_exact(self, cov, caps):
        bl = make(cov, caps).add_absolute_view("EM", 0.15, confidence=1.0)
        assert bl.posterior_returns()["EM"] == pytest.approx(0.15, abs=1e-12)

    def test_full_confidence_relative_view_is_exact(self, cov, caps):
        bl = make(cov, caps).add_relative_view("EU", "US", 0.04, confidence=1.0)
        mu = bl.posterior_returns()
        assert mu["EU"] - mu["US"] == pytest.approx(0.04, abs=1e-12)

    @pytest.mark.parametrize("c", [0.1, 0.25, 0.5, 0.8])
    def test_idzorek_confidence_is_fraction_of_full_tilt(self, cov, caps, c):
        """With Walters' closed form, the posterior moves exactly c of the way to a 100% view."""
        prior = make(cov, caps).prior_returns
        full = make(cov, caps).add_absolute_view("JP", 0.12, confidence=1.0).posterior_returns()
        part = make(cov, caps).add_absolute_view("JP", 0.12, confidence=c).posterior_returns()
        np.testing.assert_allclose(part - prior, c * (full - prior), atol=1e-12)

    def test_default_uncertainty_equals_half_confidence(self, cov, caps):
        a = make(cov, caps).add_relative_view("US", "JP", 0.02).posterior_returns()
        b = make(cov, caps).add_relative_view("US", "JP", 0.02, confidence=0.5).posterior_returns()
        np.testing.assert_allclose(a, b)

    def test_explicit_omega(self, cov, caps):
        bl = make(cov, caps).add_absolute_view("US", 0.10, omega=0.02**2)
        assert bl.omega[0, 0] == pytest.approx(0.0004)

    def test_more_confidence_moves_posterior_further(self, cov, caps):
        moves = [
            make(cov, caps).add_absolute_view("JP", 0.15, confidence=c).posterior_returns()["JP"]
            for c in (0.2, 0.5, 0.9)
        ]
        assert moves[0] < moves[1] < moves[2] < 0.15

    def test_weight_tilts_lie_in_span_of_views(self, cov, caps):
        # He & Litterman: w_BL = w_mkt/(1+tau) + P' Lambda
        bl = make(cov, caps)
        bl.add_relative_view("US", "EU", 0.03, confidence=0.6).add_absolute_view("EM", 0.12, confidence=0.3)
        tilt = (bl.implied_weights() - bl.market_weights / (1 + bl.tau)).to_numpy()
        lam, *_ = np.linalg.lstsq(bl.P.T, tilt, rcond=None)
        np.testing.assert_allclose(bl.P.T @ lam, tilt, atol=1e-12)
        # assets not in any view keep the scaled market weight
        for a in ("JP", "BOND"):
            assert bl.implied_weights()[a] == pytest.approx(bl.market_weights[a] / (1 + bl.tau))

    def test_posterior_cov_is_symmetric_psd(self, cov, caps):
        bl = make(cov, caps).add_absolute_view("US", 0.2, confidence=0.9)
        pc = bl.posterior_cov().to_numpy()
        np.testing.assert_allclose(pc, pc.T)
        assert np.linalg.eigvalsh(pc).min() > 0

    def test_numpy_inputs_and_integer_assets(self, cov, caps):
        bl = BlackLitterman(cov.to_numpy(), market_caps=caps.to_numpy())
        bl.add_absolute_view(2, 0.1)
        assert list(bl.posterior_returns().index) == [f"Asset {i}" for i in range(5)]
        labelled = BlackLitterman(cov, market_caps=caps).add_absolute_view("JP", 0.1)
        np.testing.assert_allclose(bl.posterior_returns(), labelled.posterior_returns())


class TestDiagnostics:
    def test_summary(self, cov, caps):
        bl = make(cov, caps).add_absolute_view("EM", 0.12)
        s = bl.summary()
        assert list(s.columns) == ["market_weight", "prior", "posterior", "view_impact", "bl_weight"]
        np.testing.assert_allclose(s["posterior"] - s["prior"], s["view_impact"])

    def test_view_summary(self, cov, caps):
        bl = make(cov, caps).add_absolute_view("EM", 0.12, confidence=1.0).add_relative_view("US", "EU", 0.01)
        vs = bl.view_summary()
        assert len(vs) == 2
        assert vs["posterior_implied"].iloc[0] == pytest.approx(0.12)
        assert vs.index[0] == "EM = 12.0%"

    def test_clear_views(self, cov, caps):
        bl = make(cov, caps).add_absolute_view("EM", 0.12)
        bl.clear_views()
        np.testing.assert_allclose(bl.posterior_returns(), bl.prior_returns)


class TestValidation:
    def test_non_square(self):
        with pytest.raises(ValueError, match="square"):
            BlackLitterman(np.ones((2, 3)))

    def test_asymmetric(self):
        with pytest.raises(ValueError, match="symmetric"):
            BlackLitterman(np.array([[1.0, 0.2], [0.1, 1.0]]))

    def test_not_psd(self):
        with pytest.raises(ValueError, match="semi-definite"):
            BlackLitterman(np.array([[1.0, 2.0], [2.0, 1.0]]))

    def test_bad_tau(self, cov):
        with pytest.raises(ValueError, match="tau"):
            BlackLitterman(cov, tau=0)

    def test_missing_caps(self, cov, caps):
        with pytest.raises(ValueError, match="missing"):
            BlackLitterman(cov, market_caps=caps.drop("JP"))

    def test_negative_caps(self, cov, caps):
        with pytest.raises(ValueError, match="non-negative"):
            BlackLitterman(cov, market_caps=-caps)

    @pytest.mark.parametrize("c", [0, -0.1, 1.5])
    def test_bad_confidence(self, cov, c):
        with pytest.raises(ValueError, match="confidence"):
            BlackLitterman(cov).add_absolute_view("US", 0.1, confidence=c)

    def test_confidence_and_omega(self, cov):
        with pytest.raises(ValueError, match="either"):
            BlackLitterman(cov).add_absolute_view("US", 0.1, confidence=0.5, omega=0.1)

    def test_unknown_asset(self, cov):
        with pytest.raises(KeyError):
            BlackLitterman(cov).add_absolute_view("MARS", 0.1)

    def test_empty_view(self, cov):
        with pytest.raises(ValueError, match="non-zero"):
            BlackLitterman(cov).add_view(np.zeros(5), 0.1)

    def test_wrong_length_view(self, cov):
        with pytest.raises(ValueError, match="length"):
            BlackLitterman(cov).add_view([1, 0], 0.1)
