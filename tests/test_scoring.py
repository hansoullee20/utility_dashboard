"""Tests for anomaly scoring pipeline — P0-6.

Covers:
  - Single-file weight redistribution (P0-1)
  - Edge cases in create_change_columns
  - Z-score min_valid behavior
  - HTML escaping helper (P0-2)
  - build_anomaly_df with missing sheets
  - Risk level thresholds
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest


# ── HTML escaping ────────────────────────────────────────────────────────────

class TestEsc:
    def test_basic_html_escape(self):
        from utils import esc
        assert esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"

    def test_ampersand(self):
        from utils import esc
        assert esc("A&B") == "A&amp;B"

    def test_none(self):
        from utils import esc
        assert esc(None) == ""

    def test_numeric_passthrough(self):
        from utils import esc
        assert esc(42) == "42"

    def test_korean_brand_name(self):
        from utils import esc
        assert esc("깨비옥") == "깨비옥"


# ── create_change_columns edge cases ─────────────────────────────────────────

class TestChangeColumns:
    def _make_df(self, prev, curr):
        return pd.DataFrame({
            "water_previous": [prev],
            "water_current": [curr],
            "water_usage_m3": [curr],
        })

    def test_zero_to_zero(self):
        from features import create_change_columns
        df = create_change_columns(self._make_df(0, 0))
        assert pd.isna(df["water_pct"].iloc[0])
        assert df["water_change"].iloc[0] == 0.0

    def test_zero_to_nonzero(self):
        from features import create_change_columns
        df = create_change_columns(self._make_df(0, 5.0))
        assert df["water_change"].iloc[0] == 5.0
        assert pd.isna(df["water_pct"].iloc[0])  # infinite growth → NaN

    def test_nonzero_to_zero(self):
        from features import create_change_columns
        df = create_change_columns(self._make_df(5.0, 0))
        assert df["water_change"].iloc[0] == -5.0
        assert df["water_pct"].iloc[0] == -100.0

    def test_nan_to_value(self):
        from features import create_change_columns
        df = create_change_columns(self._make_df(np.nan, 5.0))
        assert pd.isna(df["water_change"].iloc[0])
        assert pd.isna(df["water_pct"].iloc[0])

    def test_value_to_nan(self):
        from features import create_change_columns
        df = create_change_columns(self._make_df(5.0, np.nan))
        assert pd.isna(df["water_change"].iloc[0])

    def test_normal_change(self):
        from features import create_change_columns
        df = create_change_columns(self._make_df(100, 150))
        assert df["water_change"].iloc[0] == 50.0
        assert df["water_pct"].iloc[0] == 50.0


# ── Z-score ──────────────────────────────────────────────────────────────────

class TestZscore:
    def test_min_valid_default(self):
        from utils import zscore
        s = pd.Series([1.0, 2.0])  # n=2, below min_valid=3
        result = zscore(s)
        assert result.isna().all()

    def test_min_valid_met(self):
        from utils import zscore
        s = pd.Series([1.0, 2.0, 3.0])  # n=3, exactly min_valid
        result = zscore(s)
        assert not result.isna().all()

    def test_all_same_values(self):
        from utils import zscore
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = zscore(s)
        assert (result == 0.0).all()

    def test_nan_ignored(self):
        from utils import zscore
        s = pd.Series([1.0, np.nan, 2.0, 3.0])  # 3 valid
        result = zscore(s)
        assert pd.isna(result.iloc[1])  # NaN stays NaN
        assert result.iloc[0] < 0  # lowest value → negative z


# ── MAD z-score ──────────────────────────────────────────────────────────────

class TestMadZscore:
    def test_min_valid(self):
        from utils import mad_zscore
        assert mad_zscore(pd.Series([1.0, 2.0])).isna().all()

    def test_constant_series(self):
        from utils import mad_zscore
        assert (mad_zscore(pd.Series([5.0] * 5)) == 0.0).all()

    def test_normal_data_consistent_with_classical(self):
        """On roughly normal data, MAD-z and classical z should rank the same."""
        from utils import mad_zscore, zscore
        rng = np.random.RandomState(0)
        s = pd.Series(rng.normal(100, 15, 50))
        z_classic = zscore(s)
        z_mad = mad_zscore(s)
        # Rankings should be identical
        assert list(z_classic.rank()) == list(z_mad.rank())

    def test_skew_resistance(self):
        """A single huge outlier should NOT drag other tenants' scores to 0."""
        from utils import mad_zscore, zscore
        # 20 similar unit costs around 5000원/m³, one outlier at 50000
        s = pd.Series([4800, 5100, 4950, 5200, 4900, 5050, 5150, 4850,
                       5000, 5100, 4900, 5000, 5150, 4850, 5050, 4950,
                       5100, 4950, 5000, 5100, 50000])
        z_classic = zscore(s)
        z_mad = mad_zscore(s)
        # Index 0 is a tenant slightly below median (4800 vs 5000).
        # Classical z gets distorted by the 50000 outlier dragging
        # the mean upward; the normal tenants' scores collapse toward
        # 0. MAD z preserves the signal.
        assert abs(z_mad.iloc[0]) > abs(z_classic.iloc[0])
        # Outlier itself should still be flagged by both, very high on MAD.
        assert z_mad.iloc[-1] > 10  # MAD picks up the outlier sharply


# ── Won formatting ───────────────────────────────────────────────────────────

class TestFmtWon:
    def test_billions(self):
        from utils import fmt_won
        assert "억" in fmt_won(150_000_000)

    def test_ten_thousands(self):
        from utils import fmt_won
        assert "만" in fmt_won(450_000)

    def test_thousands(self):
        from utils import fmt_won
        assert "천" in fmt_won(3_500)

    def test_small(self):
        from utils import fmt_won
        assert "원" in fmt_won(500)

    def test_signed(self):
        from utils import fmt_won
        result = fmt_won(450_000, signed=True)
        assert result.startswith("+")

    def test_negative(self):
        from utils import fmt_won
        result = fmt_won(-450_000, signed=True)
        assert "-" in result


# ── Anomaly scoring — single file mode (P0-1) ───────────────────────────────

def _make_meter_df(n=20, with_mom=False):
    """Create a minimal meter DataFrame for testing."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "brand": [f"brand_{i}" for i in range(n)],
        "building": ["A"] * (n // 2) + ["B"] * (n - n // 2),
        "floor": ["1F"] * n,
        "size_m2": rng.uniform(30, 200, n).round(1),
        "size_py": rng.uniform(10, 60, n).round(1),
        "water_current": rng.uniform(0, 50, n).round(1),
        "hwater_current": rng.uniform(0, 30, n).round(1),
        "elect_current": rng.uniform(0, 500, n).round(1),
        "heat_current": rng.uniform(0, 20, n).round(1),
    })
    if with_mom:
        for pfx in ["water", "hwater", "elect", "heat"]:
            df[f"{pfx}_previous"] = (df[f"{pfx}_current"] * rng.uniform(0.5, 1.5, n)).round(1)
            df[f"{pfx}_change"] = df[f"{pfx}_current"] - df[f"{pfx}_previous"]
            denom = df[f"{pfx}_previous"].replace(0, np.nan)
            df[f"{pfx}_pct"] = ((df[f"{pfx}_change"] / denom) * 100).round(2)
    return df


class TestAnomalySingleFile:
    """When no MoM data exists, spike+consumption weights should redistribute."""

    def test_spike_weight_excluded_without_mom(self):
        from anomaly_features import build_anomaly_df
        df = _make_meter_df(20, with_mom=False)
        result = build_anomaly_df(df)
        # Spike and consumption should all be 0 (no change data)
        assert (result["spike_score"] == 0).all()
        # Without any supplementary sheets and no MoM, only consistency
        # contributes — brands with nonzero usage get 0.0 (correct).
        # The key property: spike weight is NOT dragging down scores that
        # DO have signal in other dimensions (tested in reachable test below).

    def test_risk_levels_reachable_without_mom(self):
        """With redistributed weights, 주의/위험 should be reachable."""
        from anomaly_features import build_anomaly_df
        df = _make_meter_df(20, with_mom=False)
        # Force some brands to have high consistency_score (all zeros)
        df.loc[0, ["water_current", "hwater_current", "elect_current", "heat_current"]] = 0
        df.loc[1, ["water_current", "hwater_current", "elect_current", "heat_current"]] = 0
        result = build_anomaly_df(df)
        top = result.iloc[0]
        # All-zero brand should get consistency_score=1.0
        assert top["consistency_score"] == 1.0
        # With weight redistribution, composite should be significant
        assert top["composite_score"] >= 0.20

    def test_spike_weight_included_with_mom(self):
        from anomaly_features import build_anomaly_df
        df = _make_meter_df(20, with_mom=True)
        result = build_anomaly_df(df)
        # Some brands should have non-zero spike scores
        assert result["spike_score"].gt(0).any()

    def test_new_tenant_per_utility_spike_suppressed(self):
        """First-appearance for a single utility must not leak a spike floor.

        Brand 0 is established except on water (previous=NaN). The water
        spike contribution must be exactly 0, but other utilities still score.
        """
        from anomaly_features import build_anomaly_df
        df = _make_meter_df(20, with_mom=True)
        df.loc[0, "water_previous"] = np.nan
        df.loc[0, "water_change"] = np.nan
        df.loc[0, "water_pct"] = np.nan
        result = build_anomaly_df(df)
        brand0 = result[result["brand"] == "brand_0"].iloc[0]
        assert brand0["water_is_new"] == True
        assert brand0["water_spike_pct"] != brand0["water_spike_pct"] or brand0.get("water_spike_pct") == 0  # NaN or 0
        # Row is not fully new — only one utility is new
        assert brand0["is_new_tenant"] == False

    def test_new_tenant_row_flag_when_all_utilities_new(self):
        """All-new brand gets is_new_tenant=True and zero spike score."""
        from anomaly_features import build_anomaly_df
        df = _make_meter_df(20, with_mom=True)
        for pfx in ["water", "hwater", "elect", "heat"]:
            df.loc[0, f"{pfx}_previous"] = np.nan
            df.loc[0, f"{pfx}_change"] = np.nan
            df.loc[0, f"{pfx}_pct"] = np.nan
        result = build_anomaly_df(df)
        brand0 = result[result["brand"] == "brand_0"].iloc[0]
        assert brand0["is_new_tenant"] == True
        assert brand0["spike_score"] == 0.0

    def test_spike_dim_available_when_pct_columns_present(self):
        """Regression: availability must be structural (columns exist),
        not signal-based (some brand scored > 0).

        A calm month with valid MoM data but no actual spikes must still
        count the spike dimension in the composite weighting — otherwise
        the entire 40% spike weight silently redistributes to other dims.
        """
        from anomaly_features import build_anomaly_df
        df = _make_meter_df(20, with_mom=True)
        # Force a calm month: every brand's previous == current → 0% change
        for pfx in ["water", "hwater", "elect", "heat"]:
            df[f"{pfx}_previous"] = df[f"{pfx}_current"]
            df[f"{pfx}_change"] = 0.0
            df[f"{pfx}_pct"] = 0.0
        result = build_anomaly_df(df)
        # Spike must remain in the available dims even though no brand spiked
        assert "spike_score" in result.attrs.get("_available_dims", [])
        assert "spike_score" not in result.attrs.get("_excluded_dims", [])


class TestAnomalyMissingSheets:
    """Weight redistribution when optional sheets are absent."""

    def test_no_billing(self):
        from anomaly_features import build_anomaly_df
        df = _make_meter_df(20, with_mom=True)
        result = build_anomaly_df(df, billing_df=None)
        # cost_score should not contribute
        assert "cost_score" in result.columns
        # composite should still work
        assert result["composite_score"].notna().all()

    def test_no_elec(self):
        from anomaly_features import build_anomaly_df
        df = _make_meter_df(20, with_mom=True)
        result = build_anomaly_df(df, elec_df=None)
        assert result["composite_score"].notna().all()

    def test_all_missing(self):
        """Single file, no supplementary sheets — should still produce scores."""
        from anomaly_features import build_anomaly_df
        df = _make_meter_df(20, with_mom=False)
        result = build_anomaly_df(df)
        assert not result.empty
        assert result["composite_score"].notna().all()
        assert "risk_level" in result.columns


class TestRiskThresholds:
    def test_threshold_boundaries(self):
        from anomaly_features import build_anomaly_df
        from utils import RISK_DANGER, RISK_CAUTION, RISK_OBSERVE, RISK_NORMAL
        df = _make_meter_df(20, with_mom=True)
        result = build_anomaly_df(df)
        for _, row in result.iterrows():
            s = row["composite_score"]
            r = row["risk_level"]
            if s >= 0.65:
                assert r == RISK_DANGER
            elif s >= 0.40:
                assert r == RISK_CAUTION
            elif s >= 0.20:
                assert r == RISK_OBSERVE
            else:
                assert r == RISK_NORMAL


# ── Join coverage ────────────────────────────────────────────────────────────

class TestJoinCoverage:
    """build_unit_costs should report unmatched brands."""

    def test_unmatched_brands_tracked(self):
        from cross_features import build_unit_costs
        # meter_df has brand_X, billing_df does not
        meter = pd.DataFrame({
            "brand": ["A", "B", "C"],
            "building": ["X", "X", "X"],
            "size_m2": [100, 100, 100],
            "water_usage_m3": [10, 20, 30],
        })
        billing = pd.DataFrame({
            "brand": ["A", "B"],  # C is missing
            "building": ["X", "X"],
            "water_total": [5.0, 10.0],
            "elect_total": [3.0, 6.0],
            "heat_total": [1.0, 2.0],
            "hotwater_excl": [0.5, 1.0],
            "hotwater_comm": [0.1, 0.2],
            "total": [9.6, 19.2],
        })
        result = build_unit_costs(meter, billing)
        # C should still be in result (left join) but with NaN costs
        assert len(result) == 3
        assert result.attrs.get("_unmatched_brands") is not None
        assert ("C", "X") in result.attrs["_unmatched_brands"]
        # C's cost columns should be NaN
        c_row = result[result["brand"] == "C"].iloc[0]
        assert pd.isna(c_row.get("water_unit_cost", np.nan))

    def test_full_match_no_unmatched(self):
        from cross_features import build_unit_costs
        meter = pd.DataFrame({
            "brand": ["A", "B"],
            "building": ["X", "X"],
            "size_m2": [100, 100],
            "water_usage_m3": [10, 20],
        })
        billing = pd.DataFrame({
            "brand": ["A", "B"],
            "building": ["X", "X"],
            "water_total": [5.0, 10.0],
            "elect_total": [3.0, 6.0],
            "heat_total": [1.0, 2.0],
            "hotwater_excl": [0.5, 1.0],
            "hotwater_comm": [0.1, 0.2],
            "total": [9.6, 19.2],
        })
        result = build_unit_costs(meter, billing)
        assert len(result.attrs.get("_unmatched_brands", [])) == 0


# ── Multi-building brand aggregation ─────────────────────────────────────────

class TestMultiBuildingAggregation:
    """Same brand in different buildings must NOT be merged."""

    def test_brand_building_separation(self):
        from features import aggregate_by_brand
        df = pd.DataFrame({
            "brand": ["올리브영", "올리브영", "기타"],
            "building": ["A", "B", "A"],
            "floor": ["1F", "2F", "1F"],
            "size_m2": [50, 60, 40],
            "size_py": [15, 18, 12],
            "water_current": [10, 20, 5],
            "hwater_current": [5, 10, 3],
            "elect_current": [100, 200, 50],
            "heat_current": [3, 6, 2],
        })
        result = aggregate_by_brand(df)
        # 올리브영 should appear as 2 separate rows (A and B)
        oly = result[result["brand"] == "올리브영"]
        assert len(oly) == 2
        assert set(oly["building"]) == {"A", "B"}


# ── Anomaly metadata propagation ─────────────────────────────────────────────

class TestAnomalyMetadata:
    def test_excluded_dims_propagated(self):
        from anomaly_features import build_anomaly_df
        df = _make_meter_df(10, with_mom=False)
        result = build_anomaly_df(df)
        assert "spike_score" in result.attrs.get("_excluded_dims", [])
        assert "consistency_score" in result.attrs.get("_available_dims", [])


# ── Real corpus smoke test ───────────────────────────────────────────────────

class TestRealCorpus:
    """Smoke test with the actual data file if available."""

    @pytest.fixture
    def xlsm_data(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utility_report.xlsm")
        if not os.path.exists(path):
            pytest.skip("utility_report.xlsm not found")
        with open(path, "rb") as f:
            return f.read()

    def test_full_pipeline(self, xlsm_data):
        from data import read_billing_sheet, read_electricity_sheet, read_water_sheet, read_hotwater_sheet
        from meter_view import load_raw_meter_df
        from features import aggregate_by_brand
        from anomaly_features import build_anomaly_df
        from utils import add_per_area_cols

        fname = "utility_report.xlsm"
        raw_df = load_raw_meter_df(fname, {fname: xlsm_data}, "검침 내역")
        assert len(raw_df) > 0

        agg_df = aggregate_by_brand(raw_df)
        add_per_area_cols(agg_df)

        billing_df = read_billing_sheet(fname, xlsm_data, "수도광열비 부과 내역 ")
        elec_df = read_electricity_sheet(fname, xlsm_data, "전체 전기 사용내역")
        water_df = read_water_sheet(fname, xlsm_data, "수도 사용 내역")
        hotwater_df = read_hotwater_sheet(fname, xlsm_data, "온수 사용 내역")

        anomaly_df = build_anomaly_df(
            meter_df=agg_df,
            billing_df=billing_df,
            elec_df=elec_df,
            water_df=water_df,
            hotwater_df=hotwater_df,
        )

        # Basic shape checks
        assert len(anomaly_df) == len(agg_df)
        assert "composite_score" in anomaly_df.columns
        assert "risk_level" in anomaly_df.columns
        assert "reason" in anomaly_df.columns

        # Scores should be in [0, 1]
        assert anomaly_df["composite_score"].between(0, 1).all()

        # With single file (no MoM), spike redistribution should allow
        # non-trivial scores
        assert anomaly_df["composite_score"].max() > 0.20
