"""anomaly_features.py — Cross-sheet anomaly feature engineering for 이상감지 분析.

Aggregates anomaly signals from every available sheet into a single per-brand
DataFrame with component scores [0, 1] and a weighted composite_score.

Anomaly dimensions
------------------
consumption  (40 %) — quadrant classification per utility from 검침내역
             HH=4 · HL=3 · LH=2 · Normal=1 · LL=0  → sum, normalised
cost         (35 %) — unit cost Z-scores from 수도광열비 부과 내역
             max |Z| across water ₩/m³, elect ₩/kWh, total 만원/m²
hvac         (15 %) — HVAC intensity from 전체 전기 사용내역
             normalised kWh/m² (IQR-aware)
consistency  (10 %) — zero-usage count across all utility columns

Public API
----------
build_anomaly_df(meter_df, billing_df, elec_df, water_df, hotwater_df, q0, q1)
    Returns per-brand DataFrame with all anomaly signals +
    composite_score [0, 1] and risk_level label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cross_features import build_unit_costs, build_elec_breakdown
from data import to_numeric_series

_UTIL_PREFIXES = ["water", "hwater", "elect", "heat"]
_QUAD_SCORE = {"HH": 4, "HL": 3, "LH": 2, "Normal": 1, "LL": 0, "No Data": 0}
_WEIGHTS = {
    "consumption_score": 0.40,
    "cost_score":        0.35,
    "hvac_score":        0.15,
    "consistency_score": 0.10,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _zscore(s: pd.Series) -> pd.Series:
    valid = s.dropna()
    if len(valid) < 3:
        return pd.Series(np.nan, index=s.index)
    mu, sigma = valid.mean(), valid.std()
    return pd.Series(0.0, index=s.index) if sigma == 0 else ((s - mu) / sigma).round(3)


def _normalize(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(hi - lo) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).round(4)


def _iqr_upper(s: pd.Series) -> float:
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    return q3 + 1.5 * (q3 - q1)


# ── 1. Consumption signals (always available from meter) ──────────────────────

def _add_consumption_signals(df: pd.DataFrame, q0: float, q1: float) -> pd.DataFrame:
    """Classify each utility into HH/HL/LH/LL/Normal and sum quadrant weights."""
    out = df.copy()
    total_quad = pd.Series(0.0, index=df.index)
    n_active = 0

    for pfx in _UTIL_PREFIXES:
        c_col = f"{pfx}_change"
        p_col = f"{pfx}_pct"
        if c_col not in df.columns:
            out[f"{pfx}_quadrant"]  = "No Data"
            out[f"{pfx}_quad_score"] = 0
            continue

        c = to_numeric_series(df[c_col])
        p = (to_numeric_series(df[p_col])
             if p_col in df.columns
             else pd.Series(np.nan, index=df.index))

        if c.dropna().empty:
            out[f"{pfx}_quadrant"]  = "No Data"
            out[f"{pfx}_quad_score"] = 0
            continue

        lo_c, hi_c = float(c.quantile(q0)), float(c.quantile(q1))
        if p.dropna().empty:
            lo_p, hi_p = 0.0, 0.0
        else:
            lo_p, hi_p = float(p.quantile(q0)), float(p.quantile(q1))

        has_data = c.notna() & p.notna()
        quad = pd.Series("Normal", index=df.index)
        quad[has_data & (c >= hi_c) & (p >= hi_p)] = "HH"
        quad[has_data & (c >= hi_c) & (p <= lo_p)] = "HL"
        quad[has_data & (c <= lo_c) & (p >= hi_p)] = "LH"
        quad[has_data & (c <= lo_c) & (p <= lo_p)] = "LL"
        quad[~has_data] = "No Data"

        scores = quad.map(_QUAD_SCORE).fillna(0)
        out[f"{pfx}_quadrant"]  = quad
        out[f"{pfx}_quad_score"] = scores
        out[f"{pfx}_change_z"]   = _zscore(c)
        if p.dropna().any():
            out[f"{pfx}_pct_z"]  = _zscore(p)

        total_quad += scores
        n_active += 1

    out["consumption_raw"]   = total_quad
    out["consumption_score"] = _normalize(total_quad) if n_active else 0.0
    return out


# ── 2. Cost signals (billing sheet) ──────────────────────────────────────────

def _add_cost_signals(df: pd.DataFrame, billing_df: pd.DataFrame) -> pd.DataFrame:
    try:
        unit_df = build_unit_costs(df, billing_df)
    except Exception:
        return df.assign(cost_score=0.0)

    join_cols = ["brand", "building"] if "building" in df.columns else ["brand"]
    cost_cols = [c for c in [
        "water_unit_cost", "water_unit_z",
        "elect_unit_cost", "elect_unit_z",
        "total_cost_per_m2", "total_cost_per_m2_z",
    ] if c in unit_df.columns]

    if not cost_cols:
        return df.assign(cost_score=0.0)

    merged = df.merge(unit_df[join_cols + cost_cols], on=join_cols, how="left")

    z_cols = [c for c in ["water_unit_z", "elect_unit_z", "total_cost_per_m2_z"]
              if c in merged.columns]
    merged["cost_score"] = (
        _normalize(merged[z_cols].abs().max(axis=1).fillna(0))
        if z_cols else 0.0
    )
    return merged


# ── 3. HVAC signals (electricity detail sheet) ────────────────────────────────

def _add_hvac_signals(df: pd.DataFrame, elec_df: pd.DataFrame) -> pd.DataFrame:
    try:
        elec_br = build_elec_breakdown(elec_df, meter_df=df)
    except Exception:
        return df.assign(hvac_score=0.0)

    join_cols = ["brand", "building"] if "building" in df.columns else ["brand"]
    hvac_cols = [c for c in [
        "hvac_intensity", "ehp_pct", "hvac_pct", "base_pct", "elect_unit_cost",
    ] if c in elec_br.columns]

    if not hvac_cols:
        return df.assign(hvac_score=0.0)

    merged = df.merge(elec_br[join_cols + hvac_cols], on=join_cols, how="left")

    if "hvac_intensity" in merged.columns:
        hi_s = to_numeric_series(merged["hvac_intensity"]).fillna(0)
        merged["hvac_intensity_z"] = _zscore(hi_s)
        # IQR-aware score: flag brands above IQR upper bound more heavily
        iqr_up = _iqr_upper(hi_s.replace(0, np.nan).dropna())
        clipped = hi_s.clip(upper=iqr_up * 2)   # cap extremes so 1–2 outliers don't dominate
        merged["hvac_score"] = _normalize(clipped)
    else:
        merged["hvac_score"] = 0.0

    return merged


# ── 4. Consistency signals (zero-usage detection) ─────────────────────────────

def _add_consistency_signals(
    df: pd.DataFrame,
    water_df: pd.DataFrame | None,
    hotwater_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Count zero-usage utilities per brand across meter + optional sheets."""
    out = df.copy()
    zero_cnt = pd.Series(0, index=df.index, dtype=int)

    # Zero detection from meter current-usage columns
    for pfx in _UTIL_PREFIXES:
        cur = f"{pfx}_current"
        if cur in out.columns:
            zero_cnt += (to_numeric_series(out[cur]).fillna(0) == 0).astype(int)

    # Zero detection from water sheet
    if water_df is not None and not water_df.empty and "usage_m3" in water_df.columns:
        join_cols = [c for c in ["brand", "building"] if c in water_df.columns]
        w_agg = (water_df.groupby(join_cols)["usage_m3"].sum()
                 .reset_index().rename(columns={"usage_m3": "_w_m3"}))
        tmp = out.merge(w_agg, on=join_cols, how="left")
        zero_cnt += (tmp["_w_m3"].fillna(0) == 0).astype(int).values
        out["water_sheet_m3"] = tmp["_w_m3"].values

    # Zero detection from hot water sheet
    if hotwater_df is not None and not hotwater_df.empty and "usage_m3" in hotwater_df.columns:
        join_cols = [c for c in ["brand", "building"] if c in hotwater_df.columns]
        hw_agg = (hotwater_df.groupby(join_cols)["usage_m3"].sum()
                  .reset_index().rename(columns={"usage_m3": "_hw_m3"}))
        tmp = out.merge(hw_agg, on=join_cols, how="left")
        zero_cnt += (tmp["_hw_m3"].fillna(0) == 0).astype(int).values
        out["hotwater_sheet_m3"] = tmp["_hw_m3"].values

    out["n_zero_utilities"]  = zero_cnt.values
    out["consistency_score"] = _normalize(zero_cnt.astype(float))
    return out


# ── Master function ───────────────────────────────────────────────────────────

def build_anomaly_df(
    meter_df: pd.DataFrame,
    billing_df:  pd.DataFrame | None = None,
    elec_df:     pd.DataFrame | None = None,
    water_df:    pd.DataFrame | None = None,
    hotwater_df: pd.DataFrame | None = None,
    q0: float = 0.10,
    q1: float = 0.90,
) -> pd.DataFrame:
    """Build per-brand anomaly DataFrame combining signals from all available sheets.

    Parameters
    ----------
    meter_df    : Aggregated meter df from aggregate_by_brand().
    billing_df  : Optional — 수도광열비 부과 내역 (amounts in 만원).
    elec_df     : Optional — 전체 전기 사용내역.
    water_df    : Optional — 수도 사용 내역 (for consistency cross-check).
    hotwater_df : Optional — 온수 사용 내역.
    q0, q1      : Quantile thresholds for quadrant classification (default 10th/90th).

    Returns
    -------
    DataFrame with one row per (brand, building) sorted by composite_score desc.
    Key output columns:
        {util}_quadrant, {util}_quad_score  — per utility quadrant label/weight
        consumption_score, cost_score, hvac_score, consistency_score  ∈ [0, 1]
        composite_score ∈ [0, 1]
        risk_level: 🔴 위험 / 🟠 주의 / 🟡 관찰 / 🟢 정상
    """
    df = meter_df.copy()

    # 1. Consumption signals (always available)
    df = _add_consumption_signals(df, q0=q0, q1=q1)

    # 2. Cost signals
    if billing_df is not None and not billing_df.empty:
        df = _add_cost_signals(df, billing_df)
    else:
        df["cost_score"] = 0.0

    # 3. HVAC signals
    if elec_df is not None and not elec_df.empty:
        df = _add_hvac_signals(df, elec_df)
    else:
        df["hvac_score"] = 0.0

    # 4. Consistency signals
    df = _add_consistency_signals(df, water_df, hotwater_df)

    # 5. Weighted composite score
    active = {k: v for k, v in _WEIGHTS.items() if k in df.columns}
    total_w = sum(active.values())
    df["composite_score"] = sum(
        df[k].fillna(0) * (v / total_w) for k, v in active.items()
    ).round(4)

    # 6. Risk classification
    def _risk(s: float) -> str:
        if s >= 0.65: return "🔴 위험"
        if s >= 0.40: return "🟠 주의"
        if s >= 0.20: return "🟡 관찰"
        return "🟢 정상"

    df["risk_level"] = df["composite_score"].map(_risk)
    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)
