"""anomaly_features.py — Cross-sheet anomaly feature engineering for 이상감지 분석.

Aggregates anomaly signals from every available sheet into a single per-brand
DataFrame with component scores [0, 1] and a weighted composite_score.

Scoring Philosophy (v2 — absolute-anchored, change-focused)
------------------------------------------------------------
All scores are anchored to **fixed thresholds**, not min-max normalized.
This means a brand with 100% spike always scores ~0.88 regardless of whether
another brand spiked 500%.  Scores are comparable across months.

Every dimension measures **change from baseline or deviation from peers**,
never absolute usage levels — a restaurant naturally uses more than an office,
so absolute levels are irrelevant for anomaly detection.

Anomaly dimensions
------------------
spike        (40 %) — MoM % change magnitude per utility (peer-relative)
             Scores **excess spike** = brand % − building avg %.
             80% excess sigmoid + 20% raw sigmoid (absolute floor).
             A brand at +150% when building avg is +200% scores LOW.
             Also flags large drops (possible meter error / vacancy).
consumption  (25 %) — quadrant classification per utility from 검침내역
             HH=1.0 · HL=0.6 · LH=0.5 · Normal=0 · LL=0 → max across utilities
             Measures change relative to peer distribution, not absolute level.
cost         (20 %) — unit cost Z-scores from 수도광열비 부과 내역
             |Z| ≥ 3→1.0 · ≥ 2→0.75 · ≥ 1.5→0.50 · ≥ 1→0.25
             Z-scores compare brand to peers, not absolute cost.
hvac         (5 %)  — HVAC intensity Z-score from 전체 전기 사용내역
             Same Z-mapping as cost.
consistency  (10 %) — zero-usage + sudden-drop detection
             Counts zero-current utilities + large MoM drops.

Public API
----------
build_anomaly_df(meter_df, billing_df, elec_df, water_df, hotwater_df, q0, q1)
    Returns per-brand DataFrame with all anomaly signals +
    composite_score [0, 1] and risk_level label.

Spike columns added
-------------------
{pfx}_spike_pct   — raw MoM % change for that utility (positive = increase)
{pfx}_spike_flag  — 🔴 급등(≥100%) / 🟠 주의(≥50%) / 🟡 관찰(≥20%) / "" normal
spike_score       — absolute-anchored max spike score across all utilities [0, 1]
spike_max_pct     — the highest single-utility MoM % increase
spike_worst_util  — which utility had the largest spike
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cross_features import build_unit_costs, build_elec_breakdown
from data import to_numeric_series
from utils import (
    zscore as _zscore, mad_zscore as _mad_zscore,
    iqr_upper as _iqr_upper, z_to_grade as _ztg,
    UTIL_PREFIXES as _UTIL_PREFIXES_TUPLE, UTIL_LABELS as _UTIL_LABELS,
    RISK_DANGER, RISK_CAUTION, RISK_OBSERVE, RISK_NORMAL,
)

_UTIL_PREFIXES = list(_UTIL_PREFIXES_TUPLE)

# Quadrant → score: exponential, HH dominates
_QUAD_SCORE = {"HH": 1.0, "HL": 0.6, "LH": 0.5, "Normal": 0.0, "LL": 0.0, "No Data": 0.0}

_WEIGHTS = {
    "spike_score":       0.40,
    "consumption_score": 0.25,
    "cost_score":        0.20,
    "hvac_score":        0.05,
    "consistency_score": 0.10,
}

# Absolute spike thresholds (% change from previous month)
_SPIKE_CRITICAL = 100.0   # ≥ 100% increase
_SPIKE_HIGH     =  50.0   # ≥  50%
_SPIKE_MEDIUM   =  20.0   # ≥  20%

# Sudden-drop threshold (large negative change → possible meter error / vacancy)
_DROP_THRESHOLD = -50.0   # ≥ 50% decrease


# ── helpers ───────────────────────────────────────────────────────────────────

def _sigmoid_score(pct: pd.Series, midpoint: float = 50.0, k: float = 0.04) -> pd.Series:
    """Map % change to [0, 1] via sigmoid, anchored to fixed thresholds.

    At midpoint the score is 0.5.  Default k=0.04 gives:
        20% → ~0.27,  50% → 0.50,  100% → ~0.88,  200% → ~0.998
    """
    return (1 / (1 + np.exp(-k * (pct - midpoint)))).round(4)


def _z_to_score(z: pd.Series) -> pd.Series:
    """Map absolute Z-score to [0, 1] via fixed thresholds.

    |Z| ≥ 3.0 → 1.0,  ≥ 2.0 → 0.75,  ≥ 1.5 → 0.50,  ≥ 1.0 → 0.25,  < 1.0 → 0.0
    Linearly interpolated within each band.
    """
    az = z.abs().fillna(0)
    score = pd.Series(0.0, index=z.index)
    # Linear interpolation within bands
    # [0, 1) → [0, 0.25)
    mask = (az >= 0) & (az < 1.0)
    score[mask] = (az[mask] / 1.0) * 0.25
    # [1, 1.5) → [0.25, 0.50)
    mask = (az >= 1.0) & (az < 1.5)
    score[mask] = 0.25 + ((az[mask] - 1.0) / 0.5) * 0.25
    # [1.5, 2.0) → [0.50, 0.75)
    mask = (az >= 1.5) & (az < 2.0)
    score[mask] = 0.50 + ((az[mask] - 1.5) / 0.5) * 0.25
    # [2.0, 3.0) → [0.75, 1.0)
    mask = (az >= 2.0) & (az < 3.0)
    score[mask] = 0.75 + ((az[mask] - 2.0) / 1.0) * 0.25
    # ≥ 3.0 → 1.0
    score[az >= 3.0] = 1.0
    return score.round(4)


# ── 0. Spike signals — absolute MoM change detection ─────────────────────────

def _add_spike_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Detect abrupt MoM usage changes using peer-relative sigmoid scoring.

    Primary signal is **excess spike** = brand % change − building average %.
    A brand moving with the herd (e.g., +150% when building avg is +200%)
    scores low; a brand spiking far above peers scores high.

    A small floor score (20% of raw sigmoid) is kept for extreme absolute
    spikes — +500% is still noteworthy even if peers averaged +400%.

    Both increases AND large drops are flagged (drops may indicate meter error).
    """
    out = df.copy()
    has_building = "building" in df.columns

    # ── Per-utility raw % changes ─────────────────────────────────────────
    spike_scores: list[pd.Series] = []
    per_util_pcts: dict[str, pd.Series] = {}

    for pfx in _UTIL_PREFIXES:
        pct_col = f"{pfx}_pct"
        chg_col = f"{pfx}_change"
        prev_col = f"{pfx}_previous"
        out_col = f"{pfx}_spike_pct"

        if pct_col not in df.columns:
            continue

        p = to_numeric_series(df[pct_col])   # % change (can be NaN for new tenants)
        c = to_numeric_series(df[chg_col]) if chg_col in df.columns else pd.Series(np.nan, index=df.index)

        # First-appearance detection: previous value missing entirely.
        # Distinguishes "new tenant / new meter" (previous=NaN) from
        # "zero-base growth" (previous=0, pct=NaN from div-by-zero).
        if prev_col in df.columns:
            is_new_util = to_numeric_series(df[prev_col]).isna()
        else:
            is_new_util = pd.Series(False, index=df.index)
        out[f"{pfx}_is_new"] = is_new_util

        out[out_col] = p.round(1)   # store raw (signed) for display
        per_util_pcts[pfx] = p

        # Severity flag (for display — separate from scoring)
        def _flag(v):
            if pd.isna(v) or v <= 0:
                return ""
            if v >= _SPIKE_CRITICAL:
                return "🔴 급등"
            if v >= _SPIKE_HIGH:
                return "🟠 주의"
            if v >= _SPIKE_MEDIUM:
                return "🟡 관찰"
            return ""

        out[f"{pfx}_spike_flag"] = p.apply(_flag)

        # ── Per-utility building average (for display) ────────────────
        if has_building:
            bldg = df["building"]
            pos_p = p.clip(lower=0).fillna(0)
            bldg_sum = bldg.map(pos_p.groupby(bldg).sum())
            bldg_cnt = bldg.groupby(bldg).transform("count")
            loo_avg = ((bldg_sum - pos_p) / (bldg_cnt - 1)).where(bldg_cnt >= 2)
            out[f"{pfx}_bldg_avg_pct"] = loo_avg.round(1)

        # ── Raw sigmoid on absolute spike (used for floor) ────────────
        pos_pct = p.clip(lower=0).fillna(0)
        raw_score = _sigmoid_score(pos_pct)

        # Also score large drops (meter error / vacancy signal)
        drop_score = _sigmoid_score(p.abs().fillna(0).where(p < _DROP_THRESHOLD, 0),
                                    midpoint=50.0, k=0.03)  # softer curve for drops

        # Combined raw: max of increase or drop signal
        raw_util_score = pd.concat([raw_score, drop_score], axis=1).max(axis=1)

        # Attenuate tiny-base spikes: if absolute change is very small,
        # reduce score (e.g., 0.1 → 1.0 m³ is +900% but not alarming)
        if c.dropna().any():
            c_abs = c.abs().fillna(0)
            median_change = c_abs[c_abs > 0].median()
            if median_change is not None and not pd.isna(median_change) and median_change > 0:
                small_base = c_abs < (median_change * 0.1)
                raw_util_score = raw_util_score.where(~small_base, raw_util_score * 0.5)

        # ── Peer-relative excess score ────────────────────────────────
        if has_building:
            bldg_avg = out[f"{pfx}_bldg_avg_pct"].fillna(0)
            excess = (pos_pct - bldg_avg).clip(lower=0)
            excess_score = _sigmoid_score(excess)
            # Blend: 80% excess (peer-relative) + 20% raw (absolute floor)
            util_score = (0.8 * excess_score + 0.2 * raw_util_score).round(4)
        else:
            # No building info — fall back to raw absolute scoring
            util_score = raw_util_score

        # Suppress first-appearance spikes: a brand that did not exist in
        # the previous file has no baseline, so any non-zero "change" is
        # noise from tenant turnover, not a billing anomaly. Set the
        # contribution to exactly 0.0, not the sigmoid floor.
        util_score = util_score.where(~is_new_util, 0.0).round(4)

        spike_scores.append(util_score.rename(out_col))

    if spike_scores:
        spike_mat = pd.concat(spike_scores, axis=1)
        base_spike = spike_mat.max(axis=1).round(4)

        # Which utility spiked most, and by how much
        raw_pct_cols = [f"{pfx}_spike_pct" for pfx in _UTIL_PREFIXES
                        if f"{pfx}_spike_pct" in out.columns]
        raw_mat = out[raw_pct_cols].clip(lower=0).fillna(0)
        out["spike_max_pct"]   = raw_mat.max(axis=1).round(1)
        worst_idx = raw_mat.idxmax(axis=1)
        out["spike_worst_util"] = worst_idx.map(
            lambda c: _UTIL_LABELS.get(c.replace("_spike_pct", ""), c) if isinstance(c, str) else ""
        )

        # Overall peer context (max spike across utilities)
        if has_building:
            _peer_context = _compute_peer_context(out, raw_pct_cols)
            for col in _peer_context.columns:
                out[col] = _peer_context[col]

        out["spike_score"] = base_spike.round(4)
    else:
        out["spike_score"]     = 0.0
        out["spike_max_pct"]   = 0.0
        out["spike_worst_util"] = ""

    # Row-level first-appearance flag: all tracked utilities are first-appearance.
    new_flag_cols = [f"{pfx}_is_new" for pfx in _UTIL_PREFIXES
                     if f"{pfx}_is_new" in out.columns]
    if new_flag_cols:
        out["is_new_tenant"] = out[new_flag_cols].all(axis=1)
    else:
        out["is_new_tenant"] = False

    return out


def _compute_peer_context(df: pd.DataFrame, raw_pct_cols: list[str]) -> pd.DataFrame:
    """Compute per-brand spike vs building-average spike ratio (vectorized).

    Uses leave-one-out mean: (building_sum - brand_val) / (building_count - 1).
    """
    max_spike = df[raw_pct_cols].clip(lower=0).fillna(0).max(axis=1)
    bldg = df["building"]
    bldg_sum = bldg.map(max_spike.groupby(bldg).sum())
    bldg_cnt = bldg.groupby(bldg).transform("count")
    # Leave-one-out average: exclude this brand from its building mean
    loo_avg = ((bldg_sum - max_spike) / (bldg_cnt - 1)).where(bldg_cnt >= 2).round(1)
    ratio = (max_spike / loo_avg.replace(0, np.nan)).round(1)
    return pd.DataFrame({"spike_bldg_avg_pct": loo_avg, "spike_peer_ratio": ratio},
                        index=df.index)


# ── 1. Consumption signals (always available from meter) ──────────────────────

def _add_consumption_signals(df: pd.DataFrame, q0: float, q1: float) -> pd.DataFrame:
    """Classify each utility into HH/HL/LH/LL/Normal quadrants.

    Quadrant analysis uses *change* and *% change* — NOT absolute usage levels.
    A brand is HH when its usage CHANGE and % CHANGE are both in the top decile.
    This is inherently peer-relative: it answers "did this brand change more than
    its peers?" regardless of business type or absolute usage.

    Score = max quadrant score across utilities (one HH is more concerning than
    four Normals).
    """
    out = df.copy()
    quad_scores: list[pd.Series] = []
    n_active = 0

    for pfx in _UTIL_PREFIXES:
        c_col = f"{pfx}_change"
        p_col = f"{pfx}_pct"
        if c_col not in df.columns:
            out[f"{pfx}_quadrant"]  = "No Data"
            out[f"{pfx}_quad_score"] = 0.0
            continue

        c = to_numeric_series(df[c_col])
        p = (to_numeric_series(df[p_col])
             if p_col in df.columns
             else pd.Series(np.nan, index=df.index))

        if c.dropna().empty:
            out[f"{pfx}_quadrant"]  = "No Data"
            out[f"{pfx}_quad_score"] = 0.0
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

        scores = quad.map(_QUAD_SCORE).fillna(0.0)
        out[f"{pfx}_quadrant"]  = quad
        out[f"{pfx}_quad_score"] = scores
        out[f"{pfx}_change_z"]   = _zscore(c)
        if p.dropna().any():
            out[f"{pfx}_pct_z"]  = _zscore(p)

        quad_scores.append(scores)
        n_active += 1

    if n_active and quad_scores:
        # Max across utilities — one HH is more alarming than multiple Normals
        score_mat = pd.concat(quad_scores, axis=1)
        out["consumption_raw"]   = score_mat.sum(axis=1).round(2)
        out["consumption_score"] = score_mat.max(axis=1).round(4)
    else:
        out["consumption_raw"]   = 0.0
        out["consumption_score"] = 0.0
    return out


# ── 2. Cost signals (billing sheet) ──────────────────────────────────────────

def _add_cost_signals(df: pd.DataFrame, billing_df: pd.DataFrame) -> pd.DataFrame:
    """Score unit cost deviation from peers using Z-score thresholds.

    Unit cost (₩/m³, ₩/kWh, 만원/m²) normalizes for business size —
    a large restaurant pays more total but shouldn't pay more PER UNIT.
    """
    try:
        unit_df = build_unit_costs(df, billing_df)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Cost analysis failed: %s", e)
        return df.assign(cost_score=np.nan, _cost_available=False)

    join_cols = ["brand", "building"] if "building" in df.columns else ["brand"]
    cost_cols = [c for c in [
        "water_unit_cost", "water_unit_z",
        "elect_unit_cost", "elect_unit_z",
        "total_cost_per_m2", "total_cost_per_m2_z",
        "total_cost_per_py", "total_cost_per_py_z",
    ] if c in unit_df.columns]

    if not cost_cols:
        return df.assign(cost_score=0.0, _cost_available=False)

    merged = df.merge(unit_df[join_cols + cost_cols], on=join_cols, how="left")

    z_cols = [c for c in ["water_unit_z", "elect_unit_z", "total_cost_per_py_z",
                           "total_cost_per_m2_z"]
              if c in merged.columns]
    if z_cols:
        max_z = merged[z_cols].abs().max(axis=1).fillna(0)
        merged["cost_score"] = _z_to_score(max_z)
    else:
        merged["cost_score"] = 0.0
    merged["_cost_available"] = True
    # Propagate join coverage metadata
    if hasattr(unit_df, "attrs"):
        merged.attrs["_cost_unmatched"] = unit_df.attrs.get("_unmatched_brands", [])
        merged.attrs["_cost_coverage"] = unit_df.attrs.get("_join_coverage", 1.0)
    return merged


# ── 3. HVAC signals (electricity detail sheet) ────────────────────────────────

def _add_hvac_signals(df: pd.DataFrame, elec_df: pd.DataFrame) -> pd.DataFrame:
    """Score HVAC intensity deviation from peers using Z-score thresholds.

    HVAC intensity = kWh/m² — normalized by floor area so business size
    doesn't affect the score.
    """
    try:
        elec_br = build_elec_breakdown(elec_df, meter_df=df)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("HVAC analysis failed: %s", e)
        return df.assign(hvac_score=np.nan, _hvac_available=False)

    join_cols = ["brand", "building"] if "building" in df.columns else ["brand"]
    hvac_cols = [c for c in [
        "hvac_intensity", "ehp_pct", "hvac_pct", "base_pct", "elect_unit_cost",
    ] if c in elec_br.columns]

    if not hvac_cols:
        return df.assign(hvac_score=0.0, _hvac_available=False)

    merged = df.merge(elec_br[join_cols + hvac_cols], on=join_cols, how="left")

    if "hvac_intensity" in merged.columns:
        hi_s = to_numeric_series(merged["hvac_intensity"]).fillna(0)
        # MAD-based z is skew-resistant: a single outlier HVAC tenant
        # no longer distorts the median/MAD and collapses other tenants
        # toward 0.
        z = _mad_zscore(hi_s)
        merged["hvac_intensity_z"] = z
        merged["hvac_score"] = _z_to_score(z)
    else:
        merged["hvac_score"] = 0.0
    merged["_hvac_available"] = True
    return merged


# ── 4. Consistency signals (zero-usage + sudden-drop detection) ──────────────

def _add_consistency_signals(
    df: pd.DataFrame,
    water_df: pd.DataFrame | None,
    hotwater_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Detect data quality anomalies: zero usage + sudden large drops.

    Zero current-usage and large negative MoM changes both suggest
    meter issues, vacancy, or data errors — regardless of business type.
    """
    out = df.copy()
    zero_cnt = pd.Series(0, index=df.index, dtype=int)
    drop_cnt = pd.Series(0, index=df.index, dtype=int)

    for pfx in _UTIL_PREFIXES:
        # Zero detection from meter current-usage columns
        cur = f"{pfx}_current"
        if cur in out.columns:
            zero_cnt += (to_numeric_series(out[cur]).fillna(0) == 0).astype(int)

        # Large drop detection from MoM pct change
        pct_col = f"{pfx}_pct"
        if pct_col in out.columns:
            p = to_numeric_series(out[pct_col])
            drop_cnt += (p < _DROP_THRESHOLD).fillna(False).astype(int)

    out["n_zero_utilities"] = zero_cnt.values
    out["n_drop_utilities"] = drop_cnt.values

    # Score: direct mapping — no min-max normalization
    # 0 issues → 0.0, 1 → 0.25, 2 → 0.50, 3 → 0.75, 4+ → 1.0
    total_issues = (zero_cnt + drop_cnt).clip(upper=4)
    out["consistency_score"] = (total_issues / 4.0).round(4)
    return out


# ── Reason flags ─────────────────────────────────────────────────────────

def _build_reason_flags(df: pd.DataFrame) -> pd.Series:
    """Build a concise human-readable '이유' string per brand (vectorized).

    Example output: "급등 +84%(수도) vs건물 7.2x · 수도단가 Z+2.8 · HH(수도,전기)"
    """
    def _row_reason(r):
        parts: list[str] = []
        # 1. Spike + peer context (show building avg so user sees context)
        pct = r.get("spike_max_pct", 0) or 0
        if pct >= _SPIKE_MEDIUM:
            s = f"급등 +{pct:.0f}%({r.get('spike_worst_util', '') or ''})"
            bavg = r.get("spike_bldg_avg_pct")
            pr = r.get("spike_peer_ratio")
            if bavg is not None and not pd.isna(bavg):
                s += f" 건물평균 +{bavg:.0f}%"
                if pr is not None and not pd.isna(pr) and pr >= 2.0:
                    s += f"({pr:.1f}x)"
            parts.append(s)
        # 1b. Large drops
        n_drops = r.get("n_drop_utilities", 0) or 0
        if n_drops >= 1:
            drop_utils = []
            for pfx, lbl in _UTIL_LABELS.items():
                sp = r.get(f"{pfx}_spike_pct")
                if sp is not None and not pd.isna(sp) and sp < _DROP_THRESHOLD:
                    drop_utils.append(f"{lbl}{sp:.0f}%")
            if drop_utils:
                parts.append(f"급감 {','.join(drop_utils)}")
        # 2. Worst unit cost Z ≥ 1.5 — shown as business-friendly grade
        _Z = [("수도단가", r.get("water_unit_z")),
              ("전기단가", r.get("elect_unit_z")),
              ("평당비용", r.get("total_cost_per_py_z") or r.get("total_cost_per_m2_z"))]
        _Z = [(l, float(v)) for l, v in _Z if v is not None and not pd.isna(v) and abs(v) >= 1.5]
        if _Z:
            l, v = max(_Z, key=lambda x: abs(x[1]))
            parts.append(f"{l} {_ztg(v)}")
        # 3. HH quadrants
        hh = [lbl for pfx, lbl in _UTIL_LABELS.items() if r.get(f"{pfx}_quadrant") == "HH"]
        if hh:
            parts.append(f"HH({','.join(hh)})")
        # 4. HVAC intensity ≥ 2σ
        hz = r.get("hvac_intensity_z")
        if hz is not None and not pd.isna(hz) and abs(hz) >= 2.0:
            parts.append(f"HVAC {_ztg(hz)}")
        # 5. Zero-usage ≥ 2
        nz = r.get("n_zero_utilities", 0) or 0
        if nz >= 2:
            parts.append(f"미계량 {nz}건")
        return " · ".join(parts) if parts else "—"

    return df.apply(_row_reason, axis=1)


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

    # 0. Spike signals (absolute MoM — primary anomaly signal)
    df = _add_spike_signals(df)

    # 1. Consumption signals (always available — change-based quadrants)
    df = _add_consumption_signals(df, q0=q0, q1=q1)

    # 2. Cost signals (unit cost Z-scores — size-normalized)
    if billing_df is not None and not billing_df.empty:
        df = _add_cost_signals(df, billing_df)
    else:
        df["cost_score"] = 0.0
        df["_cost_available"] = False

    # 3. HVAC signals (intensity Z-scores — size-normalized)
    if elec_df is not None and not elec_df.empty:
        df = _add_hvac_signals(df, elec_df)
    else:
        df["hvac_score"] = 0.0
        df["_hvac_available"] = False

    # 4. Consistency signals (zero-usage + drop detection)
    df = _add_consistency_signals(df, water_df, hotwater_df)

    # 5. Weighted composite score — redistribute weights for missing data
    available_dims = {}
    _cost_avail = df["_cost_available"].any() if "_cost_available" in df.columns else True
    _hvac_avail = df["_hvac_available"].any() if "_hvac_available" in df.columns else True
    # Structural availability: does MoM data actually exist in the input?
    # This is NOT "did any brand score above zero" — a calm month with valid
    # MoM data but no spikes must still count spike weight, or a normal month
    # silently redistributes the entire 40% spike weight to other dimensions.
    _spike_avail = any(
        f"{pfx}_pct" in df.columns and df[f"{pfx}_pct"].notna().any()
        for pfx in _UTIL_PREFIXES
    )
    # Consumption quadrants need change columns too
    _consumption_avail = any(
        f"{pfx}_change" in df.columns and df[f"{pfx}_change"].notna().any()
        for pfx in _UTIL_PREFIXES
    )
    for k, v in _WEIGHTS.items():
        if k == "spike_score" and not _spike_avail:
            continue  # no MoM data — exclude from weighting
        if k == "consumption_score" and not _consumption_avail:
            continue  # no change data — quadrants are all "No Data"
        if k == "cost_score" and not _cost_avail:
            continue  # billing sheet not loaded — exclude from weighting
        if k == "hvac_score" and not _hvac_avail:
            continue  # electricity sheet not loaded — exclude from weighting
        available_dims[k] = v

    total_w = sum(available_dims.values())
    df["composite_score"] = sum(
        df[k].fillna(0) * (v / total_w) for k, v in available_dims.items()
    ).round(4)

    # Clean up internal flags
    df.drop(columns=[c for c in ["_cost_available", "_hvac_available"] if c in df.columns],
            inplace=True)

    # 6. Reason flags — human-readable summary of WHY a brand is flagged
    df["reason"] = _build_reason_flags(df)

    # 7. Risk classification (absolute thresholds on absolute-anchored scores)
    def _risk(s: float) -> str:
        if s >= 0.65: return RISK_DANGER
        if s >= 0.40: return RISK_CAUTION
        if s >= 0.20: return RISK_OBSERVE
        return RISK_NORMAL

    df["risk_level"] = df["composite_score"].map(_risk)

    # Propagate join-coverage metadata for UI display
    result = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    result.attrs["_cost_unmatched"] = df.attrs.get("_cost_unmatched", [])
    result.attrs["_cost_coverage"] = df.attrs.get("_cost_coverage", 1.0)
    result.attrs["_available_dims"] = list(available_dims.keys())
    result.attrs["_excluded_dims"] = [k for k in _WEIGHTS if k not in available_dims]
    return result
