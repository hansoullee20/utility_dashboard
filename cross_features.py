"""cross_features.py — Cross-sheet feature engineering.

Joins meter readings + billing + electricity detail to produce enriched
per-brand metrics for anomaly detection and cost analysis.

Billing amounts come in from read_billing_sheet already divided by 10,000
(stored as 만원).  Electricity amounts from read_electricity_sheet are raw ₩.
"""
import numpy as np
import pandas as pd

from data import to_numeric_series
from utils import zscore as _zscore


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = to_numeric_series(df[c])
    return df


def _agg_sum(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    group_cols = ["brand", "building"] if "building" in df.columns else ["brand"]
    present = [c for c in num_cols if c in df.columns]
    agg = df.groupby(group_cols, as_index=False)[present].sum(min_count=1)
    if "brand_raw" in df.columns:
        raw = df.groupby(group_cols)["brand_raw"].first().reset_index()
        agg = agg.merge(raw, on=group_cols, how="left")
    return agg


# ── public API ────────────────────────────────────────────────────────────────

def build_unit_costs(meter_df: pd.DataFrame, billing_df: pd.DataFrame) -> pd.DataFrame:
    """Join meter readings with billing sheet to compute per-unit costs.

    Parameters
    ----------
    meter_df : aggregated meter data (cur_df from meter_view, post-aggregation)
    billing_df : output of read_billing_sheet — amounts in 만원 (÷10,000)

    Returns
    -------
    DataFrame with one row per (brand, building):
        size_m2, water_usage_m3, water_billed, water_unit_cost (₩/m³),
        elect_usage_kwh, elect_billed, elect_unit_cost (₩/kWh),
        heat_billed, total_billed (만원), total_cost_per_m2 (만원/m²),
        water_unit_z, elect_unit_z, total_cost_per_m2_z
    """
    bill_cols = [
        "water_total", "elect_total", "heat_total",
        "hotwater_excl", "hotwater_comm", "total",
    ]
    bill_agg = _agg_sum(billing_df, bill_cols)
    _to_num(bill_agg, bill_cols)

    meter_cols = [
        "brand", "building", "size_m2",
        "water_usage_m3", "hwater_usage_m3",
        "elect_usage_kw", "heat_usage_m3_mwh",
    ]
    meter_side = meter_df[[c for c in meter_cols if c in meter_df.columns]].copy()
    _to_num(meter_side, [c for c in meter_cols if c not in {"brand", "building"}])

    join_on = ["brand", "building"] if "building" in meter_side.columns else ["brand"]
    merged = meter_side.merge(bill_agg, on=join_on, how="left")
    # Track unmatched brands (in meter but not in billing)
    _bill_keys = set(bill_agg[join_on].apply(tuple, axis=1)) if len(join_on) > 1 else set(bill_agg["brand"])
    _meter_keys = set(meter_side[join_on].apply(tuple, axis=1)) if len(join_on) > 1 else set(meter_side["brand"])
    merged.attrs["_unmatched_brands"] = sorted(_meter_keys - _bill_keys)
    merged.attrs["_join_coverage"] = len(_meter_keys & _bill_keys) / max(len(_meter_keys), 1)

    size = merged["size_m2"].replace(0, np.nan) if "size_m2" in merged.columns else None

    # Water: billing.water_total 만원 × 10,000 → ₩ ÷ usage_m3 → ₩/m³
    if "water_total" in merged.columns and "water_usage_m3" in merged.columns:
        usage = merged["water_usage_m3"].replace(0, np.nan)
        merged["water_unit_cost"] = (merged["water_total"] * 10_000 / usage).round(0)

    # Electricity: billing.elect_total 만원 × 10,000 → ₩ ÷ usage_kw → ₩/kWh
    if "elect_total" in merged.columns and "elect_usage_kw" in merged.columns:
        usage = merged["elect_usage_kw"].replace(0, np.nan)
        merged["elect_unit_cost"] = (merged["elect_total"] * 10_000 / usage).round(0)

    # Total cost per m² / per 평  (keep in 만원 units for readability)
    _PY_FACTOR = 3.3058
    if "total" in merged.columns and size is not None:
        merged["total_cost_per_m2"] = (merged["total"] / size).round(4)
        merged["total_cost_per_py"] = (merged["total"] / size * _PY_FACTOR).round(4)

    # Z-scores
    for col, zcol in [
        ("water_unit_cost",   "water_unit_z"),
        ("elect_unit_cost",   "elect_unit_z"),
        ("total_cost_per_py", "total_cost_per_py_z"),
        ("total_cost_per_m2", "total_cost_per_m2_z"),
    ]:
        if col in merged.columns:
            merged[zcol] = _zscore(merged[col])

    return merged.reset_index(drop=True)


def build_elec_breakdown(
    elec_df: pd.DataFrame,
    meter_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute HVAC/EHP electricity breakdown ratios per brand.

    Parameters
    ----------
    elec_df  : output of read_electricity_sheet — amounts in raw ₩
    meter_df : optional; used only for size_m2 join (hvac_intensity)

    Returns
    -------
    DataFrame with one row per (brand, building):
        kwh_total, kwh_ehp, ehp_pct,
        kwh_hvac (ehp+fcu+ahu), hvac_pct,
        kwh_base (elec01+elec02+pump+fan), base_pct,
        elect_unit_cost (₩/kWh from grand_total),
        hvac_intensity (kWh/m²) if meter_df supplied
    """
    kwh_cols = [
        "kwh_elec02", "kwh_elec01", "kwh_kitchen_fan",
        "kwh_fcu", "kwh_ahu", "kwh_ehp", "kwh_pump", "kwh_total",
    ]
    fee_cols = ["grand_total"]
    agg = _agg_sum(elec_df, kwh_cols + fee_cols)
    _to_num(agg, kwh_cols + fee_cols)

    total = agg["kwh_total"].replace(0, np.nan) if "kwh_total" in agg.columns else None

    if total is not None:
        if "kwh_ehp" in agg.columns:
            agg["ehp_pct"] = (agg["kwh_ehp"] / total * 100).round(1)

        hvac_parts = [c for c in ["kwh_ehp", "kwh_fcu", "kwh_ahu"] if c in agg.columns]
        if hvac_parts:
            agg["kwh_hvac"] = agg[hvac_parts].sum(axis=1)
            agg["hvac_pct"] = (agg["kwh_hvac"] / total * 100).round(1)

        base_parts = [
            c for c in ["kwh_elec01", "kwh_elec02", "kwh_pump", "kwh_kitchen_fan"]
            if c in agg.columns
        ]
        if base_parts:
            agg["kwh_base"] = agg[base_parts].sum(axis=1)
            agg["base_pct"] = (agg["kwh_base"] / total * 100).round(1)

    # Unit cost from electricity sheet: grand_total (₩) / kwh_total
    if "grand_total" in agg.columns and total is not None:
        agg["elect_unit_cost"] = (agg["grand_total"] / total).round(0)

    # HVAC intensity requires size_m2
    if meter_df is not None and "size_m2" in meter_df.columns and "kwh_hvac" in agg.columns:
        join_on = ["brand", "building"] if "building" in agg.columns else ["brand"]
        size_side = (
            meter_df[join_on + ["size_m2"]]
            .groupby(join_on, as_index=False)["size_m2"]
            .first()
        )
        agg = agg.merge(size_side, on=join_on, how="left")
        size = to_numeric_series(agg["size_m2"]).replace(0, np.nan)
        agg["hvac_intensity"] = (agg["kwh_hvac"] / size).round(2)

    return agg.reset_index(drop=True)


def build_water_breakdown(
    water_df: pd.DataFrame,
    meter_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute water fee category breakdown per brand.

    Parameters
    ----------
    water_df : output of read_water_sheet — amounts in raw ₩
    meter_df : optional; used only for size_m2 join (water_intensity)

    Returns
    -------
    DataFrame with one row per (brand, building):
        usage_m3, water_fee (water_excl+comm), sewage_fee (sewage_excl+comm),
        levy_fee (levy_excl+comm), pipe_fee_comm, total,
        water_pct, sewage_pct, levy_pct, pipe_pct,
        avg_unit_price, water_intensity (m³/m²) if meter_df supplied
    """
    fee_cols = [
        "usage_m3", "pipe_fee_comm",
        "water_excl", "water_comm",
        "sewage_excl", "sewage_comm",
        "levy_excl", "levy_comm",
        "total_excl", "total_comm", "total",
        "avg_unit_price",
    ]
    agg = _agg_sum(water_df, fee_cols)
    _to_num(agg, fee_cols)

    # Combined fee categories
    for dst, parts in [
        ("water_fee",  ["water_excl", "water_comm"]),
        ("sewage_fee", ["sewage_excl", "sewage_comm"]),
        ("levy_fee",   ["levy_excl", "levy_comm"]),
    ]:
        present = [c for c in parts if c in agg.columns]
        if present:
            agg[dst] = agg[present].sum(axis=1)

    # Percentages of total
    total = agg["total"].replace(0, np.nan) if "total" in agg.columns else None
    if total is not None:
        for src, pct_col in [
            ("water_fee",    "water_pct"),
            ("sewage_fee",   "sewage_pct"),
            ("levy_fee",     "levy_pct"),
            ("pipe_fee_comm", "pipe_pct"),
        ]:
            if src in agg.columns:
                agg[pct_col] = (agg[src] / total * 100).round(1)

    # Water intensity (m³/m²)
    if meter_df is not None and "size_m2" in meter_df.columns and "usage_m3" in agg.columns:
        join_on = ["brand", "building"] if "building" in agg.columns else ["brand"]
        size_side = (
            meter_df[join_on + ["size_m2"]]
            .groupby(join_on, as_index=False)["size_m2"]
            .first()
        )
        agg = agg.merge(size_side, on=join_on, how="left")
        size = to_numeric_series(agg["size_m2"]).replace(0, np.nan)
        agg["water_intensity"] = (to_numeric_series(agg["usage_m3"]) / size).round(4)

    return agg.reset_index(drop=True)
