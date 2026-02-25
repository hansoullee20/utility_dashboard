import io
import numpy as np
import pandas as pd
from typing import List, Optional
from data import to_numeric_series
import streamlit as st

def sanitize(q0: float, q1: float):
    q0 = max(0.0, min(q0, 0.999))
    q1 = max(0.001, min(q1, 1.0))
    if q0 >= q1:
        q0 = q1 - 0.01
    return q0, q1


def create_change_columns(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("water_previous", "water_current", "water_usage_m3", "water_change", "water_pct"),
        ("hwater_previous", "hwater_current", "hwater_usage_m3", "hwater_change", "hwater_pct"),
        ("elect_previous", "elect_current", "elect_usage_kw", "elect_change", "elect_pct"),
        ("heat_previous", "heat_current", "heat_usage_m3_mwh", "heat_change", "heat_pct"),
    ]

    df = df.copy()
    for prev, curr, usage, diff, pct in specs:
        if all(c in df.columns for c in (prev, curr, usage)):
            v_prev = to_numeric_series(df[prev])
            v_curr = to_numeric_series(df[curr])

            # If current exists but previous is NaN, treat previous as 0 for change only
            v_prev_for_change = v_prev.fillna(0).where(v_curr.notna(), other=np.nan)
            d = v_curr - v_prev_for_change

            # Pct stays NaN when previous is NaN or 0
            p = (d / v_prev.replace(0, np.nan)) * 100

            idx = df.columns.get_loc(usage)
            df.insert(idx + 1, diff, d)
            df.insert(idx + 2, pct, p.round(2))
    
    return df


def aggregate_by_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate unit-level rows into one row per brand, summing usage and
    recalculating change/pct from the summed previous and current values."""

    specs = [
        ("water_previous",  "water_current",  "water_usage_m3",     "water_change",  "water_pct"),
        ("hwater_previous", "hwater_current", "hwater_usage_m3",    "hwater_change", "hwater_pct"),
        ("elect_previous",  "elect_current",  "elect_usage_kw",     "elect_change",  "elect_pct"),
        ("heat_previous",   "heat_current",   "heat_usage_m3_mwh",  "heat_change",   "heat_pct"),
    ]

    df = df.copy()

    # Coerce all numeric utility columns
    sum_cols = []
    for prev, curr, usage, change, pct in specs:
        for c in [prev, curr, usage, change, pct]:
            if c in df.columns:
                df[c] = to_numeric_series(df[c])
                if c not in (change, pct):
                    sum_cols.append(c)

    size_cols = [c for c in ["size_m2", "size_py"] if c in df.columns]
    sum_cols += size_cols
    sum_cols = list(dict.fromkeys(sum_cols))  # deduplicate, preserve order

    agg = df.groupby("brand", as_index=False)[sum_cols].sum(min_count=1)

    # Recalculate change and pct from summed previous/current
    for prev, curr, usage, change, pct in specs:
        if prev in agg.columns and curr in agg.columns:
            d = agg[curr] - agg[prev]
            p = (d / agg[prev].replace(0, np.nan)) * 100
            if usage in agg.columns:
                idx = agg.columns.get_loc(usage)
                agg.insert(idx + 1, change, d)
                agg.insert(idx + 2, pct, p.round(2))
            else:
                agg[change] = d
                agg[pct] = p.round(2)

    # Add building summary (e.g. "A, B") if available
    if "building" in df.columns:
        bldg = df.groupby("brand")["building"].apply(
            lambda x: ", ".join(sorted(x.dropna().astype(str).unique()))
        ).reset_index()
        agg = agg.merge(bldg, on="brand", how="left")

    # Add floor summary (e.g. "1F, 2F, 3F") if available
    if "floor" in df.columns:
        flr = df.groupby("brand")["floor"].apply(
            lambda x: ", ".join(sorted(x.dropna().astype(str).unique()))
        ).reset_index()
        agg = agg.merge(flr, on="brand", how="left")

    return agg


def parse_floor_value(floor_str: str) -> list:
    """Parse a compound floor string into a list of individual floor values.
    Handles '/' separators (e.g. '1F/2F') and '~' ranges (e.g. '2F~5F')."""
    import re
    floor_str = str(floor_str).strip()

    if "/" in floor_str:
        return [f.strip() for f in floor_str.split("/") if f.strip()]

    if "~" in floor_str:
        parts = floor_str.split("~")
        if len(parts) == 2:
            s, e = parts[0].strip(), parts[1].strip()
            ms = re.match(r"([A-Za-z]*)(\d+)([A-Za-z]*)", s)
            me = re.match(r"([A-Za-z]*)(\d+)([A-Za-z]*)", e)
            if ms and me:
                pre_s, n_s, suf_s = ms.groups()
                pre_e, n_e, suf_e = me.groups()
                if pre_s == pre_e and suf_s == suf_e:
                    n_s, n_e = int(n_s), int(n_e)
                    step = 1 if n_e >= n_s else -1
                    return [f"{pre_s}{n}{suf_s}" for n in range(n_s, n_e + step, step)]

    return [floor_str]


def get_simple_floors(df: pd.DataFrame) -> list:
    """Return sorted list of all individual floor values after parsing compound values."""
    if "floor" not in df.columns:
        return []
    floors = set()
    for v in df["floor"].dropna().unique():
        floors.update(parse_floor_value(str(v)))
    return sorted(floors)


def split_brand_by_floor(agg_df: pd.DataFrame, ref_df: pd.DataFrame, selected_floors: list) -> pd.DataFrame:
    """
    Split each brand's aggregated row equally across their simple floors.
    Uses parsed floor values to handle compound entries like '1F/2F' or '2F~5F'.
    """
    numeric_cols = [
        c for c in agg_df.columns
        if pd.api.types.is_numeric_dtype(agg_df[c])
        and c not in ["size_m2", "size_py"]
    ]

    # Build brand -> set of simple floors from ref_df
    brand_floors: dict = {}
    for _, row in ref_df.dropna(subset=["floor"]).iterrows():
        brand = row["brand"]
        parsed = parse_floor_value(str(row["floor"]))
        brand_floors.setdefault(brand, set()).update(parsed)

    selected_set = set(selected_floors)

    rows = []
    for _, brand_row in agg_df.iterrows():
        brand = brand_row["brand"]
        all_floors = sorted(brand_floors.get(brand, set()))
        n_floors = len(all_floors) if all_floors else 1
        floors_to_show = [f for f in all_floors if f in selected_set]

        if not floors_to_show:
            continue

        for floor in floors_to_show:
            row = brand_row.copy()
            row["floor"] = floor
            for c in numeric_cols:
                row[c] = round(brand_row[c] / n_floors, 4) if pd.notna(brand_row[c]) else np.nan
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=agg_df.columns)

    return pd.DataFrame(rows).reset_index(drop=True)


def detect_usage_col(df: pd.DataFrame, prefix: str) -> Optional[str]:
    for c in (f"{prefix}_usage_m3", f"{prefix}_usage_kw", f"{prefix}_usage_m3_mwh"):
        if c in df.columns:
            return c
    return None


def display_cols_for_prefix(df: pd.DataFrame, prefix: str) -> List[str]:
    usage = detect_usage_col(df, prefix)
    cols = ["brand", "building", "floor", "size_m2", "size_py"]
    cols += [
        c for c in [
            f"{prefix}_previous",
            f"{prefix}_current",
            usage,
            f"{prefix}_change",
            f"{prefix}_pct",
        ] if c in df.columns
    ]
    return cols


def sort_df(df: pd.DataFrame, cols, asc, mode: str):
    if mode != "extreme" or df.empty:
        return df
    return df.sort_values(cols, ascending=asc)

def cols_brand_then_category(df, prefix: str, mode: str = "change") -> list[str]:
    df_cols = list(df.columns)

    change_col = f"{prefix}_change"
    pct_col    = f"{prefix}_pct"
    prev_col   = f"{prefix}_previous"
    curr_col   = f"{prefix}_current"

    usage_candidates = [
        f"{prefix}_usage_m3",
        f"{prefix}_usage_kw",
        f"{prefix}_usage_m3_mwh",
    ]
    usage_col = next((c for c in usage_candidates if c in df_cols), None)

    # Category core order
    if mode == "change":
        core = [change_col, pct_col, prev_col, curr_col, usage_col]
    else:  # mode == "pct"
        core = [pct_col, change_col, prev_col, curr_col, usage_col]

    core = [c for c in core if c and c in df_cols]

    # Optional context AFTER category (or set context=[] if you want none)
    context = [c for c in ["building", "floor", "size_m2", "size_py"] if c in df_cols]

    cols = []
    if "brand" in df_cols:
        cols.append("brand")
    cols += [c for c in core if c not in cols]
    cols += [c for c in context if c not in cols]  # move context to the end

    return cols

def add_display_index(df, name="No"):
    df = df.reset_index(drop=True)
    df.insert(0, name, range(1, len(df) + 1))
    return df




def download_df_as_excel(df: pd.DataFrame, filename: str, sheet_name: str = "data"):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:  # uses openpyxl by default
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)

    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
