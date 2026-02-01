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

            d = v_curr - v_prev
            p = (d / v_prev.replace(0, np.nan)) * 100

            idx = df.columns.get_loc(usage)
            df.insert(idx + 1, diff, d)
            df.insert(idx + 2, pct, p.round(2))
    
    return df


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
