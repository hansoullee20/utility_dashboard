import io
import numpy as np
import pandas as pd
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

    group_cols = ["brand", "building"] if "building" in df.columns else ["brand"]
    agg = df.groupby(group_cols, as_index=False)[sum_cols].sum(min_count=1)

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

    # Add floor summary (e.g. "1F, 2F, 3F") if available
    if "floor" in df.columns:
        flr = df.groupby(group_cols)["floor"].apply(
            lambda x: ", ".join(sorted(x.dropna().astype(str).unique()))
        ).reset_index()
        agg = agg.merge(flr, on=group_cols, how="left")

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

    # Build brand -> set of simple floors via explode (no iterrows)
    floor_df = ref_df.dropna(subset=["floor"])[["brand", "floor"]].copy()
    floor_df["parsed"] = floor_df["floor"].astype(str).apply(parse_floor_value)
    floor_df = floor_df.explode("parsed")
    brand_floors: dict[str, set] = floor_df.groupby("brand")["parsed"].apply(set).to_dict()

    selected_set = set(selected_floors)

    # Per-brand: total floor count + which selected floors to show
    brand_meta: dict[str, tuple[int, list]] = {}
    for brand, floors in brand_floors.items():
        all_floors = sorted(floors)
        to_show = [f for f in all_floors if f in selected_set]
        if to_show:
            brand_meta[brand] = (len(all_floors), to_show)

    if not brand_meta:
        return pd.DataFrame(columns=agg_df.columns)

    # Filter, attach floor lists, explode, divide numeric cols
    out = agg_df[agg_df["brand"].isin(brand_meta)].copy()
    if out.empty:
        return pd.DataFrame(columns=agg_df.columns)

    out["__n__"] = out["brand"].map(lambda b: brand_meta[b][0])
    out["floor"] = out["brand"].map(lambda b: brand_meta[b][1])
    out = out.explode("floor")

    for c in numeric_cols:
        out[c] = (out[c] / out["__n__"]).round(4)

    return out.drop(columns=["__n__"]).reset_index(drop=True)



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


def apply_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Clean MultiIndex headers from the 검침 내역 sheet into a flat English-named DataFrame."""
    df = df.copy()

    if df.shape[1] > 0:
        df = df.drop(df.columns[0], axis=1)

    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c).replace(" ", "").replace("\n", "") for c in df.columns]
        return df

    # Drop "구분" block
    if "구분" in df.columns.get_level_values(0):
        df = df.drop(columns="구분", level=0, errors="ignore")

    # Under "동별 건물 면적 현황" keep only 건물/층수/전용면적
    target = "동별 건물 면적 현황"
    if target in df.columns.get_level_values(0):
        cols_to_drop = [
            (target,) + sub
            for sub in df[target].columns
            if sub[0] not in ("건물", "층수", "전용면적")
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        df.columns = df.columns.remove_unused_levels()

    # Cut columns at '전기 \n배율'
    all_lvl0 = df.columns.get_level_values(0).unique()
    if "전기 \n배율" in list(all_lvl0):
        pos = all_lvl0.get_loc("전기 \n배율")
        df = df[all_lvl0[:pos]]

    # Normalize level-0 names
    lvl0_unique = df.columns.get_level_values(0).unique()
    name_map = {old: str(old).replace(" ", "").replace("\n", "") for old in lvl0_unique}
    df.columns = pd.MultiIndex.from_tuples([(name_map[col[0]], *col[1:]) for col in df.columns])

    # Translate level-1 category names
    df = df.rename(columns={
        "급수 지침": "수도", "온수(급탕) 지침": "온수",
        "전기 지침": "전기", "FCU (냉,난방 지침)": "열요금",
    }, level=1)

    # Flatten and rename to English
    df_flat = df.droplevel(0, axis=1)
    df_flat.columns = [
        str(col[0]).replace(" ", "").replace("\n", "") if isinstance(col, tuple)
        else str(col).replace(" ", "").replace("\n", "")
        for col in df_flat.columns
    ]
    new_names = [
        "building", "floor", "size_m2", "size_py", "brand",
        "water_previous", "water_current", "water_usage_m3",
        "hwater_previous", "hwater_current", "hwater_usage_m3",
        "elect_previous", "elect_current", "elect_usage_kw",
        "heat_previous", "heat_current", "heat_usage_m3_mwh",
    ]
    if len(df_flat.columns) == len(new_names):
        df_flat.columns = new_names
    else:
        df_flat = df_flat.rename(columns=dict(zip(df_flat.columns, new_names)))
    return df_flat
