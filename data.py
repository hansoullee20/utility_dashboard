import io
import numpy as np
import pandas as pd
import streamlit as st


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def st_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            if df[c].apply(lambda x: isinstance(x, (list, dict))).any():
                df[c] = df[c].astype(str)
    return df


@st.cache_data(show_spinner=False)
def get_sheet_names(name: str, data: bytes) -> list:
    """Fast: only reads sheet names without loading any data."""
    name = name.lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.ExcelFile(io.BytesIO(data)).sheet_names
    return ["__single__"]


@st.cache_data(show_spinner="Loading sheet...")
def read_sheet(name: str, data: bytes, sheet: str) -> pd.DataFrame:
    """Lazy: reads only the selected sheet."""
    name = name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))

    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(data))

    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=[2, 3, 4])

    raise ValueError("Unsupported file type")


BILLING_SHEET_NAME  = "수도광열비 부과 내역"
EHP_OAC_SHEET_NAME  = "EHP(OAC)검침자료"


@st.cache_data(show_spinner="Loading billing sheet...")
def read_billing_sheet(name: str, data: bytes, sheet: str) -> pd.DataFrame:
    """Parse 수도광열비 부과 내역 sheet into a clean flat DataFrame."""
    raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, engine="openpyxl")

    # Data rows start at index 5; filter to only rows where col 2 (building) is A/B/C/D
    data_rows = raw.iloc[5:].copy()
    data_rows = data_rows[
        data_rows[2].astype(str).str.strip().isin({"A", "B", "C", "D"})
    ].copy()

    col_map = {
        2:  "building",
        3:  "floor",
        4:  "unit",
        5:  "size_m2",
        9:  "brand",
        10: "water_excl",
        11: "water_comm",
        12: "water_total",
        13: "elect_excl",
        14: "elect_comm",
        15: "elect_total",
        16: "hotwater_excl",
        17: "hotwater_comm",
        18: "hvac_excl",
        19: "hvac_comm",
        20: "heat_total",
        21: "total_excl",
        22: "total_comm",
        23: "total",
    }

    present_cols = [c for c in col_map if c in data_rows.columns]
    df = data_rows[present_cols].copy()
    df.columns = [col_map[c] for c in present_cols]

    # Drop rows where brand is missing
    brand_str = df["brand"].astype(str).str.strip()
    df = df[~brand_str.isin({"nan", "", "NaN"}) & df["brand"].notna()].copy()

    # Coerce numeric columns
    # Cost/amount columns: blank or dash means no charge → fill with 0
    # size_m2: keep NaN if missing (0 would break per-area calculations)
    str_cols = {"building", "floor", "unit", "brand"}
    cost_cols = {
        "water_excl", "water_comm", "water_total",
        "elect_excl", "elect_comm", "elect_total",
        "hotwater_excl", "hotwater_comm",
        "hvac_excl", "hvac_comm",
        "heat_total", "total_excl", "total_comm", "total",
    }
    _blank = {"", "-", "–", "—", "nan", "NaN", "N/A", "n/a"}
    for c in df.columns:
        if c in str_cols:
            continue
        if c in cost_cols:
            cleaned = df[c].astype(str).str.strip().replace(_blank, "0")
            df[c] = (pd.to_numeric(cleaned.str.replace(",", "", regex=False), errors="coerce").fillna(0) / 10_000).round(2)
        else:
            df[c] = to_numeric_series(df[c])

    # Clean string columns
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df.reset_index(drop=True)


import re as _re


def _parse_ehp_col_map(raw: pd.DataFrame) -> dict:
    """Scan the raw sheet header rows and return {col_index: (year, month)}.

    The Excel header has two quirks we handle:
      1. Only the first month of each year contains the year (e.g. "2019.1월검침").
         The remaining 11 months just say "2월검침", "3월검침", …
      2. The earliest year (2018) has NO year label at all — all 12 cells just
         say "N월 검침" with no year prefix.

    Algorithm
    ---------
    • Find the header row with the most "N월" cells.
    • Scan left-to-right: cells matching "YYYY.N월" start a new year; cells
      matching only "N월" inherit the current year.
    • After the scan, any cells BEFORE the first explicit year label are
      unlabeled months.  We assume they form exactly 12 months of (first_year − 1).
    """
    ym_pat   = _re.compile(r'(20\d{2})[.년\s]*(\d{1,2})월')
    m_pat    = _re.compile(r'(\d{1,2})월')

    # Pick the row with the most month-like values
    best_row_idx, best_count = 0, 0
    for ri in range(min(6, len(raw))):
        cnt = sum(1 for v in raw.iloc[ri] if m_pat.search(str(v)))
        if cnt > best_count:
            best_count, best_row_idx = cnt, ri
    if best_count == 0:
        return {}
    header = raw.iloc[best_row_idx]

    col_to_ym: dict = {}
    current_year = None

    for ci, val in enumerate(header):
        s = str(val).strip()
        ym_m = ym_pat.search(s)
        m_m  = m_pat.search(s)
        if ym_m:
            current_year = int(ym_m.group(1))
            col_to_ym[ci] = (current_year, int(ym_m.group(2)))
        elif m_m and current_year is not None:
            col_to_ym[ci] = (current_year, int(m_m.group(1)))

    # Handle unlabeled leading months (the earliest year has no year header).
    # Find columns BEFORE the first explicit year label that still have "N월".
    if col_to_ym:
        first_labeled_ci = min(col_to_ym)
        first_year = col_to_ym[first_labeled_ci][0]
        prev_year  = first_year - 1
        unlabeled: list[tuple[int, int]] = []
        for ci in range(first_labeled_ci):
            s = str(header.iloc[ci]).strip()
            m_m = m_pat.search(s)
            if m_m:
                unlabeled.append((ci, int(m_m.group(1))))
        # Only assign if the count is a reasonable year-block (≤ 12 months)
        if 0 < len(unlabeled) <= 12:
            for ci, month in unlabeled:
                col_to_ym[ci] = (prev_year, month)

    return col_to_ym


@st.cache_data(show_spinner="Loading EHP sheet...")
def read_ehp_oac_sheet(name: str, data: bytes, sheet: str) -> pd.DataFrame:
    """Parse only the ▣ OAC 전기 사용량 table inside EHP(OAC)검침자료.

    Returns a wide DataFrame with one row per EHP unit:
      building, panel_name, equipment_no, capacity_kw, brand,
      cum_YYYY_MM  (one column per year-month, derived from Excel headers)
    """
    full = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, engine="openpyxl")

    # ── Locate the ▣ OAC 전기 사용량 section ──────────────────────────────────
    table_start = None
    for ri in range(len(full)):
        row_vals = full.iloc[ri].astype(str)
        if row_vals.str.contains("OAC 전기 사용량", na=False).any():
            table_start = ri
            break
    if table_start is None:
        return pd.DataFrame()

    # Slice from that row onwards
    raw_orig = full.iloc[table_start:].reset_index(drop=True)

    # ── Parse year-month headers from the table slice ─────────────────────────
    col_to_ym = _parse_ehp_col_map(raw_orig)
    if not col_to_ym:
        return pd.DataFrame()

    # ── Forward-fill merged cells (building, brand etc. often merged) ─────────
    raw = raw_orig.ffill(axis=0)

    # ── Identify data rows: rows where the first cum column has a numeric value ─
    first_cum_col = min(col_to_ym.keys())
    data_mask = pd.to_numeric(
        raw[first_cum_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).notna()
    data_rows = raw[data_mask].copy()

    if data_rows.empty:
        return pd.DataFrame()

    # ── Detect info columns immediately left of the first cum column ──────────
    # Layout: … building, panel_name, equipment_no, capacity_kw, brand | cum cols …
    brand_col = first_cum_col - 1
    cap_col   = first_cum_col - 2
    equip_col = first_cum_col - 3
    panel_col = first_cum_col - 4
    bldg_col  = first_cum_col - 5

    def _col(ci):
        return data_rows[ci].astype(str).str.strip().values if ci >= 0 and ci in data_rows.columns else ""

    df = pd.DataFrame(index=range(len(data_rows)))
    df["building"]     = _col(bldg_col)
    df["panel_name"]   = _col(panel_col)
    df["equipment_no"] = _col(equip_col)
    df["capacity_kw"]  = pd.to_numeric(data_rows[cap_col], errors="coerce").values if cap_col >= 0 and cap_col in data_rows.columns else np.nan
    df["brand"]        = _col(brand_col)

    # Clean strings
    for c in ["building", "panel_name", "equipment_no", "brand"]:
        df[c] = df[c].astype(str).str.strip()

    # Drop rows with missing brand
    brand_str = df["brand"]
    df = df[~brand_str.isin({"nan", "", "NaN"}) & brand_str.notna()].copy()

    # ── Add cumulative reading columns ────────────────────────────────────────
    for ci, (year, month) in sorted(col_to_ym.items()):
        col_name = f"cum_{year}_{month:02d}"
        if ci in data_rows.columns and col_name not in df.columns:
            df[col_name] = pd.to_numeric(
                data_rows[ci].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ).values

    return df.reset_index(drop=True)


def apply_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the exact MultiIndex cleaning pipeline you described and output a flat
    dataframe with standardized English column names including 'building'.
    """

    df = df.copy()

    # Safety: drop first column if it's a blank/index-like column
    if df.shape[1] > 0:
        df = df.drop(df.columns[0], axis=1)

    if not isinstance(df.columns, pd.MultiIndex):
        # If the input is not MultiIndex, just clean strings and return.
        df.columns = [str(c).replace(" ", "").replace("\n", "") for c in df.columns]
        return df

    # 0) Drop Level-0 block "구분" if present (your step)
    target_lvl0 = "구분"
    if target_lvl0 in df.columns.get_level_values(0):
        df = df.drop(columns=target_lvl0, level=0, errors="ignore")

    # 1) Under "동별 건물 면적 현황" keep only 건물/층수/전용면적
    target_lvl0 = "동별 건물 면적 현황"
    keep_lvl1 = ["건물", "층수", "전용면적"]

    if target_lvl0 in df.columns.get_level_values(0):
        all_lvl1_under_target = df[target_lvl0].columns
        cols_to_drop = [
            (target_lvl0,) + sub
            for sub in all_lvl1_under_target
            if sub[0] not in keep_lvl1
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        df.columns = df.columns.remove_unused_levels()

    # 2) Cut off everything at '전기 \n배율' in level 0 (keep cols before it)
    all_lvl0_cols = df.columns.get_level_values(0).unique()
    if "전기 \n배율" in list(all_lvl0_cols):
        pos = all_lvl0_cols.get_loc("전기 \n배율")
        cols_to_keep = all_lvl0_cols[:pos]
        df = df[cols_to_keep]

        name_map = {old: str(old).replace(" ", "").replace("\n", "") for old in cols_to_keep}
        df.columns = pd.MultiIndex.from_tuples(
            [(name_map[col[0]], *col[1:]) for col in df.columns]
        )
    else:
        # If not present, still clean level-0 names
        lvl0_unique = df.columns.get_level_values(0).unique()
        name_map = {old: str(old).replace(" ", "").replace("\n", "") for old in lvl0_unique}
        df.columns = pd.MultiIndex.from_tuples(
            [(name_map[col[0]], *col[1:]) for col in df.columns]
        )

    # 3) Translate level-1 category names (stay MultiIndex)
    df = df.rename(
        columns={
            "급수 지침": "수도",
            "온수(급탕) 지침": "온수",
            "전기 지침": "전기",
            "FCU (냉,난방 지침)": "열요금",
        },
        level=1,
    )

    # 4) Drop only top level and flatten/clean
    df_drop = df.droplevel(0, axis=1)

    cleaned_cols = [
        str(col[0]).replace(" ", "").replace("\n", "") if isinstance(col, tuple)
        else str(col).replace(" ", "").replace("\n", "")
        for col in df_drop.columns
    ]
    df_drop.columns = cleaned_cols

    # 5) Final rename to English
    new_names = [
        "building", "floor", "size_m2", "size_py", "brand",
        "water_previous", "water_current", "water_usage_m3",
        "hwater_previous", "hwater_current", "hwater_usage_m3",
        "elect_previous", "elect_current", "elect_usage_kw",
        "heat_previous", "heat_current", "heat_usage_m3_mwh",
    ]

    if len(df_drop.columns) == len(new_names):
        df_drop.columns = new_names
    else:
        # Fallback: rename as much as possible in order (your existing behavior)
        rename_map = dict(zip(df_drop.columns, new_names))
        df_drop = df_drop.rename(columns=rename_map)

    return df_drop


# def apply_header_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
#     df = raw_df.drop(raw_df.columns[0], axis=1)

#     df.columns = [
#         str(c[0]).replace(" ", "").replace("\n", "")
#         if isinstance(c, tuple)
#         else str(c).replace(" ", "").replace("\n", "")
#         for c in df.columns
#     ]

#     new_cols = [
#         "building", "floor", "size_m2", "size_py", "brand",
#         "water_previous", "water_current", "water_usage_m3",
#         "hwater_previous", "hwater_current", "hwater_usage_m3",
#         "elect_previous", "elect_current", "elect_usage_kw",
#         "heat_previous", "heat_current", "heat_usage_m3_mwh",
#     ]

#     if len(df.columns) == len(new_cols):
#         df.columns = new_cols

#     return df
