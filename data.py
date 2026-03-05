import io
import re as _re_billing
import numpy as np
import pandas as pd
import streamlit as st


def get_billing_period(name: str, data: bytes, ehp_sheet: str = "EHP(OAC)검침자료") -> str | None:
    """Extract the billing period from the EHP sheet's year-month column headers.

    The EHP(OAC)검침자료 sheet has explicit headers like "2026.1월검침",
    "2025.12월검침" etc. The most recent (largest) year-month = billing period.
    Returns e.g. "2026년 1월", or None if the EHP sheet is not present.
    """
    try:
        xl = pd.ExcelFile(io.BytesIO(data), engine="calamine")
        if ehp_sheet not in xl.sheet_names:
            return None
        raw = xl.parse(ehp_sheet, header=None)
    except Exception:
        return None

    col_to_ym = _parse_ehp_col_map(raw)
    if not col_to_ym:
        return None

    year, month = max(col_to_ym.values())
    return f"{year}년 {month}월"


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
        return pd.ExcelFile(io.BytesIO(data), engine="calamine").sheet_names
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
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=[2, 3, 4], engine="calamine")

    raise ValueError("Unsupported file type")


BILLING_SHEET_NAME  = "수도광열비 부과 내역"
EHP_OAC_SHEET_NAME  = "EHP(OAC)검침자료"


@st.cache_data(show_spinner="Loading billing sheet...")
def read_billing_sheet(name: str, data: bytes, sheet: str) -> pd.DataFrame:
    """Parse 수도광열비 부과 내역 sheet into a clean flat DataFrame."""
    raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, engine="calamine")

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


def _label_columns_with_year(headers: pd.Series) -> list[str]:
    """Produce unique column names like '2018_1월', '2019_2월', etc.
    For any group of month columns with no year label, the year is
    (next labeled year - 1)."""
    ym_pat = _re.compile(r'(20\d{2})')
    m_pat  = _re.compile(r'(\d{1,2})월')
    values = [str(v).strip() for v in headers]

    # Pass 1: build a map of col_index -> explicit year label
    explicit_year = {}
    for i, s in enumerate(values):
        m = ym_pat.search(s)
        if m:
            explicit_year[i] = int(m.group(1))

    # Pass 2: for each position, resolve its year:
    # - if it has an explicit year, use it
    # - otherwise find the next explicit year and subtract 1
    def resolve_year(i):
        if i in explicit_year:
            return explicit_year[i]
        for j in sorted(explicit_year):
            if j > i:
                return explicit_year[j] - 1
        return None

    result = []
    for i, s in enumerate(values):
        m_m = m_pat.search(s)
        if m_m:
            yr = resolve_year(i)
            result.append(f"{yr}_{int(m_m.group(1))}월" if yr else s)
        else:
            result.append(s)

    return result


def read_ehp_raw_slice(name: str, data: bytes, sheet: str) -> pd.DataFrame:
    """Return the raw unprocessed slice: rows from OAC table start, columns M–DG.
    Row 0 (section title) is skipped; row 1 becomes the header with year-prefixed names."""
    full = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, engine="calamine")
    table_start = None
    table_end = None
    for ri in range(len(full)):
        row_str = full.iloc[ri].astype(str)
        if table_start is None:
            if row_str.str.contains("OAC 전기 사용량", na=False).any():
                table_start = ri
        else:
            if row_str.str.contains("▣", na=False).any():
                table_end = ri
                break
    if table_start is None:
        return pd.DataFrame()

    # Start from col N (13) — drops 전기 사용량 (col M)
    sliced = full.iloc[table_start + 1:table_end, 13:111].reset_index(drop=True)
    sliced.columns = _label_columns_with_year(sliced.iloc[0])
    sliced = sliced.iloc[1:].reset_index(drop=True)
    sliced = sliced.dropna(how="all").reset_index(drop=True)
    # Keep only rows that have a 계량기 번호 (meter number) in the last column (DG → index 97).
    meter_no = sliced.iloc[:, 97].astype(str).str.strip()
    valid = ~meter_no.isin({"nan", "", "NaN"})
    sliced = sliced[valid].reset_index(drop=True)
    return sliced


def compute_monthly_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly usage: current month reading − previous month reading.
    Crosses year boundaries naturally. First month column → NaN.
    계량기 번호 is preserved as the first column."""
    yr_pat = _re.compile(r'^(20\d{2})_')
    month_cols = [c for c in df.columns if yr_pat.match(str(c))]
    readings = df[month_cols].apply(
        lambda col: pd.to_numeric(col.astype(str).str.replace(",", "", regex=False), errors="coerce")
    )
    usage = readings.diff(axis=1)
    usage.columns = month_cols
    if "계량기 번호" in df.columns:
        usage.insert(0, "계량기 번호", df["계량기 번호"].values)
    return usage


def group_raw_slice_by_year(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return {year: df} with 계량기 번호 prepended to each year's month columns."""
    yr_pat = _re.compile(r'^(20\d{2})_')
    years = [int(yr_pat.match(str(c)).group(1)) if yr_pat.match(str(c)) else None for c in df.columns]
    key_col = ["계량기 번호"] if "계량기 번호" in df.columns else []
    result: dict[int, pd.DataFrame] = {}
    i = 0
    while i < len(years):
        yr = years[i]
        if yr is None:
            i += 1
            continue
        j = i
        while j < len(years) and years[j] == yr:
            j += 1
        result[yr] = df[key_col + list(df.columns[i:j])]
        i = j
    return result


def read_ehp_oac_sheet(name: str, data: bytes, sheet: str) -> pd.DataFrame:
    """Parse only the ▣ OAC 전기 사용량 table inside EHP(OAC)검침자료.

    Returns a wide DataFrame with one row per meter (계량기번호):
      meter_no, building, brand, panel_name, equipment_no, capacity_kw,
      cum_YYYY_MM  (one column per year-month, derived from Excel headers)
    """
    full = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, engine="calamine")

    # ── Locate the ▣ OAC 전기 사용량 section ──────────────────────────────────
    table_start = None
    for ri in range(len(full)):
        row_vals = full.iloc[ri].astype(str)
        if row_vals.str.contains("OAC 전기 사용량", na=False).any():
            table_start = ri
            break
    if table_start is None:
        return pd.DataFrame()

    # ── Step 1: slice rows from table_start, columns M–DG (indices 12–110) ────
    # The header rows are at the top of this slice; data rows follow below.
    raw_orig = full.iloc[table_start:, 12:111].reset_index(drop=True)
    raw_orig.columns = range(len(raw_orig.columns))   # normalise to 0-based so col_to_ym keys match
    del full

    # ── Parse year-month headers from the sliced columns ──────────────────────
    # col_to_ym keys are now 0-based within the slice (0 = col M, 98 = col DG)
    col_to_ym = _parse_ehp_col_map(raw_orig)
    if not col_to_ym:
        return pd.DataFrame()

    # ── Detect info columns by scanning header rows for Korean labels ─────────
    _label_map = {
        "계량기": "meter_no",
        "판넬":   "panel_name",
        "장비":   "equipment_no",
        "용량":   "capacity_kw",
        "상호":   "brand",
        "브랜드": "brand",
    }
    key_to_col: dict[str, int] = {}
    for ri in range(min(10, len(raw_orig))):
        for ci, val in enumerate(raw_orig.iloc[ri]):
            sv = str(val).strip()
            for label, key in _label_map.items():
                if label in sv and key not in key_to_col:
                    key_to_col[key] = ci

    # ── Forward-fill merged cells (building, brand etc. often merged) ─────────
    raw_orig.ffill(axis=0, inplace=True)
    raw = raw_orig

    # ── Identify data rows: rows where the first cum column has a numeric value ─
    first_cum_col = min(col_to_ym.keys())
    data_mask = pd.to_numeric(
        raw[first_cum_col].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).notna()
    data_rows = raw[data_mask].reset_index(drop=True)
    del raw, raw_orig

    if data_rows.empty:
        return pd.DataFrame()

    # ── Build output DataFrame all at once to avoid fragmentation ────────────
    def _str_col(key):
        ci = key_to_col.get(key)
        if ci is not None and ci in data_rows.columns:
            return data_rows[ci].astype(str).str.strip().values
        return None

    cols: dict = {}
    cols["meter_no"] = _str_col("meter_no") if "meter_no" in key_to_col else ""
    for key, col_name in [
        ("building",     "building"),
        ("brand",        "brand"),
        ("panel_name",   "panel_name"),
        ("equipment_no", "equipment_no"),
    ]:
        vals = _str_col(key)
        if vals is not None:
            cols[col_name] = vals

    cap_ci = key_to_col.get("capacity_kw")
    if cap_ci is not None and cap_ci in data_rows.columns:
        cols["capacity_kw"] = pd.to_numeric(data_rows[cap_ci], errors="coerce").values

    seen_cum: set = set()
    for ci, (year, month) in sorted(col_to_ym.items()):
        col_name = f"cum_{year}_{month:02d}"
        if ci in data_rows.columns and col_name not in seen_cum:
            seen_cum.add(col_name)
            cols[col_name] = pd.to_numeric(
                data_rows[ci].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            ).values

    df = pd.DataFrame(cols)

    # ── Drop rows where meter_no is missing ───────────────────────────────────
    meter_str = df["meter_no"].astype(str).str.strip()
    df = df[~meter_str.isin({"nan", "", "NaN"}) & meter_str.notna()].copy()

    # Clean string columns (only those that were actually created)
    for c in ["meter_no", "building", "brand", "panel_name", "equipment_no"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df.reset_index(drop=True)


_EHP_YEARS  = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
_EHP_MONTHS = list(range(1, 13))


@st.cache_data(show_spinner="Analyzing EHP data...", max_entries=1)
def build_ehp_analysis(name: str, data: bytes, sheet: str) -> tuple[dict, pd.DataFrame]:
    """Parse EHP sheet and return (year_dfs, annual) — cached by file identity."""
    df = read_ehp_oac_sheet(name, data, sheet)
    if df.empty:
        return {}, pd.DataFrame()

    # Compute monthly usage from cumulative readings
    prev_col = None
    usage_cols: dict = {}
    for y in _EHP_YEARS:
        for m in _EHP_MONTHS:
            cum_col   = f"cum_{y}_{m:02d}"
            usage_col = f"usage_{y}_{m:02d}"
            if cum_col in df.columns:
                if prev_col and prev_col in df.columns:
                    usage_cols[usage_col] = (df[cum_col] - df[prev_col]).clip(lower=0).values
                else:
                    usage_cols[usage_col] = np.nan
                prev_col = cum_col

    if not usage_cols:
        return {}, pd.DataFrame()

    meta_cols = ["meter_no"] + [c for c in ["brand", "capacity_kw"] if c in df.columns]
    df_unit = pd.concat([df[meta_cols], pd.DataFrame(usage_cols, index=df.index)], axis=1)
    del df

    # Build per-year DataFrames
    year_dfs: dict[int, pd.DataFrame] = {}
    for y in _EHP_YEARS:
        avail = [m for m in _EHP_MONTHS if f"usage_{y}_{m:02d}" in df_unit.columns]
        if not avail:
            continue
        src = [f"usage_{y}_{m:02d}" for m in avail]
        ydf = df_unit[meta_cols + src].copy()
        ydf = ydf.rename(columns={f"usage_{y}_{m:02d}": f"{m}월" for m in avail})
        mon_cols = [f"{m}월" for m in avail]
        ydf["연간합계"] = ydf[mon_cols].sum(axis=1, min_count=1).round(0)
        year_dfs[y] = ydf.sort_values("연간합계", ascending=False).reset_index(drop=True)

    del df_unit

    # Build annual totals (meter × year)
    if not year_dfs:
        return {}, pd.DataFrame()

    id_cols = ["meter_no"] + (["brand"] if "brand" in meta_cols else [])
    all_meters = pd.concat(
        [ydf[id_cols] for ydf in year_dfs.values()]
    ).drop_duplicates().reset_index(drop=True)

    if "capacity_kw" in meta_cols:
        for ydf in year_dfs.values():
            if "capacity_kw" in ydf.columns:
                all_meters = all_meters.merge(ydf[id_cols + ["capacity_kw"]], on=id_cols, how="left")
                break

    for y, ydf in year_dfs.items():
        all_meters = all_meters.merge(
            ydf[id_cols + ["연간합계"]].rename(columns={"연간합계": str(y)}),
            on=id_cols, how="left",
        )

    year_str_cols = [str(y) for y in year_dfs]
    all_meters["Total"] = all_meters[year_str_cols].sum(axis=1, min_count=1).round(0)
    annual = all_meters.sort_values("Total", ascending=False).reset_index(drop=True)

    return year_dfs, annual


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
