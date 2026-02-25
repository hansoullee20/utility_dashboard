import io
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
def read_upload(name: str, data: bytes):
    bio = io.BytesIO(data)
    name = name.lower()

    if name.endswith(".csv"):
        return {"__single__": pd.read_csv(bio)}

    if name.endswith(".parquet"):
        return {"__single__": pd.read_parquet(bio)}

    if name.endswith((".xlsx", ".xls", ".xlsm")):
        xls = pd.ExcelFile(bio)
        return {
            sh: pd.read_excel(io.BytesIO(data), sheet_name=sh, header=[2, 3, 4])
            for sh in xls.sheet_names
        }

    raise ValueError("Unsupported file type")


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
