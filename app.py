import streamlit as st
import pandas as pd
from typing import Dict

from data import read_upload, apply_header_rows, to_numeric_series, st_safe
from features import (
    create_change_columns,
    sanitize,
    sort_df,
    display_cols_for_prefix,
    cols_brand_then_category,
    add_display_index,
    download_df_as_excel
)
from viz import plot_hist_with_tails


def main():
    st.set_page_config(page_title="Utility Outlier Dashboard", layout="wide")
    st.title("Utility Outlier Dashboard")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Upload")
        uploads = st.file_uploader(
            "Upload CSV/XLSX/Parquet",
            type=["csv", "xlsx", "xls", "parquet"],
            accept_multiple_files=True,
        )

        st.divider()
        st.header("Thresholds")
        bins_change = st.slider("Bins (chg)", 5, 200, 30, 1)
        bins_pct = st.slider("Bins (pct)", 5, 200, 30, 1)
        q_change = st.slider(
            "Change quantiles",
            0.0,
            1.0,
            value=(0.20, 0.80),
            step=0.01,
            key="q_change",
        )
        q_pct = st.slider(
            "Pct quantiles",
            0.0,
            1.0,
            value=(0.20, 0.80),
            step=0.01,
            key="q_pct",
        )

        # Keeping for now (even if mostly redundant with explicit sorting)
        row_sort_mode = st.selectbox("Row sort", ["keep", "extreme"], index=0)
        debug = st.checkbox("Debug", value=False)

    if not uploads:
        st.info("Upload at least one file.")
        st.stop()

    # ---------------- Load files ----------------
    files: Dict[str, Dict[str, pd.DataFrame]] = {}
    for f in uploads:
        try:
            files[f.name] = read_upload(f.name, f.getvalue())
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")

    if not files:
        st.stop()

    file_name = st.selectbox("Select file", list(files.keys()))
    sheet_name = st.selectbox("Select sheet", list(files[file_name].keys()), index=0)
    raw_df = files[file_name][sheet_name]

    # ---------------- Preprocess ----------------
    try:
        df = apply_header_rows(raw_df)
        df = create_change_columns(df)

        # replicate notebook row trimming / footer removal
        valid_buildings = {"A", "B", "C", "D"}
        df["building"] = df["building"].astype(str).str.strip()
        df = df[df["building"].isin(valid_buildings)].copy()

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    for col in ["building", "brand"]:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    # ---------------- Building split ----------------
    dfs = {"df_all": df}
    # print(df.head())
    # print(df.head())


    for b in ["A", "B", "C", "D"]:
        dfs[f"df_{b.lower()}"] = df[df["building"].astype(str).str.strip() == b].copy()

    df_key = st.selectbox("Select DF", list(dfs.keys()))
    cur_df = dfs[df_key].copy()

    if debug:
        st.dataframe(st_safe(cur_df.head(20)), width="stretch", hide_index=True)
        

    # ---------------- Category selection ----------------
    allowed = ["water", "hwater", "elect", "heat"]
    present = [p for p in allowed if f"{p}_change" in cur_df.columns]

    if not present:
        st.error("No utility categories found.")
        st.stop()

    prefix = st.selectbox("Category", present)
    change_col, pct_col = f"{prefix}_change", f"{prefix}_pct"

    cur_df[change_col] = to_numeric_series(cur_df[change_col])
    cur_df[pct_col] = to_numeric_series(cur_df[pct_col])

    valid = cur_df[["brand", "building", change_col, pct_col]].dropna()
    if valid.empty:
        st.error("No numeric data.")
        st.stop()

    s_change = valid[change_col]
    s_pct = valid[pct_col]

    # ---------------- Thresholds ----------------
    q0c, q1c = sanitize(*q_change)
    q0p, q1p = sanitize(*q_pct)

    lo_c, hi_c = valid[change_col].quantile([q0c, q1c])
    lo_p, hi_p = valid[pct_col].quantile([q0p, q1p])

    # ---------------- Plots ----------------
    c1, c2 = st.columns(2)

    with c1:
        stats_change = plot_hist_with_tails(
            s_change, bins_change, float(lo_c), float(hi_c), f"Change: {change_col}"
        )
        if stats_change:
            st.caption(
                f"n={stats_change['n']} | mean={stats_change['mean']:.4g} | std={stats_change['std']:.4g} | "
                f"min={stats_change['min']:.4g} | p20={stats_change['p20']:.4g} | median={stats_change['median']:.4g} | "
                f"p80={stats_change['p80']:.4g} | max={stats_change['max']:.4g}"
            )

    with c2:
        stats_pct = plot_hist_with_tails(
            s_pct, bins_pct, float(lo_p), float(hi_p), f"Pct: {pct_col}"
        )
        if stats_pct:
            st.caption(
                f"n={stats_pct['n']} | mean={stats_pct['mean']:.4g} | std={stats_pct['std']:.4g} | "
                f"min={stats_pct['min']:.4g} | p20={stats_pct['p20']:.4g} | median={stats_pct['median']:.4g} | "
                f"p80={stats_pct['p80']:.4g} | max={stats_pct['max']:.4g}"
            )

    # ---------------- Tables (per-graph, top->bottom, sorted high->low) ----------------
    # Inclusive bounds to include ties at the cutoff
    chg_top = cur_df[cur_df[change_col] >= hi_c].copy()
    chg_bot = cur_df[cur_df[change_col] <= lo_c].copy()
    pct_top = cur_df[cur_df[pct_col] >= hi_p].copy()
    pct_bot = cur_df[cur_df[pct_col] <= lo_p].copy()

    # Sort: always highest -> lowest
    chg_top = chg_top.sort_values(change_col, ascending=False).copy()
    chg_bot = chg_bot.sort_values(change_col, ascending=False).copy()
    pct_top = pct_top.sort_values(pct_col, ascending=False).copy()
    pct_bot = pct_bot.sort_values(pct_col, ascending=False).copy()

    left, right = st.columns(2)

    with left:
        st.subheader(f"{change_col} (Top/Bottom)")

        st.markdown(f"**Top 20% (>= {float(hi_c):.4g})** — sorted high→low ({len(chg_top)})")
        chg_cols_top = cols_brand_then_category(chg_top, prefix, mode="change")
        chg_top_view = add_display_index(chg_top[chg_cols_top])
        st.dataframe(st_safe(chg_top_view), width="stretch", hide_index=True)
        download_df_as_excel(
            chg_top_view,
            filename=f"{df_key}_{prefix}_change_top20.xlsx",
            sheet_name="change_top20",
        )

        st.markdown(f"**Bottom 20% (<= {float(lo_c):.4g})** — sorted high→low ({len(chg_bot)})")
        chg_cols_bot = cols_brand_then_category(chg_bot, prefix, mode="change")
        chg_bot_view = add_display_index(chg_bot[chg_cols_bot])
        st.dataframe(st_safe(chg_bot_view), width="stretch", hide_index=True)
        download_df_as_excel(
            chg_bot_view,
            filename=f"{df_key}_{prefix}_change_bottom20.xlsx",
            sheet_name="change_bottom20",
        )

    with right:
        st.subheader(f"{pct_col} (Top/Bottom)")

        st.markdown(f"**Top 20% (>= {float(hi_p):.4g})** — sorted high→low ({len(pct_top)})")
        pct_cols_top = cols_brand_then_category(pct_top, prefix, mode="pct")
        pct_top_view = add_display_index(pct_top[pct_cols_top])
        st.dataframe(st_safe(pct_top_view), width="stretch", hide_index=True)
        download_df_as_excel(
            pct_top_view,
            filename=f"{df_key}_{prefix}_pct_top20.xlsx",
            sheet_name="pct_top20",
        )

        st.markdown(f"**Bottom 20% (<= {float(lo_p):.4g})** — sorted high→low ({len(pct_bot)})")
        pct_cols_bot = cols_brand_then_category(pct_bot, prefix, mode="pct")
        pct_bot_view = add_display_index(pct_bot[pct_cols_bot])
        st.dataframe(st_safe(pct_bot_view), width="stretch", hide_index=True)
        download_df_as_excel(
            pct_bot_view,
            filename=f"{df_key}_{prefix}_pct_bottom20.xlsx",
            sheet_name="pct_bottom20",
        )

    # ---------------- Overlap (Change × Pct quadrants) ----------------
    st.subheader("Overlap (Change × Pct quadrants)")

    chg_low = cur_df[change_col] <= lo_c
    chg_high = cur_df[change_col] >= hi_c
    pct_low = cur_df[pct_col] <= lo_p
    pct_high = cur_df[pct_col] >= hi_p

    q_LL = cur_df.loc[chg_low & pct_low].copy()
    q_LH = cur_df.loc[chg_low & pct_high].copy()
    q_HL = cur_df.loc[chg_high & pct_low].copy()
    q_HH = cur_df.loc[chg_high & pct_high].copy()

    # Sort each quadrant (high->low) using pct then change
    q_LL = q_LL.sort_values([pct_col, change_col], ascending=False).copy()
    q_LH = q_LH.sort_values([pct_col, change_col], ascending=False).copy()
    q_HL = q_HL.sort_values([pct_col, change_col], ascending=False).copy()
    q_HH = q_HH.sort_values([pct_col, change_col], ascending=False).copy()

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"**Change LOW · Pct LOW** ({len(q_LL)})")
        q_cols = cols_brand_then_category(q_LL, prefix, mode="change")
        q_view = add_display_index(q_LL[q_cols])
        st.dataframe(st_safe(add_display_index(q_LL[q_cols])), width="stretch", hide_index=True)
        download_df_as_excel(
        q_view,
        filename=f"{df_key}_{prefix}_overlap_LL.xlsx",
        sheet_name="overlap_LL",
    )
    with r2:
        st.markdown(f"**Change LOW · Pct HIGH** ({len(q_LH)})")
        q_cols = cols_brand_then_category(q_LH, prefix, mode="change")
        q_view = add_display_index(q_LH[q_cols])
        st.dataframe(st_safe(add_display_index(q_LH[q_cols])), width="stretch", hide_index=True)
        download_df_as_excel(
        q_view,
        filename=f"{df_key}_{prefix}_overlap_LH.xlsx",
        sheet_name="overlap_LH",
    )

    r3, r4 = st.columns(2)
    with r3:
        st.markdown(f"**Change HIGH · Pct LOW** ({len(q_HL)})")
        q_cols = cols_brand_then_category(q_HL, prefix, mode="change")
        q_view = add_display_index(q_HL[q_cols])
        st.dataframe(st_safe(q_view), width="stretch", hide_index=True)
        download_df_as_excel(
            q_view,
            filename=f"{df_key}_{prefix}_overlap_HL.xlsx",
            sheet_name="overlap_HL",
        )

    with r4:
        st.markdown(f"**Change HIGH · Pct HIGH** ({len(q_HH)})")
        q_cols = cols_brand_then_category(q_HH, prefix, mode="change")
        q_view = add_display_index(q_HH[q_cols])
        st.dataframe(st_safe(q_view), width="stretch", hide_index=True)
        download_df_as_excel(
            q_view,
            filename=f"{df_key}_{prefix}_overlap_HH.xlsx",
            sheet_name="overlap_HH",
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.exception(e)
        raise



# import io
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import streamlit as st

# st.set_page_config(page_title="Outlier Dashboard", layout="wide")

# # ----------------------------
# # Core helpers
# # ----------------------------
# def sanitize(q0: float, q1: float) -> Tuple[float, float]:
#     q0 = max(0.0, min(float(q0), 0.999))
#     q1 = max(0.001, min(float(q1), 1.0))
#     if q0 >= q1:
#         q0 = max(0.0, q1 - 0.01)
#     return q0, q1

# def to_numeric_series(x: pd.Series) -> pd.Series:
#     return pd.to_numeric(x.astype(str).str.replace(",", "", regex=False), errors="coerce")

# # FIX 1: Helper to make dataframes "Safe" for Streamlit/Arrow display
# def st_safe(df: pd.DataFrame) -> pd.DataFrame:
#     """Converts a dataframe to string types for display to prevent Arrow serialization errors."""
#     return df.astype(str)

# def available_prefixes(df: pd.DataFrame) -> List[str]:
#     cols = df.columns.astype(str).tolist()
#     prefixes = sorted({c.split("_")[0] for c in cols if c.endswith("_change")})
#     prefixes = [p for p in prefixes if f"{p}_pct" in cols]
#     return prefixes

# def detect_usage_col(df: pd.DataFrame, prefix: str) -> Optional[str]:
#     for c in [f"{prefix}_usage_m3", f"{prefix}_usage_kw", f"{prefix}_usage_m3_mwh"]:
#         if c in df.columns:
#             return c
#     return None

# def display_cols_for_prefix(df: pd.DataFrame, prefix: str) -> List[str]:
#     change_col = f"{prefix}_change"
#     pct_col = f"{prefix}_pct"
#     usage_col = detect_usage_col(df, prefix)
#     df_cols = list(df.columns)
#     base = [c for c in ["brand", "building", "floor", "size_m2", "size_py"] if c in df_cols]
#     cat_block = []
#     for c in [f"{prefix}_previous", f"{prefix}_current", usage_col, change_col, pct_col]:
#         if c and c in df_cols:
#             cat_block.append(c)
#     extra = [c for c in df_cols[:10] if c not in base and c not in cat_block]
#     final = []
#     for c in base + extra + cat_block:
#         if c in df_cols and c not in final:
#             final.append(c)
#     return final

# def plot_hist_with_tails(s: pd.Series, bins: int, lo: float, hi: float, title: str):
#     vals = to_numeric_series(s).dropna().values
#     if vals.size == 0:
#         st.info(f"No numeric values for {title}")
#         return
#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.hist(vals, bins=bins, edgecolor="black", alpha=0.7)
#     xmin, xmax = float(np.nanmin(vals)), float(np.nanmax(vals))
#     if lo > xmin:
#         ax.axvspan(xmin, lo, alpha=0.25, zorder=3)
#     if hi < xmax:
#         ax.axvspan(hi, xmax, alpha=0.25, zorder=3)
#     ax.axvline(lo, linestyle="--", linewidth=2, zorder=4)
#     ax.axvline(hi, linestyle="--", linewidth=2, zorder=4)
#     ax.set_title(title)
#     st.pyplot(fig, clear_figure=True)

# def sort_df(df_part: pd.DataFrame, sort_cols, ascending, mode: str):
#     if mode != "extreme" or df_part.empty:
#         return df_part
#     return df_part.sort_values(sort_cols, ascending=ascending)

# def create_change_columns(df: pd.DataFrame) -> pd.DataFrame:
#     tasks = [
#         ("water_previous", "water_current", "water_usage_m3", "water_change", "water_pct"),
#         ("hwater_previous", "hwater_current", "hwater_usage_m3", "hwater_change", "hwater_pct"),
#         ("elect_previous", "elect_current", "elect_usage_kw", "elect_change", "elect_pct"),
#         ("heat_previous", "heat_current", "heat_usage_m3_mwh", "heat_change", "heat_pct"),
#     ]
#     df = df.copy()
#     for prev, curr, target, diff_name, pct_name in tasks:
#         if prev in df.columns and curr in df.columns and target in df.columns:
#             v_curr = to_numeric_series(df[curr])
#             v_prev = to_numeric_series(df[prev])
#             diff_series = v_curr - v_prev
#             pct_series = (diff_series.div(v_prev.replace(0, np.nan)) * 100).round(2)
#             pos = df.columns.get_loc(target)
#             if diff_name not in df.columns:
#                 df.insert(loc=pos + 1, column=diff_name, value=diff_series)
#             if pct_name not in df.columns:
#                 df.insert(loc=pos + 2, column=pct_name, value=pct_series)
#     return df

# def apply_header_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
#     target_lvl0 = "동별 건물 면적 현황"
#     keep_lvl1 = ["건물", "층수", "전용면적"]
#     df_drop = raw_df.copy()
#     # Dropping the first column safely by index
#     df_drop = df_drop.drop(df_drop.columns[0], axis=1)
#     if target_lvl0 in df_drop.columns.get_level_values(0):
#         all_lvl1_under_target = df_drop[target_lvl0].columns
#         cols_to_drop = [(target_lvl0,) + sub for sub in all_lvl1_under_target if sub[0] not in keep_lvl1]
#         df = df_drop.drop(columns=cols_to_drop, errors="ignore")
#     else:
#         df = df_drop
#     all_lvl0_cols = df.columns.get_level_values(0).unique()
#     if "전기 \n배율" in list(all_lvl0_cols):
#         pos = all_lvl0_cols.get_loc("전기 \n배율")
#         cols_to_keep = all_lvl0_cols[:pos]
#         df = df[cols_to_keep]
#         name_map = {old: str(old).replace(" ", "").replace("\n", "") for old in cols_to_keep}
#         df.columns = pd.MultiIndex.from_tuples([(name_map[col[0]], *col[1:]) for col in df.columns])
#     else:
#         lvl0_unique = df.columns.get_level_values(0).unique()
#         name_map = {old: str(old).replace(" ", "").replace("\n", "") for old in lvl0_unique}
#         df.columns = pd.MultiIndex.from_tuples([(name_map[col[0]], *col[1:]) for col in df.columns])
#     df = df.rename(columns={"급수 지침": "수도", "온수(급탕) 지침": "온수", "전기 지침": "전기", "FCU (냉,난방 지침)": "열요금"}, level=1)
#     df_drop = df.droplevel(0, axis=1)
#     cleaned_cols = [str(col[0]).replace(" ", "").replace("\n", "") if isinstance(col, tuple) else str(col).replace(" ", "").replace("\n", "") for col in df_drop.columns]
#     df_drop.columns = cleaned_cols
#     new_names = ["building", "floor", "size_m2", "size_py", "brand", "water_previous", "water_current", "water_usage_m3", "hwater_previous", "hwater_current", "hwater_usage_m3", "elect_previous", "elect_current", "elect_usage_kw", "heat_previous", "heat_current", "heat_usage_m3_mwh"]
#     if len(df_drop.columns) == len(new_names):
#         df_drop.columns = new_names
#     else:
#         rename_map = dict(zip(df_drop.columns, new_names))
#         df_drop = df_drop.rename(columns=rename_map)
#     return df_drop

# @st.cache_data(show_spinner=False)
# def read_upload(name: str, data: bytes) -> Dict[str, pd.DataFrame]:
#     lname = name.lower()
#     bio = io.BytesIO(data)
#     if lname.endswith(".csv"):
#         return {"__single__": pd.read_csv(bio)}
#     if lname.endswith(".parquet"):
#         return {"__single__": pd.read_parquet(bio)}
#     if lname.endswith(".xlsx") or lname.endswith(".xls"):
#         xls = pd.ExcelFile(bio)
#         out: Dict[str, pd.DataFrame] = {}
#         for sh in xls.sheet_names:
#             out[sh] = pd.read_excel(io.BytesIO(data), sheet_name=sh, header=[2, 3, 4])
#         return out
#     raise ValueError("Unsupported file type. Upload .csv, .xlsx, .xls, or .parquet.")

# # ----------------------------
# # App UI
# # ----------------------------
# st.title("Utility Outlier Dashboard")

# with st.sidebar:
#     st.header("Upload")
#     uploads = st.file_uploader("Upload CSV/XLSX/Parquet", type=["csv", "xlsx", "xls", "parquet"], accept_multiple_files=True)
#     st.divider()
#     st.header("Thresholds")
#     bins_change = st.slider("Bins (chg)", 5, 200, 30, 1)
#     bins_pct = st.slider("Bins (pct)", 5, 200, 30, 1)
#     q_change = st.slider("Change quantiles", 0.0, 1.0, (0.20, 0.80), 0.01)
#     q_pct = st.slider("Pct quantiles", 0.0, 1.0, (0.20, 0.80), 0.01)
#     row_sort_mode = st.selectbox("Row sort", ["keep", "extreme"], index=0)
#     debug = st.checkbox("Debug", value=False)

# if not uploads:
#     st.info("Upload at least one file.")
#     st.stop()

# files: Dict[str, Dict[str, pd.DataFrame]] = {}
# for f in uploads:
#     try:
#         files[f.name] = read_upload(f.name, f.getvalue())
#     except Exception as e:
#         st.error(f"Failed to read {f.name}: {e}")

# if not files:
#     st.stop()

# file_name = st.selectbox("Select file", list(files.keys()))
# sheet_name = st.selectbox("Select sheet", list(files[file_name].keys()), index=0)
# raw_df = files[file_name][sheet_name]

# if not isinstance(raw_df, pd.DataFrame):
#     st.error("Excel parsing failed.")
#     st.stop()

# try:
#     df = apply_header_rows(raw_df)
#     df = create_change_columns(df)
# except Exception as e:
#     st.error(f"Pipeline failed: {e}")
#     st.stop()

# for c in ["building", "brand"]:
#     if c not in df.columns:
#         st.error(f"Missing required column: {c}")
#         st.stop()

# dfs: Dict[str, pd.DataFrame] = {"df_all": df}
# for b in ["A", "B", "C", "D"]:
#     dfs[f"df_{b.lower()}"] = df[df["building"].astype(str).str.strip() == b].copy()

# df_key = st.selectbox("Select DF", list(dfs.keys()), index=0)
# cur_df = dfs[df_key].copy()

# # FIX 2: Replaced use_container_width with width="stretch"
# if debug:
#     st.write("df columns:", list(cur_df.columns))
#     # Using st_safe to prevent Arrow crashing on mixed types in debug view
#     st.dataframe(st_safe(cur_df.head(20)), width="stretch", hide_index=True)

# allowed = ["water", "hwater", "elect", "heat"]
# present = [p for p in allowed if f"{p}_change" in cur_df.columns]

# if not present:
#     st.error("No categories found.")
#     st.stop()

# prefix = st.selectbox("Category", present, index=0)
# change_col, pct_col = f"{prefix}_change", f"{prefix}_pct"

# cur_df[change_col] = to_numeric_series(cur_df[change_col])
# cur_df[pct_col] = to_numeric_series(cur_df[pct_col])

# both = cur_df[["brand", "building", change_col, pct_col]].dropna()
# if both.empty:
#     st.error(f"No numeric values in {prefix}.")
#     st.stop()

# s_change, s_pct = both[change_col], both[pct_col]
# q0c, q1c = sanitize(*q_change)
# q0p, q1p = sanitize(*q_pct)
# lo_c, hi_c = float(s_change.quantile(q0c)), float(s_change.quantile(q1c))
# lo_p, hi_p = float(s_pct.quantile(q0p)), float(s_pct.quantile(q1p))

# # Main UI Display
# h1, h2 = st.columns(2)
# with h1:
#     plot_hist_with_tails(s_change, bins_change, lo_c, hi_c, f"Change: {change_col}")
# with h2:
#     plot_hist_with_tails(s_pct, bins_pct, lo_p, hi_p, f"Pct: {pct_col}")

# cols_to_show = display_cols_for_prefix(cur_df, prefix)
# chg_low_mask, chg_high_mask = cur_df[change_col] < lo_c, cur_df[change_col] > hi_c
# pct_low_mask, pct_high_mask = cur_df[pct_col] < lo_p, cur_df[pct_col] > hi_p

# # FIX 3: Applied st_safe() to all dataframe displays to prevent Arrow Errors
# st.subheader("Change tails")
# a, b = st.columns(2)
# with a:
#     df_low = sort_df(cur_df.loc[chg_low_mask].copy(), [change_col], [True], row_sort_mode)
#     st.markdown(f"**Change LOW** ({len(df_low)})")
#     st.dataframe(st_safe(df_low[cols_to_show]), width="stretch", hide_index=True)
# with b:
#     df_high = sort_df(cur_df.loc[chg_high_mask].copy(), [change_col], [False], row_sort_mode)
#     st.markdown(f"**Change HIGH** ({len(df_high)})")
#     st.dataframe(st_safe(df_high[cols_to_show]), width="stretch", hide_index=True)

# st.subheader("Pct tails")
# a, b = st.columns(2)
# with a:
#     df_p_low = sort_df(cur_df.loc[pct_low_mask].copy(), [pct_col], [True], row_sort_mode)
#     st.markdown(f"**Pct LOW** ({len(df_p_low)})")
#     st.dataframe(st_safe(df_p_low[cols_to_show]), width="stretch", hide_index=True)
# with b:
#     df_p_high = sort_df(cur_df.loc[pct_high_mask].copy(), [pct_col], [False], row_sort_mode)
#     st.markdown(f"**Pct HIGH** ({len(df_p_high)})")
#     st.dataframe(st_safe(df_p_high[cols_to_show]), width="stretch", hide_index=True)

# # Overlap quadrants
# st.subheader("Overlap (Change × Pct quadrants)")
# q_LL = cur_df.loc[chg_low_mask & pct_low_mask].copy()
# q_HH = cur_df.loc[chg_high_mask & pct_high_mask].copy()

# r1, r2 = st.columns(2)
# with r1:
#     st.markdown(f"**Double Low** ({len(q_LL)})")
#     st.dataframe(st_safe(q_LL[cols_to_show]), width="stretch", hide_index=True)
# with r2:
#     st.markdown(f"**Double High** ({len(q_HH)})")
#     st.dataframe(st_safe(q_HH[cols_to_show]), width="stretch", hide_index=True)