import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
from typing import Dict

from data import get_sheet_names, read_sheet, apply_header_rows, to_numeric_series, st_safe, read_billing_sheet, BILLING_SHEET_NAME, EHP_OAC_SHEET_NAME, EHP_BILLING_SHEET_NAME, read_ehp_billing_sheet
from billing import render_billing_view
from ehp import render_ehp_view, render_ehp_billing_view
from features import (
    create_change_columns,
    aggregate_by_brand,
    split_brand_by_floor,
    get_simple_floors,
    sanitize,
    sort_df,
    display_cols_for_prefix,
    cols_brand_then_category,
    add_display_index,
    download_df_as_excel
)
from viz import plot_hist_with_tails


def main():
    st.set_page_config(page_title="Utility Analysis Dashboard", layout="wide")
    st.title("Utility Analysis Dashboard")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Upload")
        uploads = st.file_uploader(
            "Upload CSV/XLSX/Parquet",
            type=["csv", "xlsx", "xls", "xlsm", "parquet"],
            accept_multiple_files=True,
        )

        st.divider()
        st.header("⚙️ Settings")

        # ---- Presets ----
        preset_map = {"Default (20%)": 20, "Gentle (10%)": 10, "Dense (30%)": 30}

        def apply_preset():
            val = preset_map.get(st.session_state["preset_select"])
            if val is not None:
                st.session_state["tail"]       = val
                st.session_state["tail_input"] = val

        preset = st.selectbox(
            "⚡ Quick presets",
            ["Custom", "Default (20%)", "Gentle (10%)", "Dense (30%)"],
            index=0,
            key="preset_select",
            on_change=apply_preset,
            help="Pick a preset to quickly adjust tail percentage",
        )

        st.divider()

        # ---- Bins ----
        if "bins" not in st.session_state:
            st.session_state["bins"] = 50
        if "bins_input" not in st.session_state:
            st.session_state["bins_input"] = 50

        def sync_bins_slider():
            st.session_state["bins_input"] = st.session_state["bins"]

        def sync_bins_input():
            st.session_state["bins"] = st.session_state["bins_input"]

        b1, b2 = st.columns([3, 1])
        with b1:
            st.slider("Bins", 5, 200, step=1, key="bins", on_change=sync_bins_slider)
        with b2:
            st.number_input("Bins value", 5, 200, step=1, key="bins_input", label_visibility="hidden", on_change=sync_bins_input)
        bins = st.session_state["bins"]

        # ---- Tail % ----
        if "tail" not in st.session_state:
            st.session_state["tail"] = 20
        if "tail_input" not in st.session_state:
            st.session_state["tail_input"] = 20

        def sync_tail_slider():
            st.session_state["tail_input"] = st.session_state["tail"]

        def sync_tail_input():
            st.session_state["tail"] = st.session_state["tail_input"]

        t1, t2 = st.columns([3, 1])
        with t1:
            st.slider(
                "Tail %", 1, 50, step=1, key="tail",
                on_change=sync_tail_slider,
                help="Show the bottom N% and top N% of both change and pct values",
            )
        with t2:
            st.number_input("Tail value", 1, 50, step=1, key="tail_input", label_visibility="hidden", on_change=sync_tail_input)
        tail = st.session_state["tail"]

        # Convert tail % to quantile bounds (shared for both change and pct)
        q_change = (tail / 100.0, 1.0 - tail / 100.0)
        q_pct    = q_change

        st.divider()
        debug = st.checkbox("Debug", value=False)

    if not uploads:
        st.info("Upload at least one file.")
        st.stop()

    # ---------------- Load files ----------------
    file_map: Dict[str, bytes] = {}
    sheet_map: Dict[str, list] = {}
    for f in uploads:
        try:
            data = f.getvalue()
            file_map[f.name] = data
            sheet_map[f.name] = get_sheet_names(f.name, data)
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")

    if not file_map:
        st.stop()

    file_name = st.selectbox("Select file", list(file_map.keys()))
    all_sheet_keys = sheet_map[file_name]

    SUPPORTED_SHEETS = {"검침 내역", BILLING_SHEET_NAME, EHP_OAC_SHEET_NAME, EHP_BILLING_SHEET_NAME}
    sheet_keys = [s for s in all_sheet_keys if s.strip() in SUPPORTED_SHEETS]
    if not sheet_keys:
        st.warning("No supported sheets found in this file. Expected '검침 내역' or '수도광열비 부과 내역'.")
        st.stop()

    default_sheet = "검침 내역" if "검침 내역" in sheet_keys else sheet_keys[0]
    sheet_name = st.selectbox("Select sheet", sheet_keys, index=sheet_keys.index(default_sheet), key=f"sheet_{file_name}")

    # ── Billing sheet: separate pipeline ──────────────────────────────────────
    if sheet_name.strip() == BILLING_SHEET_NAME:
        try:
            billing_df = read_billing_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse billing sheet: {e}")
            st.stop()
        render_billing_view(billing_df)
        return

    # ── EHP sheet: separate pipeline ──────────────────────────────────────────
    if sheet_name.strip() == EHP_OAC_SHEET_NAME:
        render_ehp_view(file_name, file_map[file_name], sheet_name)
        return

    # ── EHP Billing sheet ─────────────────────────────────────────────────────
    if sheet_name.strip() == EHP_BILLING_SHEET_NAME:
        try:
            ehp_billing_df = read_ehp_billing_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse EHP billing sheet: {e}")
            st.stop()
        render_ehp_billing_view(ehp_billing_df)
        return
    # ─────────────────────────────────────────────────────────────────────────

    try:
        raw_df = read_sheet(file_name, file_map[file_name], sheet_name)
    except Exception as e:
        st.error(f"Failed to read {file_name}: {e}")
        st.stop()

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

    # ---------------- Building & Floor filters ----------------
    all_buildings = sorted(df["building"].dropna().unique().tolist())
    all_floors = get_simple_floors(df)

    building_options = ["All"] + all_buildings
    floor_options    = ["All"] + all_floors

    def on_building_change():
        sel = st.session_state["building_select"]
        if not sel:
            st.session_state["building_select"] = ["All"]
        elif sel[-1] == "All":
            st.session_state["building_select"] = ["All"]
        elif "All" in sel:
            st.session_state["building_select"] = [s for s in sel if s != "All"]

    def on_floor_change():
        sel = st.session_state["floor_select"]
        if not sel:
            st.session_state["floor_select"] = ["All"]
        elif sel[-1] == "All":
            st.session_state["floor_select"] = ["All"]
        elif "All" in sel:
            st.session_state["floor_select"] = [s for s in sel if s != "All"]

    fc1, fc2 = st.columns(2)
    with fc1:
        selected_buildings = st.multiselect(
            "Building", building_options, default=["All"],
            key="building_select", on_change=on_building_change,
        )
    with fc2:
        selected_floors = st.multiselect(
            "Floor", floor_options, default=["All"],
            key="floor_select", on_change=on_floor_change,
        )

    active_buildings = all_buildings if "All" in selected_buildings else selected_buildings
    active_floors    = all_floors    if "All" in selected_floors    else selected_floors

    # Always filter by building only — floor filtering is handled after aggregation
    ref_df = df[df["building"].isin(active_buildings)].copy()
    floors_filtered = "All" not in selected_floors

    if ref_df.empty:
        st.warning("No data for the selected building.")
        st.stop()

    # Aggregate by brand using all floors, then split if specific floors selected
    agg_df = aggregate_by_brand(ref_df)
    if floors_filtered:
        cur_df = split_brand_by_floor(agg_df, ref_df, active_floors)
    else:
        cur_df = agg_df

    if cur_df.empty:
        st.warning("No data for the selected floor combination.")
        st.stop()

    # ---------------- Per-size derived columns ----------------
    usage_cols = {
        "water_current":  ("water_usage_per_m2",  "water_usage_per_py"),
        "hwater_current": ("hwater_usage_per_m2", "hwater_usage_per_py"),
        "elect_current":  ("elect_usage_per_m2",  "elect_usage_per_py"),
        "heat_current":   ("heat_usage_per_m2",   "heat_usage_per_py"),
    }
    size_m2 = to_numeric_series(cur_df["size_m2"]).replace(0, float("nan")) if "size_m2" in cur_df.columns else None
    size_py = to_numeric_series(cur_df["size_py"]).replace(0, float("nan")) if "size_py" in cur_df.columns else None
    for usage_col, (per_m2_col, per_py_col) in usage_cols.items():
        if usage_col in cur_df.columns:
            usage = to_numeric_series(cur_df[usage_col])
            if size_m2 is not None:
                cur_df[per_m2_col] = (usage / size_m2).round(4)
            if size_py is not None:
                cur_df[per_py_col] = (usage / size_py).round(4)

    bldg_tag = "all" if "All" in selected_buildings else "_".join(selected_buildings)
    df_key = f"bldg_{bldg_tag}"

    if debug:
        st.write("floors_filtered:", floors_filtered)
        st.write("active_floors:", active_floors)
        st.write("ref_df floors (unique):", sorted(ref_df["floor"].dropna().unique().tolist()) if "floor" in ref_df.columns else "no floor col")
        st.dataframe(st_safe(cur_df.head(20)), width="stretch", hide_index=True)
        

    # ---------------- 공실 filter (logic; widget rendered just above tabs) ----------------
    has_gongshil = cur_df["brand"].astype(str).str.contains("공실", na=False).any()
    gongshil_mode = st.session_state.get("gongshil_mode_radio", "All")
    if gongshil_mode == "공실 only":
        cur_df = cur_df[cur_df["brand"].astype(str).str.contains("공실", na=False)].copy()
        if cur_df.empty:
            st.warning("No 공실 entries for the current selection.")
            st.stop()
    elif gongshil_mode == "Exclude 공실":
        cur_df = cur_df[~cur_df["brand"].astype(str).str.contains("공실", na=False)].copy()
        if cur_df.empty:
            st.warning("No entries remaining after excluding 공실.")
            st.stop()

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

    # ---------------- Quadrant masks (shared across tabs) ----------------
    chg_low_mask  = cur_df[change_col] <= lo_c
    chg_high_mask = cur_df[change_col] >= hi_c
    pct_low_mask  = cur_df[pct_col]    <= lo_p
    pct_high_mask = cur_df[pct_col]    >= hi_p

    q_HH = cur_df.loc[chg_high_mask & pct_high_mask].copy()
    q_HL = cur_df.loc[chg_high_mask & pct_low_mask].copy()
    q_LH = cur_df.loc[chg_low_mask  & pct_high_mask].copy()
    q_LL = cur_df.loc[chg_low_mask  & pct_low_mask].copy()

    q_HH = q_HH.sort_values([pct_col, change_col], ascending=False)
    q_HL = q_HL.sort_values([pct_col, change_col], ascending=False)
    q_LH = q_LH.sort_values([pct_col, change_col], ascending=False)
    q_LL = q_LL.sort_values([pct_col, change_col], ascending=False)

    # ---------------- Histograms (always visible) ----------------
    def render_stats(stats: dict):
        stats_row = pd.DataFrame([{
            "n":      stats["n"],
            "min":    round(stats["min"],    4),
            "p20":    round(stats["p20"],    4),
            "median": round(stats["median"], 4),
            "mean":   round(stats["mean"],   4),
            "std":    round(stats["std"],    4),
            "p80":    round(stats["p80"],    4),
            "max":    round(stats["max"],    4),
        }])
        st.dataframe(stats_row, hide_index=True, width="stretch")

    hist_view = st.radio(
        "Histogram", ["Quantitative Change", "Percentage Change"],
        horizontal=True, key="hist_view",
    )

    if hist_view == "Quantitative Change":
        plot_hist_with_tails(
            s_change, bins, float(lo_c), float(hi_c), f"Change: {change_col}",
            source_df=cur_df, val_col=change_col, key="hist_change",
            display_cols=cols_brand_then_category(cur_df, prefix, mode="change"),
            tail_pct=tail,
        )
    else:
        plot_hist_with_tails(
            s_pct, bins, float(lo_p), float(hi_p), f"Pct: {pct_col}",
            source_df=cur_df, val_col=pct_col, key="hist_pct",
            display_cols=cols_brand_then_category(cur_df, prefix, mode="pct"),
            tail_pct=tail,
        )

    st.radio(
        "공실 filter", ["All", "Exclude 공실", "공실 only"],
        index=0, horizontal=True,
        disabled=not has_gongshil,
        key="gongshil_mode_radio",
    )

    tab_change, tab_pct, tab_overlap, tab_ranking, tab_corr = st.tabs([
        "Quantitative Change", "Percentage Change", "Quadrant Analysis", "Brand Ranking", "Correlation"
    ])

    # ---------------- Change tab ----------------
    with tab_change:
        st.subheader(f"Quantitative Change — {change_col}")
        chg_label = f"{tail}%"
        chg_view_mode = st.radio(
            "Show", ["All", "Top", "Bottom"], index=0, horizontal=True, key="chg_view_mode"
        )

        chg_display_cols = cols_brand_then_category(cur_df, prefix, mode="change")

        if chg_view_mode == "All":
            chg_all = cur_df[chg_display_cols].dropna(subset=[change_col]).sort_values(change_col, ascending=False).copy()
            chg_all_view = add_display_index(chg_all)
            st.markdown(f"**All entries** — sorted high→low ({len(chg_all)})")
            st.dataframe(st_safe(chg_all_view), width="stretch", hide_index=True, height=35 * len(chg_all_view) + 38)
            download_df_as_excel(chg_all_view, filename=f"{df_key}_{prefix}_change_all.xlsx", sheet_name="change_all")

        elif chg_view_mode == "Top":
            st.markdown(f"**Top {chg_label} (>= {float(hi_c):.4g})** — sorted high→low ({len(chg_top)})")
            chg_top_view = add_display_index(chg_top[cols_brand_then_category(chg_top, prefix, mode="change")])
            st.dataframe(st_safe(chg_top_view), width="stretch", hide_index=True)
            download_df_as_excel(chg_top_view, filename=f"{df_key}_{prefix}_change_top.xlsx", sheet_name="change_top")

        else:  # Bottom
            st.markdown(f"**Bottom {chg_label} (<= {float(lo_c):.4g})** — sorted high→low ({len(chg_bot)})")
            chg_bot_view = add_display_index(chg_bot[cols_brand_then_category(chg_bot, prefix, mode="change")])
            st.dataframe(st_safe(chg_bot_view), width="stretch", hide_index=True)
            download_df_as_excel(chg_bot_view, filename=f"{df_key}_{prefix}_change_bottom.xlsx", sheet_name="change_bottom")

        chg_nan = cur_df[cur_df[change_col].isna()][chg_display_cols].copy()
        if not chg_nan.empty:
            st.divider()
            st.markdown(f"**No Data (NaN)** — missing quantitative change ({len(chg_nan)})")
            chg_nan_view = add_display_index(chg_nan)
            st.dataframe(st_safe(chg_nan_view), width="stretch", hide_index=True)
            download_df_as_excel(chg_nan_view, filename=f"{df_key}_{prefix}_change_nan.xlsx", sheet_name="change_nan")

    # ---------------- Pct tab ----------------
    with tab_pct:
        st.subheader(f"Percentage Change — {pct_col}")
        pct_label = f"{tail}%"
        pct_view_mode = st.radio(
            "Show", ["All", "Top", "Bottom"], index=0, horizontal=True, key="pct_view_mode"
        )

        pct_display_cols = cols_brand_then_category(cur_df, prefix, mode="pct")

        if pct_view_mode == "All":
            pct_all = cur_df[pct_display_cols].dropna(subset=[pct_col]).sort_values(pct_col, ascending=False).copy()
            pct_all_view = add_display_index(pct_all)
            st.markdown(f"**All entries** — sorted high→low ({len(pct_all)})")
            st.dataframe(st_safe(pct_all_view), width="stretch", hide_index=True, height=35 * len(pct_all_view) + 38)
            download_df_as_excel(pct_all_view, filename=f"{df_key}_{prefix}_pct_all.xlsx", sheet_name="pct_all")

        elif pct_view_mode == "Top":
            st.markdown(f"**Top {pct_label} (>= {float(hi_p):.4g})** — sorted high→low ({len(pct_top)})")
            pct_top_view = add_display_index(pct_top[cols_brand_then_category(pct_top, prefix, mode="pct")])
            st.dataframe(st_safe(pct_top_view), width="stretch", hide_index=True)
            download_df_as_excel(pct_top_view, filename=f"{df_key}_{prefix}_pct_top.xlsx", sheet_name="pct_top")

        else:  # Bottom
            st.markdown(f"**Bottom {pct_label} (<= {float(lo_p):.4g})** — sorted high→low ({len(pct_bot)})")
            pct_bot_view = add_display_index(pct_bot[cols_brand_then_category(pct_bot, prefix, mode="pct")])
            st.dataframe(st_safe(pct_bot_view), width="stretch", hide_index=True)
            download_df_as_excel(pct_bot_view, filename=f"{df_key}_{prefix}_pct_bottom.xlsx", sheet_name="pct_bottom")

        pct_nan = cur_df[cur_df[pct_col].isna()][pct_display_cols].copy()
        if not pct_nan.empty:
            st.divider()
            st.markdown(f"**No Data (NaN)** — missing percentage change ({len(pct_nan)})")
            pct_nan_view = add_display_index(pct_nan)
            st.dataframe(st_safe(pct_nan_view), width="stretch", hide_index=True)
            download_df_as_excel(pct_nan_view, filename=f"{df_key}_{prefix}_pct_nan.xlsx", sheet_name="pct_nan")

    # ---------------- Overlap tab ----------------
    with tab_overlap:
        st.subheader("Outlier Quadrant Analysis — Change × Pct Cross-Filter")

        st.markdown(f"**Critical Surge** — Change HIGH · Pct HIGH ({len(q_HH)})")
        q_cols = cols_brand_then_category(q_HH, prefix, mode="change")
        q_view = add_display_index(q_HH[q_cols])
        st.dataframe(st_safe(q_view), width="stretch", hide_index=True)
        download_df_as_excel(q_view, filename=f"{df_key}_{prefix}_overlap_HH.xlsx", sheet_name="overlap_HH")

        st.divider()

        st.markdown(f"**Large Base, Moderate Surge** — Change HIGH · Pct LOW ({len(q_HL)})")
        q_cols = cols_brand_then_category(q_HL, prefix, mode="change")
        q_view = add_display_index(q_HL[q_cols])
        st.dataframe(st_safe(q_view), width="stretch", hide_index=True)
        download_df_as_excel(q_view, filename=f"{df_key}_{prefix}_overlap_HL.xlsx", sheet_name="overlap_HL")

        st.divider()

        st.markdown(f"**Small Base, Sharp Drop** — Change LOW · Pct HIGH ({len(q_LH)})")
        q_cols = cols_brand_then_category(q_LH, prefix, mode="change")
        q_view = add_display_index(q_LH[q_cols])
        st.dataframe(st_safe(q_view), width="stretch", hide_index=True)
        download_df_as_excel(q_view, filename=f"{df_key}_{prefix}_overlap_LH.xlsx", sheet_name="overlap_LH")

        st.divider()

        st.markdown(f"**Stable / No Significant Change** — Change LOW · Pct LOW ({len(q_LL)})")
        q_cols = cols_brand_then_category(q_LL, prefix, mode="change")
        q_view = add_display_index(q_LL[q_cols])
        st.dataframe(st_safe(q_view), width="stretch", hide_index=True)
        download_df_as_excel(q_view, filename=f"{df_key}_{prefix}_overlap_LL.xlsx", sheet_name="overlap_LL")

        st.divider()

        all_quadrant_idx = q_HH.index.union(q_HL.index).union(q_LH.index).union(q_LL.index)
        q_normal = cur_df.loc[~cur_df.index.isin(all_quadrant_idx)].dropna(subset=[change_col, pct_col]).copy()
        q_normal = q_normal.sort_values(change_col, ascending=False)
        q_normal_cols = cols_brand_then_category(q_normal, prefix, mode="change")
        st.markdown(f"**Normal (non-outliers)** — not in any quadrant ({len(q_normal)})")
        q_normal_view = add_display_index(q_normal[q_normal_cols])
        st.dataframe(st_safe(q_normal_view), width="stretch", hide_index=True)
        download_df_as_excel(q_normal_view, filename=f"{df_key}_{prefix}_normal.xlsx", sheet_name="normal")

    # ---------------- Brand Ranking tab ----------------
    with tab_ranking:
        st.subheader(f"Brand Significance Ranking — {prefix}")

        valid_brands = cur_df["brand"].dropna().unique()
        rows = []
        for brand in valid_brands:
            bdf = cur_df[cur_df["brand"] == brand]
            n_total = len(bdf.dropna(subset=[change_col, pct_col]))
            if n_total == 0:
                continue

            n_HH = int((bdf.index.isin(q_HH.index)).sum())
            n_HL = int((bdf.index.isin(q_HL.index)).sum())
            n_LH = int((bdf.index.isin(q_LH.index)).sum())
            n_LL = int((bdf.index.isin(q_LL.index)).sum())

            quad_score = 4 * n_HH + 3 * n_HL + 2 * n_LH

            mean_change = float(to_numeric_series(bdf[change_col]).mean())
            mean_pct    = float(to_numeric_series(bdf[pct_col]).mean())

            rows.append({
                "brand":       brand,
                "quad_score":  quad_score,
                "mean_change": mean_change,
                "mean_pct":    mean_pct,
                "n_HH":        n_HH,
                "n_HL":        n_HL,
                "n_LH":        n_LH,
                "n_LL":        n_LL,
                "n_total":     n_total,
            })

        if not rows:
            st.info("No brand data available.")
        else:
            rank_df = pd.DataFrame(rows)

            # Normalize each component to [0, 1]
            def norm(s: pd.Series) -> pd.Series:
                rng = s.max() - s.min()
                return (s - s.min()) / rng if rng > 0 else pd.Series([0.0] * len(s), index=s.index)

            rank_df["norm_quad"]   = norm(rank_df["quad_score"])
            rank_df["norm_change"] = norm(rank_df["mean_change"])
            rank_df["norm_pct"]    = norm(rank_df["mean_pct"])
            rank_df["significance_score"] = (
                (rank_df["norm_quad"] + rank_df["norm_change"] + rank_df["norm_pct"]) / 3
            ).round(4)

            rank_df = rank_df.sort_values("significance_score", ascending=False).reset_index(drop=True)
            rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))

            # Merge in full brand detail columns from cur_df
            detail_cols = cols_brand_then_category(cur_df, prefix, mode="change")
            detail_cols = [c for c in detail_cols if c != "brand"]
            brand_details = cur_df[["brand"] + detail_cols].copy()
            rank_df = rank_df.merge(brand_details, on="brand", how="left")

            ranking_cols = ["significance_score", "n_HH", "n_HL", "n_LH", "n_LL", "n_total", "mean_change", "mean_pct"]
            front_cols   = ["Rank", "brand"] + detail_cols
            display_cols = front_cols + ranking_cols
            rank_view = rank_df[display_cols].copy()
            rank_view["mean_change"] = rank_view["mean_change"].round(2)
            rank_view["mean_pct"]    = rank_view["mean_pct"].round(2)

            with st.expander("How is the significance score calculated?"):
                ex = rank_df.iloc[1] if len(rank_df) > 1 else rank_df.iloc[0]
                ex_quad_raw  = int(ex["quad_score"])
                ex_norm_quad = round(float(ex["norm_quad"]), 4)
                ex_norm_chg  = round(float(ex["norm_change"]), 4)
                ex_norm_pct  = round(float(ex["norm_pct"]), 4)
                ex_score     = round(float(ex["significance_score"]), 4)
                ex_mean_chg  = round(float(ex["mean_change"]), 2)
                ex_mean_pct  = round(float(ex["mean_pct"]), 2)

                quad_min  = int(rank_df["quad_score"].min())
                quad_max  = int(rank_df["quad_score"].max())
                chg_min   = round(float(rank_df["mean_change"].min()), 2)
                chg_max   = round(float(rank_df["mean_change"].max()), 2)
                pct_min   = round(float(rank_df["mean_pct"].min()), 2)
                pct_max   = round(float(rank_df["mean_pct"].max()), 2)

                st.markdown(f"""
**Significance Score** ranges from 0 to 1 (1 = most alarming).
It is the average of three normalized components:

---

**1. Quadrant Weight Score**
Each brand's entries are checked against the outlier quadrants for `{prefix}`:

| Quadrant | Meaning | Weight |
|---|---|---|
| HH | Critical Surge — change HIGH & pct HIGH | 4 |
| HL | Large Base, Moderate Surge — change HIGH & pct LOW | 3 |
| LH | Small Base, Sharp Drop — change LOW & pct HIGH | 2 |
| LL | Stable / No Significant Change — change LOW & pct LOW | 0 |

Raw score = `4×n_HH + 3×n_HL + 2×n_LH` → range across brands: `{quad_min}` – `{quad_max}`

**2. Mean Quantitative Change** → range: `{chg_min}` – `{chg_max}`
Average of `{change_col}` across all entries for that brand. In the default view (All floors), each brand is aggregated into one row so this equals the brand's total change directly. When specific floors are selected, the brand is split into per-floor rows and this becomes the average across those floor rows.

**3. Mean Percentage Change** → range: `{pct_min}` – `{pct_max}`
Average of `{pct_col}` across all entries for that brand. Same logic as above — equals the brand's pct in the default view, averages across floor rows when floors are filtered.

All three normalized to [0, 1], then averaged:
`significance_score = (norm_quad + norm_change + norm_pct) / 3`

---

**Example — Rank {int(ex["Rank"])} brand: `{ex["brand"]}`**

| Component | Raw value | Normalized |
|---|---|---|
| Quadrant score (4×{int(ex["n_HH"])} + 3×{int(ex["n_HL"])} + 2×{int(ex["n_LH"])}) | {ex_quad_raw} | {ex_norm_quad} |
| Mean quantitative change | {ex_mean_chg} | {ex_norm_chg} |
| Mean percentage change | {ex_mean_pct} | {ex_norm_pct} |
| **Significance score** | | **({ex_norm_quad} + {ex_norm_chg} + {ex_norm_pct}) / 3 = {ex_score}** |

---

**Column reference:**
- `n_HH / n_HL / n_LH / n_LL` — count of entries in each quadrant
- `n_total` — total entries with valid change and pct values
- `mean_change` — mean quantitative change across brand's entries
- `mean_pct` — mean percentage change across brand's entries
""")
            st.dataframe(st_safe(rank_view), width="stretch", hide_index=True,
                         height=35 * len(rank_view) + 38)
            download_df_as_excel(rank_view, filename=f"{df_key}_{prefix}_brand_ranking.xlsx", sheet_name="brand_ranking")

    # ---------------- Correlation tab ----------------
    with tab_corr:
        st.subheader("Correlation — Interactive Scatter")

        _all_numeric = sorted([
            c for c in cur_df.columns
            if pd.api.types.is_numeric_dtype(cur_df[c])
            and cur_df[c].notna().any()
        ])

        _cat_cols = {}
        for _cat, _match in [
            ("m²",          lambda c: c == "size_m2"),
            ("평 (py)",      lambda c: c == "size_py"),
            ("Water",        lambda c: c.startswith("water_")),
            ("Hot Water",    lambda c: c.startswith("hwater_")),
            ("Electricity",  lambda c: c.startswith("elect_")),
            ("Heat",         lambda c: c.startswith("heat_")),
        ]:
            cols = [c for c in _all_numeric if _match(c)]
            if cols:
                _cat_cols[_cat] = cols
        categories = list(_cat_cols.keys())

        _SUFFIX_ORDER = [
            "change", "pct",
            "current", "previous",
            "usage_m3", "usage_kw", "usage_m3_mwh",
            "usage_per_m2", "usage_per_py",
        ]
        _SUFFIX_LABELS = {
            "previous":      "Previous Usage",
            "current":       "Current Usage",
            "usage_m3":      "Usage (m³)",
            "usage_kw":      "Usage (kWh)",
            "usage_m3_mwh":  "Usage (m³/MWh)",
            "usage_per_m2":  "Usage per m²",
            "usage_per_py":  "Usage per 평",
            "change":        "Quantitative Change",
            "pct":           "Percentage Change",
        }
        _COL_LABELS = {"size_m2": "m²", "size_py": "평 (py)"}
        def _col_label(col):
            if col in _COL_LABELS:
                return _COL_LABELS[col]
            for prefix in ["water_", "hwater_", "elect_", "heat_"]:
                if col.startswith(prefix):
                    suffix = col[len(prefix):]
                    return _SUFFIX_LABELS.get(suffix, suffix)
            return col
        def _col_sort_key(col):
            for prefix in ["water_", "hwater_", "elect_", "heat_"]:
                if col.startswith(prefix):
                    suffix = col[len(prefix):]
                    try:
                        return _SUFFIX_ORDER.index(suffix)
                    except ValueError:
                        return len(_SUFFIX_ORDER)
            return -1
        _cat_cols = {cat: sorted(cols, key=_col_sort_key) for cat, cols in _cat_cols.items()}

        if sum(len(v) for v in _cat_cols.values()) < 2:
            st.info("Not enough numeric columns for correlation.")
        else:
            xc1, xc2, yc1, yc2, cc3 = st.columns(5)
            with xc1:
                x_cat = st.selectbox("X Category", categories, index=0, key="corr_x_cat")
            with xc2:
                x_col = st.selectbox("X Column", _cat_cols[x_cat], index=0, key=f"corr_x_{x_cat}", format_func=_col_label)
            with yc1:
                _y_cat_default = min(1, len(categories) - 1)
                y_cat = st.selectbox("Y Category", categories, index=_y_cat_default, key="corr_y_cat")
            with yc2:
                y_col = st.selectbox("Y Column", _cat_cols[y_cat], index=0, key=f"corr_y_{y_cat}", format_func=_col_label)
            with cc3:
                color_by = st.selectbox("Color by", ["brand", "building"], index=0, key="corr_color")

            lc1, lc2, oc1, oc2 = st.columns([1, 1, 1, 4])
            with lc1:
                log_x = st.checkbox("Log X", value=False, key="corr_log_x")
            with lc2:
                log_y = st.checkbox("Log Y", value=False, key="corr_log_y")
            with oc1:
                remove_outliers = st.checkbox("Remove outliers", value=False, key="corr_remove_outliers")
            with oc2:
                iqr_k = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1, key="corr_iqr_k", disabled=not remove_outliers)

            if x_col == y_col:
                st.info("Please select different columns for X and Y axes.")
            else:
                hover_extra = [c for c in ["brand", "building", "size_m2", "size_py"] if c in cur_df.columns and c not in [x_col, y_col]]
                corr_df = cur_df[[x_col, y_col] + hover_extra].dropna(subset=[x_col, y_col]).copy()

                if remove_outliers:
                    for col in [x_col, y_col]:
                        q1, q3 = corr_df[col].quantile(0.25), corr_df[col].quantile(0.75)
                        iqr = q3 - q1
                        corr_df = corr_df[(corr_df[col] >= q1 - iqr_k * iqr) & (corr_df[col] <= q3 + iqr_k * iqr)]

                if corr_df.empty:
                    st.warning("No data with valid values for both selected columns.")
                else:
                    # Regression on (optionally log-transformed) values
                    x_vals = corr_df[x_col].values.astype(float)
                    y_vals = corr_df[y_col].values.astype(float)
                    if log_x:
                        mask = x_vals > 0
                        x_vals, y_vals = np.log10(x_vals[mask]), y_vals[mask]
                    if log_y:
                        mask = y_vals > 0
                        x_vals, y_vals = x_vals[mask], np.log10(y_vals[mask])

                    _BLDG_COLOR_MAP = {
                        "A": "#1f77b4",
                        "B": "#d62728",
                        "C": "#2ca02c",
                        "D": "#9467bd",
                    }
                    plot_df = corr_df.copy()
                    if color_by == "building" and "building" in plot_df.columns:
                        unique_bldgs = plot_df["building"].astype(str).unique()
                        color_map = {
                            b: _BLDG_COLOR_MAP.get(b, "#aaaaaa")
                            for b in unique_bldgs
                        }
                    else:
                        color_map = None
                    category_orders = (
                        {color_by: sorted(plot_df[color_by].astype(str).unique())}
                        if color_by in plot_df.columns else None
                    )
                    fig = px.scatter(
                        plot_df,
                        x=x_col,
                        y=y_col,
                        color=color_by if color_by in corr_df.columns else None,
                        hover_data=hover_extra,
                        log_x=log_x,
                        log_y=log_y,
                        title=f"{x_col} vs {y_col}",
                        color_discrete_map=color_map,
                        category_orders=category_orders,
                    )

                    if len(x_vals) >= 2:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

                        # Build trendline points in original (non-log) space for plotly
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
                        y_line = slope * x_line + intercept
                        if log_x:
                            x_line = 10 ** x_line
                        if log_y:
                            y_line = 10 ** y_line

                        fig.add_scatter(
                            x=x_line, y=y_line,
                            mode="lines",
                            name="Trendline",
                            line=dict(color="red", width=2, dash="dash"),
                        )

                        x_label = f"log10({x_col})" if log_x else x_col
                        y_label = f"log10({y_col})" if log_y else y_col
                        sign = "+" if intercept >= 0 else "-"
                        eq_text = f"y = {slope:.4f}x {sign} {abs(intercept):.4f}"
                        fig.add_annotation(
                            xref="paper", yref="paper",
                            x=0.01, y=0.99,
                            text=eq_text,
                            showarrow=False,
                            align="left",
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="red",
                            borderwidth=1,
                            font=dict(size=12, color="red"),
                        )
                        reg_row = pd.DataFrame([{
                            "equation":   f"{y_label} = {slope:.4f} × {x_label} + {intercept:.4f}",
                            "slope":      round(slope, 6),
                            "intercept":  round(intercept, 6),
                            "R²":         round(r_value ** 2, 6),
                            "p-value":    f"{p_value:.4e}",
                            "std_err":    round(std_err, 6),
                            "n":          len(x_vals),
                        }])

                    fig.update_layout(height=550)
                    st.plotly_chart(fig, use_container_width=True)

                    if len(x_vals) >= 2:
                        st.dataframe(reg_row, hide_index=True, width="stretch")

                        # Interpretation
                        r2 = r_value ** 2
                        direction = "positive" if slope > 0 else "negative"
                        direction_meaning = (
                            f"As **{x_col}** increases, **{y_col}** tends to **increase**."
                            if slope > 0 else
                            f"As **{x_col}** increases, **{y_col}** tends to **decrease**."
                        )

                        if r2 >= 0.7:
                            strength = "very strong"
                        elif r2 >= 0.5:
                            strength = "strong"
                        elif r2 >= 0.3:
                            strength = "moderate"
                        elif r2 >= 0.1:
                            strength = "weak"
                        else:
                            strength = "very weak"

                        if p_value < 0.001:
                            sig_text = "highly statistically significant (p < 0.001)"
                        elif p_value < 0.01:
                            sig_text = "very statistically significant (p < 0.01)"
                        elif p_value < 0.05:
                            sig_text = "statistically significant (p < 0.05)"
                        else:
                            sig_text = "**not statistically significant** (p ≥ 0.05) — treat this result with caution"

                        st.markdown(f"""
**Interpretation**

There is a **{strength} {direction} linear relationship** between {x_col} and {y_col} (R² = {r2:.4f}). {direction_meaning}

The model explains **{r2*100:.1f}%** of the variance in {y_col}. The relationship is {sig_text}.

For every 1-unit increase in {x_col}, {y_col} changes by **{slope:.4f}** on average.
""")

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