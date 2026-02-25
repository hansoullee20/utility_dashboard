import streamlit as st
import pandas as pd
from typing import Dict

from data import read_upload, apply_header_rows, to_numeric_series, st_safe
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
            st.number_input("", 5, 200, step=1, key="bins_input", label_visibility="hidden", on_change=sync_bins_input)
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
            st.number_input("", 1, 50, step=1, key="tail_input", label_visibility="hidden", on_change=sync_tail_input)
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
    files: Dict[str, Dict[str, pd.DataFrame]] = {}
    for f in uploads:
        try:
            files[f.name] = read_upload(f.name, f.getvalue())
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")

    if not files:
        st.stop()

    file_name = st.selectbox("Select file", list(files.keys()))
    sheet_keys = list(files[file_name].keys())
    default_sheet = "검침 내역" if "검침 내역" in sheet_keys else sheet_keys[0]
    sheet_name = st.selectbox("Select sheet", sheet_keys, index=sheet_keys.index(default_sheet), key=f"sheet_{file_name}")
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

    # ---------------- 공실 filter ----------------
    has_gongshil = cur_df["brand"].astype(str).str.contains("공실", na=False).any()
    show_gongshil = st.toggle("공실 only", value=False, disabled=not has_gongshil)
    if show_gongshil:
        cur_df = cur_df[cur_df["brand"].astype(str).str.contains("공실", na=False)].copy()
        if cur_df.empty:
            st.warning("No 공실 entries for the current selection.")
            st.stop()

    bldg_tag = "all" if "All" in selected_buildings else "_".join(selected_buildings)
    df_key = f"bldg_{bldg_tag}"

    if debug:
        st.write("floors_filtered:", floors_filtered)
        st.write("active_floors:", active_floors)
        st.write("ref_df floors (unique):", sorted(ref_df["floor"].dropna().unique().tolist()) if "floor" in ref_df.columns else "no floor col")
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
    h1, h2 = st.columns(2)

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

    with h1:
        stats_change = plot_hist_with_tails(
            s_change, bins, float(lo_c), float(hi_c), f"Change: {change_col}"
        )
        if stats_change:
            render_stats(stats_change)
    with h2:
        stats_pct = plot_hist_with_tails(
            s_pct, bins, float(lo_p), float(hi_p), f"Pct: {pct_col}"
        )
        if stats_pct:
            render_stats(stats_pct)

    tab_change, tab_pct, tab_overlap, tab_ranking = st.tabs([
        "Quantitative Change", "Percentage Change", "Quadrant Analysis", "Brand Ranking"
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