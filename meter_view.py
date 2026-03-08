"""meter_view.py — 검침 내역 analysis view (pipeline + all tabs)."""
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from typing import Dict

from data import (
    read_sheet, apply_header_rows, get_billing_period,
    to_numeric_series, st_safe,
    read_billing_sheet, get_sheet_names,
    BILLING_SHEET_NAME,
)
from features import (
    create_change_columns, aggregate_by_brand, split_brand_by_floor,
    get_simple_floors, sanitize, sort_df, display_cols_for_prefix,
    cols_brand_then_category, add_display_index, download_df_as_excel,
)
from viz import plot_hist_with_tails
from report import generate_report_pdf


def render_meter_view(
    file_name: str,
    file_map: Dict[str, bytes],
    sheet_name: str,
    bins: int,
    tail: int,
    q_change: tuple,
    q_pct: tuple,
    debug: bool,
) -> None:
    """Full 검침 내역 analysis: pipeline, filters, histograms, 5 tabs, reconciliation."""
    try:
        raw_df = read_sheet(file_name, file_map[file_name], sheet_name)
    except Exception as e:
        st.error(f"Failed to read {file_name}: {e}")
        st.stop()

    billing_period = get_billing_period(file_name, file_map[file_name])  # e.g. "2026년 1월"

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

    # ---------------- Backward meter reading detection ----------------
    _meter_pairs = [
        ("water",  "water_previous",  "water_current",  "m³"),
        ("hwater", "hwater_previous", "hwater_current", "m³"),
        ("elect",  "elect_previous",  "elect_current",  "kWh"),
        ("heat",   "heat_previous",   "heat_current",   "m³/MWh"),
    ]
    backward_rows = []
    for prefix, prev_col, curr_col, unit in _meter_pairs:
        if prev_col not in df.columns or curr_col not in df.columns:
            continue
        prev_s = to_numeric_series(df[prev_col])
        curr_s = to_numeric_series(df[curr_col])
        mask = curr_s.notna() & prev_s.notna() & (curr_s < prev_s)
        for idx in df[mask].index:
            backward_rows.append({
                "Brand":    df.at[idx, "brand"]    if "brand"    in df.columns else "",
                "Building": df.at[idx, "building"] if "building" in df.columns else "",
                "Floor":    df.at[idx, "floor"]    if "floor"    in df.columns else "",
                "Utility":  prefix,
                "Previous": round(float(prev_s.at[idx]), 2),
                "Current":  round(float(curr_s.at[idx]), 2),
                "Drop":     round(float(curr_s.at[idx] - prev_s.at[idx]), 2),
            })
    if backward_rows:
        with st.expander(f"⚠️ Data Quality — {len(backward_rows)} backward meter reading(s) detected", expanded=True):
            st.warning(
                "The following tenants have a **current meter reading lower than the previous reading**. "
                "This is physically impossible without a meter reset and likely indicates a data entry error. "
                "These rows are still included in the analysis but their change values will appear negative."
            )
            st.dataframe(pd.DataFrame(backward_rows), hide_index=True, use_container_width=True)

    # ---------------- Filters ─────────────────────────────────────────────────
    all_buildings = sorted(df["building"].dropna().unique().tolist())
    all_floors = get_simple_floors(df)

    building_options = ["All"] + all_buildings
    floor_options    = ["All"] + all_floors

    allowed = ["water", "hwater", "elect", "heat"]
    present_all = [p for p in allowed if f"{p}_change" in df.columns]
    _CATEGORY_LABELS = {"water": "💧 수도", "hwater": "🌡️ 온수", "elect": "⚡ 전기", "heat": "🔥 난방"}
    has_gongshil_any = df["brand"].astype(str).str.contains("공실", na=False).any()

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

    fc1, fc2, fc3, fc4 = st.columns([2, 2, 1, 2])
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
    with fc3:
        prefix = st.selectbox("Category", present_all, format_func=lambda x: _CATEGORY_LABELS.get(x, x))
    with fc4:
        gongshil_mode = st.radio(
            "공실", ["All", "Exclude 공실", "공실 only"],
            index=0, horizontal=True,
            disabled=not has_gongshil_any,
            key="gongshil_mode_radio",
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

    # ---------------- Apply 공실 filter ----------------
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

    # ---------------- Category column setup ----------------
    change_col, pct_col = f"{prefix}_change", f"{prefix}_pct"

    present = [p for p in allowed if f"{p}_change" in cur_df.columns]
    if not present:
        st.error("No utility categories found.")
        st.stop()
    if change_col not in cur_df.columns or pct_col not in cur_df.columns:
        st.error(f"Category '{prefix}' not available for the current filter.")
        st.stop()

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

    # ---------------- Summary KPI bar ----------------
    _status_counts = {"Critical": set(), "Watch": set(), "Alert": set(),
                      "Stable": set(), "No Data": set()}
    _q = tail / 100.0
    for _p in present:
        _cc = f"{_p}_change"; _pc = f"{_p}_pct"
        if _cc not in cur_df.columns or _pc not in cur_df.columns:
            continue
        _sc = to_numeric_series(cur_df[_cc]); _sp = to_numeric_series(cur_df[_pc])
        _valid = _sc.notna() & _sp.notna()
        if not _valid.any():
            continue
        _hi_c = float(_sc[_valid].quantile(1 - _q)); _lo_c = float(_sc[_valid].quantile(_q))
        _hi_p = float(_sp[_valid].quantile(1 - _q)); _lo_p = float(_sp[_valid].quantile(_q))
        for _idx in cur_df.index:
            _ch = _sc.at[_idx] if _idx in _sc.index else float("nan")
            _pt = _sp.at[_idx] if _idx in _sp.index else float("nan")
            _key = (str(cur_df.at[_idx, "brand"]), str(cur_df.at[_idx, "building"]))
            import math
            if math.isnan(_ch) or math.isnan(_pt):
                _status_counts["No Data"].add(_key)
            elif _ch >= _hi_c and _pt >= _hi_p:
                _status_counts["Critical"].add(_key)
            elif _ch >= _hi_c or _pt >= _hi_p:
                _status_counts["Watch"].add(_key)
            elif _ch <= _lo_c and _pt <= _lo_p:
                _status_counts["Stable"].add(_key)
            elif _ch <= _lo_c or _pt <= _lo_p:
                _status_counts["Alert"].add(_key)

    _n_critical  = len(_status_counts["Critical"])
    _n_watch     = len(_status_counts["Watch"])
    _n_alert     = len(_status_counts["Alert"])
    _n_vacancy   = int(cur_df["brand"].astype(str).str.contains("공실", na=False).sum())
    _n_backward  = len(backward_rows)

    _k1, _k2, _k3, _k4, _k5 = st.columns(5)
    _k1.metric("Tenants", len(cur_df))
    _k2.metric("🔴 Critical",  _n_critical,  delta=None if _n_critical  == 0 else f"across any utility")
    _k3.metric("🟠 Watch",     _n_watch,     delta=None if _n_watch     == 0 else f"elevated usage")
    _k4.metric("🟡 Alert",     _n_alert,     delta=None if _n_alert     == 0 else f"sharp % rise")
    _k5.metric("🏚 공실 / ⚠ Data", f"{_n_vacancy} / {_n_backward}")
    st.divider()

    # ---------------- Summary Report download ─────────────────────────────────
    with st.expander("Download Summary Report", expanded=False):
        st.caption(
            "Generates a business-ready PDF report covering all utility types — "
            "with charts, plain-language explanations, and flagged tenants."
        )
        report_lang = st.radio(
            "Report language / 보고서 언어",
            ["한국어", "English"],
            horizontal=True,
            key="report_lang",
        )
        lang_code = "ko" if report_lang == "한국어" else "en"

        if st.button("Generate Report", key="gen_report"):
            report_context = {
                "date":      str(date.today()),
                "period":    billing_period or str(date.today()),
                "buildings": ", ".join(active_buildings) if active_buildings else "All",
                "floors":    ", ".join(active_floors) if floors_filtered else "All",
            }
            with st.spinner("보고서 생성 중…" if lang_code == "ko" else "Building PDF report…"):
                report_bytes = generate_report_pdf(
                    cur_df, present, tail,
                    context=report_context, lang=lang_code,
                )
            bldg_tag_rpt = "all" if "All" in selected_buildings else "_".join(selected_buildings)
            st.download_button(
                label="PDF 다운로드" if lang_code == "ko" else "Download PDF Report",
                data=report_bytes,
                file_name=f"utility_report_{bldg_tag_rpt}_{date.today()}.pdf",
                mime="application/pdf",
                key="dl_report",
            )

    # ---------------- Histograms ─────────────────────────────────────────────
    hist_layout = st.radio(
        "Histogram view",
        ["Side by Side", "Change only", "% Change only"],
        horizontal=True,
        key="hist_layout",
    )

    if hist_layout == "Side by Side":
        hc1, hc2 = st.columns(2)
        with hc1:
            plot_hist_with_tails(
                s_change, bins, float(lo_c), float(hi_c), f"Change: {change_col}",
                source_df=cur_df, val_col=change_col, key="hist_change",
                display_cols=cols_brand_then_category(cur_df, prefix, mode="change"),
                tail_pct=tail,
            )
        with hc2:
            plot_hist_with_tails(
                s_pct, bins, float(lo_p), float(hi_p), f"Pct: {pct_col}",
                source_df=cur_df, val_col=pct_col, key="hist_pct",
                display_cols=cols_brand_then_category(cur_df, prefix, mode="pct"),
                tail_pct=tail,
            )
    elif hist_layout == "Change only":
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

        # ── helper: map a column back to its display category ──────────────
        def _find_cat(col):
            for cat, cols in _cat_cols.items():
                if col in cols:
                    return cat
            return None

        # ── helper: which utility prefix owns a column ─────────────────────
        _UTIL_PREFIXES = ["water_", "hwater_", "elect_", "heat_"]
        def _col_util(col):
            for p in _UTIL_PREFIXES:
                if col.startswith(p):
                    return p
            return col   # size columns are their own "group"

        if sum(len(v) for v in _cat_cols.values()) < 2:
            st.info("Not enough numeric columns for correlation.")
        else:
            # ── AUTO-DISCOVERY ────────────────────────────────────────────
            st.subheader("Auto-Discover Correlations")

            # Columns worth scanning: change, pct, current + size
            _SCAN_SUFFIXES = {"change", "pct", "current"}
            _scan_cols = [
                c for c in _all_numeric
                if any(c.endswith(f"_{s}") for s in _SCAN_SUFFIXES)
                or c in ("size_m2", "size_py")
            ]

            _disc_rows = []
            for _i, _ca in enumerate(_scan_cols):
                for _cb in _scan_cols[_i + 1:]:
                    if _col_util(_ca) == _col_util(_cb):   # skip same-utility pairs
                        continue
                    _dp = cur_df[[_ca, _cb]].dropna()
                    if len(_dp) < 5:
                        continue
                    _r, _p = stats.pearsonr(_dp[_ca].values, _dp[_cb].values)
                    _disc_rows.append({
                        "X":         _ca,
                        "Y":         _cb,
                        "r":         round(_r, 3),
                        "R²":        round(_r ** 2, 3),
                        "p-value":   round(_p, 4),
                        "n":         len(_dp),
                        "Direction": "positive" if _r > 0 else "negative",
                        "Strength":  (
                            "Strong"   if abs(_r) >= 0.6 else
                            "Moderate" if abs(_r) >= 0.35 else
                            "Weak"
                        ),
                    })

            if not _disc_rows:
                st.info("Not enough cross-category data to run discovery.")
            else:
                _disc_df = (
                    pd.DataFrame(_disc_rows)
                    .sort_values("R²", ascending=False)
                    .reset_index(drop=True)
                )

                # Filter controls
                _dc1, _dc2, _dc3 = st.columns([2, 2, 3])
                with _dc1:
                    _min_r2 = st.slider(
                        "Min R²", 0.0, 1.0, 0.05, 0.05, key="disc_min_r2",
                        help="Only show pairs where R² is at least this value",
                    )
                with _dc2:
                    _show_nonsig = st.checkbox(
                        "Include p ≥ 0.05", value=False, key="disc_show_nonsig",
                        help="Also show pairs that are not statistically significant",
                    )
                with _dc3:
                    _strength_filter = st.multiselect(
                        "Strength filter", ["Strong", "Moderate", "Weak"],
                        default=["Strong", "Moderate"], key="disc_strength",
                    )

                _shown = _disc_df[
                    (_disc_df["R²"] >= _min_r2) &
                    (_disc_df["Strength"].isin(_strength_filter)) &
                    (_show_nonsig | (_disc_df["p-value"] < 0.05))
                ].reset_index(drop=True)

                if _shown.empty:
                    st.info("No pairs match the current filters.")
                else:
                    st.caption(
                        f"{len(_shown)} pair(s) found · "
                        "Select a row to load it into the scatter below"
                    )
                    _disc_event = st.dataframe(
                        _shown,
                        hide_index=True,
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="single-row",
                        column_config={
                            "r":       st.column_config.NumberColumn("r",   format="%.3f"),
                            "R²":      st.column_config.NumberColumn("R²",  format="%.3f"),
                            "p-value": st.column_config.NumberColumn("p",   format="%.4f"),
                            "X":       st.column_config.TextColumn("X Column",  width="medium"),
                            "Y":       st.column_config.TextColumn("Y Column",  width="medium"),
                        },
                    )

                    # Load selected row into scatter selectors
                    _sel_rows = (
                        _disc_event.selection.rows
                        if _disc_event and hasattr(_disc_event, "selection")
                        else []
                    )
                    if _sel_rows:
                        _sel = _shown.iloc[_sel_rows[0]]
                        _xc  = _sel["X"];  _yc = _sel["Y"]
                        _xct = _find_cat(_xc); _yct = _find_cat(_yc)
                        if _xct and _yct:
                            st.session_state["corr_x_cat"]        = _xct
                            st.session_state[f"corr_x_{_xct}"]   = _xc
                            st.session_state["corr_y_cat"]        = _yct
                            st.session_state[f"corr_y_{_yct}"]   = _yc
                            st.success(
                                f"Loaded **{_col_label(_xc)}** vs **{_col_label(_yc)}** "
                                f"(R² = {_sel['R²']:.3f}) — scroll down to scatter"
                            )

                # Correlation heatmap of change + pct columns
                _hm_cols = [c for c in _scan_cols if c in cur_df.columns
                            and (c.endswith("_change") or c.endswith("_pct"))]
                if len(_hm_cols) >= 3:
                    _hm_data = cur_df[_hm_cols].dropna()
                    if len(_hm_data) >= 3:
                        _corr_mat = _hm_data.corr()
                        _labels   = [_col_label(c) for c in _corr_mat.columns]
                        _fig_hm   = px.imshow(
                            _corr_mat,
                            x=_labels, y=_labels,
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            text_auto=".2f",
                            title="Correlation Matrix — Change & % Change",
                            aspect="auto",
                        )
                        _fig_hm.update_layout(
                            height=420,
                            margin=dict(l=10, r=10, t=50, b=10),
                            coloraxis_colorbar=dict(title="r"),
                            font=dict(size=11),
                        )
                        _fig_hm.update_traces(textfont_size=10)
                        st.plotly_chart(_fig_hm, use_container_width=True)

            st.divider()
            st.subheader("Manual Scatter")

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

    # ---------------- Billing ↔ Meter reconciliation ─────────────────────────
    _sheet_names = get_sheet_names(file_name, file_map[file_name])
    if BILLING_SHEET_NAME in _sheet_names:
        with st.expander("Billing ↔ Meter Reconciliation", expanded=False):
            st.caption(
                "Compares the billing sheet (수도광열비 부과 내역) against meter readings. "
                "Flags tenants present in one source but missing in the other."
            )
            try:
                _bill_df = read_billing_sheet(file_name, file_map[file_name], BILLING_SHEET_NAME)
                _bill_key = (
                    _bill_df[["brand", "building"]]
                    .dropna(subset=["brand"])
                    .assign(brand=lambda d: d["brand"].astype(str).str.strip(),
                            building=lambda d: d["building"].astype(str).str.strip())
                    .drop_duplicates()
                )
                _meter_key = (
                    cur_df[["brand", "building"]]
                    .assign(brand=lambda d: d["brand"].astype(str).str.strip(),
                            building=lambda d: d["building"].astype(str).str.strip())
                    .drop_duplicates()
                )
                _bill_set  = set(zip(_bill_key["brand"],  _bill_key["building"]))
                _meter_set = set(zip(_meter_key["brand"], _meter_key["building"]))

                _billed_not_metered = sorted(_bill_set  - _meter_set)
                _metered_not_billed = sorted(_meter_set - _bill_set)

                _rc1, _rc2 = st.columns(2)
                with _rc1:
                    st.markdown(f"**Billed but no meter reading** — {len(_billed_not_metered)}")
                    if _billed_not_metered:
                        st.dataframe(
                            pd.DataFrame(_billed_not_metered, columns=["Brand", "Building"]),
                            hide_index=True, use_container_width=True,
                        )
                    else:
                        st.success("All billed tenants have meter readings.")
                with _rc2:
                    st.markdown(f"**Metered but not billed** — {len(_metered_not_billed)}")
                    if _metered_not_billed:
                        st.dataframe(
                            pd.DataFrame(_metered_not_billed, columns=["Brand", "Building"]),
                            hide_index=True, use_container_width=True,
                        )
                    else:
                        st.success("All metered tenants appear on the billing sheet.")

                # Shared tenants: compare billed total vs any zero-usage flag
                _shared = _bill_set & _meter_set
                _zero_billed = []
                for _br, _bl in sorted(_shared):
                    _brow = _bill_df[
                        (_bill_df["brand"].astype(str).str.strip() == _br) &
                        (_bill_df["building"].astype(str).str.strip() == _bl)
                    ]
                    _mrow = cur_df[
                        (cur_df["brand"].astype(str).str.strip() == _br) &
                        (cur_df["building"].astype(str).str.strip() == _bl)
                    ]
                    if _brow.empty or _mrow.empty:
                        continue
                    _total = to_numeric_series(_brow["total"].iloc[[0]]).iloc[0] if "total" in _brow.columns else float("nan")
                    _has_usage = any(
                        not pd.isna(to_numeric_series(_mrow[f"{_px}_current"]).iloc[0])
                        and to_numeric_series(_mrow[f"{_px}_current"]).iloc[0] > 0
                        for _px in present
                        if f"{_px}_current" in _mrow.columns
                    )
                    if not pd.isna(_total) and _total > 0 and not _has_usage:
                        _zero_billed.append({"Brand": _br, "Building": _bl, "Billed Total (₩)": f"{int(_total):,}"})

                if _zero_billed:
                    st.markdown(f"**Billed non-zero but zero meter usage** — {len(_zero_billed)}")
                    st.dataframe(pd.DataFrame(_zero_billed), hide_index=True, use_container_width=True)

            except Exception as _e:
                st.warning(f"Reconciliation failed: {_e}")
