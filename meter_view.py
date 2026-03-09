"""meter_view.py — 검침 내역 analysis view (pipeline + all tabs)."""
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict

from data import (
    read_sheet, get_billing_period,
    to_numeric_series, st_safe,
    get_sheet_names,
    BILLING_SHEET_NAME, EHP_OAC_SHEET_NAME,
)
from features import (
    apply_header_rows, create_change_columns, aggregate_by_brand, split_brand_by_floor,
    get_simple_floors, sanitize,
    cols_brand_then_category, add_display_index, download_df_as_excel,
)
from viz import plot_hist_with_tails
from report import generate_report_pdf
from tab_corr import render_corr_tab
from tab_efficiency import render_efficiency_tab
from tab_reconciliation import render_reconciliation
from lang import t

_ROW_PX = 35   # pixels per data row in full-height dataframes
_HDR_PX = 38   # pixels for the header row


def _full_height(n_rows: int) -> int:
    return _ROW_PX * n_rows + _HDR_PX


def load_raw_meter_df(file_name: str, file_map: Dict[str, bytes], sheet_name: str) -> pd.DataFrame:
    """Load meter sheet and return pre-aggregation row-level DataFrame."""
    raw_df = read_sheet(file_name, file_map[file_name], sheet_name)
    df = apply_header_rows(raw_df)
    df = create_change_columns(df)
    df["building"] = df["building"].astype(str).str.strip()
    return df[df["building"].isin({"A", "B", "C", "D"})].copy()


def load_meter_df(file_name: str, file_map: Dict[str, bytes], sheet_name: str) -> pd.DataFrame:
    """Load and aggregate meter data without rendering any UI.

    Returns an all-buildings, all-floors aggregated DataFrame with per-area
    columns (*_usage_per_m2, *_usage_per_py) derived from size_m2/size_py.
    """
    raw_df = read_sheet(file_name, file_map[file_name], sheet_name)
    df = apply_header_rows(raw_df)
    df = create_change_columns(df)
    df["building"] = df["building"].astype(str).str.strip()
    df = df[df["building"].isin({"A", "B", "C", "D"})].copy()
    df = aggregate_by_brand(df)

    # Derive per-area columns (same logic as render_meter_view)
    _usage_cols = {
        "water_current":  ("water_usage_per_m2",  "water_usage_per_py"),
        "hwater_current": ("hwater_usage_per_m2", "hwater_usage_per_py"),
        "elect_current":  ("elect_usage_per_m2",  "elect_usage_per_py"),
        "heat_current":   ("heat_usage_per_m2",   "heat_usage_per_py"),
    }
    size_m2 = to_numeric_series(df["size_m2"]).replace(0, float("nan")) if "size_m2" in df.columns else None
    size_py = to_numeric_series(df["size_py"]).replace(0, float("nan")) if "size_py" in df.columns else None
    for usage_col, (per_m2_col, per_py_col) in _usage_cols.items():
        if usage_col in df.columns:
            usage = to_numeric_series(df[usage_col])
            if size_m2 is not None:
                df[per_m2_col] = (usage / size_m2).round(4)
            if size_py is not None:
                df[per_py_col] = (usage / size_py).round(4)

    return df


def render_meter_view(
    file_name: str,
    file_map: Dict[str, bytes],
    sheet_name: str,
    bins: int,
    tail: int,
    q: tuple,
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
        with st.expander(f"{t('backward_expander')} ({len(backward_rows)})", expanded=True):
            st.warning(t("backward_warning"))
            st.dataframe(pd.DataFrame(backward_rows), hide_index=True, use_container_width=True)

    # ---------------- Filters ─────────────────────────────────────────────────
    all_buildings = sorted(df["building"].dropna().unique().tolist())
    all_floors = get_simple_floors(df)

    building_options = ["All"] + all_buildings
    floor_options    = ["All"] + all_floors

    allowed = ["water", "hwater", "elect", "heat"]
    present_all = [p for p in allowed if f"{p}_change" in df.columns]
    _CATEGORY_LABELS = {"water": t("cat_water"), "hwater": t("cat_hwater"), "elect": t("cat_elect"), "heat": t("cat_heat")}
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

    _VACANCY_OPTS = ["All", "Exclude Vacancy", "Vacancy Only"]
    _VACANCY_LABELS = {
        "All":             {"ko": "전체",     "en": "All"},
        "Exclude Vacancy": {"ko": "공실 제외", "en": "Exclude Vacancy"},
        "Vacancy Only":    {"ko": "공실만",    "en": "Vacancy Only"},
    }
    fc1, fc2, fc3, fc4 = st.columns([2, 2, 1, 2])
    with fc1:
        selected_buildings = st.multiselect(
            t("building"), building_options, default=["All"],
            key="building_select", on_change=on_building_change,
        )
    with fc2:
        selected_floors = st.multiselect(
            t("floor"), floor_options, default=["All"],
            key="floor_select", on_change=on_floor_change,
        )
    with fc3:
        prefix = st.selectbox(t("category"), present_all, format_func=lambda x: _CATEGORY_LABELS.get(x, x))
    with fc4:
        import streamlit as _st
        _lang = _st.session_state.get("lang", "ko")
        gongshil_mode = st.radio(
            t("vacancy"), _VACANCY_OPTS,
            format_func=lambda x: _VACANCY_LABELS[x][_lang],
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
        st.warning(t("no_data_building"))
        st.stop()

    # Aggregate by brand using all floors, then split if specific floors selected
    agg_df = aggregate_by_brand(ref_df)
    if floors_filtered:
        cur_df = split_brand_by_floor(agg_df, ref_df, active_floors)
    else:
        cur_df = agg_df

    if cur_df.empty:
        st.warning(t("no_data_floor"))
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
    if gongshil_mode == "Vacancy Only":
        cur_df = cur_df[cur_df["brand"].astype(str).str.contains("공실", na=False)].copy()
        if cur_df.empty:
            st.warning("No 공실 entries for the current selection.")
            st.stop()
    elif gongshil_mode == "Exclude Vacancy":
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
        st.error(t("no_numeric"))
        st.stop()

    s_change = valid[change_col]
    s_pct = valid[pct_col]

    # ---------------- Thresholds ----------------
    q0c, q1c = sanitize(*q)
    q0p, q1p = sanitize(*q)

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
    _status_counts: dict[str, set] = {"Critical": set(), "Watch": set(), "Alert": set(),
                                      "Stable": set(), "No Data": set()}
    _q = tail / 100.0
    _keys = list(zip(cur_df["brand"].astype(str).values, cur_df["building"].astype(str).values))
    for _p in present:
        _cc = f"{_p}_change"; _pc = f"{_p}_pct"
        if _cc not in cur_df.columns or _pc not in cur_df.columns:
            continue
        _sc = to_numeric_series(cur_df[_cc]).values
        _sp = to_numeric_series(cur_df[_pc]).values
        _valid = ~(np.isnan(_sc) | np.isnan(_sp))
        if not _valid.any():
            continue
        _hi_c = float(np.quantile(_sc[_valid], 1 - _q))
        _lo_c = float(np.quantile(_sc[_valid], _q))
        _hi_p = float(np.quantile(_sp[_valid], 1 - _q))
        _lo_p = float(np.quantile(_sp[_valid], _q))

        _nan_m      = ~_valid
        _critical_m = _valid & (_sc >= _hi_c) & (_sp >= _hi_p)
        _watch_m    = _valid & ~_critical_m & ((_sc >= _hi_c) | (_sp >= _hi_p))
        _stable_m   = _valid & (_sc <= _lo_c) & (_sp <= _lo_p)
        _alert_m    = _valid & ~_stable_m & ((_sc <= _lo_c) | (_sp <= _lo_p))

        _status_counts["No Data"].update(k for k, m in zip(_keys, _nan_m)      if m)
        _status_counts["Critical"].update(k for k, m in zip(_keys, _critical_m) if m)
        _status_counts["Watch"].update(k for k, m in zip(_keys, _watch_m)       if m)
        _status_counts["Stable"].update(k for k, m in zip(_keys, _stable_m)     if m)
        _status_counts["Alert"].update(k for k, m in zip(_keys, _alert_m)       if m)

    _n_critical  = len(_status_counts["Critical"])
    _n_watch     = len(_status_counts["Watch"])
    _n_alert     = len(_status_counts["Alert"])
    _n_vacancy   = int(cur_df["brand"].astype(str).str.contains("공실", na=False).sum())
    _n_backward  = len(backward_rows)

    _k1, _k2, _k3, _k4, _k5 = st.columns(5)
    _k1.metric(t("tenants"), len(cur_df))
    _k2.metric(t("critical"), _n_critical,  delta=None if _n_critical == 0 else t("kpi_across"))
    _k3.metric(t("watch"),    _n_watch,     delta=None if _n_watch    == 0 else t("kpi_elevated"))
    _k4.metric(t("alert"),    _n_alert,     delta=None if _n_alert    == 0 else t("kpi_sharp_rise"))
    _k5.metric("🏚 공실 / ⚠ Data", f"{_n_vacancy} / {_n_backward}")
    st.divider()

    # ---------------- Summary Report download ─────────────────────────────────
    with st.expander(t("download_report"), expanded=False):
        st.caption(t("report_caption"))
        report_lang = st.radio(
            t("report_lang"),
            ["한국어", "English"],
            horizontal=True,
            key="report_lang",
        )
        lang_code = "ko" if report_lang == "한국어" else "en"

        if st.button(t("gen_report_btn"), key="gen_report"):
            report_context = {
                "date":      str(date.today()),
                "period":    billing_period or str(date.today()),
                "buildings": ", ".join(active_buildings) if active_buildings else "All",
                "floors":    ", ".join(active_floors) if floors_filtered else "All",
            }
            with st.spinner(t("report_spinning")):
                report_bytes = generate_report_pdf(
                    cur_df, present, tail,
                    context=report_context, lang=lang_code,
                )
            bldg_tag_rpt = "all" if "All" in selected_buildings else "_".join(selected_buildings)
            st.download_button(
                label=t("dl_pdf_btn"),
                data=report_bytes,
                file_name=f"utility_report_{bldg_tag_rpt}_{date.today()}.pdf",
                mime="application/pdf",
                key="dl_report",
            )

    # ---------------- Histograms ─────────────────────────────────────────────
    _HIST_OPTS = ["Side by Side", "Change only", "% Change only"]
    _HIST_LABELS = {
        "Side by Side":  {"ko": "나란히",    "en": "Side by Side"},
        "Change only":   {"ko": "변화량만",  "en": "Change only"},
        "% Change only": {"ko": "% 변화만", "en": "% Change only"},
    }
    _hlang = st.session_state.get("lang", "ko")
    hist_layout = st.radio(
        t("hist_view"),
        _HIST_OPTS,
        format_func=lambda x: _HIST_LABELS[x][_hlang],
        horizontal=True,
        key="hist_layout",
    )

    def _hist_controls(key_prefix: str):
        """Render slider+input bins & tail controls; return (bins, lo, hi, tail_pct)."""
        bk, bik = f"{key_prefix}_bins", f"{key_prefix}_bins_i"
        tk, tik = f"{key_prefix}_tail", f"{key_prefix}_tail_i"
        if bk  not in st.session_state: st.session_state[bk]  = 50
        if bik not in st.session_state: st.session_state[bik] = 50
        if tk  not in st.session_state: st.session_state[tk]  = 20
        if tik not in st.session_state: st.session_state[tik] = 20

        def _sync_bs(): st.session_state[bik] = st.session_state[bk]
        def _sync_bi(): st.session_state[bk]  = st.session_state[bik]
        def _sync_ts(): st.session_state[tik] = st.session_state[tk]
        def _sync_ti(): st.session_state[tk]  = st.session_state[tik]

        _b1, _b2 = st.columns([3, 1])
        with _b1:
            st.slider("Bins", 5, 200, value=st.session_state[bk], step=5, key=bk, on_change=_sync_bs)
        with _b2:
            st.number_input("Bins", 5, 200, value=st.session_state[bik], step=5, key=bik,
                            label_visibility="hidden", on_change=_sync_bi)

        _t1, _t2 = st.columns([3, 1])
        with _t1:
            st.slider("Tail %", 1, 50, value=st.session_state[tk], step=1, key=tk, on_change=_sync_ts)
        with _t2:
            st.number_input("Tail %", 1, 50, value=st.session_state[tik], step=1, key=tik,
                            label_visibility="hidden", on_change=_sync_ti)

        _s = s_change if "chg" in key_prefix else s_pct
        _t = int(st.session_state[tk])
        _lo, _hi = _s.quantile([_t / 100, 1 - _t / 100])
        return int(st.session_state[bk]), float(_lo), float(_hi), _t

    if hist_layout == "Side by Side":
        hc1, hc2 = st.columns(2)
        with hc1:
            _b_c, _lo_c2, _hi_c2, _t_c = _hist_controls("chg")
            plot_hist_with_tails(
                s_change, _b_c, _lo_c2, _hi_c2, f"Change: {change_col}",
                source_df=cur_df, val_col=change_col, key="hist_change",
                display_cols=cols_brand_then_category(cur_df, prefix, mode="change"),
                tail_pct=_t_c,
            )
        with hc2:
            _b_p, _lo_p2, _hi_p2, _t_p = _hist_controls("pct")
            plot_hist_with_tails(
                s_pct, _b_p, _lo_p2, _hi_p2, f"Pct: {pct_col}",
                source_df=cur_df, val_col=pct_col, key="hist_pct",
                display_cols=cols_brand_then_category(cur_df, prefix, mode="pct"),
                tail_pct=_t_p,
            )
    elif hist_layout == "Change only":
        _b_c, _lo_c2, _hi_c2, _t_c = _hist_controls("chg")
        plot_hist_with_tails(
            s_change, _b_c, _lo_c2, _hi_c2, f"Change: {change_col}",
            source_df=cur_df, val_col=change_col, key="hist_change",
            display_cols=cols_brand_then_category(cur_df, prefix, mode="change"),
            tail_pct=_t_c,
        )
    else:
        _b_p, _lo_p2, _hi_p2, _t_p = _hist_controls("pct")
        plot_hist_with_tails(
            s_pct, _b_p, _lo_p2, _hi_p2, f"Pct: {pct_col}",
            source_df=cur_df, val_col=pct_col, key="hist_pct",
            display_cols=cols_brand_then_category(cur_df, prefix, mode="pct"),
            tail_pct=_t_p,
        )

    _sheet_names = get_sheet_names(file_name, file_map[file_name])
    _ehp_sheet = EHP_OAC_SHEET_NAME if EHP_OAC_SHEET_NAME in _sheet_names else None

    tab_change, tab_pct, tab_overlap, tab_ranking, tab_corr = st.tabs([
        t("tab_change"), t("tab_pct"),
        t("tab_quadrant"), t("tab_ranking"), t("tab_corr"),
    ])

    # ---------------- Change tab ----------------
    with tab_change:
        st.subheader(f"{t('tab_change')} — {change_col}")
        chg_label = f"{tail}%"
        _show_opts = ["All", "Top", "Bottom"]
        _show_labels = {"All": {"ko": "전체", "en": "All"}, "Top": {"ko": "상위", "en": "Top"}, "Bottom": {"ko": "하위", "en": "Bottom"}}
        _slang = st.session_state.get("lang", "ko")
        chg_view_mode = st.radio(
            t("show"), _show_opts, format_func=lambda x: _show_labels[x][_slang],
            index=0, horizontal=True, key="chg_view_mode"
        )

        chg_display_cols = cols_brand_then_category(cur_df, prefix, mode="change")

        if chg_view_mode == "All":
            chg_all = cur_df[chg_display_cols].dropna(subset=[change_col]).sort_values(change_col, ascending=False).copy()
            chg_all_view = add_display_index(chg_all)
            st.markdown(f"**{t('all_entries')}** — {t('sorted_hl')} ({len(chg_all)})")
            st.dataframe(st_safe(chg_all_view), width="stretch", hide_index=True, height=_full_height(len(chg_all_view)))
            download_df_as_excel(chg_all_view, filename=f"{df_key}_{prefix}_change_all.xlsx", sheet_name="change_all")

        elif chg_view_mode == "Top":
            st.markdown(f"**{t('show_top')} {chg_label} (>= {float(hi_c):.4g})** — {t('sorted_hl')} ({len(chg_top)})")
            chg_top_view = add_display_index(chg_top[cols_brand_then_category(chg_top, prefix, mode="change")])
            st.dataframe(st_safe(chg_top_view), width="stretch", hide_index=True)
            download_df_as_excel(chg_top_view, filename=f"{df_key}_{prefix}_change_top.xlsx", sheet_name="change_top")

        else:  # Bottom
            st.markdown(f"**{t('show_bottom')} {chg_label} (<= {float(lo_c):.4g})** — {t('sorted_hl')} ({len(chg_bot)})")
            chg_bot_view = add_display_index(chg_bot[cols_brand_then_category(chg_bot, prefix, mode="change")])
            st.dataframe(st_safe(chg_bot_view), width="stretch", hide_index=True)
            download_df_as_excel(chg_bot_view, filename=f"{df_key}_{prefix}_change_bottom.xlsx", sheet_name="change_bottom")

        chg_nan = cur_df[cur_df[change_col].isna()][chg_display_cols].copy()
        if not chg_nan.empty:
            st.divider()
            st.markdown(f"**{t('no_data_nan')}** — {t('missing_change')} ({len(chg_nan)})")
            chg_nan_view = add_display_index(chg_nan)
            st.dataframe(st_safe(chg_nan_view), width="stretch", hide_index=True)
            download_df_as_excel(chg_nan_view, filename=f"{df_key}_{prefix}_change_nan.xlsx", sheet_name="change_nan")

    # ---------------- Pct tab ----------------
    with tab_pct:
        st.subheader(f"{t('tab_pct')} — {pct_col}")
        pct_label = f"{tail}%"
        _slang2 = st.session_state.get("lang", "ko")
        pct_view_mode = st.radio(
            t("show"), _show_opts, format_func=lambda x: _show_labels[x][_slang2],
            index=0, horizontal=True, key="pct_view_mode"
        )

        pct_display_cols = cols_brand_then_category(cur_df, prefix, mode="pct")

        if pct_view_mode == "All":
            pct_all = cur_df[pct_display_cols].dropna(subset=[pct_col]).sort_values(pct_col, ascending=False).copy()
            pct_all_view = add_display_index(pct_all)
            st.markdown(f"**{t('all_entries')}** — {t('sorted_hl')} ({len(pct_all)})")
            st.dataframe(st_safe(pct_all_view), width="stretch", hide_index=True, height=_full_height(len(pct_all_view)))
            download_df_as_excel(pct_all_view, filename=f"{df_key}_{prefix}_pct_all.xlsx", sheet_name="pct_all")

        elif pct_view_mode == "Top":
            st.markdown(f"**{t('show_top')} {pct_label} (>= {float(hi_p):.4g})** — {t('sorted_hl')} ({len(pct_top)})")
            pct_top_view = add_display_index(pct_top[cols_brand_then_category(pct_top, prefix, mode="pct")])
            st.dataframe(st_safe(pct_top_view), width="stretch", hide_index=True)
            download_df_as_excel(pct_top_view, filename=f"{df_key}_{prefix}_pct_top.xlsx", sheet_name="pct_top")

        else:  # Bottom
            st.markdown(f"**{t('show_bottom')} {pct_label} (<= {float(lo_p):.4g})** — {t('sorted_hl')} ({len(pct_bot)})")
            pct_bot_view = add_display_index(pct_bot[cols_brand_then_category(pct_bot, prefix, mode="pct")])
            st.dataframe(st_safe(pct_bot_view), width="stretch", hide_index=True)
            download_df_as_excel(pct_bot_view, filename=f"{df_key}_{prefix}_pct_bottom.xlsx", sheet_name="pct_bottom")

        pct_nan = cur_df[cur_df[pct_col].isna()][pct_display_cols].copy()
        if not pct_nan.empty:
            st.divider()
            st.markdown(f"**{t('no_data_nan')}** — {t('missing_pct')} ({len(pct_nan)})")
            pct_nan_view = add_display_index(pct_nan)
            st.dataframe(st_safe(pct_nan_view), width="stretch", hide_index=True)
            download_df_as_excel(pct_nan_view, filename=f"{df_key}_{prefix}_pct_nan.xlsx", sheet_name="pct_nan")

    # ---------------- Overlap tab ----------------
    with tab_overlap:
        st.subheader(t("quadrant_title"))

        st.markdown(f"{t('q_HH')} ({len(q_HH)})")
        q_cols = cols_brand_then_category(q_HH, prefix, mode="change")
        q_view = add_display_index(q_HH[q_cols])
        st.dataframe(st_safe(q_view), width="stretch", hide_index=True)
        download_df_as_excel(q_view, filename=f"{df_key}_{prefix}_overlap_HH.xlsx", sheet_name="overlap_HH")

        st.divider()

        st.markdown(f"{t('q_HL')} ({len(q_HL)})")
        q_cols = cols_brand_then_category(q_HL, prefix, mode="change")
        q_view = add_display_index(q_HL[q_cols])
        st.dataframe(st_safe(q_view), width="stretch", hide_index=True)
        download_df_as_excel(q_view, filename=f"{df_key}_{prefix}_overlap_HL.xlsx", sheet_name="overlap_HL")

        st.divider()

        st.markdown(f"{t('q_LH')} ({len(q_LH)})")
        q_cols = cols_brand_then_category(q_LH, prefix, mode="change")
        q_view = add_display_index(q_LH[q_cols])
        st.dataframe(st_safe(q_view), width="stretch", hide_index=True)
        download_df_as_excel(q_view, filename=f"{df_key}_{prefix}_overlap_LH.xlsx", sheet_name="overlap_LH")

        st.divider()

        st.markdown(f"{t('q_LL')} ({len(q_LL)})")
        q_cols = cols_brand_then_category(q_LL, prefix, mode="change")
        q_view = add_display_index(q_LL[q_cols])
        st.dataframe(st_safe(q_view), width="stretch", hide_index=True)
        download_df_as_excel(q_view, filename=f"{df_key}_{prefix}_overlap_LL.xlsx", sheet_name="overlap_LL")

        st.divider()

        all_quadrant_idx = q_HH.index.union(q_HL.index).union(q_LH.index).union(q_LL.index)
        q_normal = cur_df.loc[~cur_df.index.isin(all_quadrant_idx)].dropna(subset=[change_col, pct_col]).copy()
        q_normal = q_normal.sort_values(change_col, ascending=False)
        q_normal_cols = cols_brand_then_category(q_normal, prefix, mode="change")
        st.markdown(f"{t('q_normal')} ({len(q_normal)})")
        q_normal_view = add_display_index(q_normal[q_normal_cols])
        st.dataframe(st_safe(q_normal_view), width="stretch", hide_index=True)
        download_df_as_excel(q_normal_view, filename=f"{df_key}_{prefix}_normal.xlsx", sheet_name="normal")

    # ---------------- Brand Ranking tab ----------------
    with tab_ranking:
        st.subheader(f"{t('ranking_title')} — {prefix}")

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
            st.info(t("no_brand_data"))
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

            with st.expander(t("score_explain")):
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
                         height=_full_height(len(rank_view)))
            download_df_as_excel(rank_view, filename=f"{df_key}_{prefix}_brand_ranking.xlsx", sheet_name="brand_ranking")

    with tab_corr:
        render_corr_tab(cur_df)

    # ---------------- Billing ↔ Meter reconciliation ─────────────────────────
    render_reconciliation(file_name, file_map, cur_df, present, _sheet_names)

