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
    apply_header_rows, create_change_columns, build_from_two_files,
    aggregate_by_brand, split_brand_by_floor,
    get_simple_floors, sanitize,
    cols_brand_then_category, add_display_index, download_df_as_excel,
)
from filters import brand_search_bar
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


def _load_and_clean(file_name: str, file_map: Dict[str, bytes], sheet_name: str) -> pd.DataFrame:
    """Load meter sheet, apply headers, filter to valid buildings."""
    raw_df = read_sheet(file_name, file_map[file_name], sheet_name)
    df = apply_header_rows(raw_df)
    df["building"] = df["building"].astype(str).str.strip()
    return df[df["building"].isin({"A", "B", "C", "D"})].copy()


def load_raw_meter_df(
    file_name: str, file_map: Dict[str, bytes], sheet_name: str,
    prev_file_name: str | None = None, prev_sheet_name: str | None = None,
) -> pd.DataFrame:
    """Load meter sheet and return pre-aggregation row-level DataFrame.

    If prev_file_name is provided, merges the previous month's data so that
    *_previous = last month's usage and *_current = this month's usage, enabling
    true month-over-month change computation.
    """
    df_cur = _load_and_clean(file_name, file_map, sheet_name)
    df_prev = None
    if prev_file_name and prev_sheet_name:
        df_prev = _load_and_clean(prev_file_name, file_map, prev_sheet_name)
    df = build_from_two_files(df_cur, df_prev)
    return create_change_columns(df)


def load_meter_df(
    file_name: str, file_map: Dict[str, bytes], sheet_name: str,
    prev_file_name: str | None = None, prev_sheet_name: str | None = None,
) -> pd.DataFrame:
    """Load and aggregate meter data without rendering any UI."""
    df = load_raw_meter_df(file_name, file_map, sheet_name, prev_file_name, prev_sheet_name)
    df = aggregate_by_brand(df)

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


_CAT_LABELS = {"water": "수도", "hwater": "온수", "elect": "전기", "heat": "열"}
_QUAD_EMOJI = {"HH": "🔴 HH", "HL": "🟠 HL", "LH": "🟡 LH", "LL": "🟢 LL", "—": "—"}


def _render_all_outliers(cur_df: pd.DataFrame, present: list, tail: int, t) -> None:
    """Cross-category outlier summary: brands flagged as tail in any category."""
    q = tail / 100.0
    rows = []
    for _, row in cur_df.iterrows():
        brand = row.get("brand", "")
        entry = {"brand": brand}
        if "building" in cur_df.columns:
            entry["building"] = row.get("building", "")
        if "floor" in cur_df.columns:
            entry["floor"] = row.get("floor", "")
        outlier_count = 0
        for p in present:
            cc, pc = f"{p}_change", f"{p}_pct"
            c_all = to_numeric_series(cur_df[cc]).dropna()
            p_all = to_numeric_series(cur_df[pc]).dropna()
            if c_all.empty or p_all.empty:
                entry[_CAT_LABELS.get(p, p)] = "—"
                continue
            lo_c, hi_c = c_all.quantile(q), c_all.quantile(1 - q)
            lo_p, hi_p = p_all.quantile(q), p_all.quantile(1 - q)
            cv = to_numeric_series(pd.Series([row.get(cc)]))[0]
            pv = to_numeric_series(pd.Series([row.get(pc)]))[0]
            if pd.isna(cv) or pd.isna(pv):
                quad = "—"
            elif cv >= hi_c and pv >= hi_p:
                quad = "HH"; outlier_count += 1
            elif cv >= hi_c and pv <= lo_p:
                quad = "HL"; outlier_count += 1
            elif cv <= lo_c and pv >= hi_p:
                quad = "LH"; outlier_count += 1
            elif cv <= lo_c and pv <= lo_p:
                quad = "LL"
            else:
                quad = "—"
            entry[_CAT_LABELS.get(p, p)] = _QUAD_EMOJI.get(quad, quad)
        entry["이상치 수"] = outlier_count
        rows.append(entry)

    # Pre-compute per-category quadrant membership for reuse across tabs
    _cat_thresholds = {}
    for p in present:
        cc, pc = f"{p}_change", f"{p}_pct"
        c_all = to_numeric_series(cur_df[cc]).dropna()
        p_all = to_numeric_series(cur_df[pc]).dropna()
        if not c_all.empty and not p_all.empty:
            _cat_thresholds[p] = (
                c_all.quantile(q), c_all.quantile(1 - q),
                p_all.quantile(q), p_all.quantile(1 - q),
            )

    def _quadrant(p, row):
        if p not in _cat_thresholds:
            return "—"
        lo_c, hi_c, lo_p, hi_p = _cat_thresholds[p]
        cc, pc = f"{p}_change", f"{p}_pct"
        cv = to_numeric_series(pd.Series([row.get(cc)]))[0]
        pv = to_numeric_series(pd.Series([row.get(pc)]))[0]
        if pd.isna(cv) or pd.isna(pv):
            return "—"
        if cv >= hi_c and pv >= hi_p:   return "HH"
        if cv >= hi_c and pv <= lo_p:   return "HL"
        if cv <= lo_c and pv >= hi_p:   return "LH"
        if cv <= lo_c and pv <= lo_p:   return "LL"
        return "—"

    out = pd.DataFrame(rows).sort_values("이상치 수", ascending=False).reset_index(drop=True)
    outliers_only = out[out["이상치 수"] > 0]

    brand_search_bar("meter")

    st.caption(
        f"Tail {tail}% 기준 — 🔴 HH: 변화·비율 모두 상위 | 🟠 HL: 변화 상위·비율 하위 | "
        f"🟡 LH: 변화 하위·비율 상위 | 🟢 LL: 변화·비율 모두 하위"
    )

    _tab_labels = ["전체"] + [_CAT_LABELS.get(p, p) for p in present]
    _tabs = st.tabs(_tab_labels)

    # ── 전체 tab ──────────────────────────────────────────────────────────────
    with _tabs[0]:
        k1, k2, k3 = st.columns(3)
        k1.metric("전체 브랜드", len(out))
        k2.metric("이상치 브랜드", len(outliers_only))
        k3.metric("정상 브랜드", len(out) - len(outliers_only))

        _show_all = st.checkbox("전체 브랜드 표시 (이상치 + 정상)", value=False, key="all_outlier_show_all")
        display_df = out if _show_all else outliers_only
        if display_df.empty:
            st.info("이상치로 분류된 브랜드가 없습니다.")
        else:
            display_df = display_df.reset_index(drop=True)
            display_df.insert(0, "No", range(1, len(display_df) + 1))
            st.dataframe(display_df, hide_index=True, use_container_width=True,
                         height=_full_height(len(display_df)))

    # ── Per-category tabs ──────────────────────────────────────────────────────
    for _tab, p in zip(_tabs[1:], present):
        with _tab:
            cc, pc = f"{p}_change", f"{p}_pct"
            detail_cols = ["brand"] + (["building"] if "building" in cur_df.columns else []) + \
                          (["floor"] if "floor" in cur_df.columns else []) + \
                          ([cc] if cc in cur_df.columns else []) + \
                          ([pc] if pc in cur_df.columns else [])

            def _quad_df(quad_key, _p=p, _cc=cc):
                mask = cur_df.apply(lambda r: _quadrant(_p, r) == quad_key, axis=1)
                sub = cur_df[mask][detail_cols].copy()
                sub = sub.sort_values(_cc, ascending=False) if _cc in sub.columns else sub
                sub.insert(0, "No", range(1, len(sub) + 1))
                return sub

            # ── Histograms ────────────────────────────────────────────────────
            if p in _cat_thresholds:
                _lo_c, _hi_c, _lo_p, _hi_p = _cat_thresholds[p]
                _sc = to_numeric_series(cur_df[cc]).dropna()
                _sp = to_numeric_series(cur_df[pc]).dropna()
                _hlang = st.session_state.get("lang", "ko")
                _all_hist_labels = {
                    "% Change only": {"ko": "% 변화만", "en": "% Change only"},
                    "Change only":   {"ko": "변화량만",  "en": "Change only"},
                    "Side by Side":  {"ko": "나란히",    "en": "Side by Side"},
                }
                _all_hist_sel = st.radio(
                    t("hist_view"),
                    ["% Change only", "Change only", "Side by Side"],
                    format_func=lambda x: _all_hist_labels[x][_hlang],
                    horizontal=True, key=f"all_hist_layout_{p}",
                )
                _ak = st.slider("IQR 배수 (k)", 0.5, 3.0, 1.5, 0.25,
                                key=f"all_iqr_k_{p}",
                                help="이상치 기준: Q1 − k×IQR  /  Q3 + k×IQR")
                _cq1, _cq3, _ciqr, _clo, _chi = (
                    float(_sc.quantile(.25)), float(_sc.quantile(.75)),
                    float(_sc.quantile(.75) - _sc.quantile(.25)), 0, 0)
                _ciqr = _cq3 - _cq1; _clo = _cq1 - _ak * _ciqr; _chi = _cq3 + _ak * _ciqr
                _pq1, _pq3, _piqr, _plo, _phi = (
                    float(_sp.quantile(.25)), float(_sp.quantile(.75)),
                    float(_sp.quantile(.75) - _sp.quantile(.25)), 0, 0)
                _piqr = _pq3 - _pq1; _plo = _pq1 - _ak * _piqr; _phi = _pq3 + _ak * _piqr

                def _show_eq_c():
                    st.markdown(f"$$Q_1={_cq1:,.0f},\\;Q_3={_cq3:,.0f},\\;IQR={_ciqr:,.0f}$$\n\n"
                                f"$$\\text{{Lower}}={_clo:,.0f},\\;\\text{{Upper}}={_chi:,.0f}\\;(k={_ak})$$")
                def _show_eq_p():
                    st.markdown(f"$$Q_1={_pq1:,.0f},\\;Q_3={_pq3:,.0f},\\;IQR={_piqr:,.0f}$$\n\n"
                                f"$$\\text{{Lower}}={_plo:,.0f},\\;\\text{{Upper}}={_phi:,.0f}\\;(k={_ak})$$")

                if _all_hist_sel == "Side by Side":
                    _hc1, _hc2 = st.columns(2)
                    with _hc1:
                        _show_eq_c()
                        plot_hist_with_tails(_sc, 50, _clo, _chi, f"Change: {cc}",
                                             source_df=cur_df, val_col=cc, key=f"all_hist_chg_{p}",
                                             display_cols=detail_cols)
                    with _hc2:
                        _show_eq_p()
                        plot_hist_with_tails(_sp, 50, _plo, _phi, f"Pct: {pc}",
                                             source_df=cur_df, val_col=pc, key=f"all_hist_pct_{p}",
                                             display_cols=detail_cols)
                elif _all_hist_sel == "Change only":
                    _show_eq_c()
                    plot_hist_with_tails(_sc, 50, _clo, _chi, f"Change: {cc}",
                                         source_df=cur_df, val_col=cc, key=f"all_hist_chg_{p}",
                                         display_cols=detail_cols)
                else:  # % Change only
                    _show_eq_p()
                    plot_hist_with_tails(_sp, 50, _plo, _phi, f"Pct: {pc}",
                                         source_df=cur_df, val_col=pc, key=f"all_hist_pct_{p}",
                                         display_cols=detail_cols)
                st.divider()

            for quad, label in [
                ("HH", "🔴 HH — 변화·비율 모두 상위"),
                ("HL", "🟠 HL — 변화 상위, 비율 하위"),
                ("LH", "🟡 LH — 변화 하위, 비율 상위"),
                ("LL", "🟢 LL — 변화·비율 모두 하위"),
            ]:
                qdf = _quad_df(quad)
                st.markdown(f"**{label}** ({len(qdf)})")
                if qdf.empty:
                    st.caption("해당 없음")
                else:
                    st.dataframe(qdf, hide_index=True, use_container_width=True,
                                 height=_full_height(len(qdf)))
                st.divider()


def render_meter_view(
    file_name: str,
    file_map: Dict[str, bytes],
    sheet_name: str,
    bins: int,
    tail: int,
    q: tuple,
    debug: bool,
    prev_file_name: str | None = None,
    prev_sheet_name: str | None = None,
    billing_period: str | None = None,
    prev_billing_period: str | None = None,
) -> None:
    """Full 검침 내역 analysis: pipeline, filters, histograms, 5 tabs, reconciliation."""
    billing_period = billing_period or get_billing_period(file_name, file_map[file_name])

    if prev_billing_period and billing_period:
        st.caption(f"비교: {prev_billing_period} → {billing_period}")

    # ---------------- Preprocess ----------------
    try:
        df_cur = _load_and_clean(file_name, file_map, sheet_name)
        df_prev = None
        if prev_file_name and prev_sheet_name:
            df_prev = _load_and_clean(prev_file_name, file_map, prev_sheet_name)
        df = build_from_two_files(df_cur, df_prev)
        df = create_change_columns(df)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    for col in ["building", "brand"]:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    # ---------------- Backward meter reading detection ----------------
    # Uses the original cumulative readings (renamed by build_from_two_files)
    _meter_pairs = [
        ("water",  "water_meter_prev",  "water_meter_curr",  "m³"),
        ("hwater", "hwater_meter_prev", "hwater_meter_curr", "m³"),
        ("elect",  "elect_meter_prev",  "elect_meter_curr",  "kWh"),
        ("heat",   "heat_meter_prev",   "heat_meter_curr",   "m³/MWh"),
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
    _CATEGORY_LABELS = {
        "water": t("cat_water"), "hwater": t("cat_hwater"),
        "elect": t("cat_elect"), "heat": t("cat_heat"),
        "__all__": "🚨 전체 이상치",
    }
    category_options = (["__all__"] if len(present_all) > 1 else []) + present_all
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
    # Read from session state — widget is rendered above tabs via brand_search_bar()
    brand_search = st.session_state.get("meter_brand_search", "").strip().lower()

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
        prefix = st.selectbox(t("category"), category_options, format_func=lambda x: _CATEGORY_LABELS.get(x, x))
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

    if brand_search:
        cur_df = cur_df[cur_df["brand"].astype(str).str.lower().str.contains(brand_search, na=False)].copy()
        if cur_df.empty:
            st.warning(f"'{brand_search}' 검색 결과가 없습니다.")
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

    # ---------------- Overall outliers view ----------------------------------------
    present = [p for p in allowed if f"{p}_change" in cur_df.columns]
    if not present:
        st.error("No utility categories found.")
        st.stop()

    if prefix == "__all__":
        _render_all_outliers(cur_df, present, tail, t)
        return

    # ---------------- Category column setup ----------------
    change_col, pct_col = f"{prefix}_change", f"{prefix}_pct"

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

    with st.expander("📖 위험 등급 기준 설명", expanded=False):
        st.markdown(f"""
각 등급은 **Tail {tail}%** 설정을 기준으로 전체 입점 업체의 변화량(MoM) 및 변화율(%)의 분포에서 결정됩니다.
상위/하위 기준선 = 각 지표의 상위/하위 {tail}번째 백분위수.

| 등급 | 조건 | 해석 |
|------|------|------|
| 🔴 **위험** | 변화량 **AND** 변화율 모두 상위 {tail}% 이상 | 사용량이 절대적으로도, 비율로도 급증 — 즉각 점검 필요 |
| 🟠 **주의** | 변화량 또는 변화율 중 하나만 상위 {tail}% 이상 | 한 지표가 이상치 — 추가 모니터링 필요 |
| 🟡 **경보** | 변화량 또는 변화율 중 하나가 하위 {tail}% 이하 (단, 둘 다 하위는 아님) | 사용량 급감 — 공실 전환·계량기 오류 의심 |
| 🟢 **안정** | 변화량 **AND** 변화율 모두 하위 {tail}% 이하 | 사용량이 절대적으로도, 비율로도 크게 감소 |

> 동일 업체가 복수의 유틸리티(수도·전기·열 등)에서 각각 다른 등급을 받을 수 있습니다.
        """)

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
    _HIST_OPTS = ["% Change only", "Change only", "Side by Side"]
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

    def _iqr_bounds(s: pd.Series, k: float):
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        return q1, q3, iqr, q1 - k * iqr, q3 + k * iqr

    def _iqr_equation(q1, q3, iqr, lo, hi, k, key_suffix):
        st.markdown(
            f"$$Q_1={q1:,.0f},\\;Q_3={q3:,.0f},\\;IQR={iqr:,.0f}$$\n\n"
            f"$$\\text{{Lower}}={lo:,.0f},\\quad\\text{{Upper}}={hi:,.0f}\\;(k={k})$$"
        )

    def _hist_controls(key_prefix: str):
        """Render bins slider + IQR k slider; return (bins, lo, hi)."""
        bk, bik = f"{key_prefix}_bins", f"{key_prefix}_bins_i"
        if bk  not in st.session_state: st.session_state[bk]  = 50
        if bik not in st.session_state: st.session_state[bik] = 50

        def _sync_bs(): st.session_state[bik] = st.session_state[bk]
        def _sync_bi(): st.session_state[bk]  = st.session_state[bik]

        _b1, _b2 = st.columns([3, 1])
        with _b1:
            st.slider("Bins", 5, 200, value=st.session_state[bk], step=5, key=bk, on_change=_sync_bs)
        with _b2:
            st.number_input("Bins", 5, 200, value=st.session_state[bik], step=5, key=bik,
                            label_visibility="hidden", on_change=_sync_bi)

        _k = st.slider("IQR 배수 (k)", 0.5, 3.0, 1.5, 0.25, key=f"{key_prefix}_iqr_k",
                       help="이상치 기준: Q1 − k×IQR  /  Q3 + k×IQR")
        _s = s_change if "chg" in key_prefix else s_pct
        _q1, _q3, _iqr, _lo, _hi = _iqr_bounds(_s, _k)
        _iqr_equation(_q1, _q3, _iqr, _lo, _hi, _k, key_prefix)
        return int(st.session_state[bk]), float(_lo), float(_hi)

    if hist_layout == "Side by Side":
        hc1, hc2 = st.columns(2)
        with hc1:
            _b_c, _lo_c2, _hi_c2 = _hist_controls("chg")
            plot_hist_with_tails(
                s_change, _b_c, _lo_c2, _hi_c2, f"Change: {change_col}",
                source_df=cur_df, val_col=change_col, key="hist_change",
                display_cols=cols_brand_then_category(cur_df, prefix, mode="change"),
            )
        with hc2:
            _b_p, _lo_p2, _hi_p2 = _hist_controls("pct")
            plot_hist_with_tails(
                s_pct, _b_p, _lo_p2, _hi_p2, f"Pct: {pct_col}",
                source_df=cur_df, val_col=pct_col, key="hist_pct",
                display_cols=cols_brand_then_category(cur_df, prefix, mode="pct"),
            )
    elif hist_layout == "Change only":
        _b_c, _lo_c2, _hi_c2 = _hist_controls("chg")
        plot_hist_with_tails(
            s_change, _b_c, _lo_c2, _hi_c2, f"Change: {change_col}",
            source_df=cur_df, val_col=change_col, key="hist_change",
            display_cols=cols_brand_then_category(cur_df, prefix, mode="change"),
        )
    else:
        _b_p, _lo_p2, _hi_p2 = _hist_controls("pct")
        plot_hist_with_tails(
            s_pct, _b_p, _lo_p2, _hi_p2, f"Pct: {pct_col}",
            source_df=cur_df, val_col=pct_col, key="hist_pct",
            display_cols=cols_brand_then_category(cur_df, prefix, mode="pct"),
        )

    _sheet_names = get_sheet_names(file_name, file_map[file_name])
    _ehp_sheet = EHP_OAC_SHEET_NAME if EHP_OAC_SHEET_NAME in _sheet_names else None

    brand_search_bar("meter")
    tab_data, tab_ranking, tab_corr = st.tabs([
        t("tab_change"), t("tab_ranking"), t("tab_corr"),
    ])

    # ---------------- Data tab (Change + Pct merged) ----------------
    with tab_data:
        _show_opts = ["All", "Top", "Bottom"]
        _show_labels = {"All": {"ko": "전체", "en": "All"}, "Top": {"ko": "상위", "en": "Top"}, "Bottom": {"ko": "하위", "en": "Bottom"}}
        _slang = st.session_state.get("lang", "ko")
        _dtype_labels = {"change": {"ko": "변화량", "en": "Change"}, "pct": {"ko": "변화율 %", "en": "Pct %"}}

        _dcol1, _dcol2 = st.columns([1, 2])
        with _dcol1:
            _dtype = st.radio(
                "보기", ["change", "pct"],
                format_func=lambda x: _dtype_labels[x][_slang],
                horizontal=True, key="data_tab_dtype",
            )
        with _dcol2:
            _view_mode = st.radio(
                t("show"), _show_opts, format_func=lambda x: _show_labels[x][_slang],
                index=0, horizontal=True, key="data_tab_view_mode",
            )

        if _dtype == "change":
            _col, _lo_t, _hi_t = change_col, lo_c, hi_c
            _top_df, _bot_df = chg_top, chg_bot
            _display_cols = cols_brand_then_category(cur_df, prefix, mode="change")
            _label = f"{tail}%"
        else:
            _col, _lo_t, _hi_t = pct_col, lo_p, hi_p
            _top_df, _bot_df = pct_top, pct_bot
            _display_cols = cols_brand_then_category(cur_df, prefix, mode="pct")
            _label = f"{tail}%"

        st.subheader(f"{_dtype_labels[_dtype][_slang]} — {_col}")

        if _view_mode == "All":
            _all = cur_df[_display_cols].dropna(subset=[_col]).sort_values(_col, ascending=False).copy()
            _all_view = add_display_index(_all)
            st.markdown(f"**{t('all_entries')}** — {t('sorted_hl')} ({len(_all)})")
            st.dataframe(st_safe(_all_view), width="stretch", hide_index=True, height=_full_height(len(_all_view)))
            download_df_as_excel(_all_view, filename=f"{df_key}_{prefix}_{_dtype}_all.xlsx", sheet_name=f"{_dtype}_all")
        elif _view_mode == "Top":
            st.markdown(f"**{t('show_top')} {_label} (>= {float(_hi_t):.4g})** — {t('sorted_hl')} ({len(_top_df)})")
            _top_view = add_display_index(_top_df[cols_brand_then_category(_top_df, prefix, mode=_dtype)])
            st.dataframe(st_safe(_top_view), width="stretch", hide_index=True)
            download_df_as_excel(_top_view, filename=f"{df_key}_{prefix}_{_dtype}_top.xlsx", sheet_name=f"{_dtype}_top")
        else:
            st.markdown(f"**{t('show_bottom')} {_label} (<= {float(_lo_t):.4g})** — {t('sorted_hl')} ({len(_bot_df)})")
            _bot_view = add_display_index(_bot_df[cols_brand_then_category(_bot_df, prefix, mode=_dtype)])
            st.dataframe(st_safe(_bot_view), width="stretch", hide_index=True)
            download_df_as_excel(_bot_view, filename=f"{df_key}_{prefix}_{_dtype}_bottom.xlsx", sheet_name=f"{_dtype}_bottom")

        _nan_df = cur_df[cur_df[_col].isna()][_display_cols].copy()
        if not _nan_df.empty:
            st.divider()
            st.markdown(f"**{t('no_data_nan')}** ({len(_nan_df)})")
            _nan_view = add_display_index(_nan_df)
            st.dataframe(st_safe(_nan_view), width="stretch", hide_index=True)
            download_df_as_excel(_nan_view, filename=f"{df_key}_{prefix}_{_dtype}_nan.xlsx", sheet_name=f"{_dtype}_nan")

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

