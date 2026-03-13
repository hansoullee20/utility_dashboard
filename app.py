"""app.py — Utility Analysis Dashboard: page config, routing, orchestration.

Three-tier navigation (question-driven, not data-source-driven):
  Tier 1  🚨 점검 대상   — "Who to investigate?" (anomaly detection + vacancy)
  Tier 2  📊 현황 분석   — "Why?" (summary, period comparison, cost, efficiency)
  Tier 3  🏢 브랜드     — Single brand deep-dive
  Tier 4  📋 원본 데이터 — "Show me the raw data" (per-sheet detail views)
"""
import re
from datetime import date as _date
from typing import Dict

import streamlit as st

from data import (
    get_sheet_names, get_billing_period,
    read_billing_sheet, BILLING_SHEET_NAME,
    EHP_OAC_SHEET_NAME, HVAC_SHEET_NAME,
    read_hvac_sheet,
    read_water_sheet, WATER_SHEET_NAME,
    read_hotwater_sheet, HOTWATER_SHEET_NAME,
    read_electricity_sheet, ELECTRICITY_SHEET_NAME,
)
from billing import render_billing_view
from ehp import render_ehp_view
from water import render_water_view
from hotwater import render_hotwater_view
from electricity import render_electricity_view
from summary import render_summary_view
from sidebar import setup_sidebar
from meter_view import render_meter_view, load_raw_meter_df
from filters import render_meter_filters, show_filter_widgets, apply_sheet_filter
from data import to_numeric_series as _to_num
from tab_cross import render_cross_tab
from tab_efficiency import render_efficiency_tab
from tab_anomaly import render_anomaly_tab
from tab_mom import render_mom_tab
from lang import t


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_sheet(sheet_list: list[str], target: str) -> str | None:
    return next((k for k in sheet_list if k.strip() == target), None)


def _try_load_sheet(reader, sheet_const, fname, file_map, sheet_map):
    """Safely load a single sheet, returning None on failure."""
    fdata = file_map.get(fname)
    if fdata is None:
        return None
    key = _find_sheet(sheet_map.get(fname, []), sheet_const)
    if key is None:
        return None
    try:
        return reader(fname, fdata, key)
    except Exception as e:
        st.warning(f"{sheet_const} 로드 실패 ({fname}): {e}")
        return None


def _load_meter_data(file_name, file_map, sheet_map, all_sheet_keys,
                     prev_file, key_prefix):
    """Load 검침 내역, apply filters, compute per-area columns.

    Returns (cur_df, present, split_bldg, ref_df, ehp_sheet) or stops on error.
    """
    meter_sheet = _find_sheet(all_sheet_keys, "검침 내역")
    prev_meter_sheet = _find_sheet(
        sheet_map.get(prev_file, []), "검침 내역"
    ) if prev_file else None

    try:
        raw_df = load_raw_meter_df(
            file_name, file_map, meter_sheet,
            prev_file_name=prev_file, prev_sheet_name=prev_meter_sheet,
        )
    except Exception as e:
        st.error(f"{t('meter_load_fail')}: {e}")
        st.stop()

    cur_df, active_buildings, active_floors, gong_mode, split_bldg, ref_df = render_meter_filters(
        raw_df, key_prefix=key_prefix,
    )

    # Per-area columns (needed by efficiency + anomaly)
    from utils import add_per_area_cols
    add_per_area_cols(cur_df)

    present = [p for p in ["water", "hwater", "elect", "heat"]
               if f"{p}_change" in cur_df.columns]
    ehp_sheet = EHP_OAC_SHEET_NAME if EHP_OAC_SHEET_NAME in all_sheet_keys else None

    # Populate all brand names for watchlist management
    if "brand" in cur_df.columns:
        _existing = set(st.session_state.get("_all_brand_names", []))
        _existing.update(cur_df["brand"].dropna().astype(str).unique())
        st.session_state["_all_brand_names"] = sorted(_existing)

    return cur_df, present, split_bldg, ref_df, ehp_sheet


def _load_prev_sheet(reader, sheet_const, prev_file, file_map, sheet_map):
    """Load a sheet from the previous-month file, returning None on failure."""
    if not prev_file:
        return None
    return _try_load_sheet(reader, sheet_const, prev_file, file_map, sheet_map)


def _load_meter_data_silent(file_name, file_map, sheet_map, all_sheet_keys, prev_file):
    """Load meter data without rendering filter UI (for PDF generation)."""
    meter_sheet = _find_sheet(all_sheet_keys, "검침 내역")
    if not meter_sheet:
        return None, []
    prev_meter_sheet = _find_sheet(
        sheet_map.get(prev_file, []), "검침 내역"
    ) if prev_file else None

    try:
        cur_df = load_raw_meter_df(
            file_name, file_map, meter_sheet,
            prev_file_name=prev_file, prev_sheet_name=prev_meter_sheet,
        )
    except Exception:
        return None, []

    # Per-area columns
    from utils import add_per_area_cols
    add_per_area_cols(cur_df)

    present = [p for p in ["water", "hwater", "elect", "heat"]
               if f"{p}_change" in cur_df.columns]
    return cur_df, present


def _load_all_sheets(file_name, file_data, all_sheet_keys):
    """Load billing/electricity/water/hotwater sheets silently."""
    from utils import load_all_sheets
    return load_all_sheets(file_name, file_data, all_sheet_keys, silent=True)


def _generate_report(scope, file_name, file_map, sheet_map, all_sheet_keys,
                     prev_file, file_periods, tail):
    """Generate PDF bytes for the given scope."""
    from anomaly_features import build_anomaly_df
    from cross_features import build_unit_costs, build_elec_breakdown
    from biz_report import (
        generate_anomaly_pdf, generate_cross_pdf, generate_efficiency_pdf,
        generate_comprehensive_pdf, generate_insight_pdf,
    )
    from report import generate_report_pdf

    context = {
        "period": file_periods.get(file_name),
        "date": str(_date.today()),
    }
    file_data = file_map[file_name]

    if scope == "상세":
        cur_df, present = _load_meter_data_silent(
            file_name, file_map, sheet_map, all_sheet_keys, prev_file)
        if cur_df is None:
            return None
        return generate_report_pdf(cur_df, present, tail, context, lang="ko")

    # Load common data for anomaly/insight/comprehensive
    cur_df, present = _load_meter_data_silent(
        file_name, file_map, sheet_map, all_sheet_keys, prev_file)
    sheets = _load_all_sheets(file_name, file_data, all_sheet_keys)

    # Build anomaly df
    anomaly_df = None
    if cur_df is not None:
        try:
            anomaly_df = build_anomaly_df(
                meter_df=cur_df,
                billing_df=sheets.get(BILLING_SHEET_NAME),
                elec_df=sheets.get(ELECTRICITY_SHEET_NAME),
                water_df=sheets.get(WATER_SHEET_NAME),
                hotwater_df=sheets.get(HOTWATER_SHEET_NAME),
            )
        except Exception:
            pass

    # Build cross features
    unit_df = elec_br = None
    billing_df = sheets.get(BILLING_SHEET_NAME)
    elec_df = sheets.get(ELECTRICITY_SHEET_NAME)
    if billing_df is not None and cur_df is not None:
        try:
            unit_df = build_unit_costs(cur_df, billing_df)
            if unit_df is not None and unit_df.empty:
                unit_df = None
        except Exception:
            pass
    if elec_df is not None:
        try:
            elec_br = build_elec_breakdown(elec_df, meter_df=cur_df)
            if elec_br is not None and elec_br.empty:
                elec_br = None
        except Exception:
            pass

    if scope == "이상감지":
        if anomaly_df is None or anomaly_df.empty:
            return None
        return generate_anomaly_pdf(anomaly_df, context)

    if scope == "인사이트":
        return generate_insight_pdf(unit_df, elec_br, cur_df, present, context)

    # 종합
    return generate_comprehensive_pdf(
        anomaly_df=anomaly_df,
        unit_df=unit_df,
        elec_br_df=elec_br,
        cur_df=cur_df,
        present=present,
        context=context,
    )


def _generate_excel_export(file_name, file_map, sheet_map, all_sheet_keys, prev_file):
    """Generate a unified Excel workbook with analysis results across all modules."""
    import io
    import pandas as pd

    cur_df, present = _load_meter_data_silent(
        file_name, file_map, sheet_map, all_sheet_keys, prev_file)
    sheets = _load_all_sheets(file_name, file_map[file_name], all_sheet_keys)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # 1. Anomaly results
        if cur_df is not None:
            try:
                from anomaly_features import build_anomaly_df
                adf = build_anomaly_df(
                    meter_df=cur_df,
                    billing_df=sheets.get(BILLING_SHEET_NAME),
                    elec_df=sheets.get(ELECTRICITY_SHEET_NAME),
                    water_df=sheets.get(WATER_SHEET_NAME),
                    hotwater_df=sheets.get(HOTWATER_SHEET_NAME),
                )
                if adf is not None and not adf.empty:
                    adf.to_excel(writer, sheet_name="이상감지", index=False)
            except Exception:
                pass

        # 2. Unit cost analysis
        billing_df = sheets.get(BILLING_SHEET_NAME)
        if billing_df is not None and cur_df is not None:
            try:
                from cross_features import build_unit_costs
                udf = build_unit_costs(cur_df, billing_df)
                if udf is not None and not udf.empty:
                    udf.to_excel(writer, sheet_name="단가분석", index=False)
            except Exception:
                pass

        # 3. Electricity breakdown
        elec_df = sheets.get(ELECTRICITY_SHEET_NAME)
        if elec_df is not None:
            try:
                from cross_features import build_elec_breakdown
                ebr = build_elec_breakdown(elec_df, meter_df=cur_df)
                if ebr is not None and not ebr.empty:
                    ebr.to_excel(writer, sheet_name="전기분류", index=False)
            except Exception:
                pass

        # 4. Meter data (검침 내역)
        if cur_df is not None:
            cur_df.to_excel(writer, sheet_name="검침내역", index=False)

        # 5. Raw sheets
        for label, key in [("수도", WATER_SHEET_NAME), ("온수", HOTWATER_SHEET_NAME),
                           ("전기", ELECTRICITY_SHEET_NAME), ("청구", BILLING_SHEET_NAME)]:
            df = sheets.get(key)
            if df is not None and not df.empty:
                df.to_excel(writer, sheet_name=label, index=False)

    buf.seek(0)
    return buf.getvalue()


# ── Tier renderers ───────────────────────────────────────────────────────────

def _render_tier1_anomaly(file_name, file_map, sheet_map, all_sheet_keys,
                          prev_file, file_periods, meter_files, yoy_file=None):
    """Tier 1: 점검 대상 — 'Who to investigate?'"""
    if "검침 내역" not in all_sheet_keys:
        st.info("이상감지를 위해 **검침 내역** 시트가 필요합니다.")
        return

    cur_df, present, split_bldg, _, _ = _load_meter_data(
        file_name, file_map, sheet_map, all_sheet_keys, prev_file,
        key_prefix="t1",
    )
    from filters import brand_search_bar, render_active_filters
    brand_search_bar("t1")
    render_active_filters("t1")

    # Anomaly detection — single page, no sub-tabs
    prev_data = file_map.get(prev_file) if prev_file else None
    prev_sheets = sheet_map.get(prev_file, []) if prev_file else None
    prev_lbl = "전월"
    yoy_data = file_map.get(yoy_file) if yoy_file else None
    yoy_sheets = sheet_map.get(yoy_file, []) if yoy_file else None
    yoy_lbl = "전년동월"
    render_anomaly_tab(
        cur_df, file_name, file_map[file_name], all_sheet_keys,
        split_by_building=split_bldg,
        prev_file_data=prev_data,
        prev_sheet_keys=prev_sheets,
        prev_label=prev_lbl,
        yoy_file_data=yoy_data,
        yoy_sheet_keys=yoy_sheets,
        yoy_label=yoy_lbl,
    )


def _render_tier2_insight(file_name, file_map, sheet_map, all_sheet_keys,
                          prev_file, file_periods, yoy_file=None,
                          meter_files=None):
    """Tier 2: 현황 분석 — 'Why is it anomalous?'"""
    has_meter = "검침 내역" in all_sheet_keys
    _UTIL_SHEETS = {WATER_SHEET_NAME, HOTWATER_SHEET_NAME, ELECTRICITY_SHEET_NAME}
    has_util = any(s.strip() in _UTIL_SHEETS for s in all_sheet_keys)

    if not has_meter and not has_util:
        st.info("인사이트 분석을 위해 검침 내역 또는 유틸리티 시트가 필요합니다.")
        return

    # Load meter data (renders filter widgets) BEFORE tabs
    cur_df = ref_df = None
    present = []
    split_bldg = False
    ehp_sheet = None
    from filters import render_active_filters as _raf
    if has_meter:
        cur_df, present, split_bldg, ref_df, ehp_sheet = _load_meter_data(
            file_name, file_map, sheet_map, all_sheet_keys, prev_file,
            key_prefix="t2",
        )
        _raf("t2")
    elif has_util:
        # No meter sheet — render standalone filters for utility summary
        import pandas as _pd_u
        _load_u = lambda r, c: _try_load_sheet(r, c, file_name, file_map, sheet_map)
        _u_parts = [d for d in [
            _load_u(read_water_sheet, WATER_SHEET_NAME),
            _load_u(read_hotwater_sheet, HOTWATER_SHEET_NAME),
            _load_u(read_electricity_sheet, ELECTRICITY_SHEET_NAME),
        ] if d is not None and not d.empty]
        if _u_parts:
            _u_ref = _pd_u.concat(_u_parts, ignore_index=True)
            show_filter_widgets(_u_ref, "t2")

    # Build tab list dynamically
    tab_labels = []
    tab_keys = []
    if has_util:
        tab_labels.append("📊 유틸리티 분석")
        tab_keys.append("summary")
    if has_meter:
        tab_labels += ["💰 비용 분석", "⚡ 효율 분석"]
        tab_keys += ["cost", "eff"]
        tab_labels.append("📈 기간 비교")
        tab_keys.append("period")
        if prev_file or yoy_file:
            tab_labels.append("🔄 공실 현황")
            tab_keys.append("vacancy")

    import streamlit_antd_components as _sac2
    _t2_sel = _sac2.segmented(
        [_sac2.SegmentedItem(label=lbl) for lbl in tab_labels],
        key="t2_seg", use_container_width=True,
    )
    st.divider()

    _t2_key = tab_keys[tab_labels.index(_t2_sel)] if _t2_sel in tab_labels else tab_keys[0]

    # Load YoY meter data if available
    yoy_cur_df = None
    if yoy_file and has_meter:
        _yoy_sheets = sheet_map.get(yoy_file, [])
        yoy_cur_df, _ = _load_meter_data_silent(
            yoy_file, file_map, _yoy_sheets, _yoy_sheets, None,
        )

    if _t2_key == "cost" and cur_df is not None:
        render_cross_tab(
            cur_df, file_name, file_map[file_name], all_sheet_keys,
            split_by_building=split_bldg,
            yoy_df=yoy_cur_df,
            yoy_file=yoy_file,
            yoy_file_data=file_map.get(yoy_file) if yoy_file else None,
            yoy_sheet_names=sheet_map.get(yoy_file, []),
            billing_period=file_periods.get(file_name),
            yoy_billing_period=file_periods.get(yoy_file) if yoy_file else None,
        )

    elif _t2_key == "eff" and cur_df is not None:
        render_efficiency_tab(
            cur_df, present,
            file_name=file_name,
            file_data=file_map[file_name],
            ehp_sheet=ehp_sheet,
            split_by_building=split_bldg,
            yoy_df=yoy_cur_df,
            billing_period=file_periods.get(file_name),
            yoy_billing_period=file_periods.get(yoy_file) if yoy_file else None,
            sheet_names=all_sheet_keys,
        )

    elif _t2_key == "period" and cur_df is not None:
        import streamlit_antd_components as _sac_period
        _period_modes = []
        if prev_file:
            _period_modes.append("📈 전월 대비")
        if yoy_file:
            _period_modes.append("📅 전년 대비")
        if not _period_modes:
            st.info("비교 데이터가 없습니다. 전월 또는 전년 파일을 함께 업로드하세요.")
        else:
            _period_sel = _sac_period.segmented(
                [_sac_period.SegmentedItem(label=lbl) for lbl in _period_modes],
                key="t2_period_seg", use_container_width=True,
            ) if len(_period_modes) > 1 else _period_modes[0]

            if _period_sel == "📈 전월 대비":
                render_mom_tab(
                    cur_df, present,
                    billing_period=file_periods.get(file_name),
                    prev_billing_period=file_periods.get(prev_file) if prev_file else None,
                    prev_file=prev_file,
                    all_files=meter_files if meter_files and len(meter_files) >= 3 else None,
                    file_map=file_map,
                    file_periods=file_periods,
                    sheet_map=sheet_map,
                )
            elif _period_sel == "📅 전년 대비":
                from tab_yoy import render_yoy_tab
                render_yoy_tab(
                    cur_df, present,
                    billing_period=file_periods.get(file_name),
                    yoy_file=yoy_file,
                    yoy_period=file_periods.get(yoy_file) if yoy_file else None,
                    file_map=file_map,
                    sheet_map=sheet_map,
                )


    elif _t2_key == "vacancy" and cur_df is not None:
        from tab_vacancy import render_vacancy_tab
        render_vacancy_tab(
            cur_df,
            prev_file=prev_file,
            yoy_file=yoy_file,
            file_map=file_map,
            sheet_map=sheet_map,
            billing_period=file_periods.get(file_name),
            prev_billing_period=file_periods.get(prev_file) if prev_file else None,
            yoy_period=file_periods.get(yoy_file) if yoy_file else None,
        )

    elif _t2_key == "summary":
        _load = lambda r, c, fn=None: _try_load_sheet(
            r, c, fn or file_name, file_map, sheet_map)

        w_df  = _load(read_water_sheet,       WATER_SHEET_NAME)
        hw_df = _load(read_hotwater_sheet,     HOTWATER_SHEET_NAME)
        el_df = _load(read_electricity_sheet,  ELECTRICITY_SHEET_NAME)
        b_df  = _load(read_billing_sheet,      BILLING_SHEET_NAME)

        if all(d is None for d in [w_df, hw_df, el_df]):
            st.error(t("no_util_sheets"))
            return

        pw_df  = _load(read_water_sheet,       WATER_SHEET_NAME,       prev_file) if prev_file else None
        phw_df = _load(read_hotwater_sheet,     HOTWATER_SHEET_NAME,    prev_file) if prev_file else None
        pel_df = _load(read_electricity_sheet,  ELECTRICITY_SHEET_NAME, prev_file) if prev_file else None
        pb_df  = _load(read_billing_sheet,      BILLING_SHEET_NAME,     prev_file) if prev_file else None

        yw_df  = _load(read_water_sheet,       WATER_SHEET_NAME,       yoy_file) if yoy_file else None
        yhw_df = _load(read_hotwater_sheet,     HOTWATER_SHEET_NAME,    yoy_file) if yoy_file else None
        yel_df = _load(read_electricity_sheet,  ELECTRICITY_SHEET_NAME, yoy_file) if yoy_file else None
        yb_df  = _load(read_billing_sheet,      BILLING_SHEET_NAME,     yoy_file) if yoy_file else None

        _sb = st.session_state.get("t2_building", ["All"])
        _sf = st.session_state.get("t2_floor", ["All"])
        _gm = st.session_state.get("t2_gongshil", "All")
        _bs = st.session_state.get("t2_brand_search", "").strip().lower()
        _split = not ("All" in _sb and "All" in _sf)
        def _flt(d):
            if d is None or d.empty:
                return d
            return apply_sheet_filter(d, _sb, _sf, _gm, _bs)
        w_df, hw_df, el_df = _flt(w_df), _flt(hw_df), _flt(el_df)
        b_df = _flt(b_df)
        pw_df, phw_df, pel_df = _flt(pw_df), _flt(phw_df), _flt(pel_df)
        pb_df = _flt(pb_df)
        yw_df, yhw_df, yel_df = _flt(yw_df), _flt(yhw_df), _flt(yel_df)
        yb_df = _flt(yb_df)

        render_summary_view(
            w_df, hw_df, el_df,
            split_by_building=_split,
            prev_water_df=pw_df, prev_hotwater_df=phw_df,
            prev_elec_df=pel_df,
            billing_period=file_periods.get(file_name),
            prev_billing_period=file_periods.get(prev_file) if prev_file else None,
            yoy_water_df=yw_df, yoy_hotwater_df=yhw_df,
            yoy_elec_df=yel_df,
            yoy_billing_period=file_periods.get(yoy_file) if yoy_file else None,
            billing_df=b_df, prev_billing_df=pb_df, yoy_billing_df=yb_df,
            meter_df=cur_df,
        )


def _render_tier3_detail(file_name, file_map, sheet_map, all_sheet_keys,
                         sheet_keys, prev_file, file_periods, bins, tail, q, debug,
                         yoy_file=None):
    """Tier 3: Detail Viewing — 'Show me the raw data'"""
    _PROFILE_LABEL = "🏢 브랜드 프로필"
    default_sheet = "검침 내역" if "검침 내역" in sheet_keys else sheet_keys[0]
    sheet_name = st.selectbox(
        t("select_sheet"), sheet_keys,
        index=sheet_keys.index(default_sheet),
        key=f"sheet_{file_name}",
    )

    stripped = sheet_name.strip()

    def _load_yoy_sheet(reader, sheet_const):
        """Load a sheet from the YoY file, returning None on failure."""
        if not yoy_file:
            return None
        return _try_load_sheet(reader, sheet_const, yoy_file, file_map, sheet_map)

    _yoy_period = file_periods.get(yoy_file) if yoy_file else None

    # ── Route: HVAC ──────────────────────────────────────────────────────────
    if stripped == HVAC_SHEET_NAME:
        from billing import render_hvac_view
        try:
            hvac_df = read_hvac_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse HVAC sheet: {e}")
            st.stop()
        render_hvac_view(hvac_df)
        return

    # ── Route: Hot water ─────────────────────────────────────────────────────
    if stripped == HOTWATER_SHEET_NAME:
        try:
            hw_df = read_hotwater_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse hot water sheet: {e}")
            st.stop()
        prev_hw = _load_prev_sheet(read_hotwater_sheet, HOTWATER_SHEET_NAME,
                                   prev_file, file_map, sheet_map)
        render_hotwater_view(
            hw_df, prev_df=prev_hw,
            billing_period=file_periods.get(file_name),
            prev_billing_period=file_periods.get(prev_file) if prev_file else None,
            yoy_df=_load_yoy_sheet(read_hotwater_sheet, HOTWATER_SHEET_NAME),
            yoy_billing_period=_yoy_period,
        )
        return

    # ── Route: Electricity ───────────────────────────────────────────────────
    if stripped == ELECTRICITY_SHEET_NAME:
        try:
            elec_df = read_electricity_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse electricity sheet: {e}")
            st.stop()
        prev_elec = _load_prev_sheet(read_electricity_sheet, ELECTRICITY_SHEET_NAME,
                                     prev_file, file_map, sheet_map)
        render_electricity_view(
            elec_df, prev_df=prev_elec,
            billing_period=file_periods.get(file_name),
            prev_billing_period=file_periods.get(prev_file) if prev_file else None,
            yoy_df=_load_yoy_sheet(read_electricity_sheet, ELECTRICITY_SHEET_NAME),
            yoy_billing_period=_yoy_period,
        )
        return

    # ── Route: Water ─────────────────────────────────────────────────────────
    if stripped == WATER_SHEET_NAME:
        try:
            water_df = read_water_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse water sheet: {e}")
            st.stop()
        prev_water = _load_prev_sheet(read_water_sheet, WATER_SHEET_NAME,
                                      prev_file, file_map, sheet_map)
        render_water_view(
            water_df, prev_df=prev_water,
            billing_period=file_periods.get(file_name),
            prev_billing_period=file_periods.get(prev_file) if prev_file else None,
            yoy_df=_load_yoy_sheet(read_water_sheet, WATER_SHEET_NAME),
            yoy_billing_period=_yoy_period,
        )
        return

    # ── Route: Billing ───────────────────────────────────────────────────────
    if stripped == BILLING_SHEET_NAME:
        try:
            billing_df = read_billing_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse billing sheet: {e}")
            st.stop()
        prev_billing = _load_prev_sheet(read_billing_sheet, BILLING_SHEET_NAME,
                                        prev_file, file_map, sheet_map)
        render_billing_view(
            billing_df, prev_df=prev_billing,
            billing_period=file_periods.get(file_name),
            prev_billing_period=file_periods.get(prev_file) if prev_file else None,
            yoy_df=_load_yoy_sheet(read_billing_sheet, BILLING_SHEET_NAME),
            yoy_billing_period=_yoy_period,
        )
        return

    # ── Route: EHP ───────────────────────────────────────────────────────────
    if stripped == EHP_OAC_SHEET_NAME:
        render_ehp_view(file_name, file_map[file_name], sheet_name)
        return

    # ── Route: 검침 내역 ─────────────────────────────────────────────────────
    prev_meter_sheet = _find_sheet(
        sheet_map.get(prev_file, []), "검침 내역"
    ) if prev_file else None
    render_meter_view(
        file_name, file_map, sheet_name, bins, tail, q, debug,
        prev_file_name=prev_file, prev_sheet_name=prev_meter_sheet,
        billing_period=file_periods.get(file_name),
        prev_billing_period=file_periods.get(prev_file) if prev_file else None,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="유틸리티 분석", layout="wide",
                       page_icon="📊")
    st.markdown("""
<style>
/* ── Layout: clean, breathable ── */
.block-container { padding-top: 0.6rem !important; padding-bottom: 1rem !important; }
header[data-testid="stHeader"] { height: 0; }

/* ── Typography: Inter-style hierarchy ── */
h1 { font-weight: 800 !important; letter-spacing: -0.03em !important; font-size: 1.65rem !important; }
h2 { font-weight: 700 !important; letter-spacing: -0.02em !important; font-size: 1.2rem !important; }
h3 { font-weight: 700 !important; font-size: 1.05rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid rgba(128,128,128,0.15);
}
.stTabs [data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
}
.stTabs [aria-selected="true"] {
    font-weight: 700 !important;
    border-bottom: 2.5px solid #4C72B0 !important;
}

/* ── Metrics: refined + compact ── */
[data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
}
[data-testid="stMetricDelta"] {
    font-size: 0.74rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
}

/* ── Container borders: subtle lift ── */
[data-testid="stExpander"],
div[data-testid="stVerticalBlock"] > div[style*="border"] {
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(128,128,128,0.08), 0 4px 12px rgba(128,128,128,0.04);
}

/* ── DataFrames: clean, tight rows ── */
.stDataFrame [data-testid="stDataFrameResizable"] {
    font-size: 0.83rem !important;
    border-radius: 8px;
}

/* ── Sidebar: refined (no forced background — respects dark mode) ── */
section[data-testid="stSidebar"] hr {
    margin: 0.5rem 0 !important;
    opacity: 0.15;
}
section[data-testid="stSidebar"] .stSubheader {
    font-size: 0.88rem !important;
    font-weight: 700 !important;
}

/* ── Chart containers: elevated ── */
[data-testid="stPlotlyChart"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Buttons: premium feel ── */
.stButton button, .stDownloadButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1);
    letter-spacing: 0.01em;
}
.stButton button:hover, .stDownloadButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* ── Selectbox/multiselect: rounded ── */
[data-baseweb="select"] > div {
    border-radius: 8px !important;
    transition: border-color 0.2s ease;
}

/* ── Slider: accent color ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #4C72B0 !important;
}

/* ── Caption: subtle ── */
.stCaption { opacity: 0.65; }

/* ── Dividers: lighter ── */
hr { opacity: 0.15 !important; margin: 0.8rem 0 !important; }

/* ── Info/warning/success boxes: refined ── */
.stAlert { border-radius: 10px !important; font-size: 0.88rem !important; }
</style>
""", unsafe_allow_html=True)
    st.title("Utility Analysis Dashboard")

    uploads, bins, tail, q = setup_sidebar()

    if not uploads:
        st.info(t("upload_prompt"))
        st.stop()

    # ── Load files ───────────────────────────────────────────────────────────
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

    # ── Detect billing periods and sort files ────────────────────────────────
    def _parse_period(p: str | None) -> tuple:
        if not p:
            return (0, 0)
        m = re.match(r"(\d+)년\s*(\d+)월", p)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    file_periods: Dict[str, str | None] = {
        fname: get_billing_period(fname, fdata)
        for fname, fdata in file_map.items()
    }
    sorted_files = sorted(file_map.keys(),
                          key=lambda f: _parse_period(file_periods[f]),
                          reverse=True)
    meter_files = [f for f in sorted_files if _parse_period(file_periods[f]) > (0, 0)]
    prev_file = meter_files[1] if len(meter_files) > 1 else None

    # Detect YoY file: same month, previous year
    def _find_yoy_file(target_file: str) -> str | None:
        y, m = _parse_period(file_periods.get(target_file))
        if y == 0:
            return None
        for f in meter_files:
            fy, fm = _parse_period(file_periods.get(f))
            if fy == y - 1 and fm == m:
                return f
        return None

    def _file_label(fname: str) -> str:
        period = file_periods.get(fname)
        return f"{period} — {fname}" if period else fname

    file_name = st.selectbox(t("select_file"), sorted_files, index=0,
                             format_func=_file_label)

    if prev_file is None:
        st.warning("이전 달 파일이 없습니다 — 월별 변화량을 계산할 수 없습니다.")

    all_sheet_keys = sheet_map[file_name]

    SUPPORTED_SHEETS = {
        "검침 내역", BILLING_SHEET_NAME, EHP_OAC_SHEET_NAME, HVAC_SHEET_NAME,
        WATER_SHEET_NAME, HOTWATER_SHEET_NAME, ELECTRICITY_SHEET_NAME,
    }
    sheet_keys = [s for s in all_sheet_keys if s.strip() in SUPPORTED_SHEETS]
    if not sheet_keys:
        st.warning(t("no_sheets_warn"))
        st.stop()

    # ── Quick risk count for nav badge ─────────────────────────────────────
    _risk_badge = ""
    if "검침 내역" in all_sheet_keys:
        _cache_key = f"_risk_cnt_{file_name}"
        if _cache_key not in st.session_state:
            _qdf, _qpresent = _load_meter_data_silent(
                file_name, file_map, sheet_map, all_sheet_keys, prev_file)
            if _qdf is not None:
                try:
                    from anomaly_features import build_anomaly_df as _q_build
                    _q_sheets = _load_all_sheets(file_name, file_map[file_name], all_sheet_keys)
                    _q_adf = _q_build(meter_df=_qdf,
                                      billing_df=_q_sheets.get(BILLING_SHEET_NAME),
                                      elec_df=_q_sheets.get(ELECTRICITY_SHEET_NAME),
                                      water_df=_q_sheets.get(WATER_SHEET_NAME),
                                      hotwater_df=_q_sheets.get(HOTWATER_SHEET_NAME))
                    _n_risk = int((_q_adf["risk_level"].isin(["🔴 위험", "🟠 주의"])).sum())
                    st.session_state[_cache_key] = _n_risk
                except Exception:
                    st.session_state[_cache_key] = 0
        _n_risk = st.session_state.get(_cache_key, 0)
        _risk_badge = f" ({_n_risk})" if _n_risk else ""

    # ── Handle brand→profile navigation ────────────────────────────────────
    _NAV_ANOMALY = t("nav_anomaly")
    _NAV_INSIGHT = t("nav_insight")
    _NAV_PROFILE = t("nav_profile")
    _NAV_DETAIL  = t("nav_detail")
    _NAV_ANOMALY_BADGE = f"{_NAV_ANOMALY}{_risk_badge}"

    import streamlit_antd_components as sac
    _nav_key = f"nav_{file_name}"
    _valid_labels = {_NAV_ANOMALY_BADGE, _NAV_INSIGHT, _NAV_PROFILE, _NAV_DETAIL}

    # Handle "back from profile" request (must happen BEFORE widget)
    if st.session_state.pop("_profile_do_return", False):
        _return_to = st.session_state.pop("_profile_return_nav", None)
        if _return_to:
            st.session_state[_nav_key] = _return_to

    # If brand was clicked, switch to profile (must happen BEFORE widget)
    if st.session_state.get("_goto_profile_brand"):
        _prev_nav = st.session_state.get(_nav_key)
        if _prev_nav and _prev_nav != _NAV_PROFILE:
            st.session_state["_profile_return_nav"] = _prev_nav
        st.session_state[_nav_key] = _NAV_PROFILE
    elif st.session_state.get(_nav_key) not in _valid_labels:
        st.session_state[_nav_key] = _NAV_ANOMALY_BADGE
    with st.sidebar:
        nav_mode = sac.tabs(
            [sac.TabsItem(label=_NAV_ANOMALY_BADGE),
             sac.TabsItem(label=_NAV_INSIGHT),
             sac.TabsItem(label=_NAV_PROFILE),
             sac.TabsItem(label=_NAV_DETAIL)],
            index=0,
            position="left",
            height=220,
            key=_nav_key,
        )

        # ── "← 돌아가기" button (visible when navigated to profile via shortcut)
        _return_nav = st.session_state.get("_profile_return_nav")
        if _return_nav and nav_mode == _NAV_PROFILE:
            if st.button("← 돌아가기", key="sidebar_back_btn",
                         use_container_width=True, type="primary"):
                st.session_state["_profile_do_return"] = True
                st.rerun()

        # ── Brands of Interest (관심 브랜드) ──────────────────────────────
        st.divider()
        _watchlist = st.session_state.get("_brand_watchlist", [])
        if _watchlist:
            st.subheader("⭐ 관심 브랜드")
            # Quick-click buttons for each watchlisted brand
            for _wi, _wb in enumerate(_watchlist):
                _wc1, _wc2 = st.columns([4, 1])
                with _wc1:
                    if st.button(
                        f"🔍 {_wb}", key=f"watchlist_goto_{_wi}",
                        use_container_width=True,
                    ):
                        st.session_state["_goto_profile_brand"] = _wb
                        st.rerun()
                with _wc2:
                    if st.button("✕", key=f"watchlist_rm_{_wi}",
                                 help=f"{_wb} 제거"):
                        _watchlist.remove(_wb)
                        st.session_state["_brand_watchlist"] = _watchlist
                        st.rerun()

            # Manage watchlist
            with st.expander("관심 브랜드 관리"):
                # Add from all brands
                _all_brands_for_wl = sorted(
                    set(st.session_state.get("_all_brand_names", []))
                    - set(_watchlist)
                )
                if _all_brands_for_wl:
                    _add_brand = st.selectbox(
                        "브랜드 추가", [""] + _all_brands_for_wl,
                        key="watchlist_add_sel",
                        label_visibility="collapsed",
                        placeholder="브랜드 선택하여 추가…",
                    )
                    if _add_brand:
                        _watchlist.append(_add_brand)
                        st.session_state["_brand_watchlist"] = _watchlist
                        st.rerun()
                if st.button("전체 초기화", key="watchlist_clear"):
                    st.session_state["_brand_watchlist"] = []
                    st.rerun()

        # ── Report generation ─────────────────────────────────────────────
        st.divider()
        st.subheader("📄 보고서 생성")
        _REPORT_OPTIONS = ["종합 리포트", "이상감지 리포트", "비용·효율 리포트", "검침 상세"]
        _SCOPE_MAP = {"종합 리포트": "종합", "이상감지 리포트": "이상감지",
                      "비용·효율 리포트": "인사이트", "검침 상세": "상세"}
        _report_scope_label = st.radio(
            "보고서 범위",
            _REPORT_OPTIONS,
            key="report_scope",
            horizontal=True,
        )
        _report_scope = _SCOPE_MAP[_report_scope_label]
        _report_pdf_key = f"sidebar_report_{file_name}"
        if st.button("📄 보고서 생성", key="gen_report_sidebar", use_container_width=True):
            with st.spinner("보고서 생성 중…"):
                _pdf = _generate_report(
                    _report_scope, file_name, file_map, sheet_map,
                    all_sheet_keys, prev_file, file_periods, tail,
                )
                if _pdf:
                    st.session_state[_report_pdf_key] = _pdf
                    _fn_map = {"종합": "종합_리포트", "이상감지": "이상감지_리포트",
                               "인사이트": "비용효율_리포트", "상세": "검침상세_리포트"}
                    st.session_state[f"{_report_pdf_key}_fn"] = f"{_fn_map[_report_scope]}.pdf"
                    st.success("생성 완료!")
                else:
                    st.error("데이터가 부족하여 보고서를 생성할 수 없습니다.")
        if _report_pdf_key in st.session_state:
            st.download_button(
                "⬇️ PDF 다운로드",
                st.session_state[_report_pdf_key],
                file_name=st.session_state.get(f"{_report_pdf_key}_fn", "보고서.pdf"),
                mime="application/pdf",
                key="dl_report_sidebar",
                use_container_width=True,
            )

        # ── Excel export ────────────────────────────────────────────────
        st.subheader("📊 데이터 내보내기")
        _xl_key = f"sidebar_xl_{file_name}"
        if st.button("📊 Excel 내보내기", key="gen_xl_sidebar", use_container_width=True):
            with st.spinner("Excel 생성 중…"):
                try:
                    _xl = _generate_excel_export(
                        file_name, file_map, sheet_map, all_sheet_keys, prev_file,
                    )
                    st.session_state[_xl_key] = _xl
                    st.success("생성 완료!")
                except Exception as _e:
                    st.error(f"Excel 생성 실패: {_e}")
        if _xl_key in st.session_state:
            st.download_button(
                "⬇️ Excel 다운로드",
                st.session_state[_xl_key],
                file_name="유틸리티_분석.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_xl_sidebar",
                use_container_width=True,
            )

        st.divider()
        debug = st.checkbox(t("debug"), value=False)

    # ── Route to tier ────────────────────────────────────────────────────────
    yoy_file = _find_yoy_file(file_name)

    # Clear return-to-profile state when user navigates away from profile manually
    if nav_mode != _NAV_PROFILE:
        st.session_state.pop("_profile_return_nav", None)

    if nav_mode == _NAV_ANOMALY_BADGE:
        _render_tier1_anomaly(
            file_name, file_map, sheet_map, all_sheet_keys,
            prev_file, file_periods, meter_files, yoy_file=yoy_file,
        )

    elif nav_mode == _NAV_INSIGHT:
        _render_tier2_insight(
            file_name, file_map, sheet_map, all_sheet_keys,
            prev_file, file_periods, yoy_file=yoy_file,
            meter_files=meter_files,
        )

    elif nav_mode == _NAV_PROFILE:
        # Handle brand navigation from anomaly tab
        _goto = st.session_state.pop("_goto_profile_brand", None)
        if _goto:
            st.session_state["profile_brand_select"] = _goto

        if "검침 내역" not in all_sheet_keys:
            st.info("브랜드 프로필을 위해 검침 내역 시트가 필요합니다.")
        else:
            from brand_profile import render_brand_profile_tab
            cur_df, present, split_bldg, ref_df, _ = _load_meter_data(
                file_name, file_map, sheet_map, all_sheet_keys, prev_file,
                key_prefix="t_profile",
            )
            # Build anomaly_df for profile badge
            _profile_anom_df = None
            try:
                from anomaly_features import build_anomaly_df as _build_anom
                _profile_sheets = {
                    k: _try_load_sheet(r, k, file_name, file_map, sheet_map)
                    for r, k in [
                        (read_billing_sheet, BILLING_SHEET_NAME),
                        (read_electricity_sheet, ELECTRICITY_SHEET_NAME),
                        (read_water_sheet, WATER_SHEET_NAME),
                        (read_hotwater_sheet, HOTWATER_SHEET_NAME),
                    ]
                }
                _profile_anom_df = _build_anom(
                    meter_df=cur_df,
                    billing_df=_profile_sheets.get(BILLING_SHEET_NAME),
                    elec_df=_profile_sheets.get(ELECTRICITY_SHEET_NAME),
                    water_df=_profile_sheets.get(WATER_SHEET_NAME),
                    hotwater_df=_profile_sheets.get(HOTWATER_SHEET_NAME),
                )
            except Exception:
                pass

            render_brand_profile_tab(
                cur_df, ref_df, present,
                tail=st.session_state.get("tail", 20),
                billing_period=file_periods.get(file_name),
                prev_billing_period=file_periods.get(prev_file) if prev_file else None,
                billing_df=_try_load_sheet(read_billing_sheet, BILLING_SHEET_NAME,
                                          file_name, file_map, sheet_map),
                water_df=_try_load_sheet(read_water_sheet, WATER_SHEET_NAME,
                                        file_name, file_map, sheet_map),
                hotwater_df=_try_load_sheet(read_hotwater_sheet, HOTWATER_SHEET_NAME,
                                           file_name, file_map, sheet_map),
                electricity_df=_try_load_sheet(read_electricity_sheet, ELECTRICITY_SHEET_NAME,
                                              file_name, file_map, sheet_map),
                anomaly_df=_profile_anom_df,
            )

    elif nav_mode == _NAV_DETAIL:
        _render_tier3_detail(
            file_name, file_map, sheet_map, all_sheet_keys,
            sheet_keys, prev_file, file_periods, bins, tail, q, debug,
            yoy_file=yoy_file,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.exception(e)
        raise
