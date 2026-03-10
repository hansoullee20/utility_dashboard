"""app.py — Utility Analysis Dashboard: page config, routing, orchestration."""
import re
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
from meter_view import render_meter_view, load_meter_df, load_raw_meter_df
from filters import render_meter_filters, render_sheet_filters, show_filter_widgets, apply_sheet_filter
from data import to_numeric_series as _to_num
from tab_cross import render_cross_tab
from tab_efficiency import render_efficiency_tab
from tab_anomaly import render_anomaly_tab
from lang import t


def main():
    st.set_page_config(page_title="Utility Analysis Dashboard", layout="wide")
    st.title("Utility Analysis Dashboard")

    uploads, bins, tail, q, debug = setup_sidebar()

    if not uploads:
        st.info(t("upload_prompt"))
        st.stop()

    # ── Load files ─────────────────────────────────────────────────────────────
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

    # ── Detect billing periods and sort files ──────────────────────────────────
    def _parse_period(p: str | None) -> tuple:
        if not p:
            return (0, 0)
        m = re.match(r"(\d+)년\s*(\d+)월", p)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    _file_periods: Dict[str, str | None] = {
        fname: get_billing_period(fname, fdata)
        for fname, fdata in file_map.items()
    }
    # Sort all files: most recent first
    _sorted_files = sorted(file_map.keys(), key=lambda f: _parse_period(_file_periods[f]), reverse=True)

    # Current = most recent file with a detected period; previous = second most recent
    _meter_files = [f for f in _sorted_files if _parse_period(_file_periods[f]) > (0, 0)]
    _current_file  = _meter_files[0] if _meter_files else _sorted_files[0]
    _prev_file     = _meter_files[1] if len(_meter_files) > 1 else None

    def _file_label(fname: str) -> str:
        period = _file_periods.get(fname)
        return f"{period} — {fname}" if period else fname

    file_name = st.selectbox(t("select_file"), _sorted_files, index=0, format_func=_file_label)

    if _prev_file is None:
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

    # ── Top-level navigation ───────────────────────────────────────────────────
    _UTIL_SHEETS = {WATER_SHEET_NAME, HOTWATER_SHEET_NAME, ELECTRICITY_SHEET_NAME}
    _has_util    = any(s.strip() in _UTIL_SHEETS for s in all_sheet_keys)
    _has_meter   = "검침 내역" in all_sheet_keys

    # Build analysis options using current-language labels
    _OPT_SUMMARY = t("summary_analysis")
    _OPT_BIZ     = t("biz_analysis")
    _analysis_options = []
    if _has_util:
        _analysis_options.append(_OPT_SUMMARY)
    if _has_meter:
        _analysis_options.append(_OPT_BIZ)

    _NAV_SHEET    = t("nav_sheet_view")
    _NAV_ANALYSIS = t("nav_analysis")
    _NAV_PROFILE  = "브랜드 프로필"
    _nav_options  = [_NAV_SHEET]
    if _analysis_options:
        _nav_options.append(_NAV_ANALYSIS)
    if _has_meter:
        _nav_options.append(_NAV_PROFILE)

    _nc1, _nc2 = st.columns([2, 5])
    with _nc1:
        nav_mode = st.radio(
            "Navigation", _nav_options,
            horizontal=True, label_visibility="collapsed",
            key=f"nav_{file_name}",
        )

    # ── 분석 branch ────────────────────────────────────────────────────────────
    if nav_mode == _NAV_ANALYSIS:
        with _nc2:
            analysis_name = st.selectbox(
                t("analysis_select"), _analysis_options,
                key=f"analysis_{file_name}",
            )

        if analysis_name == _OPT_SUMMARY:
            st.header(t("summary_header"))
            def _try_load(reader, sheet_const):
                key = next((k for k in all_sheet_keys if k.strip() == sheet_const), None)
                if key is None:
                    return None
                try:
                    return reader(file_name, file_map[file_name], key)
                except Exception as e:
                    st.warning(f"{sheet_const} 로드 실패 (제외됨): {e}")
                    return None
            _w_df  = _try_load(read_water_sheet,       WATER_SHEET_NAME)
            _hw_df = _try_load(read_hotwater_sheet,    HOTWATER_SHEET_NAME)
            _el_df = _try_load(read_electricity_sheet, ELECTRICITY_SHEET_NAME)
            if all(d is None for d in [_w_df, _hw_df, _el_df]):
                st.error(t("no_util_sheets"))
                st.stop()
            # Build a combined reference df for the filter widgets
            import pandas as _pd
            _ref = _pd.concat(
                [d[["building", "floor", "brand"]] for d in [_w_df, _hw_df, _el_df]
                 if d is not None and all(c in d.columns for c in ["building", "floor", "brand"])],
                ignore_index=True,
            ).drop_duplicates()
            _sel_bldg, _sel_floor, _gong_mode, _brand_search = show_filter_widgets(_ref, key_prefix="summary")
            _split_bldg = not ("All" in _sel_bldg and "All" in _sel_floor)
            render_summary_view(
                apply_sheet_filter(_w_df,  _sel_bldg, _sel_floor, _gong_mode, _brand_search) if _w_df  is not None else None,
                apply_sheet_filter(_hw_df, _sel_bldg, _sel_floor, _gong_mode, _brand_search) if _hw_df is not None else None,
                apply_sheet_filter(_el_df, _sel_bldg, _sel_floor, _gong_mode, _brand_search) if _el_df is not None else None,
                split_by_building=_split_bldg,
            )
            return

        if analysis_name == _OPT_BIZ:
            st.header(t("biz_header"))
            _meter_sheet = next((k for k in all_sheet_keys if k.strip() == "검침 내역"), None)
            _prev_meter_sheet = None
            if _prev_file:
                _prev_meter_sheet = next(
                    (k for k in sheet_map.get(_prev_file, []) if k.strip() == "검침 내역"), None
                )
            try:
                _raw_df = load_raw_meter_df(
                    file_name, file_map, _meter_sheet,
                    prev_file_name=_prev_file, prev_sheet_name=_prev_meter_sheet,
                )
            except Exception as e:
                st.error(f"{t('meter_load_fail')}: {e}")
                st.stop()

            _cur_df, _, _, _, _split_bldg, _ = render_meter_filters(_raw_df, key_prefix="biz")

            # ── Per-area columns (needed by efficiency tab) ─────────────────────
            _size_m2 = _to_num(_cur_df["size_m2"]).replace(0, float("nan")) if "size_m2" in _cur_df.columns else None
            _size_py = _to_num(_cur_df["size_py"]).replace(0, float("nan")) if "size_py" in _cur_df.columns else None
            for _uc, (_pm2, _ppy) in {
                "water_current":  ("water_usage_per_m2",  "water_usage_per_py"),
                "hwater_current": ("hwater_usage_per_m2", "hwater_usage_per_py"),
                "elect_current":  ("elect_usage_per_m2",  "elect_usage_per_py"),
                "heat_current":   ("heat_usage_per_m2",   "heat_usage_per_py"),
            }.items():
                if _uc in _cur_df.columns:
                    _u = _to_num(_cur_df[_uc])
                    if _size_m2 is not None:
                        _cur_df[_pm2] = (_u / _size_m2).round(4)
                    if _size_py is not None:
                        _cur_df[_ppy] = (_u / _size_py).round(4)
            _allowed = ["water", "hwater", "elect", "heat"]
            _present = [p for p in _allowed if f"{p}_change" in _cur_df.columns]
            _ehp_sheet = EHP_OAC_SHEET_NAME if EHP_OAC_SHEET_NAME in all_sheet_keys else None

            # ── Tabs ────────────────────────────────────────────────────────────
            from filters import brand_search_bar as _bsb
            _bsb("biz")
            _tab_cost, _tab_eff, _tab_anom = st.tabs([
                t("biz_tab_cost"), t("biz_tab_eff"), t("biz_tab_anom"),
            ])
            with _tab_cost:
                render_cross_tab(_cur_df, file_name, file_map[file_name], all_sheet_keys,
                                 split_by_building=_split_bldg)
            with _tab_eff:
                render_efficiency_tab(
                    _cur_df, _present,
                    file_name=file_name,
                    file_data=file_map[file_name],
                    ehp_sheet=_ehp_sheet,
                    split_by_building=_split_bldg,
                )
            with _tab_anom:
                render_anomaly_tab(
                    _cur_df, file_name, file_map[file_name], all_sheet_keys,
                    split_by_building=_split_bldg,
                )
            return

    # ── 브랜드 프로필 branch ───────────────────────────────────────────────────
    if nav_mode == _NAV_PROFILE:
        from brand_profile import render_brand_profile_tab
        from features import aggregate_by_brand as _agg
        st.header("브랜드 프로필")
        _meter_sheet = next((k for k in all_sheet_keys if k.strip() == "검침 내역"), None)
        _prev_meter_sheet = next(
            (k for k in sheet_map.get(_prev_file, []) if k.strip() == "검침 내역"), None
        ) if _prev_file else None
        try:
            _raw_df = load_raw_meter_df(
                file_name, file_map, _meter_sheet,
                prev_file_name=_prev_file, prev_sheet_name=_prev_meter_sheet,
            )
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
            st.stop()
        _cur_df, _, _, _, _, _ref_df = render_meter_filters(_raw_df, key_prefix="profile")
        _allowed  = ["water", "hwater", "elect", "heat"]
        _present  = [p for p in _allowed if f"{p}_change" in _cur_df.columns]

        def _try_load_profile(reader, sheet_const):
            key = next((k for k in all_sheet_keys if k.strip() == sheet_const), None)
            if key is None:
                return None
            try:
                return reader(file_name, file_map[file_name], key)
            except Exception:
                return None

        render_brand_profile_tab(
            _cur_df, _ref_df, _present,
            tail=st.session_state.get("tail", 20),
            billing_period=_file_periods.get(file_name),
            prev_billing_period=_file_periods.get(_prev_file) if _prev_file else None,
            billing_df=_try_load_profile(read_billing_sheet,     BILLING_SHEET_NAME),
            water_df=_try_load_profile(read_water_sheet,         WATER_SHEET_NAME),
            hotwater_df=_try_load_profile(read_hotwater_sheet,   HOTWATER_SHEET_NAME),
            electricity_df=_try_load_profile(read_electricity_sheet, ELECTRICITY_SHEET_NAME),
        )
        return

    # ── 시트 보기 branch ────────────────────────────────────────────────────────
    with _nc2:
        default_sheet = "검침 내역" if "검침 내역" in sheet_keys else sheet_keys[0]
        sheet_name = st.selectbox(
            t("select_sheet"), sheet_keys,
            index=sheet_keys.index(default_sheet),
            key=f"sheet_{file_name}",
        )

    # ── Route: HVAC ────────────────────────────────────────────────────────────
    if sheet_name.strip() == HVAC_SHEET_NAME:
        from billing import render_hvac_view
        try:
            hvac_df = read_hvac_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse HVAC sheet: {e}")
            st.stop()
        render_hvac_view(hvac_df)
        return

    # ── Route: Hot water ───────────────────────────────────────────────────────
    if sheet_name.strip() == HOTWATER_SHEET_NAME:
        try:
            hw_df = read_hotwater_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse hot water sheet: {e}")
            st.stop()
        render_hotwater_view(hw_df)
        return

    # ── Route: Electricity ─────────────────────────────────────────────────────
    if sheet_name.strip() == ELECTRICITY_SHEET_NAME:
        try:
            elec_df = read_electricity_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse electricity sheet: {e}")
            st.stop()
        render_electricity_view(elec_df)
        return

    # ── Route: Water ───────────────────────────────────────────────────────────
    if sheet_name.strip() == WATER_SHEET_NAME:
        try:
            water_df = read_water_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse water sheet: {e}")
            st.stop()
        render_water_view(water_df)
        return

    # ── Route: Billing ─────────────────────────────────────────────────────────
    if sheet_name.strip() == BILLING_SHEET_NAME:
        try:
            billing_df = read_billing_sheet(file_name, file_map[file_name], sheet_name)
        except Exception as e:
            st.error(f"Failed to parse billing sheet: {e}")
            st.stop()
        render_billing_view(billing_df)
        return

    # ── Route: EHP ─────────────────────────────────────────────────────────────
    if sheet_name.strip() == EHP_OAC_SHEET_NAME:
        render_ehp_view(file_name, file_map[file_name], sheet_name)
        return

    # ── Route: 검침 내역 ────────────────────────────────────────────────────────
    _prev_meter_sheet = None
    if _prev_file:
        _prev_meter_sheet = next(
            (k for k in sheet_map.get(_prev_file, []) if k.strip() == "검침 내역"), None
        )
    render_meter_view(
        file_name, file_map, sheet_name, bins, tail, q, debug,
        prev_file_name=_prev_file, prev_sheet_name=_prev_meter_sheet,
        billing_period=_file_periods.get(file_name),
        prev_billing_period=_file_periods.get(_prev_file) if _prev_file else None,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.exception(e)
        raise
