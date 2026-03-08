"""app.py — Utility Analysis Dashboard: page config, routing, orchestration."""
from typing import Dict

import streamlit as st

from data import (
    get_sheet_names,
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
from meter_view import render_meter_view


def main():
    st.set_page_config(page_title="Utility Analysis Dashboard", layout="wide")
    st.title("Utility Analysis Dashboard")

    uploads, bins, tail, q_change, q_pct, debug = setup_sidebar()

    if not uploads:
        st.info("Upload at least one file.")
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

    file_name = st.selectbox("Select file", list(file_map.keys()))
    all_sheet_keys = sheet_map[file_name]

    SUPPORTED_SHEETS = {
        "검침 내역", BILLING_SHEET_NAME, EHP_OAC_SHEET_NAME, HVAC_SHEET_NAME,
        WATER_SHEET_NAME, HOTWATER_SHEET_NAME, ELECTRICITY_SHEET_NAME,
    }
    sheet_keys = [s for s in all_sheet_keys if s.strip() in SUPPORTED_SHEETS]
    if not sheet_keys:
        st.warning("No supported sheets found in this file. Expected '검침 내역' or '수도광열비 부과 내역'.")
        st.stop()

    _SUMMARY_VIRTUAL = "📊 통합 분석"
    _UTIL_SHEET_NAMES = {WATER_SHEET_NAME, HOTWATER_SHEET_NAME, ELECTRICITY_SHEET_NAME}
    _has_any_util = any(s.strip() in _UTIL_SHEET_NAMES for s in all_sheet_keys)
    display_keys = sheet_keys + ([_SUMMARY_VIRTUAL] if _has_any_util else [])

    default_sheet = "검침 내역" if "검침 내역" in display_keys else display_keys[0]
    sheet_name = st.selectbox(
        "Select sheet", display_keys,
        index=display_keys.index(default_sheet),
        key=f"sheet_{file_name}",
    )

    # ── Route: 통합 분석 ───────────────────────────────────────────────────────
    if sheet_name == _SUMMARY_VIRTUAL:
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
            st.error("로드 가능한 유틸리티 시트가 없습니다.")
            st.stop()
        render_summary_view(_w_df, _hw_df, _el_df)
        return

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
    render_meter_view(file_name, file_map, sheet_name, bins, tail, q_change, q_pct, debug)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.exception(e)
        raise
