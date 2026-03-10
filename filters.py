"""filters.py — Shared building / floor / 공실 filter widgets.

Public API
----------
show_filter_widgets(ref_df, key_prefix)
    Renders the three filter controls and returns (sel_bldg, sel_floor, gong_mode).
    Use this when you need to apply the same selections to multiple DataFrames.

apply_sheet_filter(df, sel_bldg, sel_floor, gong_mode)
    Applies selections to a row-level DataFrame and returns the filtered copy.

render_sheet_filters(df, key_prefix)
    Convenience wrapper: show_filter_widgets + apply_sheet_filter in one call.
    For water, hotwater, electricity views (each row = one tenant unit).

render_meter_filters(raw_df, key_prefix)
    For meter-based analysis views.
    Runs aggregate_by_brand + split_brand_by_floor + 공실 filter.
    Returns (cur_df, active_buildings, active_floors, gong_mode).
"""
import streamlit as st

from features import (
    get_simple_floors, parse_floor_value,
    aggregate_by_brand, split_brand_by_floor,
)
from lang import t

_VACANCY_OPTS = ["All", "Exclude Vacancy", "Vacancy Only"]
_VACANCY_LABELS = {
    "All":             {"ko": "전체",     "en": "All"},
    "Exclude Vacancy": {"ko": "공실 제외", "en": "Exclude Vacancy"},
    "Vacancy Only":    {"ko": "공실만",    "en": "Vacancy Only"},
}


def _all_toggle(key: str) -> callable:
    def cb():
        sel = st.session_state[key]
        if not sel:
            st.session_state[key] = ["All"]
        elif sel[-1] == "All":
            st.session_state[key] = ["All"]
        elif "All" in sel:
            st.session_state[key] = [s for s in sel if s != "All"]
    return cb


def _apply_gongshil(df, mode: str):
    if "brand" not in df.columns:
        return df
    mask = df["brand"].astype(str).str.contains("공실", na=False)
    if mode == "Exclude Vacancy":
        return df[~mask].copy()
    if mode == "Vacancy Only":
        return df[mask].copy()
    return df


# ── Primitive: show widgets only ─────────────────────────────────────────────

def show_filter_widgets(ref_df, key_prefix: str = ""):
    """Render brand search + building / floor / 공실 widgets and return selections.

    Returns
    -------
    (sel_bldg, sel_floor, gong_mode, brand_search)
        brand_search : lowercase search string (empty = no filter)
    """
    all_buildings = (
        sorted(ref_df["building"].dropna().astype(str).str.strip().unique())
        if "building" in ref_df.columns else []
    )
    all_floors = get_simple_floors(ref_df) if "floor" in ref_df.columns else []
    has_gong = (
        ref_df["brand"].astype(str).str.contains("공실", na=False).any()
        if "brand" in ref_df.columns else False
    )
    _lang = st.session_state.get("lang", "ko")

    bkey = f"{key_prefix}_building"
    fkey = f"{key_prefix}_floor"
    gkey = f"{key_prefix}_gongshil"
    skey = f"{key_prefix}_brand_search"

    # Read from session state — the widget is rendered above tabs via brand_search_bar()
    brand_search = st.session_state.get(skey, "").strip().lower()

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_bldg = st.multiselect(
            t("building"), ["All"] + all_buildings, default=["All"],
            key=bkey, on_change=_all_toggle(bkey),
        )
    with fc2:
        sel_floor = st.multiselect(
            t("floor"), ["All"] + all_floors, default=["All"],
            key=fkey, on_change=_all_toggle(fkey),
        )
    with fc3:
        gong_mode = st.radio(
            t("vacancy"), _VACANCY_OPTS,
            format_func=lambda x: _VACANCY_LABELS[x][_lang],
            index=0, horizontal=True, disabled=not has_gong,
            key=gkey,
        )

    return sel_bldg, sel_floor, gong_mode, brand_search


# ── Primitive: apply selections ───────────────────────────────────────────────

def apply_sheet_filter(df, sel_bldg, sel_floor, gong_mode: str, brand_search: str = ""):
    """Apply brand search + building / floor / 공실 selections to a row-level DataFrame."""
    if brand_search and "brand" in df.columns:
        df = df[df["brand"].astype(str).str.lower().str.contains(brand_search, na=False)].copy()

    if "building" in df.columns and "All" not in sel_bldg and sel_bldg:
        df = df[df["building"].astype(str).str.strip().isin(sel_bldg)].copy()

    if "floor" in df.columns and "All" not in sel_floor and sel_floor:
        sel_set = set(sel_floor)
        df = df[
            df["floor"].apply(
                lambda v: bool(set(parse_floor_value(str(v))) & sel_set)
            )
        ].copy()

    return _apply_gongshil(df, gong_mode)


# ── Convenience wrapper for single-df sheet views ────────────────────────────

def render_sheet_filters(df, key_prefix: str = ""):
    """Show filter widgets and apply them to a single row-level DataFrame."""
    sel_bldg, sel_floor, gong_mode, brand_search = show_filter_widgets(df, key_prefix)
    return apply_sheet_filter(df, sel_bldg, sel_floor, gong_mode, brand_search)


# ── Meter-based analysis views ────────────────────────────────────────────────

def render_meter_filters(raw_df, key_prefix: str = ""):
    """Building / floor / 공실 filters for meter-based analysis views.

    Pipeline: show widgets → filter by building → aggregate_by_brand
              → split_brand_by_floor (if needed) → 공실 filter.

    Returns (cur_df, active_buildings, active_floors, gong_mode).
    Calls st.stop() if the result is empty.
    """
    all_buildings = (
        sorted(raw_df["building"].dropna().unique().tolist())
        if "building" in raw_df.columns else []
    )
    all_floors = get_simple_floors(raw_df) if "floor" in raw_df.columns else []

    sel_bldg, sel_floor, gong_mode, brand_search = show_filter_widgets(raw_df, key_prefix)

    active_buildings = all_buildings if "All" in sel_bldg else sel_bldg
    active_floors    = all_floors    if "All" in sel_floor else sel_floor
    floors_filtered  = "All" not in sel_floor

    ref_df = (
        raw_df[raw_df["building"].isin(active_buildings)].copy()
        if "building" in raw_df.columns else raw_df.copy()
    )
    if ref_df.empty:
        st.warning(t("no_data_building"))
        st.stop()

    cur_df = aggregate_by_brand(ref_df)
    if floors_filtered:
        cur_df = split_brand_by_floor(cur_df, ref_df, active_floors)

    if cur_df.empty:
        st.warning(t("no_data_floor"))
        st.stop()

    cur_df = _apply_gongshil(cur_df, gong_mode)
    if cur_df.empty:
        st.warning(t("vacancy_only") if gong_mode == "Vacancy Only" else t("vacancy_exclude"))
        st.stop()

    if brand_search and "brand" in cur_df.columns:
        cur_df = cur_df[cur_df["brand"].astype(str).str.lower().str.contains(brand_search, na=False)].copy()
        if cur_df.empty:
            st.warning(f"'{brand_search}' 검색 결과가 없습니다.")
            st.stop()

    split_by_building = not ("All" in sel_bldg and "All" in sel_floor)
    return cur_df, active_buildings, active_floors, gong_mode, split_by_building, ref_df


def brand_search_bar(key_prefix: str = "") -> None:
    """Render the brand search text input. Call this just above st.tabs().

    The filter itself is applied earlier via session state in show_filter_widgets()
    or apply_sheet_filter(). This function only renders the visible widget.
    """
    skey = f"{key_prefix}_brand_search"
    st.text_input("🔍 브랜드 검색", placeholder="브랜드명 입력...", key=skey)
