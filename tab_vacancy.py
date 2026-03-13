"""tab_vacancy.py — 입퇴점 · 공실 현황 tab.

Compares brand lists between two periods (MoM or YoY) to detect:
  - 공실 유지 / 신규 공실 / 공실 해소
  - 입점 / 퇴점 / 이전 (relocated)
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def _is_vacant(df: pd.DataFrame) -> pd.Series:
    return df["brand"].str.strip().str.match(r"^공\s*실", na=False)


def _analyze_turnover(
    cur_df: pd.DataFrame,
    prev_df: pd.DataFrame,
) -> dict | None:
    """Compare brand lists and return turnover analysis dict, or None."""
    merge_keys = [k for k in ["brand", "building"]
                  if k in cur_df.columns and k in prev_df.columns]
    if not merge_keys:
        return None

    cur_side = cur_df[
        merge_keys + [c for c in ["floor"] if c in cur_df.columns]
    ].drop_duplicates(subset=merge_keys)
    prev_side = prev_df[
        merge_keys + [c for c in ["floor"] if c in prev_df.columns]
    ].drop_duplicates(subset=merge_keys)
    if "floor" in prev_side.columns:
        prev_side = prev_side.rename(columns={"floor": "floor_prev"})

    merged = cur_side.merge(prev_side, on=merge_keys, how="outer", indicator=True)

    matched = merged[merged["_merge"] == "both"].drop(columns=["_merge"]).copy()
    left_only = merged[merged["_merge"] == "left_only"].copy()
    right_only = merged[merged["_merge"] == "right_only"].copy()

    vacant_both = matched[_is_vacant(matched)]
    vacant_new = left_only[_is_vacant(left_only)]
    vacant_filled = right_only[_is_vacant(right_only)]

    new_raw = left_only[~_is_vacant(left_only)]
    closed_raw = right_only[~_is_vacant(right_only)]

    relocated_names = set(new_raw["brand"].str.strip()) & set(closed_raw["brand"].str.strip())
    new_brands = new_raw[~new_raw["brand"].str.strip().isin(relocated_names)]
    closed_brands = closed_raw[~closed_raw["brand"].str.strip().isin(relocated_names)]
    relocated_in = new_raw[new_raw["brand"].str.strip().isin(relocated_names)]
    relocated_out = closed_raw[closed_raw["brand"].str.strip().isin(relocated_names)]

    return {
        "left_only": left_only,
        "right_only": right_only,
        "vacant_both": vacant_both,
        "vacant_new": vacant_new,
        "vacant_filled": vacant_filled,
        "new_brands": new_brands,
        "closed_brands": closed_brands,
        "relocated_in": relocated_in,
        "relocated_out": relocated_out,
        "relocated_names": relocated_names,
    }


def _render_analysis(
    result: dict,
    prev_label: str = "전월",
    curr_label: str = "이번달",
) -> None:
    """Render vacancy + turnover KPIs and tables from analysis result."""
    vacant_both = result["vacant_both"]
    vacant_new = result["vacant_new"]
    vacant_filled = result["vacant_filled"]
    new_brands = result["new_brands"]
    closed_brands = result["closed_brands"]
    relocated_in = result["relocated_in"]
    relocated_out = result["relocated_out"]
    relocated_names = result["relocated_names"]
    left_only = result["left_only"]
    right_only = result["right_only"]

    n_vacant_both = len(vacant_both)
    n_vacant_new = len(vacant_new)
    n_vacant_filled = len(vacant_filled)
    n_in = len(new_brands)
    n_out = len(closed_brands)
    n_relocated = len(relocated_names)

    has_vacancy = n_vacant_both + n_vacant_new + n_vacant_filled > 0
    has_turnover = n_in + n_out + n_relocated > 0

    if not has_vacancy and not has_turnover:
        st.info("공실 변동 및 입퇴점 변동이 없습니다.")
        return

    # ── Vacancy KPIs ──────────────────────────────────────────────────────
    if has_vacancy:
        st.subheader("🏢 공실 현황")
        vc = st.columns(3)
        vc[0].metric("공실 유지", f"{n_vacant_both}개",
                     help=f"{prev_label}·{curr_label} 모두 공실인 호실")
        vc[1].metric("신규 공실", f"{n_vacant_new}개",
                     help=f"{prev_label}에는 입점 상태였으나 {curr_label} 공실로 전환",
                     delta=f"+{n_vacant_new}" if n_vacant_new else None,
                     delta_color="inverse")
        vc[2].metric("공실 해소", f"{n_vacant_filled}개",
                     help=f"{prev_label}에 공실이었으나 {curr_label} 입점 완료",
                     delta=f"-{n_vacant_filled}" if n_vacant_filled else None,
                     delta_color="inverse")

        _vac_ren = {"brand": "브랜드", "building": "건물", "floor": "층"}
        _vac_cols = [c for c in ["brand", "building", "floor"] if c in left_only.columns]

        _vc1, _vc2, _vc3 = st.columns(3)
        with _vc1:
            st.caption(f"🏚️ 공실 유지 ({n_vacant_both}개)")
            if n_vacant_both:
                _both_cols = [c for c in ["brand", "building", "floor"] if c in vacant_both.columns]
                disp = vacant_both[_both_cols].copy()
                disp.columns = [_vac_ren.get(c, c) for c in disp.columns]
                st.dataframe(disp.reset_index(drop=True), hide_index=True, use_container_width=True)
            else:
                st.info("없음")
        with _vc2:
            st.caption(f"📉 신규 공실 ({n_vacant_new}개)")
            if n_vacant_new:
                disp = vacant_new[_vac_cols].copy()
                disp.columns = [_vac_ren.get(c, c) for c in disp.columns]
                st.dataframe(disp.reset_index(drop=True), hide_index=True, use_container_width=True)
            else:
                st.info("없음")
        with _vc3:
            st.caption(f"📈 공실 해소 ({n_vacant_filled}개)")
            if n_vacant_filled:
                _rcols = [c for c in ["brand", "building", "floor_prev"] if c in right_only.columns]
                disp = vacant_filled[_rcols].copy()
                disp.columns = [_vac_ren.get(c, c) if c != "floor_prev" else "층" for c in disp.columns]
                st.dataframe(disp.reset_index(drop=True), hide_index=True, use_container_width=True)
            else:
                st.info("없음")

    # ── Tenant turnover ───────────────────────────────────────────────────
    if has_turnover:
        if has_vacancy:
            st.divider()
        st.subheader("🔄 입퇴점 현황")
        _parts = []
        if n_in:
            _parts.append(f"입점 {n_in}개")
        if n_out:
            _parts.append(f"퇴점 {n_out}개")
        if n_relocated:
            _parts.append(f"이전 {n_relocated}개")
        st.markdown(f"**{' · '.join(_parts)}**")
        st.caption(f"{prev_label} 대비 브랜드 매칭 기준 (공실 제외)")

        _col_ren = {"brand": "브랜드", "building": "건물", "floor": "층"}

        if n_relocated:
            _bcols = [c for c in ["brand", "building", "floor"] if c in relocated_in.columns]
            reloc_disp = relocated_out[
                [c for c in ["brand", "building", "floor_prev"] if c in relocated_out.columns]
            ].rename(columns={"floor_prev": "floor"}).merge(
                relocated_in[_bcols], on="brand", suffixes=(f"({prev_label})", f"({curr_label})"),
            )
            reloc_disp.columns = [
                c.replace("brand", "브랜드").replace("building", "건물").replace("floor", "층")
                for c in reloc_disp.columns
            ]
            st.markdown(f"**🔀 이전** ({n_relocated}개) — 위치 변경")
            st.dataframe(reloc_disp.reset_index(drop=True), hide_index=True, use_container_width=True)

        _c1, _c2 = st.columns(2)
        with _c1:
            if n_in:
                st.markdown(f"**🟢 입점** ({n_in}개)")
                disp = new_brands[[c for c in ["brand", "building", "floor"] if c in new_brands.columns]].copy()
                disp.columns = [_col_ren.get(c, c) for c in disp.columns]
                st.dataframe(disp.reset_index(drop=True), hide_index=True, use_container_width=True)
            else:
                st.info("신규 입점 없음")
        with _c2:
            if n_out:
                st.markdown(f"**🔴 퇴점** ({n_out}개)")
                _out_cols = [c for c in ["brand", "building", "floor_prev"] if c in closed_brands.columns]
                disp = closed_brands[_out_cols].copy()
                disp.columns = [_col_ren.get(c, c) if c != "floor_prev" else "층" for c in disp.columns]
                st.dataframe(disp.reset_index(drop=True), hide_index=True, use_container_width=True)
            else:
                st.info("퇴점 없음")


# ── Public render ─────────────────────────────────────────────────────────────

def render_vacancy_tab(
    cur_df: pd.DataFrame,
    prev_file: str | None = None,
    yoy_file: str | None = None,
    file_map: dict[str, bytes] | None = None,
    sheet_map: dict[str, list] | None = None,
    billing_period: str | None = None,
    prev_billing_period: str | None = None,
    yoy_period: str | None = None,
) -> None:
    """Render vacancy/turnover tab with MoM and/or YoY comparison."""
    from tab_mom import _load_brand_usage_for_file

    if not file_map or not sheet_map:
        st.info("파일 데이터가 필요합니다.")
        return

    _modes = []
    if prev_file:
        _modes.append("📈 전월 대비")
    if yoy_file:
        _modes.append("📅 전년 대비")

    if not _modes:
        st.info("비교 데이터가 없습니다. 전월 또는 전년 파일을 함께 업로드하세요.")
        return

    import streamlit_antd_components as _sac_v
    _mode = _sac_v.segmented(
        [_sac_v.SegmentedItem(label=lbl) for lbl in _modes],
        key="vacancy_cmp_mode", use_container_width=True,
    ) if len(_modes) > 1 else _modes[0]

    if _mode == "📈 전월 대비":
        prev_df = _load_brand_usage_for_file(prev_file, file_map, sheet_map)
        if prev_df is None or prev_df.empty:
            st.warning("전월 파일에서 데이터를 로드할 수 없습니다.")
            return
        _period_str = (
            f"{prev_billing_period} → {billing_period}"
            if billing_period and prev_billing_period
            else "전월 대비"
        )
        st.caption(f"기간: {_period_str}")
        result = _analyze_turnover(cur_df, prev_df)
        if result:
            _render_analysis(result, prev_label="전월", curr_label="이번달")
        else:
            st.info("브랜드 매칭 결과가 없습니다.")

    elif _mode == "📅 전년 대비":
        yoy_df = _load_brand_usage_for_file(yoy_file, file_map, sheet_map)
        if yoy_df is None or yoy_df.empty:
            st.warning("전년 파일에서 데이터를 로드할 수 없습니다.")
            return
        _period_str = (
            f"{yoy_period} → {billing_period}"
            if billing_period and yoy_period
            else "전년 대비"
        )
        st.caption(f"기간: {_period_str}")
        result = _analyze_turnover(cur_df, yoy_df)
        if result:
            _render_analysis(result, prev_label="전년", curr_label="올해")
        else:
            st.info("브랜드 매칭 결과가 없습니다.")
