"""tab_anomaly.py — 이상감지 분석 (Anomaly Detection Analysis) UI.

Focused investigation view:
  1. KPI row             — risk-level brand counts
  2. Master table        — who to investigate and WHY (above fold, no expander)
  3. Visual ranking      — composite bar chart + heatmap
  4. Detail tabs:
       📈 급등 감지   — MoM spike detection with peer context (unique)
       🔍 일관성 검사 — zero-usage + 집계/부과 brand reconciliation
  5. Reference           — PDF, scoring method, raw data

Cost / HVAC / consumption detail → Tier 2 인사이트 (no duplication).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from anomaly_features import (
    build_anomaly_df, _UTIL_PREFIXES, _UTIL_LABELS,
    _SPIKE_CRITICAL, _SPIKE_HIGH, _SPIKE_MEDIUM,
)
from data import (
    read_billing_sheet,    BILLING_SHEET_NAME,
    read_electricity_sheet, ELECTRICITY_SHEET_NAME,
    read_water_sheet,       WATER_SHEET_NAME,
    read_hotwater_sheet,    HOTWATER_SHEET_NAME,
)
from features import add_display_index, download_df_as_excel
from biz_report import render_pdf_buttons, generate_anomaly_pdf
from tab_mgmt import render_mgmt_report

from utils import BLD_COLOR as _BLDG_COLOR, RISK_COLOR as _RISK_COLOR, UTIL_LABELS_UI as _UTIL_LABELS_UI

_SCORE_CSCALE = [
    [0.00, "#2ca02c"],
    [0.35, "#F0C040"],
    [0.60, "#DD8A00"],
    [1.00, "#C44E52"],
]


# ── Insight summary ──────────────────────────────────────────────────────────

def _render_overall_trend(anomaly_df: pd.DataFrame) -> None:
    """Render overall + per-building trend context with toggleable outlier view.

    Two modes for manual comparison:
    - 전체 평균 기준: brand deviation = brand_pct − overall_avg
    - 건물 평균 기준: brand deviation = brand_pct − building_avg

    Shows summary stats + top deviating brands in each mode so the user
    can judge which baseline catches real outliers better.
    """
    from data import to_numeric_series

    avail_pfx = [p for p in _UTIL_PREFIXES if f"{p}_pct" in anomaly_df.columns]
    if not avail_pfx:
        return

    has_building = "building" in anomaly_df.columns

    # ── Compute overall averages ──────────────────────────────────────────
    overall_avgs: dict[str, float] = {}
    for pfx in avail_pfx:
        avg = to_numeric_series(anomaly_df[f"{pfx}_pct"]).mean()
        if pd.notna(avg):
            overall_avgs[pfx] = avg

    # ── Compute building averages ─────────────────────────────────────────
    bldg_avgs: dict[str, dict[str, float]] = {}  # {building: {pfx: avg}}
    buildings: list[str] = []
    if has_building:
        buildings = sorted(anomaly_df["building"].dropna().unique())
        for bldg in buildings:
            bdf = anomaly_df[anomaly_df["building"] == bldg]
            bldg_avgs[bldg] = {}
            for pfx in avail_pfx:
                avg = to_numeric_series(bdf[f"{pfx}_pct"]).mean()
                if pd.notna(avg):
                    bldg_avgs[bldg][pfx] = avg

    # ── Summary line: overall averages ────────────────────────────────────
    overall_parts: list[str] = []
    for pfx in avail_pfx:
        lbl = _UTIL_LABELS_UI.get(pfx, pfx)
        avg = overall_avgs.get(pfx)
        if avg is not None:
            clr = "#C44E52" if avg > 10 else "#2ca02c" if avg < -10 else "#888"
            overall_parts.append(
                f'<span style="color:{clr};font-weight:600">{lbl} {avg:+.1f}%</span>'
            )
    py_parts: list[str] = []
    for pfx in avail_pfx:
        lbl = _UTIL_LABELS_UI.get(pfx, pfx)
        py_col = f"{pfx}_usage_per_py"
        if py_col in anomaly_df.columns:
            avg_py = to_numeric_series(anomaly_df[py_col]).mean()
            if pd.notna(avg_py):
                py_parts.append(f"{lbl} {avg_py:.2f}")

    if not overall_parts:
        return

    # ── Styled metric cards for overall averages ────────────────────────
    st.markdown(
        '<p style="margin:0 0 6px;font-size:0.88rem;font-weight:700;color:inherit;opacity:0.7">'
        '📊 전월 대비 전체 평균 변화율</p>',
        unsafe_allow_html=True,
    )
    _metric_cols = st.columns(len(avail_pfx))
    for _mi, pfx in enumerate(avail_pfx):
        lbl = _UTIL_LABELS_UI.get(pfx, pfx)
        avg = overall_avgs.get(pfx)
        if avg is None or pd.isna(avg):
            continue
        # Per-평 value
        py_col = f"{pfx}_usage_per_py"
        py_str = ""
        if py_col in anomaly_df.columns:
            py_avg = to_numeric_series(anomaly_df[py_col]).mean()
            if pd.notna(py_avg):
                py_str = f"{py_avg:.2f}/평"

        clr = "#C44E52" if avg > 10 else "#2ca02c" if avg < -10 else "#888"
        arrow = "▲" if avg > 0 else "▼" if avg < 0 else ""

        # Distortion note for this utility
        distortion = ""
        if has_building and len(buildings) >= 2:
            worst_bldg, worst_diff = None, 0
            for bldg in buildings:
                b_avg = bldg_avgs.get(bldg, {}).get(pfx)
                if b_avg is None or pd.isna(b_avg):
                    continue
                diff = b_avg - avg
                if abs(diff) > abs(worst_diff):
                    worst_bldg, worst_diff = bldg, diff
            if worst_bldg and abs(worst_diff) >= 15:
                d_clr = "#C44E52" if worst_diff > 0 else "#2ca02c"
                distortion = (
                    f'<div style="font-size:0.68rem;color:{d_clr};margin-top:3px">'
                    f'⚠ {worst_bldg}동 {worst_diff:+.0f}%p</div>'
                )

        with _metric_cols[_mi]:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,{clr}08,{clr}03);'
                f'border:1px solid {clr}20;border-radius:10px;padding:12px 14px;text-align:center">'
                f'<div style="font-size:0.78rem;color:inherit;opacity:0.55;margin-bottom:2px">{lbl}</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:{clr};line-height:1.1">'
                f'{arrow}{abs(avg):.1f}%</div>'
                f'<div style="font-size:0.7rem;color:inherit;opacity:0.45;margin-top:2px">{py_str}</div>'
                f'{distortion}'
                f'</div>',
                unsafe_allow_html=True,
            )




def _render_insight_summary(anomaly_df: pd.DataFrame, sheets: dict) -> None:
    """Compact insight summary — signals + risk brand pills."""
    from utils import z_to_grade as _ztg

    danger_df = anomaly_df[anomaly_df["risk_level"] == "🔴 위험"].copy()
    caution_df = anomaly_df[anomaly_df["risk_level"] == "🟠 주의"].copy()
    has_reason = "reason" in anomaly_df.columns

    # Auto-populate watchlist
    _watchlist: list[str] = st.session_state.get("_brand_watchlist", [])
    for _ab in danger_df["brand"].tolist() + caution_df["brand"].tolist():
        if _ab not in _watchlist:
            _watchlist.append(_ab)
    st.session_state["_brand_watchlist"] = _watchlist

    # ── Build signal badges (one-liner) ───────────────────────────────────
    badges: list[str] = []

    # Spike
    if "spike_max_pct" in anomaly_df.columns:
        n_spike = int((anomaly_df["spike_max_pct"] >= _SPIKE_HIGH).sum())
        if n_spike:
            badges.append(
                f'<span style="background:#C44E5212;color:#C44E52;border:1px solid #C44E5230;'
                f'border-radius:12px;padding:3px 10px;font-size:0.78rem;font-weight:600">'
                f'📈 급등 {n_spike}건</span>'
            )

    # Cost anomaly
    _z_cols = [c for c in ["water_unit_z", "elect_unit_z", "total_cost_per_py_z",
                            "total_cost_per_m2_z"] if c in anomaly_df.columns]
    if _z_cols:
        n_cost = int((anomaly_df[_z_cols].abs().max(axis=1) >= 2.0).sum())
        if n_cost:
            badges.append(
                f'<span style="background:#DD8A0012;color:#DD8A00;border:1px solid #DD8A0030;'
                f'border-radius:12px;padding:3px 10px;font-size:0.78rem;font-weight:600">'
                f'💰 단가이상 {n_cost}건</span>'
            )

    # Zero usage
    if "n_zero_utilities" in anomaly_df.columns:
        n_zero = int((anomaly_df["n_zero_utilities"] > 0).sum())
        if n_zero:
            badges.append(
                f'<span style="background:#F0C04012;color:#B8960A;border:1px solid #F0C04030;'
                f'border-radius:12px;padding:3px 10px;font-size:0.78rem;font-weight:600">'
                f'⚠️ 미계량 {n_zero}건</span>'
            )

    # Data sources
    src = ["검침"]
    if BILLING_SHEET_NAME in sheets:
        src.append("청구")
    if ELECTRICITY_SHEET_NAME in sheets:
        src.append("전기")
    badges.append(
        f'<span style="background:rgba(128,128,128,0.12);color:inherit;opacity:0.5;border-radius:12px;'
        f'padding:3px 10px;font-size:0.72rem">📂 {" · ".join(src)}</span>'
    )

    with st.container(border=True):
        # Signal badges row
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:10px">'
            f'<span style="font-size:0.9rem;font-weight:700;color:#4C72B0;margin-right:4px">'
            f'핵심 인사이트</span>{"".join(badges)}</div>',
            unsafe_allow_html=True,
        )

        # ── Risk brand pills ─────────────────────────────────────────────
        def _render_pills(risk_df, level, bg, label_color, key_prefix, show_label=True):
            if risk_df.empty:
                return
            level_label = {"danger": "즉시 조사", "caution": "모니터링"}[level]
            emoji = {"danger": "🔴", "caution": "🟠"}[level]
            _css_id = f"pills_{key_prefix}"
            _label_html = (
                f'<span style="color:{label_color};font-weight:700;font-size:0.82rem">'
                f'{emoji} {level_label} — {len(risk_df)}개</span>'
                if show_label else ""
            )
            st.markdown(
                f'<div id="{_css_id}"></div>'
                f'<style>'
                f'#{_css_id} ~ div [data-testid="stPopover"] > button {{'
                f'  background: {bg} !important;'
                f'  border: 1.5px solid {label_color}40 !important;'
                f'  border-radius: 20px !important;'
                f'  color: {label_color} !important;'
                f'  font-weight: 700 !important;'
                f'  font-size: 0.82rem !important;'
                f'  min-height: 38px !important;'
                f'  padding: 6px 16px !important;'
                f'}}'
                f'#{_css_id} ~ div [data-testid="stPopover"] > button:hover {{'
                f'  background: {label_color}20 !important;'
                f'  border-color: {label_color}80 !important;'
                f'}}'
                f'</style>'
                f'{_label_html}',
                unsafe_allow_html=True,
            )
            _ncols = 5
            for _row_start in range(0, len(risk_df), _ncols):
                _row_df = risk_df.iloc[_row_start:_row_start + _ncols]
                _cols = st.columns(_ncols)
                for _i, (_, _r) in enumerate(_row_df.iterrows()):
                    reason = str(_r.get("reason", "")) if has_reason else ""
                    score = _r.get("composite_score", 0)
                    spike = _r.get("spike_max_pct", 0)
                    spike_util = _r.get("spike_worst_util", "")
                    bldg_avg = _r.get("spike_bldg_avg_pct")
                    risk = _r.get("risk_level", "")
                    bldg = _r.get("building", "")
                    name = str(_r["brand"])
                    display_name = name if len(name) <= 10 else name[:9] + "…"

                    with _cols[_i]:
                        with st.popover(display_name, use_container_width=True):
                            st.markdown(f"**{name}** {bldg}동  {risk}")
                            _s1, _s2 = st.columns(2)
                            _s1.metric("복합 점수", f"{score:.3f}")
                            if spike and not pd.isna(spike):
                                _s2.metric("최대 급등", f"+{spike:.0f}% {spike_util}")
                            if bldg_avg is not None and not pd.isna(bldg_avg):
                                st.caption(f"건물평균: +{bldg_avg:.0f}%")
                            if reason:
                                st.caption(f"📋 {reason}")
                            if st.button("🏢 프로필 이동",
                                         key=f"{key_prefix}_{_row_start + _i}",
                                         use_container_width=True):
                                st.session_state["_goto_profile_brand"] = name
                                st.rerun()

        _render_pills(danger_df, "danger", "#C44E5218", "#C44E52", "goto_danger")
        if not caution_df.empty:
            with st.expander(f"🟠 모니터링 — {len(caution_df)}개", expanded=False):
                _render_pills(caution_df, "caution", "#DD8A0012", "#DD8A00", "goto_caution",
                              show_label=False)

        if danger_df.empty and caution_df.empty:
            st.success("특이 사항 없음")


# ── Zero-usage change detection (vs prev/yoy) ────────────────────────────────

def _build_zero_set(file_data: bytes, sheet_keys: list[str]) -> dict[str, set[str]]:
    """Build per-utility set of zero-usage brands from a period file.

    Returns {utility_prefix: {brand_name, ...}} for brands with current==0.
    """
    from data import read_sheet
    from features import apply_header_rows, build_from_two_files, create_change_columns, aggregate_by_brand
    from data import to_numeric_series

    meter_key = next((k for k in sheet_keys if k.strip() == "검침 내역"), None)
    if not meter_key:
        return {}
    try:
        raw = read_sheet("__zero__.xlsx", file_data, meter_key)
        df_cur = apply_header_rows(raw)
        df_cur["building"] = df_cur["building"].astype(str).str.strip()
        df_cur = df_cur[df_cur["building"].isin({"A", "B", "C", "D"})].copy()
        df = build_from_two_files(df_cur, None)
        raw_df = create_change_columns(df)
        agg = aggregate_by_brand(raw_df)
    except Exception:
        return {}

    result: dict[str, set[str]] = {}
    for pfx in _UTIL_PREFIXES:
        col = f"{pfx}_current"
        if col in agg.columns:
            zeros = set(agg.loc[to_numeric_series(agg[col]).fillna(0) == 0, "brand"].astype(str))
            if zeros:
                result[pfx] = zeros
    return result


def _render_zero_change(
    cur_df: pd.DataFrame,
    prev_file_data: bytes | None,
    prev_sheet_keys: list[str] | None,
    prev_label: str | None,
    yoy_file_data: bytes | None,
    yoy_sheet_keys: list[str] | None,
    yoy_label: str | None,
) -> None:
    """Compare zero-usage brands between current and previous/yoy periods."""
    from data import to_numeric_series

    # Current zero-usage sets
    cur_zeros: dict[str, set[str]] = {}
    for pfx in _UTIL_PREFIXES:
        col = f"{pfx}_current"
        if col in cur_df.columns:
            zeros = set(cur_df.loc[to_numeric_series(cur_df[col]).fillna(0) == 0, "brand"].astype(str))
            if zeros:
                cur_zeros[pfx] = zeros

    comparisons: list[tuple[str, dict[str, set[str]]]] = []
    if prev_file_data and prev_sheet_keys:
        prev_zeros = _build_zero_set(prev_file_data, prev_sheet_keys)
        if prev_zeros:
            comparisons.append(("전월", prev_zeros))
    if yoy_file_data and yoy_sheet_keys:
        yoy_zeros = _build_zero_set(yoy_file_data, yoy_sheet_keys)
        if yoy_zeros:
            comparisons.append(("전년", yoy_zeros))

    if not comparisons:
        st.info("전월 또는 전년 파일을 업로드하면 미계량 변화를 감지합니다.")
        return

    # Period toggle
    _period_options = [c[0] for c in comparisons]
    if len(_period_options) > 1:
        _sel_period = st.radio(
            "비교 기간", _period_options, horizontal=True, key="dq_zero_period",
        )
    else:
        _sel_period = _period_options[0]

    prev_zeros = next(pz for lbl, pz in comparisons if lbl == _sel_period)
    period_lbl = _sel_period

    # --- render for selected period ---
    _render_zero_change_period(cur_zeros, prev_zeros, period_lbl)


def _render_zero_change_period(
    cur_zeros: dict,
    prev_zeros: dict,
    period_lbl: str,
) -> None:
    """Render zero-change butterfly chart + tables + correction for one period."""
    for period_lbl, prev_zeros in [(period_lbl, prev_zeros)]:
        all_utils = sorted(set(cur_zeros.keys()) | set(prev_zeros.keys()))
        rows: list[dict] = []
        chart_data: dict[str, dict[str, int]] = {}
        for pfx in all_utils:
            label = _UTIL_LABELS_UI.get(pfx, pfx)
            cur_set = cur_zeros.get(pfx, set())
            prev_set = prev_zeros.get(pfx, set())
            new_zero = sorted(cur_set - prev_set)
            recovered = sorted(prev_set - cur_set)
            chart_data[label] = {"new": len(new_zero), "recovered": len(recovered)}
            for b in new_zero:
                rows.append({"브랜드": b, "유틸리티": label, "상태": "새로 미계량",
                             "조치": "계량기 고장·분리 확인"})
            for b in recovered:
                rows.append({"브랜드": b, "유틸리티": label, "상태": "계량 복구",
                             "조치": "정상 복구 확인"})

        n_new = sum(d["new"] for d in chart_data.values())
        n_rec = sum(d["recovered"] for d in chart_data.values())
        _net = n_new - n_rec

        if not rows:
            st.success(f"{period_lbl} 대비 미계량 변화 없음")
            continue

        # KPI badges
        _net_clr = "#C44E52" if _net > 0 else "#2ca02c" if _net < 0 else "#888"
        st.markdown(
            f'<div style="display:flex;gap:8px;margin-bottom:10px;align-items:center">'
            f'<span style="background:#C44E5212;color:#C44E52;border:1px solid #C44E5230;'
            f'border-radius:12px;padding:4px 12px;font-size:0.82rem;font-weight:600">'
            f'🔴 새로 미계량 {n_new}건</span>'
            f'<span style="background:#2ca02c12;color:#2ca02c;border:1px solid #2ca02c30;'
            f'border-radius:12px;padding:4px 12px;font-size:0.82rem;font-weight:600">'
            f'🟢 계량 복구 {n_rec}건</span>'
            f'<span style="color:{_net_clr};font-weight:700;font-size:0.85rem">'
            f'순변화 {_net:+d}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Butterfly chart
        _util_labels = list(chart_data.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="새로 미계량", y=_util_labels,
            x=[chart_data[u]["new"] for u in _util_labels],
            orientation="h", marker_color="#C44E52",
            text=[f"+{chart_data[u]['new']}" if chart_data[u]["new"] else ""
                  for u in _util_labels],
            textposition="outside", textfont=dict(size=11, color="#C44E52"),
        ))
        fig.add_trace(go.Bar(
            name="계량 복구", y=_util_labels,
            x=[-chart_data[u]["recovered"] for u in _util_labels],
            orientation="h", marker_color="#2ca02c",
            text=[f"-{chart_data[u]['recovered']}" if chart_data[u]["recovered"] else ""
                  for u in _util_labels],
            textposition="outside", textfont=dict(size=11, color="#2ca02c"),
        ))
        fig.add_vline(x=0, line_color="#888", line_width=1)
        _max_val = max(max(d["new"], d["recovered"]) for d in chart_data.values()) + 1
        fig.update_layout(
            barmode="overlay", height=200,
            margin=dict(t=10, b=20, l=10, r=10),
            xaxis=dict(range=[-_max_val, _max_val], showticklabels=False),
            yaxis=dict(tickfont=dict(size=12)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"zero_chg_{period_lbl}")

        # Detail table
        if rows:
            change_df = pd.DataFrame(rows)
            _new_df = change_df[change_df["상태"] == "새로 미계량"]
            _rec_df = change_df[change_df["상태"] == "계량 복구"]

            _c1, _c2 = st.columns(2)
            with _c1:
                if not _new_df.empty:
                    st.markdown(f"**🔴 새로 미계량 — {len(_new_df)}건**")
                    st.dataframe(
                        _new_df[["브랜드", "유틸리티", "조치"]].reset_index(drop=True),
                        hide_index=True, use_container_width=True,
                    )
            with _c2:
                if not _rec_df.empty:
                    st.markdown(f"**🟢 계량 복구 — {len(_rec_df)}건**")
                    st.dataframe(
                        _rec_df[["브랜드", "유틸리티", "조치"]].reset_index(drop=True),
                        hide_index=True, use_container_width=True,
                    )

            # Correction tool
            with st.expander("✏️ 미계량 상태 보정", expanded=False):
                st.caption(
                    "의도적 미계량(공실, 계량기 미설치 등)인 경우 아래에서 제외 처리하세요. "
                    "제외된 브랜드는 이상감지 점수에서 미계량 페널티가 적용되지 않습니다."
                )
                _exclude_key = "_dq_zero_exclude"
                _excluded = st.session_state.get(_exclude_key, [])

                _all_brands = sorted(set(
                    r["브랜드"] for r in rows
                ))
                _to_exclude = st.multiselect(
                    "의도적 미계량 브랜드 선택",
                    [b for b in _all_brands if b not in _excluded],
                    key=f"dq_exclude_sel_{period_lbl}",
                )
                _ec1, _ec2 = st.columns(2)
                with _ec1:
                    if st.button("✅ 제외 처리", key=f"dq_exclude_btn_{period_lbl}",
                                 disabled=not _to_exclude):
                        _excluded.extend(_to_exclude)
                        st.session_state[_exclude_key] = _excluded
                        st.success(f"{len(_to_exclude)}개 브랜드 제외 처리됨")
                        st.rerun()
                with _ec2:
                    if _excluded and st.button("🗑️ 제외 초기화", key=f"dq_reset_{period_lbl}"):
                        st.session_state[_exclude_key] = []
                        st.rerun()

                if _excluded:
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:inherit;opacity:0.55;margin-top:4px">'
                        f'현재 제외: {", ".join(_excluded)}</div>',
                        unsafe_allow_html=True,
                    )


# ── Cross-file spike detection (MoM / YoY) ──────────────────────────────────

def _build_cross_file_anomaly(
    cur_file_data: bytes,
    cur_sheet_keys: list[str],
    cmp_file_data: bytes,
    cmp_sheet_keys: list[str],
    label: str,
) -> pd.DataFrame | None:
    """Build anomaly_df comparing current file usage vs a comparison file.

    The comparison file's *_current values become *_previous in the merged df,
    so that change/pct reflect current-vs-comparison instead of internal MoM.
    Returns anomaly_df or None on failure.
    """
    from data import read_sheet, to_numeric_series
    from features import (
        apply_header_rows, build_from_two_files,
        create_change_columns, aggregate_by_brand,
    )

    def _load_meter(fdata, skeys, tag):
        mk = next((k for k in skeys if k.strip() == "검침 내역"), None)
        if not mk:
            return None
        raw = read_sheet(f"__{tag}__.xlsx", fdata, mk)
        df = apply_header_rows(raw)
        df["building"] = df["building"].astype(str).str.strip()
        return df[df["building"].isin({"A", "B", "C", "D"})].copy()

    try:
        df_cur = _load_meter(cur_file_data, cur_sheet_keys, "cur")
        df_cmp = _load_meter(cmp_file_data, cmp_sheet_keys, f"cmp_{label}")
        if df_cur is None or df_cmp is None:
            return None

        # build_from_two_files treats df_cmp as "previous" period
        merged = build_from_two_files(df_cur, df_cmp)
        raw_df = create_change_columns(merged)
        agg_df = aggregate_by_brand(raw_df)
    except Exception:
        return None

    # Load supporting sheets from current file for full anomaly scoring
    sheets = _load_sheets("__cur__.xlsx", cur_file_data, cur_sheet_keys)
    try:
        return build_anomaly_df(
            meter_df=agg_df,
            billing_df=sheets.get(BILLING_SHEET_NAME),
            elec_df=sheets.get(ELECTRICITY_SHEET_NAME),
            water_df=sheets.get(WATER_SHEET_NAME),
            hotwater_df=sheets.get(HOTWATER_SHEET_NAME),
        )
    except Exception:
        return None


def _handle_chart_click(ev, df: pd.DataFrame, field: str = "x",
                         match: str = "exact", trunc: int = 0) -> None:
    """Delegate to shared handler in utils_plot."""
    from utils_plot import handle_chart_click
    handle_chart_click(ev, df, field=field, trunc=trunc)


# ── Sheet loader ──────────────────────────────────────────────────────────────

def _load_sheets(file_name: str, file_data: bytes, all_sheet_keys: list[str]) -> dict:
    from utils import load_all_sheets
    return load_all_sheets(file_name, file_data, all_sheet_keys, silent=False)


# ── Section: KPI row ──────────────────────────────────────────────────────────

def _render_kpis(df: pd.DataFrame, has_billing: bool, has_elec: bool) -> None:
    counts = df["risk_level"].value_counts()
    sources = ["검침"] + (["청구"] if has_billing else []) + (["전기"] if has_elec else [])
    n_total = len(df)
    n_danger = counts.get("🔴 위험", 0)
    n_caution = counts.get("🟠 주의", 0)
    n_watch = counts.get("🟡 관찰", 0)
    n_normal = counts.get("🟢 정상", 0)

    # Risk gauge summary — modern card
    _pct_risk = (n_danger + n_caution) / n_total * 100 if n_total else 0
    _gauge_color = "#C44E52" if _pct_risk >= 30 else "#DD8A00" if _pct_risk >= 15 else "#2ca02c"

    # Progress bar visual
    _bar_w_danger = n_danger / n_total * 100 if n_total else 0
    _bar_w_caution = n_caution / n_total * 100 if n_total else 0
    _bar_w_watch = n_watch / n_total * 100 if n_total else 0
    _bar_w_normal = n_normal / n_total * 100 if n_total else 0

    st.markdown(
        f'<div style="background:linear-gradient(135deg,{_gauge_color}0C,{_gauge_color}03);'
        f'border:1px solid {_gauge_color}25;border-radius:12px;padding:16px 20px;margin-bottom:16px">'
        f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:10px">'
        f'<span style="font-size:2rem;font-weight:800;color:{_gauge_color};line-height:1">'
        f'{n_danger + n_caution}</span>'
        f'<span style="font-size:0.95rem;color:inherit;opacity:0.7;font-weight:500"> / {n_total} 브랜드 조사 필요</span>'
        f'<span style="margin-left:auto;font-size:0.78rem;color:inherit;opacity:0.5;'
        f'background:rgba(128,128,128,0.12);padding:3px 10px;border-radius:12px">'
        f'📂 {" · ".join(sources)}</span>'
        f'</div>'
        # Stacked severity bar
        f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;background:rgba(128,128,128,0.15)">'
        f'<div style="width:{_bar_w_danger}%;background:#C44E52"></div>'
        f'<div style="width:{_bar_w_caution}%;background:#DD8A00"></div>'
        f'<div style="width:{_bar_w_watch}%;background:#F0C040"></div>'
        f'<div style="width:{_bar_w_normal}%;background:#2ca02c"></div>'
        f'</div>'
        f'<div style="display:flex;gap:16px;margin-top:8px;font-size:0.75rem;color:inherit;opacity:0.55">'
        f'<span><span style="color:#C44E52;font-weight:700">{n_danger}</span> 위험</span>'
        f'<span><span style="color:#DD8A00;font-weight:700">{n_caution}</span> 주의</span>'
        f'<span><span style="color:#F0C040;font-weight:700">{n_watch}</span> 관찰</span>'
        f'<span><span style="color:#2ca02c;font-weight:700">{n_normal}</span> 정상</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Section: Composite ranked bar chart ───────────────────────────────────────

def _render_composite_bar(df: pd.DataFrame, n: int, split_by_building: bool) -> None:
    top = df.head(n).copy().iloc[::-1]  # reverse so highest is at top of h-bar
    marker_color = (
        [_BLDG_COLOR.get(str(b), "#888") for b in top["building"]]
        if split_by_building and "building" in top.columns
        else [_RISK_COLOR.get(r, "#888") for r in top["risk_level"]]
    )
    fig = go.Figure(go.Bar(
        x=top["composite_score"],
        y=[str(b)[:28] for b in top["brand"]],
        orientation="h",
        marker_color=marker_color,
        text=[f'{r}  {s:.3f}' for r, s in zip(top["risk_level"], top["composite_score"])],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>복합 이상 점수: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"복합 이상 점수 — 상위 {n}개 브랜드",
        height=max(400, n * 22 + 80),
        xaxis=dict(title="점수 [0–1]", range=[0, 1.20],
                   gridcolor="rgba(128,128,128,0.2)", griddash="dot"),
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=170, t=50, b=40),
        showlegend=False,
    )
    ev = st.plotly_chart(fig, use_container_width=True, key="anom_composite_bar", on_select="rerun")
    _handle_chart_click(ev, df, field="y", match="contains", trunc=26)


# ── Section: Anomaly heatmap ──────────────────────────────────────────────────

def _render_heatmap(df: pd.DataFrame, n: int) -> None:
    top = df.head(n).copy()

    heat_cols: list[str] = []
    col_labels: list[str] = []

    for pfx, label in _UTIL_LABELS_UI.items():
        qc = f"{pfx}_quad_score"
        if qc in top.columns:
            heat_cols.append(qc)
            col_labels.append(f"{label}\n사분면")

    for col, label in [
        ("water_unit_z",        "수도\n단가등급"),
        ("elect_unit_z",        "전기\n단가등급"),
        ("total_cost_per_py_z", "평당\n비용등급"),
        ("total_cost_per_m2_z", "총비용\n/m²등급"),
        ("hvac_intensity_z",    "HVAC\n강도등급"),
        ("n_zero_utilities",    "미계량\n항목수"),
    ]:
        if col in top.columns:
            heat_cols.append(col)
            col_labels.append(label)

    if not heat_cols:
        return

    matrix = top[heat_cols].fillna(0).copy()
    for c in heat_cols:
        if "_z" in c:
            matrix[c] = matrix[c].abs()

    norm = matrix.apply(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0,
        axis=0,
    )

    brand_labels = [str(b)[:26] for b in top["brand"]]

    fig = go.Figure(go.Heatmap(
        z=norm.values,
        x=col_labels,
        y=brand_labels,
        colorscale=_SCORE_CSCALE,
        zmin=0, zmax=1,
        customdata=matrix.values,
        hovertemplate="<b>%{y}</b><br>%{x}: %{customdata:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="강도", len=0.6,
                      tickvals=[0, 0.5, 1], ticktext=["낮음", "중간", "높음"]),
    ))
    fig.update_layout(
        title=f"이상 신호 히트맵 — 상위 {n}개 브랜드",
        height=max(400, n * 20 + 120),
        xaxis=dict(side="top", tickangle=-30),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=100, t=120, b=20),
    )
    ev = st.plotly_chart(fig, use_container_width=True, key="anom_signal_heatmap", on_select="rerun")
    _handle_chart_click(ev, df, field="y", match="contains", trunc=26)


# ── Tab: 급등 감지 (MoM Spike Detection) ─────────────────────────────────────

def _render_spike_tab(df: pd.DataFrame, split_by_building: bool,
                      key_suffix: str = "") -> None:
    sfx = key_suffix
    _is_yoy = "yoy" in sfx
    _cmp_label = "작년동월" if _is_yoy else "전월"

    spike_pct_cols = [f"{p}_spike_pct" for p in _UTIL_PREFIXES if f"{p}_spike_pct" in df.columns]
    if not spike_pct_cols:
        st.info(f"{_cmp_label} 데이터가 없어 급등 감지를 수행할 수 없습니다.")
        return

    # ── Threshold + controls row ──────────────────────────────────────────────
    _ctrl1, _ctrl2 = st.columns([3, 1])
    with _ctrl1:
        thresh = st.slider(
            f"급등 기준 ({_cmp_label} 대비 증가율 %)", 10, 300, int(_SPIKE_HIGH), step=10,
            key=f"spike_thresh{sfx}",
        )
    with _ctrl2:
        _logy = st.checkbox("Log 스케일", key=f"spike_logy{sfx}")

    # ── KPI cards with visual severity ────────────────────────────────────────
    n_critical = int((df["spike_max_pct"] >= _SPIKE_CRITICAL).sum())
    n_high     = int(((df["spike_max_pct"] >= _SPIKE_HIGH) & (df["spike_max_pct"] < _SPIKE_CRITICAL)).sum())
    n_medium   = int(((df["spike_max_pct"] >= _SPIKE_MEDIUM) & (df["spike_max_pct"] < _SPIKE_HIGH)).sum())
    n_above    = int((df["spike_max_pct"] >= thresh).sum())
    n_total    = len(df)
    n_normal   = n_total - n_critical - n_high - n_medium

    _kpi_items = [
        ("🔴 급등", n_critical, "#C44E52", f"≥{_SPIKE_CRITICAL:.0f}%"),
        ("🟠 주의", n_high, "#DD8A00", f"≥{_SPIKE_HIGH:.0f}%"),
        ("🟡 관찰", n_medium, "#F0C040", f"≥{_SPIKE_MEDIUM:.0f}%"),
        ("🟢 정상", n_normal, "#2ca02c", f"<{_SPIKE_MEDIUM:.0f}%"),
    ]
    _kpi_html = '<div style="display:flex;gap:8px;margin-bottom:16px">'
    for _lbl, _cnt, _clr, _thr in _kpi_items:
        _pct = _cnt / n_total * 100 if n_total else 0
        _kpi_html += (
            f'<div style="flex:1;background:linear-gradient(135deg,{_clr}12,{_clr}04);'
            f'border:1px solid {_clr}30;border-radius:10px;padding:12px 16px;text-align:center">'
            f'<div style="font-size:1.6rem;font-weight:800;color:{_clr};line-height:1.2">{_cnt}</div>'
            f'<div style="font-size:0.78rem;font-weight:600;color:inherit;opacity:0.7;margin-top:2px">{_lbl}</div>'
            f'<div style="font-size:0.68rem;color:inherit;opacity:0.5;margin-top:1px">{_thr} · {_pct:.0f}%</div>'
            f'</div>'
        )
    _kpi_html += '</div>'
    st.markdown(_kpi_html, unsafe_allow_html=True)

    # ── 1. All-utility overview: horizontal stacked bar ─────────────────────
    _avail_pfx = [p for p in _UTIL_PREFIXES if f"{p}_spike_pct" in df.columns]
    if len(_avail_pfx) >= 2:
        _overview_df = df[df["spike_max_pct"] >= _SPIKE_MEDIUM].nlargest(
            min(15, len(df)), "spike_max_pct"
        ).copy()
        if not _overview_df.empty:
            _n_ov = len(_overview_df)
            _ov_c1, _ov_c2 = st.columns([4, 1])
            with _ov_c1:
                st.markdown(f"**{_cmp_label} 대비 유틸리티별 급등 현황** — 상위 {_n_ov}개 브랜드")
            with _ov_c2:
                _ov_logy = st.checkbox("Log 스케일", key=f"spike_ov_logy{sfx}")
            fig_ov = go.Figure()
            _util_colors = {"water": "#4C72B0", "hwater": "#C44E52",
                            "elect": "#DD8A00", "heat": "#E377C2"}
            brands = [str(b)[:20] for b in _overview_df["brand"]]
            for p in _avail_pfx:
                col = f"{p}_spike_pct"
                lbl = _UTIL_LABELS_UI.get(p, p)
                vals = _overview_df[col].fillna(0).clip(lower=0).tolist()
                fig_ov.add_trace(go.Bar(
                    name=lbl, y=brands, x=vals,
                    orientation="h",
                    marker_color=_util_colors.get(p, "#888"),
                    text=[f"{v:.0f}%" if v >= _SPIKE_MEDIUM else "" for v in vals],
                    textposition="inside", textfont=dict(size=10, color="white"),
                    insidetextanchor="middle",
                    hovertemplate=f"<b>%{{y}}</b><br>{lbl}: %{{x:.1f}}%<extra></extra>",
                ))
            # Threshold lines
            for lvl, clr, lbl in [
                (_SPIKE_CRITICAL, "rgba(196,78,82,0.6)", f"급등 {_SPIKE_CRITICAL:.0f}%"),
                (_SPIKE_HIGH, "rgba(221,138,0,0.6)", f"주의 {_SPIKE_HIGH:.0f}%"),
            ]:
                fig_ov.add_vline(x=lvl, line_dash="dot", line_color=clr, line_width=1.5,
                                 annotation_text=lbl, annotation_position="top",
                                 annotation_font_size=10, annotation_font_color=clr)
            _ov_xaxis = dict(title="증가율 합계 (%)",
                            gridcolor="rgba(128,128,128,0.15)", griddash="dot")
            if _ov_logy:
                _ov_xaxis["type"] = "log"
            fig_ov.update_layout(
                barmode="stack",
                height=max(420, _n_ov * 36 + 100),
                margin=dict(l=10, r=80, t=40, b=40),
                xaxis=_ov_xaxis,
                yaxis=dict(autorange="reversed", categoryorder="array",
                           categoryarray=brands, tickfont=dict(size=11)),
                legend=dict(orientation="h", y=1.06, x=0, font_size=12),
                plot_bgcolor="rgba(0,0,0,0)",
                bargap=0.25,
            )
            ev_ov = st.plotly_chart(fig_ov, use_container_width=True,
                                    key=f"spike_overview{sfx}", on_select="rerun")
            _handle_chart_click(ev_ov, _overview_df, field="y", match="contains", trunc=20)

    # ── 2. Spike severity distribution (donut + scatter) ──────────────────────
    _viz1, _viz2 = st.columns([1, 2])
    with _viz1:
        st.markdown(f"**등급 분포**")
        fig_donut = go.Figure(go.Pie(
            labels=[k[0] for k in _kpi_items],
            values=[k[1] for k in _kpi_items],
            marker_colors=[k[2] for k in _kpi_items],
            hole=0.55,
            textinfo="value+percent",
            textfont_size=11,
            sort=False,
        ))
        fig_donut.update_layout(
            height=300, margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            annotations=[dict(
                text=f"<b>{n_critical + n_high}</b><br>위험",
                x=0.5, y=0.5, font_size=16, showarrow=False,
                font_color="#C44E52" if n_critical > 0 else "#DD8A00",
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True, key=f"spike_donut{sfx}")

    with _viz2:
        # Scatter: max spike pct vs peer ratio (bubble = brand)
        if "spike_peer_ratio" in df.columns:
            _sc_c1, _sc_c2 = st.columns([4, 1])
            with _sc_c1:
                st.markdown(f"**급등율 vs 건물 대비 배수** — 우상단이 가장 위험")
            with _sc_c2:
                _scat_logy = st.checkbox("Log 스케일", key=f"spike_scat_logy{sfx}")
            _scat_df = df[df["spike_max_pct"] > 0].copy()
            _scat_df["_label"] = _scat_df["brand"].astype(str).str[:15]
            _scat_df["_color"] = _scat_df["spike_max_pct"].apply(
                lambda v: "#C44E52" if v >= _SPIKE_CRITICAL
                else "#DD8A00" if v >= _SPIKE_HIGH
                else "#F0C040" if v >= _SPIKE_MEDIUM
                else "#2ca02c"
            )
            fig_scat = go.Figure()
            for _, _row in _scat_df.iterrows():
                fig_scat.add_trace(go.Scatter(
                    x=[_row["spike_max_pct"]],
                    y=[_row["spike_peer_ratio"] if pd.notna(_row["spike_peer_ratio"]) else 1],
                    mode="markers",
                    marker=dict(
                        size=10, color=_row["_color"],
                        line=dict(width=1, color="white"),
                    ),
                    name=_row["_label"],
                    hovertemplate=(
                        f"<b>{_row['_label']}</b><br>"
                        f"{_cmp_label} 대비: %{{x:.1f}}%<br>"
                        "vs건물: %{y:.1f}x<extra></extra>"
                    ),
                ))
            # Danger zone shading — use yref/xref="paper" (0–1) for far edge
            # so shapes don't pin the axis range when legend items are toggled
            fig_scat.add_shape(
                type="rect", x0=_SPIKE_CRITICAL, x1=1, y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(196,78,82,0.03)", line_width=0, layer="below",
            )
            fig_scat.add_shape(
                type="rect", x0=0, x1=1, y0=2, y1=1,
                xref="paper", yref="y",
                fillcolor="rgba(196,78,82,0.03)", line_width=0, layer="below",
            )
            fig_scat.add_hline(y=2, line_dash="dot", line_color="rgba(196,78,82,0.3)",
                               annotation_text="동종 2배", annotation_position="right",
                               annotation_font_size=9)
            fig_scat.add_vline(x=_SPIKE_HIGH, line_dash="dot", line_color="rgba(221,138,0,0.3)")
            fig_scat.update_layout(
                height=400,
                xaxis=dict(title=f"{_cmp_label} 대비 최대 증가율 (%)",
                           gridcolor="rgba(128,128,128,0.15)", griddash="dot",
                           **({"type": "log"} if _scat_logy else {})),
                yaxis=dict(title="건물 평균 대비 (배수)",
                           gridcolor="rgba(128,128,128,0.15)", griddash="dot",
                           **({"type": "log"} if _scat_logy else {})),
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=40, l=50, r=20),
                showlegend=True,
                legend=dict(
                    font_size=10,
                    itemsizing="constant",
                    orientation="v",
                    yanchor="top", y=1,
                    xanchor="left", x=1.02,
                ),
            )
            st.plotly_chart(fig_scat, use_container_width=True, key=f"spike_scatter{sfx}")
        else:
            st.markdown(f"**유틸리티별 급등 분포**")
            # Fallback: box plot per utility
            _box_data = []
            for p in _avail_pfx:
                col = f"{p}_spike_pct"
                for _, r in df.iterrows():
                    v = r.get(col)
                    if pd.notna(v) and v != 0:
                        _box_data.append({"유틸리티": _UTIL_LABELS_UI.get(p, p), "증가율": float(v)})
            if _box_data:
                _box_df = pd.DataFrame(_box_data)
                fig_box = px.box(_box_df, x="유틸리티", y="증가율",
                                 color="유틸리티", points="outliers")
                fig_box.add_hline(y=_SPIKE_HIGH, line_dash="dot", line_color="#DD8A00")
                fig_box.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)",
                                      margin=dict(t=10, b=30), showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True, key=f"spike_box{sfx}")

    st.divider()

    # ── 3. Spike brands table (with peer context) ─────────────────────────────
    spike_df = df[df["spike_max_pct"] >= thresh].copy()
    if spike_df.empty:
        st.success(f"기준({thresh}%) 초과 브랜드 없음")
    else:
        st.markdown(f"**기준 초과 브랜드 ({n_above}개)** — {_cmp_label} 대비 {thresh}% 이상 증가")
        peer_cols = [c for c in ["spike_bldg_avg_pct", "spike_peer_ratio"] if c in spike_df.columns]
        # Interleave per-utility spike_pct with its building avg
        util_detail_cols = []
        for p in _UTIL_PREFIXES:
            pct_c = f"{p}_spike_pct"
            avg_c = f"{p}_bldg_avg_pct"
            if pct_c in spike_df.columns:
                util_detail_cols.append(pct_c)
                if avg_c in spike_df.columns:
                    util_detail_cols.append(avg_c)
        disp_cols = (
            [c for c in ["brand", "building", "floor"] if c in spike_df.columns]
            + ["spike_max_pct", "spike_worst_util"]
            + peer_cols
            + util_detail_cols
        )
        col_cfg: dict = {
            "brand":              st.column_config.TextColumn("브랜드", width="medium"),
            "building":           st.column_config.TextColumn("건물", width="small"),
            "floor":              st.column_config.TextColumn("층", width="small"),
            "spike_max_pct":      st.column_config.NumberColumn("최대 증가율 (%)", format="%.1f"),
            "spike_worst_util":   st.column_config.TextColumn("급등 항목"),
            "spike_bldg_avg_pct": st.column_config.NumberColumn("건물평균(%)", format="%.1f"),
            "spike_peer_ratio":   st.column_config.NumberColumn("vs건물", format="%.1fx"),
        }
        util_labels = {f"{p}_spike_pct": f"{lbl} (%)" for p, lbl in _UTIL_LABELS_UI.items()}
        bldg_avg_labels = {f"{p}_bldg_avg_pct": f"{lbl} 건물평균(%)" for p, lbl in _UTIL_LABELS_UI.items()}
        for c, lbl in util_labels.items():
            if c in spike_df.columns:
                col_cfg[c] = st.column_config.NumberColumn(lbl, format="%.1f")
        for c, lbl in bldg_avg_labels.items():
            if c in spike_df.columns:
                col_cfg[c] = st.column_config.NumberColumn(lbl, format="%.1f")

        st.dataframe(
            spike_df[disp_cols].sort_values("spike_max_pct", ascending=False).reset_index(drop=True),
            column_config=col_cfg,
            hide_index=True,
            use_container_width=True,
        )

        # Narrative for spike brands
        if n_critical > 0:
            _top_spike = spike_df.nlargest(1, "spike_max_pct").iloc[0]
            _bavg = _top_spike.get("spike_bldg_avg_pct")
            _bavg_ctx = ""
            if _bavg is not None and not pd.isna(_bavg):
                if _top_spike["spike_max_pct"] > _bavg * 1.5:
                    _bavg_ctx = f" (건물평균 +{_bavg:.0f}% 대비 크게 상회)"
                else:
                    _bavg_ctx = f" (건물평균도 +{_bavg:.0f}%로 전반적 상승세)"
            st.warning(
                f"**{_top_spike['brand']}**의 {_top_spike.get('spike_worst_util', '유틸리티')} 사용량이 "
                f"{_cmp_label} 대비 **{_top_spike['spike_max_pct']:.0f}%** 급등했습니다{_bavg_ctx}. "
                f"누수·설비 이상·계량 오류 여부를 확인하세요."
            )

    # ── 4. Per-utility deep dive ──────────────────────────────────────────────
    st.divider()
    util_sel = st.selectbox(
        f"유틸리티별 상세 분석",
        [p for p in _UTIL_PREFIXES if f"{p}_spike_pct" in df.columns],
        format_func=lambda p: _UTIL_LABELS_UI.get(p, p),
        key=f"spike_util_sel{sfx}",
    )
    pct_col = f"{util_sel}_spike_pct"
    flag_col = f"{util_sel}_spike_flag"

    chart_df = df[["brand"] + [c for c in ["building", pct_col, flag_col] if c in df.columns]].copy()
    chart_df = chart_df[chart_df[pct_col].notna()].sort_values(pct_col, ascending=False).head(50)

    color_col = "building" if split_by_building and "building" in chart_df.columns else None

    # Color bars by severity level
    if color_col is None:
        chart_df["_severity"] = chart_df[pct_col].apply(
            lambda v: "🔴 급등" if v >= _SPIKE_CRITICAL
            else "🟠 주의" if v >= _SPIKE_HIGH
            else "🟡 관찰" if v >= _SPIKE_MEDIUM
            else "정상"
        )
        _sev_color_map = {"🔴 급등": "#C44E52", "🟠 주의": "#DD8A00",
                          "🟡 관찰": "#F0C040", "정상": "#4C72B0"}
        fig = px.bar(
            chart_df, x="brand", y=pct_col,
            color="_severity", color_discrete_map=_sev_color_map,
            title=f"{_UTIL_LABELS_UI.get(util_sel, util_sel)} {_cmp_label} 대비 증가율 (%) — 상위 50개",
            labels={pct_col: "증가율 (%)", "brand": "브랜드", "_severity": "등급"},
            log_y=_logy,
        )
    else:
        fig = px.bar(
            chart_df, x="brand", y=pct_col,
            color=color_col, color_discrete_map=_BLDG_COLOR,
            title=f"{_UTIL_LABELS_UI.get(util_sel, util_sel)} {_cmp_label} 대비 증가율 (%) — 상위 50개",
            labels={pct_col: "증가율 (%)", "brand": "브랜드"},
            log_y=_logy,
        )

    for lvl, color, label in [
        (_SPIKE_CRITICAL, "#C44E52", f"급등 {_SPIKE_CRITICAL:.0f}%"),
        (_SPIKE_HIGH,     "#DD8A00", f"주의 {_SPIKE_HIGH:.0f}%"),
        (_SPIKE_MEDIUM,   "#F0C040", f"관찰 {_SPIKE_MEDIUM:.0f}%"),
    ]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="top right",
                      annotation_font_size=9)
    _raw_pcts = df[pct_col].clip(lower=0).fillna(0)
    _overall_avg = float(_raw_pcts.mean())
    if _overall_avg > 0:
        fig.add_hline(y=_overall_avg, line_dash="dash", line_color="#4C72B0", line_width=2,
                      annotation_text=f"전체 평균 {_overall_avg:.0f}%",
                      annotation_position="bottom right",
                      annotation_font_color="#4C72B0",
                      annotation_font_size=9)
    fig.update_layout(
        height=420, xaxis_tickangle=-45,
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=55, b=90, l=50, r=20),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    ev = st.plotly_chart(fig, use_container_width=True, key=f"anom_spike_bar{sfx}", on_select="rerun")
    _handle_chart_click(ev, chart_df, field="x")


# ── Tab: 데이터 품질 (consolidated) ──────────────────────────────────────────

def _render_data_quality_tab(
    df: pd.DataFrame,
    cur_df: pd.DataFrame,
    file_name: str | None = None,
    file_data: bytes | None = None,
    all_sheet_keys: list[str] | None = None,
    prev_file_data: bytes | None = None,
    prev_sheet_keys: list[str] | None = None,
    prev_label: str | None = None,
    yoy_file_data: bytes | None = None,
    yoy_sheet_keys: list[str] | None = None,
    yoy_label: str | None = None,
) -> None:
    """Consolidated data quality view: KPI summary + tabbed detail sections."""
    from data import to_numeric_series as _tns

    st.subheader("🛡 데이터 품질 검사")

    # ── KPI summary row ───────────────────────────────────────────────────
    # Count issues
    _meter_pairs = [
        ("water", "water_previous", "water_current"),
        ("hwater", "hwater_previous", "hwater_current"),
        ("elect", "elect_previous", "elect_current"),
        ("heat", "heat_previous", "heat_current"),
    ]
    n_backward = 0
    for _, prev_col, curr_col in _meter_pairs:
        if prev_col in cur_df.columns and curr_col in cur_df.columns:
            p, c = _tns(cur_df[prev_col]), _tns(cur_df[curr_col])
            n_backward += int((c.notna() & p.notna() & (c < p)).sum())

    n_zero = int((df["n_zero_utilities"] > 0).sum()) if "n_zero_utilities" in df.columns else 0
    n_total = len(df)

    _k1, _k2, _k3 = st.columns(3)
    _k1.metric("⚠️ 역방향 검침", f"{n_backward}건")
    _k2.metric("🔍 미계량 브랜드", f"{n_zero}개")
    _k3.metric("전체 브랜드", f"{n_total}개")

    # ── Tabbed detail sections ────────────────────────────────────────────
    _dq1, _dq2, _dq3 = st.tabs(
        ["⚠️ 역방향 검침", "🔤 명칭 불일치", "🔍 미계량 검출"]
    )

    with _dq1:
        _render_backward_detection(cur_df)

    with _dq2:
        st.caption("시트 간 동일 브랜드의 명칭 불일치 검출. 불일치 시 데이터 병합에서 누락이 발생합니다.")
        _render_sheet_reconciliation(
            file_name, file_data, all_sheet_keys,
            prev_file_data=prev_file_data,
            prev_sheet_keys=prev_sheet_keys,
            prev_label=prev_label,
            yoy_file_data=yoy_file_data,
            yoy_sheet_keys=yoy_sheet_keys,
            yoy_label=yoy_label,
        )

    with _dq3:
        st.caption("유틸리티별 사용량=0 브랜드 현황 및 전월 대비 변화.")
        _render_consistency_section(
            df,
            prev_file_data=prev_file_data,
            prev_sheet_keys=prev_sheet_keys,
            prev_label=prev_label,
            yoy_file_data=yoy_file_data,
            yoy_sheet_keys=yoy_sheet_keys,
            yoy_label=yoy_label,
        )


def _render_backward_detection(cur_df: pd.DataFrame) -> None:
    """Detect meters where current reading < previous reading (physically impossible)."""
    from data import to_numeric_series
    st.caption("현재 검침값 < 이전 검침값인 경우. 계량기 교체 없이 물리적으로 불가능 — 데이터 입력 오류 가능성.")

    _meter_pairs = [
        ("water", "water_previous", "water_current", "m³"),
        ("hwater", "hwater_previous", "hwater_current", "m³"),
        ("elect", "elect_previous", "elect_current", "kWh"),
        ("heat", "heat_previous", "heat_current", "m³/MWh"),
    ]
    backward_rows = []
    for prefix, prev_col, curr_col, unit in _meter_pairs:
        if prev_col not in cur_df.columns or curr_col not in cur_df.columns:
            continue
        prev_s = to_numeric_series(cur_df[prev_col])
        curr_s = to_numeric_series(cur_df[curr_col])
        mask = curr_s.notna() & prev_s.notna() & (curr_s < prev_s)
        for idx in cur_df[mask].index:
            brand_col = "brand_raw" if "brand_raw" in cur_df.columns else "brand"
            lbl = _UTIL_LABELS_UI.get(prefix, prefix)
            backward_rows.append({
                "브랜드": cur_df.at[idx, brand_col] if brand_col in cur_df.columns else "",
                "건물": cur_df.at[idx, "building"] if "building" in cur_df.columns else "",
                "층": cur_df.at[idx, "floor"] if "floor" in cur_df.columns else "",
                "유틸리티": lbl,
                "이전 검침": f"{float(prev_s.at[idx]):,.1f} {unit}",
                "현재 검침": f"{float(curr_s.at[idx]):,.1f} {unit}",
                "차이": f"{float(curr_s.at[idx] - prev_s.at[idx]):+,.1f}",
            })

    if not backward_rows:
        st.success("역방향 검침 없음 — 모든 검침값이 정상 순서입니다.")
        return

    bw_df = pd.DataFrame(backward_rows)

    # Summary by utility
    _by_util = bw_df["유틸리티"].value_counts()
    _badges = " ".join(
        f'<span style="background:#C44E5212;color:#C44E52;border:1px solid #C44E5230;'
        f'border-radius:12px;padding:3px 10px;font-size:0.8rem;font-weight:600">'
        f'{u} {n}건</span>'
        for u, n in _by_util.items()
    )
    st.markdown(
        f'<div style="margin-bottom:10px">'
        f'<span style="font-weight:700;color:#C44E52;margin-right:8px">'
        f'⚠️ {len(backward_rows)}건 감지</span>{_badges}</div>',
        unsafe_allow_html=True,
    )

    # Table
    st.dataframe(bw_df, hide_index=True, use_container_width=True)

    # Correction tool
    with st.expander("✏️ 역방향 검침 보정", expanded=False):
        st.caption(
            "계량기 교체로 인한 정상적인 역방향인 경우 아래에서 제외 처리하세요. "
            "제외된 항목은 이상감지 분석에서 페널티가 적용되지 않습니다."
        )
        _exclude_key = "_dq_backward_exclude"
        _excluded = st.session_state.get(_exclude_key, [])

        _all_brands = sorted(bw_df["브랜드"].unique())
        _to_exclude = st.multiselect(
            "계량기 교체 확인된 브랜드",
            [b for b in _all_brands if b not in _excluded],
            key="dq_bw_exclude_sel",
        )
        _ec1, _ec2 = st.columns(2)
        with _ec1:
            if st.button("✅ 제외 처리", key="dq_bw_exclude_btn",
                         disabled=not _to_exclude):
                _excluded.extend(_to_exclude)
                st.session_state[_exclude_key] = _excluded
                st.success(f"{len(_to_exclude)}개 브랜드 제외 처리됨")
                st.rerun()
        with _ec2:
            if _excluded and st.button("🗑️ 제외 초기화", key="dq_bw_reset"):
                st.session_state[_exclude_key] = []
                st.rerun()

        if _excluded:
            st.markdown(
                f'<div style="font-size:0.78rem;color:inherit;opacity:0.55;margin-top:4px">'
                f'현재 제외: {", ".join(_excluded)}</div>',
                unsafe_allow_html=True,
            )


# ── Section: 일관성 검사 (미계량) ──────────────────────────────────────────────

def _render_consistency_section(
    df: pd.DataFrame,
    prev_file_data: bytes | None = None,
    prev_sheet_keys: list[str] | None = None,
    prev_label: str | None = None,
    yoy_file_data: bytes | None = None,
    yoy_sheet_keys: list[str] | None = None,
    yoy_label: str | None = None,
) -> None:
    # ── 1. Change detection FIRST — most actionable ────────────────────
    _render_zero_change(df, prev_file_data, prev_sheet_keys, prev_label,
                        yoy_file_data, yoy_sheet_keys, yoy_label)

    # ── 2. Current vs previous comparison chart ──────────────────────────
    from data import to_numeric_series as _tns_dq
    _cur_zeros: dict[str, int] = {}
    for pfx in _UTIL_PREFIXES:
        col = f"{pfx}_current"
        if col in df.columns:
            _cur_zeros[_UTIL_LABELS_UI.get(pfx, pfx)] = int(
                (_tns_dq(df[col]).fillna(0) == 0).sum()
            )
    _prev_zeros: dict[str, int] = {}
    if prev_file_data and prev_sheet_keys:
        _pz = _build_zero_set(prev_file_data, prev_sheet_keys)
        for pfx, brands in _pz.items():
            _prev_zeros[_UTIL_LABELS_UI.get(pfx, pfx)] = len(brands)

    if _cur_zeros:
        _labels = list(_cur_zeros.keys())
        fig = go.Figure()
        if _prev_zeros:
            fig.add_trace(go.Bar(
                name="전월", x=_labels,
                y=[_prev_zeros.get(l, 0) for l in _labels],
                marker_color="#A8C4E0",
                text=[_prev_zeros.get(l, 0) for l in _labels],
                textposition="outside",
            ))
        fig.add_trace(go.Bar(
            name="이번 달", x=_labels,
            y=[_cur_zeros.get(l, 0) for l in _labels],
            marker_color="#DD8A00",
            text=[_cur_zeros.get(l, 0) for l in _labels],
            textposition="outside",
        ))
        fig.update_layout(
            title="유틸리티별 미계량 브랜드 수" + (" — 전월 비교" if _prev_zeros else ""),
            barmode="group", height=280,
            margin=dict(t=45, b=30), yaxis_title="브랜드 수",
            legend=dict(orientation="h", y=1.1, x=0),
        )
        st.plotly_chart(fig, use_container_width=True, key="anom_zero_cmp_bar")

    # ── 3. Full zero-usage list — per utility detail ────────────────────
    from data import to_numeric_series as _tns_full
    _zero_rows = []
    for pfx in _UTIL_PREFIXES:
        col = f"{pfx}_current"
        if col not in df.columns:
            continue
        lbl = _UTIL_LABELS_UI.get(pfx, pfx)
        _vals = _tns_full(df[col]).fillna(0)
        for idx in df[_vals == 0].index:
            _zero_rows.append({
                "브랜드": df.at[idx, "brand"] if "brand" in df.columns else "",
                "건물": df.at[idx, "building"] if "building" in df.columns else "",
                "유틸리티": lbl,
            })

    if _zero_rows:
        _zdf = pd.DataFrame(_zero_rows)

        # Pivot: brand → list of zero utilities
        _pivot = _zdf.groupby(["브랜드", "건물"])["유틸리티"].apply(
            lambda x: " · ".join(sorted(x))
        ).reset_index()
        _pivot.columns = ["브랜드", "건물", "미계량 항목"]
        _pivot["미계량 수"] = _pivot["미계량 항목"].str.count("·") + 1

        # Count available utilities
        _n_avail = len([p for p in _UTIL_PREFIXES if f"{p}_current" in df.columns])

        # Separate: all-4 missing vs partial
        _all_missing = _pivot[_pivot["미계량 수"] >= _n_avail].copy()
        _partial = _pivot[_pivot["미계량 수"] < _n_avail].copy()

        st.markdown(f"**📋 전체 미계량 현황 — {len(_pivot)}개 브랜드, {len(_zdf)}건**")

        # All-4 missing — highlight first with full details
        if not _all_missing.empty:
            st.error(f"**⛔ 전 항목 미계량 — {len(_all_missing)}개 브랜드** (공실·폐업·데이터 누락 가능성)")
            # Join back to source df for details
            _detail_cols = [c for c in ["brand", "building", "floor", "size_m2", "size_py"]
                            if c in df.columns]
            _all_detail = df[df["brand"].isin(_all_missing["브랜드"])][_detail_cols].copy()
            _col_rename = {"brand": "브랜드", "building": "건물", "floor": "층",
                           "size_m2": "면적(m²)", "size_py": "면적(평)"}
            _all_detail = _all_detail.rename(columns=_col_rename)
            st.dataframe(_all_detail.reset_index(drop=True),
                         hide_index=True, use_container_width=True)

        # Summary + partial table
        _summary = _zdf["유틸리티"].value_counts().reset_index()
        _summary.columns = ["유틸리티", "미계량 수"]
        _sc1, _sc2 = st.columns([1, 3])
        with _sc1:
            st.dataframe(_summary, hide_index=True, use_container_width=True)
        with _sc2:
            _show = _partial if not _partial.empty else _pivot
            # Join floor/size details
            _detail_join = df[["brand", "building"] +
                              [c for c in ["floor", "size_m2", "size_py"] if c in df.columns]
                              ].drop_duplicates(subset=["brand", "building"])
            _show_detail = _show.merge(
                _detail_join.rename(columns={"brand": "브랜드", "building": "건물"}),
                on=["브랜드", "건물"], how="left",
            )
            _show_cols = ["브랜드", "건물"]
            if "floor" in _show_detail.columns:
                _show_detail = _show_detail.rename(columns={"floor": "층"})
                _show_cols.append("층")
            if "size_py" in _show_detail.columns:
                _show_detail = _show_detail.rename(columns={"size_py": "면적(평)"})
                _show_cols.append("면적(평)")
            _show_cols.append("미계량 항목")
            st.dataframe(
                _show_detail[_show_cols].sort_values(
                    "미계량 항목", ascending=False
                ).reset_index(drop=True),
                hide_index=True, use_container_width=True,
            )
    else:
        st.success("미계량 브랜드 없음 — 모든 유틸리티 정상 계량")

    # (Cross-sheet reconciliation moved to its own tab — 🔤 명칭 불일치)


def _render_sheet_reconciliation(
    file_name: str | None,
    file_data: bytes | None,
    all_sheet_keys: list[str] | None,
    prev_file_data: bytes | None = None,
    prev_sheet_keys: list[str] | None = None,
    prev_label: str | None = None,
    yoy_file_data: bytes | None = None,
    yoy_sheet_keys: list[str] | None = None,
    yoy_label: str | None = None,
) -> None:
    """Cross-sheet brand reconciliation: name consistency + amount verification."""
    if not file_data or not all_sheet_keys:
        st.info("파일을 업로드하면 시트 간 브랜드 명칭 불일치를 검출합니다.")
        return

    SH_A = "브랜드별 집계 내역"
    SH_B_match = next((s for s in all_sheet_keys
                       if s.strip() == "수도광열비 부과 내역"), None)
    if SH_A not in all_sheet_keys or SH_B_match is None:
        st.warning(f"교차검증에 필요한 시트를 찾을 수 없습니다 ('{SH_A}' 또는 '수도광열비 부과 내역').")
        return
    SH_B = SH_B_match.strip()

    # Show tabs for each available period
    file_pairs = [("📋 현재 파일", file_data, all_sheet_keys)]
    for extra_data, extra_sheets, extra_label, default_lbl in [
        (prev_file_data, prev_sheet_keys, prev_label, "전월"),
        (yoy_file_data, yoy_sheet_keys, yoy_label, "전년"),
    ]:
        if extra_data and extra_sheets:
            sh_b = next((s for s in extra_sheets
                         if s.strip() == "수도광열비 부과 내역"), None)
            if sh_b and SH_A in extra_sheets:
                lbl = extra_label or default_lbl
                file_pairs.append((f"📅 {lbl}", extra_data, extra_sheets))

    if len(file_pairs) > 1:
        recon_tabs = st.tabs([fp[0] for fp in file_pairs])
    else:
        recon_tabs = [st.container()]

    import io
    from brand_normalize import (
        reconcile_sheets, find_name_inconsistencies,
        normalize_brand, load_synonyms, save_synonyms,
    )
    synonyms = load_synonyms()
    n_saved = len(synonyms)

    for tab_idx, (tab_label, fdata, sheet_keys) in enumerate(file_pairs):
        with recon_tabs[tab_idx]:
            sh_b_match = next((s for s in sheet_keys
                               if s.strip() == "수도광열비 부과 내역"), None)
            if not sh_b_match:
                st.warning("수도광열비 부과 내역 시트를 찾을 수 없습니다.")
                continue

            a_raw = pd.read_excel(io.BytesIO(fdata), sheet_name=SH_A,
                                  header=None, engine="calamine")
            b_raw = pd.read_excel(io.BytesIO(fdata), sheet_name=sh_b_match,
                                  header=None, engine="calamine")

            result = reconcile_sheets(
                a_raw, b_raw,
                a_brand_col=10, b_brand_col=9,
                a_unit_col=4, b_unit_col=4,
                a_totals={"전용": 13, "공용": 14, "합계": 15},
                b_totals={"전용": 21, "공용": 22, "합계": 23},
                synonyms=synonyms,
            )
            s = result["summary"]

            _render_reconciliation_body(
                result, s, fdata, sheet_keys, synonyms, n_saved,
                SH_A, SH_B, tab_idx,
            )


def _render_reconciliation_body(
    result: dict, s: dict,
    file_data: bytes, all_sheet_keys: list[str],
    synonyms: dict, n_saved: int,
    SH_A: str, SH_B: str,
    tab_idx: int = 0,
) -> None:
    """Render reconciliation: inconsistencies → KPIs → mapping UI."""
    from brand_normalize import (
        find_name_inconsistencies, normalize_brand, load_synonyms, save_synonyms,
    )
    sfx = f"_{tab_idx}" if tab_idx else ""

    # ── 1. Name inconsistencies — most important ─────────────────────────
    inconsistencies = find_name_inconsistencies(file_data, all_sheet_keys, synonyms=synonyms)
    if inconsistencies:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            f'<span style="background:#C44E5212;color:#C44E52;border:1px solid #C44E5230;'
            f'border-radius:12px;padding:4px 12px;font-size:0.82rem;font-weight:600">'
            f'🔤 명칭 불일치 {len(inconsistencies)}건</span>'
            f'<span style="background:#DD8A0012;color:#DD8A00;border:1px solid #DD8A0030;'
            f'border-radius:12px;padding:4px 12px;font-size:0.82rem;font-weight:600">'
            f'⚠ 추정 매칭 {s["fuzzy_suggested"]}건</span>'
            f'<span style="background:#C44E5212;color:#C44E52;border:1px solid #C44E5230;'
            f'border-radius:12px;padding:4px 12px;font-size:0.82rem;font-weight:600">'
            f'💰 금액 불일치 {s["amount_mismatches"]}건</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        _SHEET_COLORS = {
            "브랜드별 집계 내역": "#4C72B0",
            "수도광열비 부과 내역": "#C44E52",
            "수도 사용 내역": "#55A868",
            "온수 사용 내역": "#DD8A00",
            "전체 전기 사용내역": "#8172B2",
        }
        def _badge(sheet: str) -> str:
            bg = _SHEET_COLORS.get(sheet, "#888")
            return (f'<span style="background:{bg};color:#fff;padding:2px 7px;'
                    f'border-radius:10px;font-size:0.82em;white-space:nowrap">'
                    f'{sheet}</span>')

        trs = []
        for item in inconsistencies:
            by_variant: dict[str, list[str]] = {}
            for sheet, raw in item["variants"].items():
                by_variant.setdefault(raw, []).append(sheet)
            parts = [
                f'{variant} {" ".join(_badge(sh) for sh in sheets)}'
                for variant, sheets in by_variant.items()
            ]
            detail = "<br>".join(parts)
            trs.append(
                f'<tr style="border-bottom:1px solid #eee">'
                f'<td style="padding:5px;vertical-align:top">{item["normalized"]}</td>'
                f'<td style="padding:5px">{detail}</td></tr>'
            )
        html = (
            '<table style="width:100%;border-collapse:collapse;font-size:0.9em">'
            '<thead><tr style="border-bottom:2px solid #ccc;text-align:left">'
            '<th style="padding:6px">정규화</th>'
            '<th style="padding:6px">표기 · 사용 시트</th>'
            '</tr></thead><tbody>'
            + "".join(trs)
            + "</tbody></table>"
        )
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.success("명칭 불일치 없음")

    # ── 2. KPI row ─────────────────────────────────────────────────────
    with st.expander("📊 매칭 현황", expanded=False):
        kc = st.columns(4)
        kc[0].metric("정확 매칭", f"{s['exact_match']}개")
        kc[1].metric("유사 매칭", f"{s['fuzzy_match']}개")
        kc[2].metric(f"{SH_A}", f"{s['a_total']}개")
        kc[3].metric(f"{SH_B}", f"{s['b_total']}개")

    # ── 3. Interactive brand mapping UI ───────────────────────────────────
    # Combine fuzzy_suggested + unmatched pairs for unified editing
    all_candidates = []
    for fs in result.get("fuzzy_suggested", []):
        all_candidates.append({
            "집계": fs["brand_a"], "부과": fs["brand_b"],
            "건물": fs["building"], "유사도": fs["similarity"],
            "norm_a": fs["norm_a"], "norm_b": fs["norm_b"],
            "source": "fuzzy",
        })
    # Add unmatched pairs that share same building+unit
    for a_item in result.get("only_a", []):
        if a_item["brand"] in ("계약손실",) or "사무실" in a_item["brand"]:
            continue
        for b_item in result.get("only_b", []):
            if a_item["building"] == b_item["building"]:
                all_candidates.append({
                    "집계": a_item["brand"], "부과": b_item["brand"],
                    "건물": a_item["building"], "유사도": 0,
                    "norm_a": normalize_brand(a_item["brand"]),
                    "norm_b": normalize_brand(b_item["brand"]),
                    "source": "unmatched",
                })

    if all_candidates:
        import streamlit_antd_components as _sac_m
        with st.expander(
            f"🔧 브랜드 매칭 교정 (저장: {n_saved}건, 후보: {len(all_candidates)}건)",
            expanded=bool(result.get("fuzzy_suggested")),
        ):
            if n_saved:
                st.caption(f"현재 {n_saved}건의 동의어 매핑이 저장되어 자동 적용 중")

            # ─ Step 1: Select which candidates to process ────────────
            labels = [
                f"{c['집계']} ↔ {c['부과']} ({c['건물']}동"
                + (f" · {c['유사도']}%" if c['유사도'] else "") + ")"
                for c in all_candidates
            ]
            mode = _sac_m.segmented(
                [_sac_m.SegmentedItem(label="전체"),
                 _sac_m.SegmentedItem(label="선택"),
                 _sac_m.SegmentedItem(label="제외")],
                key=f"brand_map_mode{sfx}", use_container_width=True,
            )

            if mode == "전체":
                selected_idx = set(range(len(all_candidates)))
            elif mode == "선택":
                picks = st.multiselect(
                    "매칭할 브랜드", labels, default=labels,
                    key=f"brand_pick{sfx}",
                )
                selected_idx = {i for i, l in enumerate(labels) if l in picks}
            else:  # 제외
                excludes = st.multiselect(
                    "제외할 브랜드", labels, default=[],
                    key=f"brand_excl{sfx}",
                )
                excluded = {i for i, l in enumerate(labels) if l in excludes}
                selected_idx = set(range(len(all_candidates))) - excluded

            selected = [all_candidates[i] for i in sorted(selected_idx)]

            if not selected:
                st.info("선택된 항목이 없습니다.")
            else:
                # ─ Step 2: Name format ───────────────────────────────
                fmt = st.radio(
                    "대표 이름 형식",
                    ["정규화 (표준)", "집계 이름 기준", "부과 이름 기준", "직접 입력"],
                    horizontal=True, key=f"name_fmt{sfx}",
                )

                pending: dict[str, str] = {}

                if fmt == "직접 입력":
                    st.caption("각 브랜드별 대표 이름을 입력하세요")
                    for i, c in enumerate(selected):
                        _c1, _c2 = st.columns([5, 5])
                        with _c1:
                            st.text(f"{c['집계']} ↔ {c['부과']}")
                        with _c2:
                            custom = st.text_input(
                                "이름", value=c["norm_a"],
                                key=f"custom_name_{i}{sfx}",
                                label_visibility="collapsed",
                            )
                        cn = normalize_brand(custom) if custom else ""
                        if cn:
                            if c["norm_a"] != cn:
                                pending[c["norm_a"]] = cn
                            if c["norm_b"] != cn:
                                pending[c["norm_b"]] = cn
                else:
                    # Preview table
                    preview_rows = []
                    for c in selected:
                        if fmt == "정규화 (표준)":
                            canon = c["norm_a"]  # normalized form
                        elif fmt == "집계 이름 기준":
                            canon = normalize_brand(c["집계"])
                        else:
                            canon = normalize_brand(c["부과"])

                        preview_rows.append({
                            "집계": c["집계"], "부과": c["부과"],
                            "건물": c["건물"], "→ 대표": canon,
                        })
                        if c["norm_a"] != canon:
                            pending[c["norm_a"]] = canon
                        if c["norm_b"] != canon:
                            pending[c["norm_b"]] = canon

                    st.dataframe(
                        pd.DataFrame(preview_rows),
                        hide_index=True, use_container_width=True,
                    )

                # ─ Step 3: Save ──────────────────────────────────────
                _s1, _s2 = st.columns(2)
                with _s1:
                    if st.button("💾 저장", type="primary",
                                 disabled=not pending, key=f"save_synonyms{sfx}"):
                        merged = {**synonyms, **pending}
                        save_synonyms(merged)
                        st.success(f"{len(pending)}건 저장 (총 {len(merged)}건)")
                        st.rerun()
                with _s2:
                    if n_saved and st.button("🗑️ 초기화", key=f"reset_synonyms{sfx}"):
                        save_synonyms({})
                        st.success("초기화 완료")
                        st.rerun()

    # Unmatched brands (remaining after synonym + fuzzy)
    _col_ren = {"brand": "브랜드", "building": "건물", "unit": "호수"}
    if result["only_a"] or result["only_b"]:
        _ua, _ub = st.columns(2)
        for col, label, data in [(_ua, SH_A, result["only_a"]),
                                  (_ub, SH_B, result["only_b"])]:
            with col:
                st.caption(f"[{label}]에만 존재 ({len(data)}개)")
                if data:
                    st.dataframe(
                        pd.DataFrame(data).rename(columns=_col_ren),
                        hide_index=True, use_container_width=True)
                else:
                    st.success("없음")

    # Amount mismatches — vertical layout: 브랜드+동 on first row, rest below
    if result["amount_mismatches"]:
        st.warning(f"금액 불일치: {s['amount_mismatches']}건")
        am_raw = result["amount_mismatches"]
        # Group by (brand, building), preserve field order
        from collections import OrderedDict
        groups: OrderedDict[tuple, list] = OrderedDict()
        for m in am_raw:
            key = (m["brand_a"], m["building"])
            groups.setdefault(key, []).append(m)
        rows = []
        for (brand, bldg), items in groups.items():
            for idx, m in enumerate(items):
                rows.append({
                    "브랜드": brand if idx == 0 else "",
                    "동": bldg if idx == 0 else "",
                    "항목": m["field"],
                    "집계": f"{m['a_value']:,.0f}",
                    "부과": f"{m['b_value']:,.0f}",
                    "차이": f"{m['a_value'] - m['b_value']:+,.0f}",
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    elif s["exact_match"] + s["fuzzy_match"] > 0:
        st.success("매칭된 모든 브랜드의 전용/공용/합계 금액 일치")


# ── Public render ─────────────────────────────────────────────────────────────


def render_data_quality_tab(
    cur_df: pd.DataFrame,
    file_name: str,
    file_data: bytes,
    all_sheet_keys: list[str],
    prev_file_data: bytes | None = None,
    prev_sheet_keys: list[str] | None = None,
    yoy_file_data: bytes | None = None,
    yoy_sheet_keys: list[str] | None = None,
) -> None:
    """Standalone data quality tab — builds anomaly_df internally."""
    with st.spinner("데이터 품질 분석 중…"):
        sheets = _load_sheets(file_name, file_data, all_sheet_keys)
        try:
            anomaly_df = build_anomaly_df(
                meter_df=cur_df,
                billing_df=sheets.get(BILLING_SHEET_NAME),
                elec_df=sheets.get(ELECTRICITY_SHEET_NAME),
                water_df=sheets.get(WATER_SHEET_NAME),
                hotwater_df=sheets.get(HOTWATER_SHEET_NAME),
            )
        except Exception as e:
            st.error(f"데이터 품질 분석 실패: {e}")
            return

    from utils import display_brand as _display_brand
    anomaly_df = _display_brand(anomaly_df)

    _render_data_quality_tab(
        anomaly_df, cur_df, file_name, file_data, all_sheet_keys,
        prev_file_data=prev_file_data,
        prev_sheet_keys=prev_sheet_keys,
        yoy_file_data=yoy_file_data,
        yoy_sheet_keys=yoy_sheet_keys,
    )


def render_anomaly_tab(
    cur_df: pd.DataFrame,
    file_name: str,
    file_data: bytes,
    all_sheet_keys: list[str],
    split_by_building: bool = True,
    prev_file_data: bytes | None = None,
    prev_sheet_keys: list[str] | None = None,
    prev_label: str | None = None,
    yoy_file_data: bytes | None = None,
    yoy_sheet_keys: list[str] | None = None,
    yoy_label: str | None = None,
) -> None:
    """Render the 이상감지 분석 view — loads immediately (no lazy-load button)."""

    with st.spinner("데이터 로드 및 이상 신호 산출 중…"):
        sheets = _load_sheets(file_name, file_data, all_sheet_keys)
        try:
            anomaly_df = build_anomaly_df(
                meter_df=cur_df,
                billing_df=sheets.get(BILLING_SHEET_NAME),
                elec_df=sheets.get(ELECTRICITY_SHEET_NAME),
                water_df=sheets.get(WATER_SHEET_NAME),
                hotwater_df=sheets.get(HOTWATER_SHEET_NAME),
            )
        except Exception as e:
            st.error(f"이상감지 분석 실패: {e}")
            return

    if anomaly_df.empty:
        st.error("이상감지 데이터를 생성할 수 없습니다.")
        return

    # Restore original brand names for display
    from utils import display_brand as _display_brand
    anomaly_df = _display_brand(anomaly_df)

    has_billing = BILLING_SHEET_NAME in sheets
    has_elec    = ELECTRICITY_SHEET_NAME in sheets

    # ── Data quality warning ─────────────────────────────────────────────
    from data import to_numeric_series as _tns_dqw
    _n_bw = 0
    for _pfx in _UTIL_PREFIXES:
        _pc, _cc = f"{_pfx}_previous", f"{_pfx}_current"
        if _pc in cur_df.columns and _cc in cur_df.columns:
            _p, _c = _tns_dqw(cur_df[_pc]), _tns_dqw(cur_df[_cc])
            _n_bw += int((_c.notna() & _p.notna() & (_c < _p)).sum())
    _n_zr = int((anomaly_df["n_zero_utilities"] > 0).sum()) if "n_zero_utilities" in anomaly_df.columns else 0
    _n_all_zero = 0
    if "n_zero_utilities" in anomaly_df.columns:
        _n_avail_u = len([p for p in _UTIL_PREFIXES if f"{p}_current" in anomaly_df.columns])
        _n_all_zero = int((anomaly_df["n_zero_utilities"] >= _n_avail_u).sum())

    _dq_issues: list[str] = []
    if _n_bw > 0:
        _dq_issues.append(f"역방향 검침 {_n_bw}건")
    if _n_all_zero > 0:
        _dq_issues.append(f"전 항목 미계량 {_n_all_zero}개")
    if _n_zr > 5:
        _dq_issues.append(f"미계량 브랜드 {_n_zr}개")

    if _dq_issues:
        _issue_badges = " ".join(
            f'<span style="background:#C44E5218;color:#C44E52;border:1px solid #C44E5240;'
            f'border-radius:8px;padding:2px 8px;font-size:0.78rem;font-weight:600">'
            f'{iss}</span>'
            for iss in _dq_issues
        )
        st.markdown(
            f'<div style="background:linear-gradient(135deg,#C44E5225,#C44E5215);'
            f'border:2px solid #C44E5260;border-radius:10px;padding:14px 18px;margin-bottom:12px">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
            f'<span style="font-size:1.3rem">🚨</span>'
            f'<span style="font-size:0.92rem;font-weight:800;color:#C44E52">'
            f'데이터 품질 문제 감지</span>'
            f'</div>'
            f'<div style="margin-bottom:8px">{_issue_badges}</div>'
            f'<div style="font-size:0.8rem;color:inherit;opacity:0.65">'
            f'분석 정확도에 직접 영향을 줍니다. '
            f'사이드바 <b>🛡 품질 검사</b>에서 데이터를 먼저 검증하세요.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 0. Overall trend — context first ────────────────────────────────
    _render_overall_trend(anomaly_df)

    # ── 0b. KPI row ──────────────────────────────────────────────────────
    _render_kpis(anomaly_df, has_billing, has_elec)

    # ── 1. Key insight summary — risk brands + signals ───────────────────
    _render_insight_summary(anomaly_df, sheets)

    # ── 2. Tabs — investigation + master table as first tab ─────────────────
    tab_master, tab_spike, tab_mgmt = st.tabs(
        ["🔍 조사 대상", "📈 급등 감지", "📋 경영 보고"]
    )

    with tab_master:
        st.caption(
            "복합 이상 점수는 사용량 급등, 단가 이상, 사분면 분류, HVAC 강도, 미계량 여부를 종합한 지표입니다. "
            "점수가 높을수록 빌링 오류 또는 설비 이상 가능성이 높으며, 상위 브랜드부터 조사를 권장합니다."
        )

        _n = st.slider("표시 브랜드 수", 10, len(anomaly_df),
                       min(20, len(anomaly_df)), key="anom_n")

        _col_bar, _col_heat = st.columns(2)
        with _col_bar:
            _render_composite_bar(anomaly_df, _n, split_by_building)
        with _col_heat:
            _render_heatmap(anomaly_df, _n)

        id_cols    = [c for c in ["brand", "building", "floor"] if c in anomaly_df.columns]
        reason_col = ["reason"] if "reason" in anomaly_df.columns else []
        key_cols   = [c for c in ["composite_score", "risk_level",
                                  "spike_max_pct", "spike_worst_util",
                                  "spike_bldg_avg_pct", "spike_peer_ratio"] if c in anomaly_df.columns]
        master_show = id_cols + key_cols + reason_col
        master_view = add_display_index(anomaly_df[master_show])
        _col_cfg = {
            "No":                  st.column_config.NumberColumn("No", width="small"),
            "brand":               st.column_config.TextColumn("브랜드", width="medium"),
            "building":            st.column_config.TextColumn("건물", width="small"),
            "floor":               st.column_config.TextColumn("층", width="small"),
            "composite_score":     st.column_config.ProgressColumn(
                "복합 점수", format="%.3f", min_value=0, max_value=1, width="small"),
            "risk_level":          st.column_config.TextColumn("위험도", width="small"),
            "spike_max_pct":       st.column_config.NumberColumn(
                "최대 증가율(%)", format="%.1f", width="small"),
            "spike_worst_util":    st.column_config.TextColumn("급등 항목", width="small"),
            "spike_bldg_avg_pct":  st.column_config.NumberColumn(
                "건물평균(%)", format="%.1f", width="small"),
            "spike_peer_ratio":    st.column_config.NumberColumn(
                "vs건물", format="%.1fx", width="small"),
            "reason":              st.column_config.TextColumn("이유", width="large"),
        }
        st.dataframe(
            master_view,
            column_config=_col_cfg,
            hide_index=True,
            use_container_width=True,
        )
        download_df_as_excel(master_view, filename="anomaly_investigation.xlsx", sheet_name="조사대상")

    with tab_spike:
        # Two comparison modes: 전월 대비 (MoM) and 작년동월 대비 (YoY)
        _has_prev = prev_file_data is not None and prev_sheet_keys is not None
        _has_yoy = yoy_file_data is not None and yoy_sheet_keys is not None

        if _has_prev or _has_yoy:
            _spike_tab_labels = ["📈 전월 대비"]
            if _has_yoy:
                _spike_tab_labels.append(f"📅 작년동월 대비")
            _spike_tabs = st.tabs(_spike_tab_labels)
        else:
            _spike_tabs = [st.container()]

        # Tab 1: 전월 대비 — existing anomaly_df (already MoM)
        with _spike_tabs[0]:
            _render_spike_tab(anomaly_df, split_by_building,
                              key_suffix="_mom")

        # Tab 2: 작년동월 대비 — cross-file comparison
        if _has_yoy and len(_spike_tabs) > 1:
            with _spike_tabs[-1]:
                with st.spinner("작년동월 대비 분석 중…"):
                    _yoy_anomaly = _build_cross_file_anomaly(
                        file_data, all_sheet_keys,
                        yoy_file_data, yoy_sheet_keys,
                        label=yoy_label or "전년동월",
                    )
                if _yoy_anomaly is not None and not _yoy_anomaly.empty:
                    _render_spike_tab(_yoy_anomaly, split_by_building,
                                      key_suffix="_yoy")
                else:
                    st.info("작년동월 파일에서 비교 가능한 검침 데이터를 찾을 수 없습니다.")

    with tab_mgmt:
        # Apply same filters as anomaly tab to utility sheets
        from filters import apply_sheet_filter as _apply_flt
        _sb = st.session_state.get("t1_building", ["All"])
        _sf = st.session_state.get("t1_floor", ["All"])
        _gm = st.session_state.get("t1_gongshil", "All")
        _bs = st.session_state.get("t1_brand_search", "").strip().lower()

        def _flt_sheet(df):
            if df is None or df.empty:
                return df
            return _apply_flt(df, _sb, _sf, _gm, _bs)

        # Load prev sheets for MoM in management report
        _prev_sheets = (
            _load_sheets("_prev_", prev_file_data, prev_sheet_keys)
            if prev_file_data and prev_sheet_keys else {}
        )
        render_mgmt_report(
            water_df=_flt_sheet(sheets.get(WATER_SHEET_NAME)),
            hotwater_df=_flt_sheet(sheets.get(HOTWATER_SHEET_NAME)),
            elec_df=_flt_sheet(sheets.get(ELECTRICITY_SHEET_NAME)),
            billing_df=_flt_sheet(sheets.get(BILLING_SHEET_NAME)),
            meter_df=cur_df,
            prev_water_df=_flt_sheet(_prev_sheets.get(WATER_SHEET_NAME)),
            prev_hotwater_df=_flt_sheet(_prev_sheets.get(HOTWATER_SHEET_NAME)),
            prev_elec_df=_flt_sheet(_prev_sheets.get(ELECTRICITY_SHEET_NAME)),
            prev_billing_df=_flt_sheet(_prev_sheets.get(BILLING_SHEET_NAME)),
            split_by_building=split_by_building,
        )

    st.divider()

    # ── 5. Reference — scoring method, PDF, raw data ─────────────────────────
    _ref1, _ref2 = st.columns(2)
    with _ref1:
        _pdf_key = f"anomaly_pdf_{file_name}"
        render_pdf_buttons(
            _pdf_key,
            lambda: generate_anomaly_pdf(anomaly_df),
            "📥 PDF 리포트",
            "이상감지_리포트.pdf",
        )
    with _ref2:
        st.caption("💡 비용·HVAC·소비 상세 분석은 **📊 인사이트** 탭에서 확인하세요.")

    with st.expander("📖 이상 점수 계산 방법", expanded=False):
        st.markdown("""
**복합 점수** = 급등(30%) + 소비(25%) + 비용(25%) + HVAC(10%) + 일관성(10%)  — 각 구성 요소 [0, 1]

| 구성 요소 | 신호 | 시트 |
|---|---|---|
| **급등** ★ | 전월 대비 사용량 증가율 절대값 기준 — 🔴 ≥100% / 🟠 ≥50% / 🟡 ≥20% | 검침내역 |
| **소비** | 유틸리티별 사분면 점수 합산 정규화 (HH=4, HL=3, LH=2, Normal=1, LL=0) | 검침내역 |
| **비용** | 수도 ₩/m³, 전기 ₩/kWh, 총비용 만원/m² Z-점수의 최댓값 정규화 | 수도광열비 부과 내역 |
| **HVAC** | HVAC 강도 (kWh/m²) IQR-보정 정규화 | 전체 전기 사용내역 |
| **일관성** | 사용량=0 유틸리티 항목 수 정규화 | 검침내역 |

**위험 등급**: 🔴 위험 ≥ 0.65 · 🟠 주의 ≥ 0.40 · 🟡 관찰 ≥ 0.20 · 🟢 정상 < 0.20

**동종 비교 (vs건물)**: 같은 건물 내 다른 브랜드 평균 급등률 대비 배수. 2x 이상 = 건물 전체 추세가 아닌 해당 브랜드만의 이상.
        """)

    with st.expander("📊 원시 데이터", expanded=False):
        st.dataframe(anomaly_df.reset_index(drop=True), hide_index=True, use_container_width=True)
