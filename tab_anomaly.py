"""tab_anomaly.py — 이상감지 분析 (Anomaly Detection Analysis) UI.

Lazily loads all available utility sheets, calls build_anomaly_df(),
and renders:
  1. KPI row             — risk-level brand counts
  2. Composite bar chart — ranked brands with risk colour
  3. Anomaly heatmap     — brands × signal dimensions
  4. Tabs:
       🔺 소비 이상   — quadrant bars per utility + distribution table
       💰 비용 이상   — unit cost Z-score bars with ±2σ lines
       ❄️ HVAC 이상  — HVAC intensity + EHP / HVAC share bars
       🔍 일관성 검사 — zero-usage flags across utilities
       📋 전체 결과  — sortable master table with download
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

_BLDG_COLOR = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd"}
_RISK_COLOR = {
    "🔴 위험": "#C44E52",
    "🟠 주의": "#DD8A00",
    "🟡 관찰": "#F0C040",
    "🟢 정상": "#2ca02c",
}
_QUAD_COLOR  = {"HH": "#C44E52", "HL": "#DD8A00", "LH": "#9467bd",
                "LL": "#4C72B0", "Normal": "#AACCEE", "No Data": "#DDDDDD"}
_QUAD_LABEL  = {"HH": "🔴 HH", "HL": "🟠 HL", "LH": "🟡 LH",
                "LL": "🟢 LL", "Normal": "⚪ 정상", "No Data": "—"}
_UTIL_LABELS = {"water": "💧 수도", "hwater": "🌡 온수",
                "elect": "⚡ 전기",  "heat":   "🔥 난방"}

_SCORE_CSCALE = [
    [0.00, "#2ca02c"],
    [0.35, "#F0C040"],
    [0.60, "#DD8A00"],
    [1.00, "#C44E52"],
]


# ── Sheet loader ──────────────────────────────────────────────────────────────

def _load_sheets(file_name: str, file_data: bytes, all_sheet_keys: list[str]) -> dict:
    loaders = {
        BILLING_SHEET_NAME:     read_billing_sheet,
        ELECTRICITY_SHEET_NAME: read_electricity_sheet,
        WATER_SHEET_NAME:       read_water_sheet,
        HOTWATER_SHEET_NAME:    read_hotwater_sheet,
    }
    results = {}
    for const, loader in loaders.items():
        key = next((k for k in all_sheet_keys if k.strip() == const), None)
        if key is None:
            continue
        try:
            results[const] = loader(file_name, file_data, key)
        except Exception as e:
            st.warning(f"⚠️ {const} 로드 실패: {e}")
    return results


# ── Section: KPI row ──────────────────────────────────────────────────────────

def _render_kpis(df: pd.DataFrame, has_billing: bool, has_elec: bool) -> None:
    counts = df["risk_level"].value_counts()
    cols = st.columns(6)
    cols[0].metric("분석 브랜드", f"{len(df)}개")
    cols[1].metric("🔴 위험", f"{counts.get('🔴 위험', 0)}개")
    cols[2].metric("🟠 주의", f"{counts.get('🟠 주의', 0)}개")
    cols[3].metric("🟡 관찰", f"{counts.get('🟡 관찰', 0)}개")
    cols[4].metric("🟢 정상", f"{counts.get('🟢 정상', 0)}개")
    # Data-source badge
    sources = ["검침"] + (["청구"] if has_billing else []) + (["전기"] if has_elec else [])
    cols[5].metric("분析 데이터", " · ".join(sources))


# ── Section: Composite ranked bar chart ───────────────────────────────────────

def _render_composite_bar(df: pd.DataFrame, n: int, split_by_building: bool) -> None:
    top = df.head(n).copy()
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
        textfont=dict(size=9, color="black"),
        hovertemplate="<b>%{y}</b><br>복합 이상 점수: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"복합 이상 점수 — 상위 {n}개 브랜드",
        height=max(400, n * 22 + 80),
        xaxis=dict(title="점수 [0–1]", range=[0, 1.20],
                   gridcolor="#DDDDDD", griddash="dot"),
        plot_bgcolor="white",
        margin=dict(l=10, r=170, t=50, b=40),
        showlegend=False,
    )
    _ev_comp_bar = st.plotly_chart(fig, use_container_width=True, key="anom_composite_bar", on_select="rerun")
    _sel_comp_bar = _ev_comp_bar.selection.points if _ev_comp_bar and hasattr(_ev_comp_bar, "selection") else []
    if _sel_comp_bar:
        _pt = _sel_comp_bar[0]
        _brand = _pt.get("y") or _pt.get("customdata") or _pt.get("x") or ""
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = df[df["brand"].str.contains(str(_brand)[:26], regex=False)] if _brand else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)


# ── Section: Anomaly heatmap ──────────────────────────────────────────────────

def _render_heatmap(df: pd.DataFrame, n: int) -> None:
    top = df.head(n).copy()

    heat_cols: list[str] = []
    col_labels: list[str] = []

    for pfx, label in _UTIL_LABELS.items():
        qc = f"{pfx}_quad_score"
        if qc in top.columns:
            heat_cols.append(qc)
            col_labels.append(f"{label}\n사분면")

    for col, label in [
        ("water_unit_z",        "수도\n단가Z"),
        ("elect_unit_z",        "전기\n단가Z"),
        ("total_cost_per_m2_z", "총비용\n/m²Z"),
        ("hvac_intensity_z",    "HVAC\n강도Z"),
        ("n_zero_utilities",    "미계량\n항목수"),
    ]:
        if col in top.columns:
            heat_cols.append(col)
            col_labels.append(label)

    if not heat_cols:
        return

    matrix = top[heat_cols].fillna(0).copy()
    # Z-score columns → take absolute value for colouring
    for c in heat_cols:
        if "_z" in c:
            matrix[c] = matrix[c].abs()

    # Normalise each column to [0, 1] for colour scale
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
    _ev_heatmap = st.plotly_chart(fig, use_container_width=True, key="anom_signal_heatmap", on_select="rerun")
    _sel_heatmap = _ev_heatmap.selection.points if _ev_heatmap and hasattr(_ev_heatmap, "selection") else []
    if _sel_heatmap:
        _pt = _sel_heatmap[0]
        _brand = _pt.get("y") or ""
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = df[df["brand"].str.contains(str(_brand)[:26], regex=False)] if _brand else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)


# ── Tab: 소비 이상 ────────────────────────────────────────────────────────────

def _render_consumption_tab(df: pd.DataFrame, split_by_building: bool) -> None:
    st.subheader("소비량 이상 분석")
    st.caption("검침내역 기반 각 유틸리티의 사분면 분류  ·  HH=위험 급등 · HL=큰 기저 급등 · LH=급락 · LL=안정")

    present = [p for p in _UTIL_PREFIXES
               if f"{p}_quadrant" in df.columns
               and df[f"{p}_quadrant"].ne("No Data").any()]
    if not present:
        st.info("소비량 변화 데이터가 없습니다.")
        return

    sel = st.selectbox("유틸리티", present,
                       format_func=lambda p: _UTIL_LABELS.get(p, p),
                       key="anom_util_sel")
    q_col = f"{sel}_quadrant"
    s_col = f"{sel}_quad_score"

    plot_df = df[["brand", "building", q_col, s_col]].sort_values(s_col, ascending=False)

    color_col = "building" if split_by_building and "building" in plot_df.columns else q_col
    color_map  = _BLDG_COLOR if split_by_building else _QUAD_COLOR
    cat_orders = None if split_by_building else {"category_orders": {q_col: list(_QUAD_COLOR)}}

    fig = px.bar(
        plot_df, x="brand", y=s_col,
        color=color_col,
        color_discrete_map=color_map,
        title=f"{_UTIL_LABELS.get(sel, sel)} — 사분면 점수 (HH=4 · HL=3 · LH=2 · Normal=1 · LL=0)",
        labels={s_col: "사분면 점수", "brand": "브랜드"},
    )
    fig.update_layout(height=400, xaxis_tickangle=-45, plot_bgcolor="white",
                      margin=dict(t=55, b=80))
    _ev_cons = st.plotly_chart(fig, use_container_width=True, key="anom_consumption_bar", on_select="rerun")
    _sel_cons = _ev_cons.selection.points if _ev_cons and hasattr(_ev_cons, "selection") else []
    if _sel_cons:
        _pt = _sel_cons[0]
        _brand = _pt.get("x") or _pt.get("customdata") or ""
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = plot_df[plot_df["brand"] == _brand] if _brand else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    # Quadrant distribution + per-utility heatmap of all quad scores
    c1, c2 = st.columns([1, 2])
    with c1:
        st.caption("사분면 분포")
        qd = df[q_col].value_counts().reset_index()
        qd.columns = ["사분면", "브랜드 수"]
        qd["사분면"] = qd["사분면"].map(_QUAD_LABEL).fillna(qd["사분면"])
        st.dataframe(qd, hide_index=True, use_container_width=True)

    with c2:
        # Mini heatmap: all utilities × all brands (top 30)
        score_cols = [f"{p}_quad_score" for p in _UTIL_PREFIXES if f"{p}_quad_score" in df.columns]
        if score_cols:
            top30 = df.head(30)[["brand"] + score_cols].set_index("brand")
            top30.columns = [_UTIL_LABELS.get(c.replace("_quad_score", ""), c) for c in score_cols]
            norm = top30.apply(
                lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0,
                axis=0,
            )
            fig2 = go.Figure(go.Heatmap(
                z=norm.values, x=norm.columns.tolist(),
                y=[str(b)[:24] for b in norm.index],
                colorscale=_SCORE_CSCALE, zmin=0, zmax=1,
                customdata=top30.values,
                hovertemplate="<b>%{y}</b><br>%{x}: %{customdata:.0f}<extra></extra>",
                showscale=False,
            ))
            fig2.update_layout(
                title="유틸리티별 사분면 점수 (상위 30개)",
                height=max(300, len(top30) * 18 + 80),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(side="top"),
                margin=dict(l=10, r=30, t=80, b=10),
            )
            _ev_mini_hm = st.plotly_chart(fig2, use_container_width=True, key="anom_mini_heatmap", on_select="rerun")
            _sel_mini_hm = _ev_mini_hm.selection.points if _ev_mini_hm and hasattr(_ev_mini_hm, "selection") else []
            if _sel_mini_hm:
                _pt = _sel_mini_hm[0]
                _brand = _pt.get("y") or ""
                if isinstance(_brand, (list, tuple)):
                    _brand = _brand[0]
                _fdf = df[df["brand"].str.contains(str(_brand)[:24], regex=False)] if _brand else pd.DataFrame()
                if not _fdf.empty:
                    st.caption(f"선택됨: **{_brand}**")
                    st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)


# ── Tab: 비용 이상 ────────────────────────────────────────────────────────────

def _render_cost_tab(df: pd.DataFrame, split_by_building: bool) -> None:
    st.subheader("비용 이상 분析")
    st.caption("청구 데이터 기반 단위 비용 Z-점수 — |Z| ≥ 2 이상 신호")

    z_map = {
        "water_unit_z":        ("💧 수도 단가 Z",   "water_unit_cost",   "₩/m³"),
        "elect_unit_z":        ("⚡ 전기 단가 Z",   "elect_unit_cost",   "₩/kWh"),
        "total_cost_per_m2_z": ("📊 총비용/m² Z",  "total_cost_per_m2", "만원/m²"),
    }
    available = {k: v for k, v in z_map.items() if k in df.columns}
    if not available:
        st.info("비용 분析을 위해 수도광열비 부과 내역 시트가 필요합니다.")
        return

    tabs = st.tabs([v[0] for v in available.values()])
    for tab_ui, (z_col, (label, val_col, unit)) in zip(tabs, available.items()):
        with tab_ui:
            show_cols = [c for c in ["brand", "building", val_col, z_col] if c in df.columns]
            plot_df = df[show_cols].dropna(subset=[z_col]).sort_values(z_col, ascending=False)
            if plot_df.empty:
                st.info("데이터 없음")
                continue

            color_col = "building" if split_by_building and "building" in plot_df.columns else None
            fig = px.bar(
                plot_df, x="brand", y=z_col,
                color=color_col, color_discrete_map=_BLDG_COLOR,
                title=f"{label}  (|Z| ≥ 2 = 이상)",
                labels={z_col: "Z-점수", "brand": "브랜드"},
            )
            fig.add_hline(y= 2.0, line_dash="dot", line_color="#C44E52",
                          annotation_text="+2σ", annotation_position="top right")
            fig.add_hline(y=-2.0, line_dash="dot", line_color="#C44E52",
                          annotation_text="−2σ", annotation_position="bottom right")
            fig.update_layout(height=400, xaxis_tickangle=-45, plot_bgcolor="white",
                              margin=dict(t=55, b=80))
            _ev_cost = st.plotly_chart(fig, use_container_width=True, key=f"anom_cost_{z_col}", on_select="rerun")
            _sel_cost = _ev_cost.selection.points if _ev_cost and hasattr(_ev_cost, "selection") else []
            if _sel_cost:
                _pt = _sel_cost[0]
                _brand = _pt.get("x") or _pt.get("customdata") or ""
                if isinstance(_brand, (list, tuple)):
                    _brand = _brand[0]
                _fdf = plot_df[plot_df["brand"] == _brand] if _brand else pd.DataFrame()
                if not _fdf.empty:
                    st.caption(f"선택됨: **{_brand}**")
                    st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

            anomalies = plot_df[plot_df[z_col].abs() >= 2.0]
            if not anomalies.empty:
                st.warning(f"**{len(anomalies)}개** 브랜드 — ±2σ 이상 감지")
                st.dataframe(anomalies, hide_index=True, use_container_width=True)
            else:
                st.success("±2σ 범위 내 — 이상 없음")

            # Unit cost histogram
            if val_col in plot_df.columns:
                fig_h = px.histogram(plot_df, x=val_col, nbins=20,
                                     title=f"{unit} 분포",
                                     labels={val_col: unit})
                fig_h.update_layout(height=280, margin=dict(t=45, b=40),
                                    plot_bgcolor="white")
                st.plotly_chart(fig_h, use_container_width=True, key=f"anom_hist_{val_col}")


# ── Tab: HVAC 이상 ────────────────────────────────────────────────────────────

def _render_hvac_tab(df: pd.DataFrame, split_by_building: bool) -> None:
    st.subheader("HVAC 이상 분析")
    st.caption("전기 상세 내역 기반 HVAC 강도 (kWh/m²) 및 EHP 비중")

    if "hvac_intensity" not in df.columns:
        st.info("HVAC 분析을 위해 전체 전기 사용내역 시트가 필요합니다.")
        return

    plot_df = (df[["brand", "building"] + [c for c in
                   ["hvac_intensity", "hvac_intensity_z", "ehp_pct", "hvac_pct", "base_pct"]
                   if c in df.columns]]
               .dropna(subset=["hvac_intensity"])
               .sort_values("hvac_intensity", ascending=False))

    color_col = "building" if split_by_building and "building" in plot_df.columns else None

    # IQR upper bound reference line
    hi_s = plot_df["hvac_intensity"]
    iqr_up = float(hi_s.quantile(0.75) + 1.5 * (hi_s.quantile(0.75) - hi_s.quantile(0.25)))

    fig = px.bar(
        plot_df, x="brand", y="hvac_intensity",
        color=color_col, color_discrete_map=_BLDG_COLOR,
        title="HVAC 강도 (kWh/m²)",
        labels={"hvac_intensity": "kWh/m²", "brand": "브랜드"},
    )
    if iqr_up < hi_s.max() * 5:
        fig.add_hline(y=iqr_up, line_dash="dot", line_color="#DD8A00",
                      annotation_text=f"IQR 상한 {iqr_up:.1f}", annotation_position="top right")
    fig.update_layout(height=400, xaxis_tickangle=-45, plot_bgcolor="white",
                      margin=dict(t=55, b=80))
    _ev_hvac = st.plotly_chart(fig, use_container_width=True, key="anom_hvac_intensity_bar", on_select="rerun")
    _sel_hvac = _ev_hvac.selection.points if _ev_hvac and hasattr(_ev_hvac, "selection") else []
    if _sel_hvac:
        _pt = _sel_hvac[0]
        _brand = _pt.get("x") or _pt.get("customdata") or ""
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = plot_df[plot_df["brand"] == _brand] if _brand else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    # EHP / HVAC share group bar
    pct_cols = [c for c in ["ehp_pct", "hvac_pct", "base_pct"] if c in plot_df.columns]
    if len(pct_cols) >= 2:
        melt = plot_df[["brand"] + pct_cols].head(40).melt(
            id_vars="brand", var_name="구분", value_name="비중(%)"
        )
        label_map = {"ehp_pct": "EHP", "hvac_pct": "HVAC 합계", "base_pct": "기본 부하"}
        melt["구분"] = melt["구분"].map(label_map)
        fig2 = px.bar(melt, x="brand", y="비중(%)", color="구분", barmode="group",
                      title="전기 사용 카테고리 비중 (%)",
                      labels={"brand": "브랜드"})
        fig2.update_layout(height=360, xaxis_tickangle=-45, plot_bgcolor="white",
                           margin=dict(t=50, b=80))
        _ev_hvac2 = st.plotly_chart(fig2, use_container_width=True, key="anom_hvac_pct_bar", on_select="rerun")
        _sel_hvac2 = _ev_hvac2.selection.points if _ev_hvac2 and hasattr(_ev_hvac2, "selection") else []
        if _sel_hvac2:
            _pt = _sel_hvac2[0]
            _brand = _pt.get("x") or _pt.get("customdata") or ""
            if isinstance(_brand, (list, tuple)):
                _brand = _brand[0]
            _fdf = plot_df[plot_df["brand"] == _brand] if _brand else pd.DataFrame()
            if not _fdf.empty:
                st.caption(f"선택됨: **{_brand}**")
                st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    # Scatter: HVAC intensity vs EHP share
    if "ehp_pct" in plot_df.columns:
        fig3 = px.scatter(
            plot_df, x="ehp_pct", y="hvac_intensity",
            color="building" if "building" in plot_df.columns else None,
            color_discrete_map=_BLDG_COLOR,
            text="brand",
            title="EHP 비중 vs HVAC 강도",
            labels={"ehp_pct": "EHP 비중 (%)", "hvac_intensity": "HVAC 강도 (kWh/m²)"},
            size_max=12,
        )
        fig3.update_traces(textposition="top center", textfont_size=8,
                           marker=dict(size=8, opacity=0.75))
        fig3.update_layout(height=380, plot_bgcolor="white", margin=dict(t=50, b=40))
        _ev_hvac3 = st.plotly_chart(fig3, use_container_width=True, key="anom_hvac_scatter", on_select="rerun")
        _sel_hvac3 = _ev_hvac3.selection.points if _ev_hvac3 and hasattr(_ev_hvac3, "selection") else []
        if _sel_hvac3:
            _pt = _sel_hvac3[0]
            _cd = _pt.get("customdata", [])
            _brand = _cd[0] if isinstance(_cd, list) and _cd else str(_cd) if _cd else _pt.get("text") or ""
            _fdf = plot_df[plot_df["brand"] == _brand] if _brand else pd.DataFrame()
            if not _fdf.empty:
                st.caption(f"선택됨: **{_brand}**")
                st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)


# ── Tab: 급등 감지 (MoM Spike Detection) ─────────────────────────────────────

def _render_spike_tab(df: pd.DataFrame, split_by_building: bool) -> None:
    st.subheader("📈 전월 대비 급등 감지")
    st.caption(
        f"전월 대비 사용량 증가율이 🔴 {_SPIKE_CRITICAL:.0f}% 이상 / "
        f"🟠 {_SPIKE_HIGH:.0f}% 이상 / 🟡 {_SPIKE_MEDIUM:.0f}% 이상인 브랜드를 탐지합니다. "
        "다른 브랜드와의 상대 비교 없이 **절대 증가율** 기준으로 판단합니다."
    )

    spike_pct_cols = [f"{p}_spike_pct" for p in _UTIL_PREFIXES if f"{p}_spike_pct" in df.columns]
    flag_cols      = [f"{p}_spike_flag" for p in _UTIL_PREFIXES if f"{p}_spike_flag" in df.columns]
    if not spike_pct_cols:
        st.info("전월 데이터가 없어 급등 감지를 수행할 수 없습니다.")
        return

    # ── Threshold selector ────────────────────────────────────────────────────
    thresh = st.slider(
        "급등 기준 (전월 대비 증가율 %)", 10, 300, int(_SPIKE_HIGH), step=10,
        key="spike_thresh",
        help="선택한 % 이상 증가한 브랜드만 표시합니다.",
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    raw_pct = df[spike_pct_cols].clip(lower=0).fillna(0)
    n_critical = int((df["spike_max_pct"] >= _SPIKE_CRITICAL).sum())
    n_high     = int(((df["spike_max_pct"] >= _SPIKE_HIGH) & (df["spike_max_pct"] < _SPIKE_CRITICAL)).sum())
    n_medium   = int(((df["spike_max_pct"] >= _SPIKE_MEDIUM) & (df["spike_max_pct"] < _SPIKE_HIGH)).sum())
    n_above    = int((df["spike_max_pct"] >= thresh).sum())
    kc = st.columns(4)
    kc[0].metric(f"🔴 급등 (≥{_SPIKE_CRITICAL:.0f}%)", f"{n_critical}개")
    kc[1].metric(f"🟠 주의 (≥{_SPIKE_HIGH:.0f}%)",     f"{n_high}개")
    kc[2].metric(f"🟡 관찰 (≥{_SPIKE_MEDIUM:.0f}%)",   f"{n_medium}개")
    kc[3].metric(f"기준 초과 (≥{thresh}%)",             f"{n_above}개")

    # ── Spike brands table ────────────────────────────────────────────────────
    spike_df = df[df["spike_max_pct"] >= thresh].copy()
    if spike_df.empty:
        st.success(f"기준({thresh}%) 초과 브랜드 없음 — 급격한 급등 없음")
    else:
        disp_cols = (
            [c for c in ["brand", "building", "floor"] if c in spike_df.columns]
            + ["spike_max_pct", "spike_worst_util"]
            + spike_pct_cols
        )
        col_cfg: dict = {
            "spike_max_pct":    st.column_config.NumberColumn("최대 증가율 (%)", format="%.1f"),
            "spike_worst_util": st.column_config.TextColumn("급등 항목"),
        }
        util_labels = {f"{p}_spike_pct": f"{lbl} 증가율(%)" for p, lbl in _UTIL_LABELS.items()}
        for c, lbl in util_labels.items():
            if c in spike_df.columns:
                col_cfg[c] = st.column_config.NumberColumn(lbl, format="%.1f")

        st.dataframe(
            spike_df[disp_cols].sort_values("spike_max_pct", ascending=False).reset_index(drop=True),
            column_config=col_cfg,
            hide_index=True,
            use_container_width=True,
        )

    # ── Spike bar chart per utility ───────────────────────────────────────────
    st.divider()
    util_sel = st.selectbox(
        "유틸리티별 전월 대비 증가율",
        [p for p in _UTIL_PREFIXES if f"{p}_spike_pct" in df.columns],
        format_func=lambda p: _UTIL_LABELS.get(p, p),
        key="spike_util_sel",
    )
    pct_col = f"{util_sel}_spike_pct"
    flag_col = f"{util_sel}_spike_flag"

    chart_df = df[["brand"] + [c for c in ["building", pct_col, flag_col] if c in df.columns]].copy()
    chart_df = chart_df[chart_df[pct_col].notna()].sort_values(pct_col, ascending=False).head(50)

    color_col = "building" if split_by_building and "building" in chart_df.columns else None
    fig = px.bar(
        chart_df, x="brand", y=pct_col,
        color=color_col, color_discrete_map=_BLDG_COLOR,
        title=f"{_UTIL_LABELS.get(util_sel, util_sel)} 전월 대비 증가율 (%) — 상위 50개",
        labels={pct_col: "증가율 (%)", "brand": "브랜드"},
    )
    # Reference lines
    for lvl, color, label in [
        (_SPIKE_CRITICAL, "#C44E52", f"급등 {_SPIKE_CRITICAL:.0f}%"),
        (_SPIKE_HIGH,     "#DD8A00", f"주의 {_SPIKE_HIGH:.0f}%"),
        (_SPIKE_MEDIUM,   "#F0C040", f"관찰 {_SPIKE_MEDIUM:.0f}%"),
    ]:
        fig.add_hline(y=lvl, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="top right")
    fig.update_layout(
        height=420, xaxis_tickangle=-45,
        plot_bgcolor="white", margin=dict(t=55, b=90),
    )
    _ev_spike = st.plotly_chart(fig, use_container_width=True, key="anom_spike_bar", on_select="rerun")
    _sel_spike = _ev_spike.selection.points if _ev_spike and hasattr(_ev_spike, "selection") else []
    if _sel_spike:
        _pt = _sel_spike[0]
        _brand = _pt.get("x") or ""
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = chart_df[chart_df["brand"] == _brand] if _brand else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)


# ── Tab: 일관성 검사 ──────────────────────────────────────────────────────────

def _render_consistency_tab(df: pd.DataFrame) -> None:
    st.subheader("일관성 검사 — 미계량 유틸리티")
    st.caption("사용량=0으로 기록된 유틸리티 항목 수. 계량기 미설치 또는 입력 누락 가능성.")

    if "n_zero_utilities" not in df.columns:
        st.info("데이터 없음")
        return

    zero_df = (df[df["n_zero_utilities"] > 0]
               [[c for c in ["brand", "building", "floor", "n_zero_utilities",
                             "risk_level", "composite_score"] if c in df.columns]]
               .sort_values("n_zero_utilities", ascending=False))

    c1, c2 = st.columns([1, 2])
    with c1:
        total_zero = int((df["n_zero_utilities"] > 0).sum())
        st.metric("미계량 브랜드", f"{total_zero}개")
        st.metric("전 유틸리티 미계량",
                  f"{int((df['n_zero_utilities'] >= len([p for p in _UTIL_PREFIXES if f'{p}_current' in df.columns])).sum())}개")

    with c2:
        hist_df = df["n_zero_utilities"].value_counts().sort_index().reset_index()
        hist_df.columns = ["미계량 항목 수", "브랜드 수"]
        fig = px.bar(hist_df, x="미계량 항목 수", y="브랜드 수",
                     title="미계량 항목 수 분포",
                     color_discrete_sequence=["#DD8A00"])
        fig.update_layout(height=280, plot_bgcolor="white", margin=dict(t=45, b=30))
        st.plotly_chart(fig, use_container_width=True, key="anom_zero_dist_bar")

    if zero_df.empty:
        st.success("미계량 브랜드 없음 — 모든 유틸리티 정상 계량")
    else:
        st.dataframe(zero_df, hide_index=True, use_container_width=True)

    # Cross-check: water/hotwater sheet vs meter
    for sheet_col, label in [("water_sheet_m3", "💧 수도 시트 vs 검침"),
                              ("hotwater_sheet_m3", "🌡 온수 시트 vs 검침")]:
        if sheet_col not in df.columns:
            continue
        meter_col = ("water_current" if "water" in sheet_col else "hwater_current")
        if meter_col not in df.columns:
            continue
        st.caption(f"**{label}** — 시트 사용량 vs 검침 현재값 비교")
        cross = df[["brand", "building", sheet_col, meter_col]].copy()
        cross["차이"] = (cross[sheet_col] - cross[meter_col]).round(2)
        cross["불일치"] = (cross["차이"].abs() > cross[meter_col].abs() * 0.05) & cross["차이"].notna()
        mismatch = cross[cross["불일치"]].sort_values("차이", key=abs, ascending=False)
        if not mismatch.empty:
            st.warning(f"**{len(mismatch)}개** 브랜드 — 시트/검침 5% 이상 불일치")
            st.dataframe(mismatch.drop(columns=["불일치"]), hide_index=True, use_container_width=True)
        else:
            st.success("시트/검침 불일치 없음")


# ── Public render ─────────────────────────────────────────────────────────────

def render_anomaly_tab(
    cur_df: pd.DataFrame,
    file_name: str,
    file_data: bytes,
    all_sheet_keys: list[str],
    split_by_building: bool = True,
) -> None:
    """Render the 이상감지 분析 view.

    Parameters
    ----------
    cur_df           : Aggregated meter DataFrame (from render_meter_filters).
    file_name        : Excel file name (used as cache key).
    file_data        : Raw bytes of the Excel file.
    all_sheet_keys   : All sheet names found in the file.
    split_by_building: When False, do not colour bars by building.
    """
    _key = f"anomaly_loaded_{file_name}"

    if not st.session_state.get(_key):
        avail = [s for s in [BILLING_SHEET_NAME, ELECTRICITY_SHEET_NAME,
                              WATER_SHEET_NAME, HOTWATER_SHEET_NAME]
                 if any(k.strip() == s for k in all_sheet_keys)]
        src_list = ", ".join(avail) if avail else "없음"
        st.info(
            f"이상감지 분析은 **검침내역**을 기본으로, 추가 시트 ({src_list})의 비용·전기·수도 데이터를 "
            "결합하여 브랜드별 복합 이상 점수를 산출합니다."
        )
        if st.button("🔍 이상감지 분析 시작", key="btn_anomaly"):
            st.session_state[_key] = True
            st.rerun()
        return

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
            st.error(f"이상감지 분析 실패: {e}")
            return

    if anomaly_df.empty:
        st.error("이상감지 데이터를 생성할 수 없습니다.")
        return

    has_billing = BILLING_SHEET_NAME in sheets
    has_elec    = ELECTRICITY_SHEET_NAME in sheets

    # ── KPI row ───────────────────────────────────────────────────────────────
    _render_kpis(anomaly_df, has_billing, has_elec)
    st.divider()

    # ── Score breakdown legend ────────────────────────────────────────────────
    with st.expander("📖 이상 점수 계산 방법", expanded=False):
        st.markdown("""
**복합 점수** = 급등(30%) + 소비(25%) + 비용(25%) + HVAC(10%) + 일관성(10%)  — 각 구성 요소 [0, 1]

| 구성 요소 | 신호 | 시트 |
|---|---|---|
| **급등** ★ | 전월 대비 사용량 증가율 절대값 기준 — 🔴 ≥100% / 🟠 ≥50% / 🟡 ≥20% (상대 비교 없음) | 검침내역 |
| **소비** | 유틸리티별 사분면 점수 합산 정규화 (HH=4, HL=3, LH=2, Normal=1, LL=0) | 검침내역 |
| **비용** | 수도 ₩/m³, 전기 ₩/kWh, 총비용 만원/m² Z-점수의 최댓값 정규화 | 수도광열비 부과 내역 |
| **HVAC** | HVAC 강도 (kWh/m²) IQR-보정 정규화 | 전체 전기 사용내역 |
| **일관성** | 사용량=0 유틸리티 항목 수 정규화 | 검침내역 + 수도/온수 시트 |

**위험 등급**: 🔴 위험 ≥ 0.65 · 🟠 주의 ≥ 0.40 · 🟡 관찰 ≥ 0.20 · 🟢 정상 < 0.20
        """)

    # ── Composite bar + heatmap ───────────────────────────────────────────────
    _n = st.slider("표시 브랜드 수", 10, min(60, len(anomaly_df)),
                   min(10, len(anomaly_df)), key="anom_n")

    _chart_tab_bar, _chart_tab_heat = st.tabs(["📊 복합 점수 순위", "🗺️ 이상 히트맵"])
    with _chart_tab_bar:
        _render_composite_bar(anomaly_df, _n, split_by_building)
    with _chart_tab_heat:
        _render_heatmap(anomaly_df, _n)

    st.divider()

    # ── Detail tabs ───────────────────────────────────────────────────────────
    tab_spike, tab_cons, tab_cost, tab_hvac, tab_chk, tab_full = st.tabs([
        "📈 급등 감지", "🔺 소비 이상", "💰 비용 이상", "❄️ HVAC 이상", "🔍 일관성 검사", "📋 전체 결과",
    ])

    with tab_spike:
        _render_spike_tab(anomaly_df, split_by_building)

    with tab_cons:
        _render_consumption_tab(anomaly_df, split_by_building)

    with tab_cost:
        _render_cost_tab(anomaly_df, split_by_building)

    with tab_hvac:
        _render_hvac_tab(anomaly_df, split_by_building)

    with tab_chk:
        _render_consistency_tab(anomaly_df)

    with tab_full:
        st.subheader("전체 이상감지 결과")

        id_cols    = [c for c in ["brand", "building", "floor", "size_m2"] if c in anomaly_df.columns]
        score_cols = [c for c in ["composite_score", "risk_level",
                                  "spike_score", "spike_max_pct", "spike_worst_util",
                                  "consumption_score", "cost_score",
                                  "hvac_score", "consistency_score"] if c in anomaly_df.columns]
        quad_cols  = [f"{p}_quadrant" for p in _UTIL_PREFIXES if f"{p}_quadrant" in anomaly_df.columns]
        z_cols     = [c for c in ["water_unit_z", "elect_unit_z",
                                   "total_cost_per_m2_z", "hvac_intensity_z"] if c in anomaly_df.columns]
        misc_cols  = [c for c in ["n_zero_utilities"] if c in anomaly_df.columns]

        show_cols = id_cols + score_cols + quad_cols + z_cols + misc_cols
        view = add_display_index(anomaly_df[show_cols])
        st.dataframe(
            view,
            column_config={
                "composite_score":    st.column_config.ProgressColumn(
                    "복합 점수", format="%.3f", min_value=0, max_value=1),
                "spike_score":        st.column_config.ProgressColumn(
                    "급등", format="%.3f", min_value=0, max_value=1),
                "spike_max_pct":      st.column_config.NumberColumn(
                    "최대 증가율(%)", format="%.1f"),
                "spike_worst_util":   st.column_config.TextColumn("급등 항목"),
                "consumption_score":  st.column_config.ProgressColumn(
                    "소비", format="%.3f", min_value=0, max_value=1),
                "cost_score":         st.column_config.ProgressColumn(
                    "비용", format="%.3f", min_value=0, max_value=1),
                "hvac_score":         st.column_config.ProgressColumn(
                    "HVAC", format="%.3f", min_value=0, max_value=1),
                "consistency_score":  st.column_config.ProgressColumn(
                    "일관성", format="%.3f", min_value=0, max_value=1),
            },
            hide_index=True,
            use_container_width=True,
        )
        download_df_as_excel(view, filename="anomaly_analysis.xlsx", sheet_name="이상감지")

    # ── PDF download ──────────────────────────────────────────────────────────
    st.divider()
    _pdf_key = f"anomaly_pdf_{file_name}"
    _col_gen, _col_dl = st.columns([1, 2])
    with _col_gen:
        if st.button("📄 PDF 리포트 생성", key=f"gen_anomaly_pdf_{file_name}"):
            with st.spinner("PDF 생성 중…"):
                from biz_report import generate_anomaly_pdf
                st.session_state[_pdf_key] = generate_anomaly_pdf(anomaly_df)
    if _pdf_key in st.session_state:
        with _col_dl:
            st.download_button(
                "⬇️ 이상감지 리포트 다운로드",
                st.session_state[_pdf_key],
                file_name="이상감지_리포트.pdf",
                mime="application/pdf",
                key=f"dl_anomaly_pdf_{file_name}",
            )
