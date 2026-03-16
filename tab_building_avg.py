"""tab_building_avg.py — 건물 평균 분석 (Building Average Analysis) for 현황 분석."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data import to_numeric_series
from utils import UTIL_PREFIXES, UTIL_LABELS_UI, UTIL_UNITS


def render_building_avg_tab(cur_df: pd.DataFrame) -> None:
    """Full building-level average analysis with heatmap + per-평 charts."""
    if "building" not in cur_df.columns:
        st.info("건물 정보가 없습니다.")
        return

    avail_pfx = [p for p in UTIL_PREFIXES if f"{p}_pct" in cur_df.columns]
    if not avail_pfx:
        st.info("변화율 데이터가 없습니다.")
        return

    buildings = sorted(cur_df["building"].dropna().unique())
    if len(buildings) < 2:
        st.info("건물이 2개 이상이어야 비교가 가능합니다.")
        return

    st.subheader("🏢 건물별 평균 변화 분석")
    st.caption("건물별 유틸리티 평균 변화율과 평당 사용량을 비교합니다.")

    # ── Compute averages ──────────────────────────────────────────────────
    overall_avgs: dict[str, float] = {}
    bldg_avgs: dict[str, dict[str, float]] = {}
    for pfx in avail_pfx:
        overall_avgs[pfx] = to_numeric_series(cur_df[f"{pfx}_pct"]).mean()
    for bldg in buildings:
        bdf = cur_df[cur_df["building"] == bldg]
        bldg_avgs[bldg] = {}
        for pfx in avail_pfx:
            bldg_avgs[bldg][pfx] = to_numeric_series(bdf[f"{pfx}_pct"]).mean()

    # ── Build data matrix ─────────────────────────────────────────────────
    util_cols = [UTIL_LABELS_UI.get(pfx, pfx) for pfx in avail_pfx]
    heat_rows = []
    for bldg in buildings:
        row = {"건물": f"{bldg}동", "브랜드 수": len(cur_df[cur_df["building"] == bldg])}
        bdf = cur_df[cur_df["building"] == bldg]
        for pfx in avail_pfx:
            lbl = UTIL_LABELS_UI.get(pfx, pfx)
            avg = bldg_avgs[bldg].get(pfx)
            row[lbl] = round(avg, 1) if avg is not None and pd.notna(avg) else None
            py_col = f"{pfx}_usage_per_py"
            if py_col in bdf.columns:
                py_avg = to_numeric_series(bdf[py_col]).mean()
                row[f"{lbl} 평당"] = round(py_avg, 2) if pd.notna(py_avg) else None
        heat_rows.append(row)
    heat_df = pd.DataFrame(heat_rows)

    # Overall row
    overall_row = {"건물": "전체", "브랜드 수": len(cur_df)}
    for pfx in avail_pfx:
        lbl = UTIL_LABELS_UI.get(pfx, pfx)
        overall_row[lbl] = round(overall_avgs.get(pfx, 0), 1)
        py_col = f"{pfx}_usage_per_py"
        if py_col in cur_df.columns:
            overall_row[f"{lbl} 평당"] = round(to_numeric_series(cur_df[py_col]).mean(), 2)
    heat_df = pd.concat([heat_df, pd.DataFrame([overall_row])], ignore_index=True)

    py_label_cols = [f"{lbl} 평당" for lbl in util_cols if f"{lbl} 평당" in heat_df.columns]

    # ── Heatmap: 평균 변화율 ──────────────────────────────────────────────
    _raw_z = heat_df[util_cols].values
    _flat = _raw_z[~pd.isna(_raw_z)].flatten()
    _zmax = max(abs(float(np.percentile(_flat, 5))),
                abs(float(np.percentile(_flat, 95))), 10) if len(_flat) > 0 else 50
    z_clipped = [[max(-_zmax, min(_zmax, v)) if v is not None and not pd.isna(v) else None
                  for v in row] for row in _raw_z.tolist()]
    text_vals = [[f"{v:+.1f}%" if v is not None and not pd.isna(v) else "—"
                  for v in row] for row in _raw_z.tolist()]

    fig_heat = go.Figure(go.Heatmap(
        z=z_clipped, x=util_cols, y=heat_df["건물"].tolist(),
        text=text_vals, texttemplate="%{text}",
        textfont=dict(size=13, color="white"),
        colorscale=[[0, "#2ca02c"], [0.35, "#F0C040"], [0.5, "#EEEEEE"],
                    [0.65, "#DD8A00"], [1, "#C44E52"]],
        zmid=0, zmin=-_zmax, zmax=_zmax,
        colorbar=dict(title="%", thickness=12, len=0.8),
        hovertemplate="<b>%{y}</b> %{x}: %{text}<extra></extra>",
    ))
    fig_heat.update_layout(
        title="건물별 평균 변화율 (%)",
        height=max(250, len(heat_df) * 55 + 80),
        margin=dict(t=40, b=20, l=80, r=20),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True, key="ba_heat")

    # ── Per-평 subplots ───────────────────────────────────────────────────
    if py_label_cols:
        _n_utils = len(py_label_cols)
        fig_bar = make_subplots(
            rows=1, cols=_n_utils, shared_yaxes=False,
            horizontal_spacing=0.08,
            subplot_titles=[c.replace(" 평당", "") for c in py_label_cols],
        )
        _bar_colors = ["#4C72B0", "#C44E52", "#DD8A00", "#55A868"]
        for i, col in enumerate(py_label_cols):
            fig_bar.add_trace(go.Bar(
                x=heat_df["건물"], y=heat_df[col],
                marker_color=_bar_colors[i % len(_bar_colors)],
                text=heat_df[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else ""),
                textposition="outside", textfont=dict(size=9), showlegend=False,
            ), row=1, col=i + 1)
            _pfx = avail_pfx[i] if i < len(avail_pfx) else ""
            fig_bar.update_yaxes(title_text=f"{UTIL_UNITS.get(_pfx, '')}/평",
                                 title_font_size=9, row=1, col=i + 1)
        fig_bar.update_layout(
            title="건물별 평당 평균 사용량",
            height=max(280, len(heat_df) * 50 + 80),
            margin=dict(t=40, b=40, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="ba_py_bar")

    # ── Notable deviations ────────────────────────────────────────────────
    _badges: list[str] = []
    for bldg in buildings:
        for pfx in avail_pfx:
            lbl = UTIL_LABELS_UI.get(pfx, pfx)
            emoji = lbl.split(" ")[0]
            avg = bldg_avgs[bldg].get(pfx)
            if avg is None or pd.isna(avg):
                continue
            diff = avg - overall_avgs.get(pfx, 0)
            if abs(diff) < 15:
                continue
            if diff > 0:
                bg, clr, arrow = "#C44E5218", "#C44E52", "▲"
            else:
                bg, clr, arrow = "#2ca02c18", "#2ca02c", "▼"
            _badges.append(
                f'<span style="background:{bg};color:{clr};border:1px solid {clr}30;'
                f'border-radius:6px;padding:3px 8px;margin:2px;display:inline-block;'
                f'font-size:0.82rem;font-weight:600">'
                f'{emoji} {bldg}동 {arrow}{abs(avg):.0f}% '
                f'<span style="font-weight:400;font-size:0.72rem;opacity:0.8">'
                f'(전체 대비 {diff:+.0f}%p)</span></span>'
            )
    if _badges:
        st.markdown(
            f'<div style="margin-top:8px"><b>전체 대비 주요 편차:</b><br>'
            f'{"".join(_badges)}</div>',
            unsafe_allow_html=True,
        )

    # ── Data table ────────────────────────────────────────────────────────
    with st.expander("📋 상세 데이터", expanded=False):
        st.dataframe(heat_df, hide_index=True, use_container_width=True)
