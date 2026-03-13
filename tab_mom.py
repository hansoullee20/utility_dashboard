"""tab_mom.py — Month-over-month utility change summary + multi-month trends.

Uses the *_current, *_previous, *_change, *_pct columns already computed
by build_from_two_files() in the 검침내역 pipeline.

When 3+ monthly files are uploaded, renders a multi-month trend line chart
(brand × period × usage) above the pairwise MoM comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import to_numeric_series

_UTIL_META = {
    "water":  {"label": "💧 수도",  "unit": "m³"},
    "hwater": {"label": "🌡 온수",  "unit": "m³"},
    "elect":  {"label": "⚡ 전기",  "unit": "kWh"},
    "heat":   {"label": "🔥 난방",  "unit": "m³"},
}


def _fmt(v: float, unit: str) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:,.1f} {unit}"


# ── Multi-month trend section ────────────────────────────────────────────────

def _load_brand_usage_for_file(
    fname: str,
    file_map: dict[str, bytes],
    sheet_map: dict[str, list],
) -> pd.DataFrame | None:
    """Load a single file and return brand-level aggregated usage (current only)."""
    from meter_view import load_meter_df
    sheet_name = next(
        (s for s in sheet_map.get(fname, []) if "검침" in s),
        None,
    )
    if sheet_name is None:
        return None
    try:
        df = load_meter_df(fname, file_map, sheet_name)
        return df
    except Exception:
        return None


def _render_trend_section(
    present: list[str],
    all_files: list[str],
    file_map: dict[str, bytes],
    file_periods: dict[str, str | None],
    sheet_map: dict[str, list],
) -> None:
    """Render multi-month trend line charts when 3+ files are available."""
    st.subheader("📈 다월 추세 분석")
    st.caption(
        f"{len(all_files)}개 월 데이터 — 브랜드별 사용량 추이를 확인하세요."
    )

    # Load each file's brand-level usage
    period_dfs: list[tuple[str, pd.DataFrame]] = []
    for fname in reversed(all_files):  # oldest first for chronological x-axis
        period = file_periods.get(fname) or fname
        df = _load_brand_usage_for_file(fname, file_map, sheet_map)
        if df is not None:
            period_dfs.append((period, df))

    if len(period_dfs) < 3:
        st.info("추세 분석을 위해 3개 이상의 월 파일이 필요합니다.")
        return

    # Build long-format trend data via melt (vectorized)
    frames: list[pd.DataFrame] = []
    for period, df in period_dfs:
        curr_cols = {f"{p}_current": p for p in present if f"{p}_current" in df.columns}
        if not curr_cols:
            continue
        sub = df[["brand", "building"] + list(curr_cols.keys())].copy()
        sub["period"] = period
        melted = sub.melt(
            id_vars=["period", "brand", "building"],
            value_vars=list(curr_cols.keys()),
            var_name="utility", value_name="usage",
        )
        melted["utility"] = melted["utility"].map(curr_cols)
        frames.append(melted.dropna(subset=["usage"]))

    if not frames:
        st.info("추세 데이터를 구성할 수 없습니다.")
        return

    trend_df = pd.concat(frames, ignore_index=True)

    # Utility selector
    _util_opts = [p for p in present if p in trend_df["utility"].unique()]
    if not _util_opts:
        return

    _sel = st.selectbox(
        "유틸리티",
        _util_opts,
        format_func=lambda p: _UTIL_META[p]["label"],
        key="trend_util_sel",
    )
    unit = _UTIL_META[_sel]["unit"]
    util_df = trend_df[trend_df["utility"] == _sel].copy()

    # Top N brands by total usage (to keep chart readable)
    brand_totals = util_df.groupby("brand")["usage"].sum().sort_values(ascending=False)
    _max_brands = len(brand_totals)
    _n_show = st.slider("표시 브랜드 수", 3, _max_brands, min(10, _max_brands), key="trend_n_brands")
    top_brands = brand_totals.head(_n_show).index.tolist()
    plot_df = util_df[util_df["brand"].isin(top_brands)].copy()

    # Pivot for line chart: period × brand
    pivot = plot_df.pivot_table(index="period", columns="brand", values="usage", aggfunc="sum")
    # Ensure chronological order (periods come from reversed all_files)
    period_order = [p for p, _ in period_dfs]
    pivot = pivot.reindex([p for p in period_order if p in pivot.index])

    # Moving average option
    _show_ma = st.checkbox("3기간 이동평균 표시", key="trend_show_ma",
                           help="3개월 이동평균선을 추가하여 추세 방향을 파악합니다.") if len(period_dfs) >= 3 else False

    # Plot
    fig = go.Figure()
    for brand in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[brand],
            mode="lines+markers",
            name=str(brand)[:20],
            hovertemplate=f"<b>{brand}</b><br>%{{x}}: %{{y:,.1f}} {unit}<extra></extra>",
        ))
        if _show_ma and len(pivot) >= 3:
            ma = pivot[brand].rolling(3, min_periods=2).mean()
            fig.add_trace(go.Scatter(
                x=pivot.index,
                y=ma,
                mode="lines",
                name=f"{str(brand)[:16]} MA3",
                line=dict(dash="dot", width=1.5),
                opacity=0.6,
                showlegend=False,
                hovertemplate=f"<b>{brand} MA3</b><br>%{{x}}: %{{y:,.1f}} {unit}<extra></extra>",
            ))
    fig.update_layout(
        title=f"{_UTIL_META[_sel]['label']} 사용량 추이 — 상위 {_n_show}개 브랜드 ({unit})",
        height=450,
        xaxis_title="기간",
        yaxis_title=f"사용량 ({unit})",
        margin=dict(t=55, b=60, l=60, r=20),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                    font=dict(size=9)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"trend_line_{_sel}")

    # Trend summary table
    with st.expander("📋 추세 데이터 테이블", expanded=False):
        display_pivot = pivot.T.copy()
        display_pivot.index.name = "brand"
        display_pivot = display_pivot.reset_index()
        st.dataframe(display_pivot, hide_index=True, use_container_width=True)

    st.divider()


def render_mom_tab(
    cur_df: pd.DataFrame,
    present: list[str],
    billing_period: str | None = None,
    prev_billing_period: str | None = None,
    prev_file: str | None = None,
    all_files: list[str] | None = None,
    file_map: dict[str, bytes] | None = None,
    file_periods: dict[str, str | None] | None = None,
    sheet_map: dict[str, list] | None = None,
) -> None:
    """Render the month-over-month change view + multi-month trends."""

    # ── Pairwise MoM comparison — "What just changed?" ─────────────────
    _has_prev = any(f"{p}_previous" in cur_df.columns for p in present)
    period_str = (
        f"{prev_billing_period} → {billing_period}"
        if billing_period and prev_billing_period
        else billing_period or "이번 달"
    )
    st.subheader(f"📈 월별 유틸리티 변화  ({period_str})")

    if not _has_prev:
        st.info("전월 데이터가 없습니다. 두 개의 연속 월 파일을 업로드하세요.")
        return

    # ── KPI row ───────────────────────────────────────────────────────────
    _kpi_specs = [
        (p, _UTIL_META[p]["label"], _UTIL_META[p]["unit"])
        for p in present
        if f"{p}_current" in cur_df.columns and f"{p}_previous" in cur_df.columns
    ]
    if not _kpi_specs:
        st.info("변화량 데이터 없음")
        return

    cols = st.columns(len(_kpi_specs))
    for col, (p, label, unit) in zip(cols, _kpi_specs):
        _curr = to_numeric_series(cur_df[f"{p}_current"]).sum()
        _prev = to_numeric_series(cur_df[f"{p}_previous"]).sum()
        _delta = _curr - _prev
        _pct   = _delta / _prev * 100 if _prev else 0
        col.metric(
            label,
            _fmt(_curr, unit),
            delta=f"{_delta:+,.1f} {unit}  ({_pct:+.1f}%)",
            delta_color="inverse",
        )

    st.divider()

    # ── Utility selector ──────────────────────────────────────────────────
    _util_opts = [p for p, _, _ in _kpi_specs]
    _sel = st.selectbox(
        "유틸리티",
        _util_opts,
        format_func=lambda p: _UTIL_META[p]["label"],
        key="mom_util_sel",
    )
    unit = _UTIL_META[_sel]["unit"]
    chg_col  = f"{_sel}_change"
    curr_col = f"{_sel}_current"
    prev_col = f"{_sel}_previous"
    pct_col  = f"{_sel}_pct"

    _plot_df = cur_df[
        ["brand", "building"] +
        [c for c in ["floor", chg_col, curr_col, prev_col, pct_col] if c in cur_df.columns]
    ].copy()
    _plot_df[chg_col]  = to_numeric_series(_plot_df[chg_col])
    _plot_df[curr_col] = to_numeric_series(_plot_df[curr_col])
    _plot_df[prev_col] = to_numeric_series(_plot_df[prev_col])
    _plot_df = _plot_df.dropna(subset=[chg_col]).sort_values(chg_col, ascending=True).reset_index(drop=True)

    # ── Top/bottom tables (with spike severity badge) ───────────────────
    def _spike_badge(pct_val):
        """Return risk badge based on MoM % change."""
        if pd.isna(pct_val) or pct_val <= 0:
            return ""
        if pct_val >= 100:
            return "🔴"
        if pct_val >= 50:
            return "🟠"
        if pct_val >= 20:
            return "🟡"
        return ""

    def _fmt_mom_table(df):
        d = df[["brand", "building"]].copy()
        d["전월"] = df[prev_col].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
        d["이번달"] = df[curr_col].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
        d["변화"] = df[chg_col].apply(lambda v: f"{v:+,.1f}" if pd.notna(v) else "—")
        if pct_col in df.columns:
            d["변화율(%)"] = df[pct_col].apply(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
            )
            d["위험"] = df[pct_col].apply(_spike_badge)
        return d

    _n = min(10, len(_plot_df))
    _c1, _c2 = st.columns(2)
    with _c1:
        st.markdown(f"**🔴 증가 상위 {_n}개**")
        st.dataframe(_fmt_mom_table(_plot_df.nlargest(_n, chg_col)).reset_index(drop=True), hide_index=True, use_container_width=True)
    with _c2:
        st.markdown(f"**🟢 감소 상위 {_n}개**")
        st.dataframe(_fmt_mom_table(_plot_df.nsmallest(_n, chg_col)).reset_index(drop=True), hide_index=True, use_container_width=True)

    st.caption("💡 위험 표시가 있는 브랜드는 🚨 이상감지 탭에서 상세 분석을 확인하세요.")

    st.divider()

    # ── Full table ────────────────────────────────────────────────────────
    with st.expander("📋 전체 변화 목록", expanded=False):
        _all_chg_cols = []
        for p in present:
            for sfx in ["_previous", "_current", "_change", "_pct"]:
                c = f"{p}{sfx}"
                if c in cur_df.columns:
                    _all_chg_cols.append(c)
        _disp = cur_df[
            ["brand", "building"] + [c for c in ["floor"] if c in cur_df.columns] + _all_chg_cols
        ].copy()
        st.dataframe(_disp.reset_index(drop=True), hide_index=True, use_container_width=True)

    # ── Multi-month trend — "Is this part of a pattern?" ─────────────────
    if (all_files and file_map and file_periods and sheet_map
            and len(all_files) >= 3):
        _render_trend_section(present, all_files, file_map, file_periods, sheet_map)

    # ── Change bar chart (full brand list) ─────────────────────────────────
    st.divider()
    _sorted = _plot_df.sort_values(chg_col, ascending=False).reset_index(drop=True)
    _colors = _sorted[chg_col].apply(lambda v: "#C44E52" if v > 0 else "#2ca02c").tolist()
    fig = go.Figure(go.Bar(
        x=[str(b)[:18] for b in _sorted["brand"]],
        y=_sorted[chg_col],
        marker_color=_colors,
        text=_sorted[chg_col].apply(lambda v: f"{v:+,.1f}"),
        textposition="outside",
        textfont=dict(size=8, color="#222222"),
        hovertemplate="<b>%{x}</b><br>변화: %{y:+,.1f} " + unit + "<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#888888", line_width=1)
    fig.update_layout(
        title=f"{_UTIL_META[_sel]['label']} 전월 대비 변화 ({unit})",
        height=430,
        yaxis_title=f"변화 ({unit})",
        xaxis_tickangle=-45,
        margin=dict(t=55, b=100, l=50, r=20),
        showlegend=False,
        xaxis=dict(tickfont=dict(size=9)),
    )
    _ev = st.plotly_chart(fig, use_container_width=True, key=f"mom_bar_{_sel}", on_select="rerun")
    _pts = _ev.selection.points if _ev and hasattr(_ev, "selection") else []
    if _pts:
        _brand_short = _pts[0].get("x", "")
        if isinstance(_brand_short, (list, tuple)):
            _brand_short = _brand_short[0]
        _fdf = _sorted[_sorted["brand"].str[:18] == str(_brand_short)[:18]]
        if not _fdf.empty:
            st.caption(f"선택됨: **{_fdf.iloc[0]['brand']}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)
