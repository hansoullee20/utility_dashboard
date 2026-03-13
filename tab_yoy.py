"""tab_yoy.py — Year-over-year utility comparison.

Compares the current file's usage against the same month from the previous year.
Reuses the brand-level loader from tab_mom.py for consistency.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import to_numeric_series

_UTIL_META = {
    "water":  {"label": "💧 수도",  "unit": "m³",  "curr": "water_current"},
    "hwater": {"label": "🌡 온수",  "unit": "m³",  "curr": "hwater_current"},
    "elect":  {"label": "⚡ 전기",  "unit": "kWh", "curr": "elect_current"},
    "heat":   {"label": "🔥 난방",  "unit": "m³",  "curr": "heat_current"},
}


def _fmt(v: float, unit: str) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:,.1f} {unit}"


def _load_brand_usage(fname: str, file_map: dict, sheet_map: dict) -> pd.DataFrame | None:
    """Load a file and return brand-level aggregated usage."""
    from tab_mom import _load_brand_usage_for_file
    return _load_brand_usage_for_file(fname, file_map, sheet_map)


def _merge_yoy(cur_df: pd.DataFrame, yoy_df: pd.DataFrame, present: list[str]) -> pd.DataFrame:
    """Merge current and previous-year data, compute YoY change/pct."""
    merge_keys = [k for k in ["brand", "building"] if k in cur_df.columns and k in yoy_df.columns]
    if not merge_keys:
        return pd.DataFrame()

    # Prepare current side — include floor for display
    cur_cols = merge_keys.copy()
    if "floor" in cur_df.columns:
        cur_cols.append("floor")
    for p in present:
        cc = _UTIL_META[p]["curr"]
        if cc in cur_df.columns:
            cur_cols.append(cc)
    cur = cur_df[cur_cols].copy()

    # Prepare YoY side — rename current cols to *_yoy_prev
    yoy_cols = merge_keys.copy()
    if "floor" in yoy_df.columns:
        yoy_cols.append("floor")
    yoy_renames = {"floor": "floor_prev"} if "floor" in yoy_df.columns else {}
    for p in present:
        cc = _UTIL_META[p]["curr"]
        if cc in yoy_df.columns:
            yoy_cols.append(cc)
            yoy_renames[cc] = f"{p}_yoy_prev"
    yoy = yoy_df[yoy_cols].copy().rename(columns=yoy_renames)

    merged = cur.merge(yoy, on=merge_keys, how="outer", indicator=True)

    # Compute change and pct
    for p in present:
        cc = _UTIL_META[p]["curr"]
        yp = f"{p}_yoy_prev"
        if cc in merged.columns and yp in merged.columns:
            c_val = to_numeric_series(merged[cc])
            y_val = to_numeric_series(merged[yp])
            merged[f"{p}_yoy_change"] = c_val - y_val
            merged[f"{p}_yoy_pct"] = ((c_val - y_val) / y_val.replace(0, float("nan")) * 100).round(1)

    return merged


# ── Public render ─────────────────────────────────────────────────────────────

def render_yoy_tab(
    cur_df: pd.DataFrame,
    present: list[str],
    billing_period: str | None = None,
    yoy_file: str | None = None,
    yoy_period: str | None = None,
    file_map: dict[str, bytes] | None = None,
    sheet_map: dict[str, list] | None = None,
) -> None:
    """Render the year-over-year comparison tab."""

    if not yoy_file or not file_map or not sheet_map:
        st.info("전년 동월 데이터가 없습니다. 전년도 파일을 함께 업로드하세요.")
        return

    period_str = (
        f"{yoy_period} → {billing_period}"
        if billing_period and yoy_period
        else billing_period or "이번 달"
    )
    st.subheader(f"📅 전년 동월 대비 변화  ({period_str})")

    # Load previous year data
    with st.spinner("전년 데이터 로드 중…"):
        yoy_df = _load_brand_usage(yoy_file, file_map, sheet_map)

    if yoy_df is None or yoy_df.empty:
        st.warning("전년 파일에서 검침 데이터를 로드할 수 없습니다.")
        return

    merged = _merge_yoy(cur_df, yoy_df, present)
    if merged.empty:
        st.warning("브랜드 매칭 결과가 없습니다.")
        return

    # Separate matched vs unmatched
    matched = merged[merged["_merge"] == "both"].drop(columns=["_merge"]).copy()

    # ── KPI row ───────────────────────────────────────────────────────────
    _kpi_specs = [
        (p, _UTIL_META[p]["label"], _UTIL_META[p]["unit"])
        for p in present
        if _UTIL_META[p]["curr"] in matched.columns and f"{p}_yoy_prev" in matched.columns
    ]

    if _kpi_specs:
        # 종합 KPI — total change count across all utilities
        _total_curr = sum(
            to_numeric_series(matched[_UTIL_META[p]["curr"]]).sum()
            for p, _, _ in _kpi_specs
        )
        _total_prev = sum(
            to_numeric_series(matched[f"{p}_yoy_prev"]).sum()
            for p, _, _ in _kpi_specs
        )
        _total_delta = _total_curr - _total_prev
        _total_pct = _total_delta / _total_prev * 100 if _total_prev else 0

        cols = st.columns(1 + len(_kpi_specs))
        cols[0].metric(
            "📊 종합",
            f"{len(matched)}개 브랜드",
            delta=f"{_total_pct:+.1f}%",
            delta_color="inverse",
            help="전체 유틸리티 합산 전년 대비 변화율",
        )
        for col, (p, label, unit) in zip(cols[1:], _kpi_specs):
            _curr = to_numeric_series(matched[_UTIL_META[p]["curr"]]).sum()
            _prev = to_numeric_series(matched[f"{p}_yoy_prev"]).sum()
            _delta = _curr - _prev
            _pct = _delta / _prev * 100 if _prev else 0
            col.metric(
                label,
                _fmt(_curr, unit),
                delta=f"{_delta:+,.1f} {unit}  ({_pct:+.1f}%)",
                delta_color="inverse",
            )
        st.divider()

    # ── Utility selector ──────────────────────────────────────────────────
    _util_opts = [p for p, _, _ in _kpi_specs] if _kpi_specs else present
    if not _util_opts:
        st.info("비교 가능한 유틸리티 데이터가 없습니다.")
        return

    _sel = st.selectbox(
        "유틸리티",
        _util_opts,
        format_func=lambda p: _UTIL_META[p]["label"],
        key="yoy_util_sel",
    )
    unit = _UTIL_META[_sel]["unit"]
    chg_col = f"{_sel}_yoy_change"
    pct_col = f"{_sel}_yoy_pct"
    curr_col = _UTIL_META[_sel]["curr"]
    prev_col = f"{_sel}_yoy_prev"

    if chg_col not in matched.columns:
        st.info("해당 유틸리티의 전년 데이터가 없습니다.")
        return

    plot_df = matched[
        ["brand"] +
        [c for c in ["building", "floor", curr_col, prev_col, chg_col, pct_col] if c in matched.columns]
    ].copy()
    plot_df[chg_col] = to_numeric_series(plot_df[chg_col])
    plot_df = plot_df.dropna(subset=[chg_col]).sort_values(chg_col, ascending=True).reset_index(drop=True)

    if plot_df.empty:
        st.info("변화 데이터가 없습니다.")
        return

    # ── Change bar chart ──────────────────────────────────────────────────
    _colors = plot_df[chg_col].apply(lambda v: "#C44E52" if v > 0 else "#2ca02c").tolist()
    fig = go.Figure(go.Bar(
        x=plot_df[chg_col],
        y=plot_df["brand"],
        orientation="h",
        marker_color=_colors,
        text=plot_df[chg_col].apply(lambda v: f"{v:+,.1f}"),
        textposition="outside",
        textfont=dict(size=9, color="#222222"),
        hovertemplate="<b>%{y}</b><br>전년 대비: %{x:+,.1f} " + unit + "<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#888888", line_width=1)
    fig.update_layout(
        title=f"{_UTIL_META[_sel]['label']} 전년 동월 대비 변화 ({unit})",
        height=max(430, len(plot_df) * 22 + 80),
        xaxis_title=f"변화 ({unit})",
        margin=dict(t=55, b=40, l=10, r=130),
        showlegend=False,
        yaxis=dict(tickfont=dict(size=10)),
    )
    ev = st.plotly_chart(fig, use_container_width=True, key=f"yoy_bar_{_sel}", on_select="rerun")
    pts = ev.selection.points if ev and hasattr(ev, "selection") else []
    if pts:
        _brand = pts[0].get("y", "")
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        fdf = plot_df[plot_df["brand"] == _brand]
        if not fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    st.divider()

    # ── Top/bottom tables ─────────────────────────────────────────────────
    def _fmt_table(df):
        d = df[["brand"]].copy()
        if "building" in df.columns:
            d["building"] = df["building"]
        d["전년"] = df[prev_col].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
        d["올해"] = df[curr_col].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
        d["변화"] = df[chg_col].apply(lambda v: f"{v:+,.1f}" if pd.notna(v) else "—")
        if pct_col in df.columns:
            d["변화율(%)"] = df[pct_col].apply(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
            )
        return d

    _n = min(10, len(plot_df))
    _c1, _c2 = st.columns(2)
    with _c1:
        st.markdown(f"**🔴 증가 상위 {_n}개**")
        st.dataframe(
            _fmt_table(plot_df.nlargest(_n, chg_col)).reset_index(drop=True),
            hide_index=True, use_container_width=True,
        )
    with _c2:
        st.markdown(f"**🟢 감소 상위 {_n}개**")
        st.dataframe(
            _fmt_table(plot_df.nsmallest(_n, chg_col)).reset_index(drop=True),
            hide_index=True, use_container_width=True,
        )

    # ── Full table ────────────────────────────────────────────────────────
    with st.expander("📋 전체 전년 대비 데이터", expanded=False):
        all_cols = (
            [c for c in ["brand", "building"] if c in matched.columns]
            + [c for p in present
               for c in [_UTIL_META[p]["curr"], f"{p}_yoy_prev", f"{p}_yoy_change", f"{p}_yoy_pct"]
               if c in matched.columns]
        )
        st.dataframe(matched[all_cols].reset_index(drop=True), hide_index=True, use_container_width=True)
