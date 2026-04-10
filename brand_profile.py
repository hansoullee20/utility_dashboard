"""brand_profile.py — Comprehensive single-brand profile across all data sources."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data import to_numeric_series, st_safe
from utils_plot import handle_chart_click

_C_PREV  = "#A8C4E0"   # light blue  — previous month
_C_CURR  = "#4C72B0"   # blue        — this month
_C_AVG   = "#DD8A00"   # amber       — fleet average
_C_POS   = "#E07070"   # red-ish     — positive change (increase)
_C_NEG   = "#6AAB6A"   # green       — negative change (decrease)

from utils import UTIL_LABELS as _LABEL, UTIL_UNITS as _UNIT

_QUAD_LABEL = {
    "HH": "🔴 HH — 변화·비율 모두 상위",
    "HL": "🟠 HL — 변화 상위, 비율 하위",
    "LH": "🟡 LH — 변화 하위, 비율 상위",
    "LL": "🟢 LL — 변화·비율 모두 하위",
    "—":  "— 정상",
}


def _quad(cv, pv, lo_c, hi_c, lo_p, hi_p) -> str:
    if pd.isna(cv) or pd.isna(pv):
        return "—"
    if cv >= hi_c and pv >= hi_p: return "HH"
    if cv >= hi_c and pv <= lo_p: return "HL"
    if cv <= lo_c and pv >= hi_p: return "LH"
    if cv <= lo_c and pv <= lo_p: return "LL"
    return "—"


def _v(row, col):
    val = row.get(col)
    try:
        return float(pd.to_numeric(val, errors="coerce"))
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v, decimals=2, sign=False):
    if pd.isna(v): return "—"
    fmt = f"{{:+,.{decimals}f}}" if sign else f"{{:,.{decimals}f}}"
    return fmt.format(v)


def _filter_brand(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Filter df to rows matching brand, stripping whitespace on both sides."""
    if "brand" not in df.columns:
        return pd.DataFrame()
    return df[df["brand"].astype(str).str.strip() == brand.strip()].copy()


def _peer_chart(
    df: pd.DataFrame,
    selected: str,
    cols: list[tuple[str, str]],
    title: str,
    unit: str = "",
) -> None:
    """One subplot per metric, each with its own y-axis, brand vs peer average.

    cols: list of (col_name, display_label).
    """
    num_cols = [c for c, _ in cols if c in df.columns]
    if not num_cols or "brand" not in df.columns:
        return

    agg = df.copy()
    for c in num_cols:
        agg[c] = to_numeric_series(agg[c])
    brand_agg = agg.groupby(agg["brand"].astype(str).str.strip())[num_cols].sum(min_count=1)

    sel = selected.strip()
    if sel not in brand_agg.index:
        return
    peers = brand_agg.drop(index=sel, errors="ignore")

    # Build per-metric data
    metrics = []
    for col, lbl in cols:
        if col not in brand_agg.columns:
            continue
        bv = brand_agg.loc[sel, col]
        pv = peers[col].mean() if not peers.empty else np.nan
        metrics.append((lbl, float(bv) if not pd.isna(bv) else 0.0,
                              float(pv) if not pd.isna(pv) else 0.0))
    if not metrics:
        return

    n = len(metrics)
    fig = make_subplots(rows=1, cols=n, subplot_titles=[m[0] for m in metrics])

    for i, (lbl, bv, pv) in enumerate(metrics, start=1):
        show_legend = i == 1
        fig.add_trace(go.Bar(
            name=sel[:20], x=[sel[:20]], y=[bv],
            marker_color=_C_CURR,
            text=[f"{bv:,.2f}"], textposition="outside", cliponaxis=False,
            legendgroup="brand", showlegend=show_legend,
        ), row=1, col=i)
        fig.add_trace(go.Bar(
            name="동종 평균", x=["동종 평균"], y=[pv],
            marker_color=_C_AVG, opacity=0.75,
            text=[f"{pv:,.2f}"], textposition="outside", cliponaxis=False,
            legendgroup="peer", showlegend=show_legend,
        ), row=1, col=i)

    fig.update_layout(
        title=dict(text=title, font_size=13),
        barmode="group",
        legend=dict(orientation="h", y=1.15, x=0),
        margin=dict(t=70, b=20, l=30, r=10),
        height=300,
    )
    fig.update_yaxes(zeroline=True, rangemode="tozero")
    _ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                          key=f"peer_{title[:8]}")
    handle_chart_click(_ev, df, brand_col="brand", field="x")


def _bar(labels, values, colors, title, unit):
    """Single grouped bar chart."""
    text = [f"{v:,.2f}" if not pd.isna(v) else "" for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=text, textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title=dict(text=title, font_size=13),
        yaxis=dict(title=unit, zeroline=True),
        margin=dict(t=40, b=20, l=40, r=10),
        height=300,
        showlegend=False,
    )
    return fig


def _render_comparison_charts(row, cur_df, present, selected_brand):
    """Two charts per category:
    1. This month: selected brand vs peer average.
    2. MoM change: selected brand's delta vs peer average delta.
    """
    if not present:
        return

    peers = cur_df[cur_df["brand"].astype(str) != selected_brand]

    # ── Chart 1: this month vs peer average ──────────────────────────────────
    _short = selected_brand[:20]
    st.subheader(f"📊 {_short} 사용량 — 동종 업체 평균 대비")
    st.caption("해당 브랜드의 이번달 사용량을 건물 내 동종 업체 평균과 비교합니다. 평균보다 높은 항목은 과소비 또는 설비 이상 가능성이 있습니다.")
    cols = st.columns(min(len(present), 2))
    for i, p in enumerate(present):
        lbl  = _LABEL.get(p, p)
        unit = _UNIT.get(p, "")
        curr     = _v(row, f"{p}_current")
        avg_curr = to_numeric_series(peers[f"{p}_current"]).mean() if f"{p}_current" in peers.columns else np.nan

        with cols[i % 2]:
            _ev = st.plotly_chart(
                _bar(
                    [_short, "동종 평균"],
                    [curr, avg_curr],
                    [_C_CURR, _C_AVG],
                    lbl, unit,
                ),
                use_container_width=True,
                on_select="rerun",
                key=f"comp_curr_{p}",
            )
            handle_chart_click(_ev, cur_df, brand_col="brand", field="x")

    # ── Chart 2: MoM change — brand vs peer average ───────────────────────────
    st.subheader("📉 전월 대비 변화량 — 동종 업체 평균 대비")
    st.caption("변화량이 동종 평균보다 크게 높으면 해당 브랜드에만 발생한 이슈(누수, 장비 고장 등)일 가능성이 높습니다.")
    chg_cats, brand_chgs, avg_chgs = [], [], []
    for p in present:
        cv  = _v(row, f"{p}_change")
        avg = to_numeric_series(peers[f"{p}_change"]).mean() if f"{p}_change" in peers.columns else np.nan
        if pd.isna(cv) and pd.isna(avg):
            continue
        chg_cats.append(_LABEL.get(p, p))
        brand_chgs.append(cv   if not pd.isna(cv)  else 0)
        avg_chgs.append(avg    if not pd.isna(avg)  else 0)

    if chg_cats:
        brand_colors = [_C_POS if v > 0 else _C_NEG for v in brand_chgs]
        avg_colors   = [_C_POS if v > 0 else _C_NEG for v in avg_chgs]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=_short,
            x=chg_cats, y=brand_chgs,
            marker_color=brand_colors,
            text=[f"{v:+,.2f}" for v in brand_chgs],
            textposition="outside", cliponaxis=False,
        ))
        fig.add_trace(go.Bar(
            name="동종 평균",
            x=chg_cats, y=avg_chgs,
            marker_color=avg_colors,
            opacity=0.55,
            text=[f"{v:+,.2f}" for v in avg_chgs],
            textposition="outside", cliponaxis=False,
        ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(zeroline=True, zerolinewidth=1.5),
            legend=dict(orientation="h", y=1.1, x=0),
            margin=dict(t=40, b=20, l=40, r=10),
            height=320,
        )
        _ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                              key="comp_mom_chg")
        handle_chart_click(_ev, cur_df, brand_col="brand", field="x")


def render_brand_profile_tab(
    cur_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    present: list[str],
    tail: int,
    billing_period: str | None = None,
    prev_billing_period: str | None = None,
    billing_df: pd.DataFrame | None = None,
    water_df: pd.DataFrame | None = None,
    hotwater_df: pd.DataFrame | None = None,
    electricity_df: pd.DataFrame | None = None,
    anomaly_df: pd.DataFrame | None = None,
) -> None:
    """Render brand profile with sub-tabs: single profile + comparison."""
    from utils import display_brand as _display_brand
    cur_df = _display_brand(cur_df)
    if anomaly_df is not None:
        anomaly_df = _display_brand(anomaly_df)
    brands = sorted(cur_df["brand"].dropna().astype(str).unique().tolist())
    if not brands:
        st.info("표시할 브랜드가 없습니다.")
        return

    tab_single, tab_compare = st.tabs(["🔍 브랜드 프로필", "⚖️ 브랜드 비교"])

    with tab_compare:
        _render_brand_comparison(
            cur_df, present, billing_period, prev_billing_period,
            billing_df, water_df, hotwater_df, electricity_df, brands,
        )

    with tab_single:
        _render_single_brand_profile(
            cur_df, ref_df, present, tail, billing_period,
            prev_billing_period, billing_df, water_df, hotwater_df,
            electricity_df, brands, anomaly_df=anomaly_df,
        )


def _render_single_brand_profile(
    cur_df, ref_df, present, tail, billing_period, prev_billing_period,
    billing_df, water_df, hotwater_df, electricity_df, brands,
    anomaly_df=None,
):
    """Single-brand deep-dive (original profile view)."""
    _search = st.session_state.get("profile_sel_search", "").strip().lower()
    brands_shown = [b for b in brands if _search in b.lower()] if _search else brands
    if not brands_shown:
        brands_shown = brands

    st.text_input("🔍 브랜드 검색", placeholder="브랜드명 입력...",
                  key="profile_sel_search")
    selected = st.selectbox("브랜드 선택", brands_shown, index=0,
                            key="profile_brand_select")

    row = cur_df[cur_df["brand"].astype(str) == selected]
    if row.empty:
        st.warning("해당 브랜드 데이터를 찾을 수 없습니다.")
        return
    row = row.iloc[0]

    # ── Header ────────────────────────────────────────────────────────────────
    bldg  = str(row.get("building", "—"))
    floor = str(row.get("floor",    "—"))
    m2    = _v(row, "size_m2")
    py    = _v(row, "size_py")
    size_str = (
        f"{m2:,.1f} m²  /  {py:,.1f} 평" if not pd.isna(m2) and not pd.isna(py)
        else f"{m2:,.1f} m²" if not pd.isna(m2) else "—"
    )
    period_str = (
        f"{prev_billing_period} → {billing_period}"
        if prev_billing_period and billing_period else billing_period or "—"
    )

    # Brand header card
    from utils import esc as _esc
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#4C72B010,#4C72B005);'
        f'border:1px solid #4C72B025;border-radius:12px;padding:16px 20px;margin-bottom:16px">'
        f'<div style="font-size:1.5rem;font-weight:800;color:inherit;margin-bottom:8px">'
        f'{_esc(selected)}</div>'
        f'<div style="display:flex;gap:24px;font-size:0.88rem;color:inherit;opacity:0.7">'
        f'<span>🏢 <b>{_esc(bldg)}</b></span>'
        f'<span>📍 <b>{_esc(floor)}</b></span>'
        f'<span>📐 <b>{_esc(size_str)}</b></span>'
        f'<span>📅 <b>{_esc(period_str)}</b></span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Anomaly status badge ──────────────────────────────────────────────────
    if anomaly_df is not None and not anomaly_df.empty:
        _anom_row = anomaly_df[anomaly_df["brand"].astype(str).str.strip() == selected.strip()]
        if not _anom_row.empty:
            _ar = _anom_row.iloc[0]
            _risk = str(_ar.get("risk_level", "—"))
            _score = float(_ar.get("composite_score", 0))
            _reason = str(_ar.get("reason", ""))
            _spike_pct = _ar.get("spike_max_pct", np.nan)
            _spike_util = str(_ar.get("spike_worst_util", ""))

            _risk_colors = {"🔴 위험": "#C44E52", "🟠 주의": "#DD8A00", "🟢 정상": "#2ca02c"}
            _risk_clr = _risk_colors.get(_risk, "#888")

            _anom_cols = st.columns([1, 1, 1, 2])
            _anom_cols[0].metric("위험도", _risk)
            _anom_cols[1].metric("복합 점수", f"{_score:.3f}")
            if not pd.isna(_spike_pct):
                _anom_cols[2].metric("최대 급등", f"{_spike_pct:.1f}%",
                                     help=f"급등 항목: {_spike_util}")
            if _reason and _reason != "nan":
                with _anom_cols[3]:
                    st.markdown(
                        f'<div style="background:{_risk_clr}10;border-left:3px solid {_risk_clr};'
                        f'padding:8px 12px;border-radius:4px;font-size:0.82rem">'
                        f'<b>이유</b>: {_esc(_reason)}</div>',
                        unsafe_allow_html=True,
                    )

    # Debug: brand name matching across sheets
    _sheet_dfs = {
        "수도광열비": billing_df,
        "수도": water_df,
        "온수": hotwater_df,
        "전기": electricity_df,
    }
    _missing = [
        name for name, df in _sheet_dfs.items()
        if df is not None and not df.empty and _filter_brand(df, selected).empty
    ]
    if _missing:
        with st.expander("⚠️ 일부 시트에서 브랜드를 찾을 수 없음"):
            st.caption(f"선택된 브랜드: **{selected}**")
            for name, df in _sheet_dfs.items():
                if df is not None and not df.empty and name in _missing:
                    similar = [b for b in df["brand"].astype(str).str.strip().unique()
                               if selected.strip().lower()[:3] in b.lower()]
                    st.caption(f"**{name}** 시트 유사 브랜드: {similar[:10] or '없음'}")

    # ── Compute quadrant thresholds ───────────────────────────────────────────
    thresholds: dict[str, tuple] = {}
    for p in present:
        cc, pc = f"{p}_change", f"{p}_pct"
        ca = to_numeric_series(cur_df[cc]).dropna()
        pa = to_numeric_series(cur_df[pc]).dropna()
        if not ca.empty and not pa.empty:
            thresholds[p] = (
                ca.quantile(tail / 100), ca.quantile(1 - tail / 100),
                pa.quantile(tail / 100), pa.quantile(1 - tail / 100),
            )

    # ── Business Insight Summary ─────────────────────────────────────────────
    _bp_insights: list[str] = []
    peers = cur_df[cur_df["brand"].astype(str) != selected]
    for p in present:
        curr_col = f"{p}_current"
        if curr_col not in cur_df.columns:
            continue
        bv = _v(row, curr_col)
        if pd.isna(bv):
            continue
        s = to_numeric_series(cur_df[curr_col]).dropna()
        if s.empty:
            continue
        pctile = float((s < bv).sum()) / len(s) * 100
        peer_avg = to_numeric_series(peers[curr_col]).mean() if curr_col in peers.columns else np.nan
        lbl = _LABEL.get(p, p)
        u = _UNIT.get(p, "")
        if not pd.isna(peer_avg) and peer_avg > 0:
            ratio = bv / peer_avg
            if ratio >= 1.5:
                _bp_insights.append(
                    f"**{lbl}** 동종 평균 대비 **{ratio:.1f}배** (상위 {100-pctile:.0f}%) → "
                    f"또래 대비 사용량이 높아 비용 절감 여지가 큽니다"
                )
            elif ratio <= 0.5:
                _bp_insights.append(
                    f"**{lbl}** 동종 평균의 **{ratio:.0%}** 수준 → "
                    f"효율적으로 운영하고 있습니다"
                )
        chg = _v(row, f"{p}_change")
        pct = _v(row, f"{p}_pct")
        if not pd.isna(pct) and abs(pct) >= 20:
            _dir = "증가" if pct > 0 else "감소"
            if pct > 0:
                _bp_insights.append(
                    f"**{lbl}** 전월 대비 **{abs(pct):.0f}% {_dir}** ({chg:+,.1f} {u}) → "
                    f"급격한 증가로 누수·장비 이상 여부를 확인하세요"
                )
            else:
                _bp_insights.append(
                    f"**{lbl}** 전월 대비 **{abs(pct):.0f}% {_dir}** ({chg:+,.1f} {u}) → "
                    f"사용량이 크게 줄어 공실·영업 축소 여부를 확인하세요"
                )
    if _bp_insights:
        with st.container(border=True):
            st.markdown(
                '<p style="margin:0 0 6px;font-size:0.9rem;font-weight:700;color:#4C72B0">'
                '비즈니스 인사이트</p>',
                unsafe_allow_html=True,
            )
            st.markdown("  \n".join(f"- {i}" for i in _bp_insights))

    # ── 비용 효율 분석 ─────────────────────────────────────────────────────────
    _has_billing = billing_df is not None and not billing_df.empty
    _bdf = _filter_brand(billing_df, selected) if _has_billing else pd.DataFrame()
    if not _bdf.empty and "total" in _bdf.columns:
        st.subheader("💵 비용 효율 분석")
        st.caption("면적 대비 비용 수준을 동종 업체와 비교하여 과납·저납 여부를 진단합니다.")
        _b_total = to_numeric_series(_bdf["total"]).sum()  # 만원
        _b_m2 = float(m2) if not pd.isna(m2) and m2 > 0 else 0
        _b_py = float(py) if not pd.isna(py) and py > 0 else 0
        _b_per_m2 = _b_total / _b_m2 if _b_m2 > 0 else np.nan
        _b_per_py = _b_total / _b_py if _b_py > 0 else np.nan

        # Peer stats
        _peer_bill = billing_df[billing_df["brand"].astype(str).str.strip() != selected.strip()].copy()
        _peer_agg = _peer_bill.groupby("brand").agg(
            total=("total", "sum"), area=("size_m2", "sum")
        )
        _peer_agg = _peer_agg[_peer_agg["area"] > 0]
        _peer_agg["per_m2"] = _peer_agg["total"] / _peer_agg["area"]
        _peer_agg["per_py"] = _peer_agg["total"] / (_peer_agg["area"] / 3.3058)
        _peer_med_m2 = _peer_agg["per_m2"].median() if not _peer_agg.empty else np.nan
        _peer_med_py = _peer_agg["per_py"].median() if not _peer_agg.empty else np.nan

        _ec1, _ec2, _ec3, _ec4 = st.columns(4)
        _ec1.metric("총 비용", f"{_b_total:,.1f} 만원")
        if not pd.isna(_b_per_m2):
            _delta_m2 = _b_per_m2 - _peer_med_m2 if not pd.isna(_peer_med_m2) else None
            _ec2.metric("만원/m²", f"{_b_per_m2:,.2f}",
                        delta=f"{_delta_m2:+,.2f} vs 중앙값" if _delta_m2 is not None else None,
                        delta_color="inverse")
        if not pd.isna(_b_per_py):
            _delta_py = _b_per_py - _peer_med_py if not pd.isna(_peer_med_py) else None
            _ec3.metric("만원/평", f"{_b_per_py:,.2f}",
                        delta=f"{_delta_py:+,.2f} vs 중앙값" if _delta_py is not None else None,
                        delta_color="inverse")
        # Percentile rank
        if not _peer_agg.empty and not pd.isna(_b_per_m2):
            _rank = int((_peer_agg["per_m2"] < _b_per_m2).sum()) + 1
            _total_n = len(_peer_agg) + 1
            _pctile = _rank / _total_n * 100
            _ec4.metric("비용 순위", f"{_rank}위 / {_total_n}개",
                        help=f"면적당 비용 기준 상위 {_pctile:.0f}%")

        # Utility cost breakdown (pie + bar)
        _cost_items = []
        for _col, _lbl in [("water_total", "상하수도"), ("elect_total", "전기"),
                            ("heat_total", "난방"), ("hotwater_excl", "온수"),
                            ("hvac_excl", "HVAC")]:
            if _col in _bdf.columns:
                _val = to_numeric_series(_bdf[_col]).sum()
                if _val > 0:
                    _cost_items.append((_lbl, float(_val)))

        if _cost_items:
            st.markdown("**유틸리티 비용 구성**")
            _ci_labels = [x[0] for x in _cost_items]
            _ci_values = [x[1] for x in _cost_items]
            _ci_colors = ["#4C72B0", "#DD8A00", "#E377C2", "#C44E52", "#2ca02c"]

            _pie_c, _bar_c = st.columns(2)
            with _pie_c:
                fig_pie = go.Figure(go.Pie(
                    labels=_ci_labels, values=_ci_values,
                    marker_colors=_ci_colors[:len(_ci_labels)],
                    textinfo="label+percent",
                    textfont_size=11,
                    hole=0.35,
                ))
                fig_pie.update_layout(
                    height=280, margin=dict(t=30, b=10, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="profile_cost_pie")

            with _bar_c:
                # Compare brand vs peer average per category
                _peer_avgs = []
                for _col, _lbl in [("water_total", "상하수도"), ("elect_total", "전기"),
                                    ("heat_total", "난방"), ("hotwater_excl", "온수"),
                                    ("hvac_excl", "HVAC")]:
                    if _col in billing_df.columns:
                        _pv = to_numeric_series(_peer_bill[_col]).mean() if not _peer_bill.empty else 0
                        _peer_avgs.append((_lbl, float(_pv)))
                if _peer_avgs:
                    _pa_dict = dict(_peer_avgs)
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Bar(
                        name=selected[:15], x=_ci_labels, y=_ci_values,
                        marker_color=_C_CURR,
                        text=[f"{v:,.1f}" for v in _ci_values],
                        textposition="outside", textfont_size=9,
                    ))
                    fig_comp.add_trace(go.Bar(
                        name="동종 평균", x=_ci_labels,
                        y=[_pa_dict.get(l, 0) for l in _ci_labels],
                        marker_color=_C_AVG, opacity=0.7,
                        text=[f"{_pa_dict.get(l,0):,.1f}" for l in _ci_labels],
                        textposition="outside", textfont_size=9,
                    ))
                    fig_comp.update_layout(
                        barmode="group", height=280,
                        yaxis_title="만원",
                        margin=dict(t=30, b=20, l=40, r=10),
                        legend=dict(orientation="h", y=1.12, x=0),
                    )
                    st.plotly_chart(fig_comp, use_container_width=True, key="profile_cost_comp")

        # Cost efficiency narrative
        if not pd.isna(_b_per_m2) and not pd.isna(_peer_med_m2) and _peer_med_m2 > 0:
            _cost_ratio = _b_per_m2 / _peer_med_m2
            if _cost_ratio >= 1.5:
                st.warning(
                    f"**{selected[:15]}**의 면적당 비용({_b_per_m2:,.2f} 만원/m²)이 "
                    f"동종 중앙값({_peer_med_m2:,.2f})의 **{_cost_ratio:.1f}배**입니다. "
                    f"특정 유틸리티 과소비 또는 요금 체계 문제가 없는지 확인이 필요합니다."
                )
            elif _cost_ratio <= 0.5:
                st.info(
                    f"**{selected[:15]}**의 면적당 비용이 동종 중앙값의 **{_cost_ratio:.0%}** 수준으로 "
                    f"매우 낮습니다. 미계량 항목이 없는지 확인하세요."
                )
            else:
                st.caption(
                    f"**{selected[:15]}**의 면적당 비용은 동종 중앙값 대비 "
                    f"{_cost_ratio:.0%} 수준으로 적정 범위입니다."
                )
        if _cost_items:
            _dominant = max(_cost_items, key=lambda x: x[1])
            _dom_pct = _dominant[1] / sum(v for _, v in _cost_items) * 100
            if _dom_pct >= 50:
                st.caption(
                    f"비용 구성에서 **{_dominant[0]}**이(가) {_dom_pct:.0f}%로 가장 큰 비중을 차지합니다. "
                    f"해당 항목의 절감이 전체 비용 관리에 가장 효과적입니다."
                )

        st.divider()

    # ── 단가 분석 (Unit Cost) ─────────────────────────────────────────────────
    _unit_cost_items = []
    # Water unit cost: ₩/m³
    _wdf_brand = _filter_brand(water_df, selected) if water_df is not None and not water_df.empty else pd.DataFrame()
    if not _wdf_brand.empty and "usage_m3" in _wdf_brand.columns and "total" in _wdf_brand.columns:
        _w_usage = to_numeric_series(_wdf_brand["usage_m3"]).sum()
        _w_cost = to_numeric_series(_wdf_brand["total"]).sum()
        if _w_usage > 0:
            _w_unit = _w_cost / _w_usage
            # Peer
            _w_peers = water_df[water_df["brand"].astype(str).str.strip() != selected.strip()].copy()
            _wp_agg = _w_peers.groupby("brand").agg(u=("usage_m3", "sum"), c=("total", "sum"))
            _wp_agg = _wp_agg[_wp_agg["u"] > 0]
            _wp_agg["unit"] = _wp_agg["c"] / _wp_agg["u"]
            _wp_med = _wp_agg["unit"].median() if not _wp_agg.empty else np.nan
            _wp_std = _wp_agg["unit"].std() if not _wp_agg.empty else np.nan
            _w_z = (_w_unit - _wp_med) / _wp_std if not pd.isna(_wp_std) and _wp_std > 0 else np.nan
            _unit_cost_items.append(("💧 수도", "₩/m³", _w_unit, _wp_med, _w_z))

    # Electricity unit cost: ₩/kWh
    _edf_brand = _filter_brand(electricity_df, selected) if electricity_df is not None and not electricity_df.empty else pd.DataFrame()
    if not _edf_brand.empty and "kwh_total" in _edf_brand.columns and "grand_total" in _edf_brand.columns:
        _e_usage = to_numeric_series(_edf_brand["kwh_total"]).sum()
        _e_cost = to_numeric_series(_edf_brand["grand_total"]).sum()
        if _e_usage > 0:
            _e_unit = _e_cost / _e_usage
            _e_peers = electricity_df[electricity_df["brand"].astype(str).str.strip() != selected.strip()].copy()
            _ep_agg = _e_peers.groupby("brand").agg(u=("kwh_total", "sum"), c=("grand_total", "sum"))
            _ep_agg = _ep_agg[_ep_agg["u"] > 0]
            _ep_agg["unit"] = _ep_agg["c"] / _ep_agg["u"]
            _ep_med = _ep_agg["unit"].median() if not _ep_agg.empty else np.nan
            _ep_std = _ep_agg["unit"].std() if not _ep_agg.empty else np.nan
            _e_z = (_e_unit - _ep_med) / _ep_std if not pd.isna(_ep_std) and _ep_std > 0 else np.nan
            _unit_cost_items.append(("⚡ 전기", "₩/kWh", _e_unit, _ep_med, _e_z))

    if _unit_cost_items:
        st.subheader("📏 단가 분석")
        st.caption("동일 사용량이라도 단가가 높으면 비용 부담이 커집니다. 동종 업체 중앙값 대비 Z-score로 이상 여부를 판단합니다.")
        _uc_cols = st.columns(len(_unit_cost_items))
        for _ui, (_u_label, _u_unit, _u_val, _u_peer, _u_z) in enumerate(_unit_cost_items):
            with _uc_cols[_ui]:
                _z_str = f"Z={_u_z:+.1f}" if not pd.isna(_u_z) else ""
                _delta_str = None
                if not pd.isna(_u_peer):
                    _diff = _u_val - _u_peer
                    _delta_str = f"{_diff:+,.0f} vs 중앙값 ({_z_str})"
                st.metric(f"{_u_label} 단가 ({_u_unit})", f"{_u_val:,.0f}",
                          delta=_delta_str, delta_color="inverse")
                if not pd.isna(_u_z):
                    _z_clr = "#C44E52" if abs(_u_z) >= 2 else "#DD8A00" if abs(_u_z) >= 1.5 else "#2ca02c"
                    _z_msg = "⚠️ 과금 이상 의심" if abs(_u_z) >= 2 else "📌 주의 필요" if abs(_u_z) >= 1.5 else "✅ 정상 범위"
                    st.markdown(
                        f'<span style="color:{_z_clr};font-size:0.82rem;font-weight:600">{_z_msg}</span>',
                        unsafe_allow_html=True,
                    )
        # Unit cost narrative
        _uc_alerts = [f"{l} Z={z:+.1f}" for l, _, _, _, z in _unit_cost_items if not pd.isna(z) and abs(z) >= 1.5]
        if _uc_alerts:
            st.info(
                f"💡 **{selected[:15]}**의 단가가 동종 업체 대비 이상 범위에 있는 항목: "
                f"{', '.join(_uc_alerts)}. 계량기 오류나 요금 체계 변경 여부를 확인하세요."
            )
        else:
            st.caption(f"**{selected[:15]}**의 단가는 동종 업체 중앙값과 유사한 정상 범위입니다.")
        st.divider()

    # ── 전기 사용 구성 (Electricity Breakdown) ─────────────────────────────────
    if not _edf_brand.empty:
        _has_ehp = "kwh_ehp" in _edf_brand.columns
        _has_fcu = "kwh_fcu" in _edf_brand.columns
        _has_elec01 = "kwh_elec01" in _edf_brand.columns
        if _has_ehp or _has_fcu:
            st.subheader("⚡ 전기 사용 구성")
            st.caption("EHP/FCU 등 냉난방(HVAC)이 전기 비용의 핵심 동인입니다. HVAC 비율이 높을수록 설비 효율이 비용에 미치는 영향이 큽니다.")
            _ekwh_total = to_numeric_series(_edf_brand["kwh_total"]).sum()
            _ekwh_ehp = to_numeric_series(_edf_brand["kwh_ehp"]).sum() if _has_ehp else 0
            _ekwh_fcu = to_numeric_series(_edf_brand.get("kwh_fcu", pd.Series(dtype=float))).sum()
            _ekwh_pump = to_numeric_series(_edf_brand.get("kwh_pump", pd.Series(dtype=float))).sum()
            _ekwh_base = _ekwh_total - _ekwh_ehp - _ekwh_fcu - _ekwh_pump
            _ekwh_hvac = _ekwh_ehp + _ekwh_fcu

            _eb_items = [("EHP", _ekwh_ehp), ("FCU", _ekwh_fcu),
                         ("펌프", _ekwh_pump), ("일반전기", max(0, _ekwh_base))]
            _eb_items = [(l, v) for l, v in _eb_items if v > 0]

            _eb1, _eb2 = st.columns(2)
            with _eb1:
                if _eb_items:
                    fig_eb = go.Figure(go.Pie(
                        labels=[x[0] for x in _eb_items],
                        values=[x[1] for x in _eb_items],
                        marker_colors=["#C44E52", "#DD8A00", "#2ca02c", "#4C72B0"],
                        textinfo="label+percent",
                        textfont_size=11,
                        hole=0.35,
                    ))
                    fig_eb.update_layout(
                        height=260, margin=dict(t=30, b=10, l=10, r=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_eb, use_container_width=True, key="profile_elec_pie")
            with _eb2:
                _hvac_pct = _ekwh_hvac / _ekwh_total * 100 if _ekwh_total > 0 else 0
                st.metric("HVAC 비율", f"{_hvac_pct:.1f}%",
                          help="EHP + FCU가 전체 전기의 몇 %인지")
                if not pd.isna(m2) and m2 > 0:
                    _hvac_intensity = _ekwh_hvac / m2
                    st.metric("HVAC 집약도", f"{_hvac_intensity:,.1f} kWh/m²")
                    # Peer HVAC intensity
                    if _has_ehp and "kwh_fcu" in electricity_df.columns:
                        _ep = electricity_df[electricity_df["brand"].astype(str).str.strip() != selected.strip()].copy()
                        _ep_agg2 = _ep.groupby("brand").agg(
                            ehp=("kwh_ehp", "sum"), fcu=("kwh_fcu", "sum"), area=("size_m2", "sum")
                        )
                        _ep_agg2 = _ep_agg2[_ep_agg2["area"] > 0]
                        _ep_agg2["hvac_int"] = (_ep_agg2["ehp"] + _ep_agg2["fcu"]) / _ep_agg2["area"]
                        _ep_hvac_med = _ep_agg2["hvac_int"].median()
                        if not pd.isna(_ep_hvac_med):
                            _ratio = _hvac_intensity / _ep_hvac_med if _ep_hvac_med > 0 else 0
                            if _ratio >= 1.5:
                                st.warning(f"동종 대비 **{_ratio:.1f}배** — HVAC 설비 점검 권장")
                            elif _ratio <= 0.5:
                                st.success(f"동종 대비 **{_ratio:.1f}배** — 효율적 운영")

                for l, v in _eb_items:
                    _pct = v / _ekwh_total * 100 if _ekwh_total > 0 else 0
                    st.caption(f"**{l}**: {v:,.0f} kWh ({_pct:.1f}%)")

            # Electricity narrative
            if _ekwh_total > 0:
                if _hvac_pct >= 60:
                    st.info(
                        f"💡 HVAC(냉난방)이 전체 전기의 **{_hvac_pct:.0f}%**를 차지합니다. "
                        f"인버터 교체, 설정온도 조정, 필터 청소 등 설비 관리가 전기 비용 절감의 핵심입니다."
                    )
                elif _hvac_pct <= 20 and _ekwh_hvac > 0:
                    st.caption(
                        f"HVAC 비중이 {_hvac_pct:.0f}%로 낮아 일반 전기(조명·기기)가 주된 비용 요인입니다."
                    )

            st.divider()

    # ── 면적당 사용 효율 (Per-m² Intensity) ───────────────────────────────────
    if not pd.isna(m2) and m2 > 0:
        _intensity_items = []
        for p in present:
            _curr_col = f"{p}_current"
            _pm2_col = f"{p}_usage_per_m2"
            if _pm2_col in cur_df.columns:
                bv = _v(row, _pm2_col)
                if pd.isna(bv):
                    continue
                s = to_numeric_series(cur_df[_pm2_col]).dropna()
                if s.empty:
                    continue
                _med = s.median()
                _rank = int((s < bv).sum()) + 1
                _pctile = _rank / len(s) * 100
                _ratio = bv / _med if _med > 0 else 0
                _intensity_items.append((_LABEL.get(p, p), _UNIT.get(p, ""), bv, _med, _ratio, _pctile, len(s)))

        if _intensity_items:
            st.subheader("📐 면적당 사용 효율")
            st.caption("동일 면적 대비 사용량을 비교하면 규모 차이를 제거하고 순수 효율성을 평가할 수 있습니다.")
            _int_cols = st.columns(len(_intensity_items))
            _overuse_items = []
            for _ii, (_i_label, _i_unit, _i_val, _i_med, _i_ratio, _i_pctile, _i_n) in enumerate(_intensity_items):
                with _int_cols[_ii]:
                    _delta = _i_val - _i_med
                    st.metric(
                        f"{_i_label} ({_i_unit}/m²)",
                        f"{_i_val:,.4f}",
                        delta=f"{_delta:+,.4f} vs 중앙값",
                        delta_color="inverse",
                    )
                    _i_clr = "#C44E52" if _i_ratio >= 2 else "#DD8A00" if _i_ratio >= 1.3 else "#2ca02c"
                    _i_msg = "⚠️ 과다 사용" if _i_ratio >= 2 else "📌 평균 이상" if _i_ratio >= 1.3 else "✅ 효율적"
                    st.markdown(
                        f'<span style="color:{_i_clr};font-size:0.82rem;font-weight:600">'
                        f'{_i_msg} (상위 {100-_i_pctile:.0f}%)</span>',
                        unsafe_allow_html=True,
                    )
                    if _i_ratio >= 1.3:
                        _overuse_items.append(_i_label)
            # Intensity narrative
            if _overuse_items:
                st.info(
                    f"💡 **{selected[:15]}**은 면적 대비 "
                    f"**{', '.join(_overuse_items)}** 사용이 동종 평균을 상회합니다. "
                    f"업종 특성(예: 음식점의 수도 사용)이 아니라면 누수·설비 점검을 권장합니다."
                )
            else:
                st.caption(f"모든 항목에서 면적당 사용 효율이 정상 범위입니다.")
            st.divider()

    # ── 검침내역: usage summary ───────────────────────────────────────────────
    st.subheader("📊 사용량 요약 (검침내역)")
    st.caption("유틸리티별 이번달·전달 사용량, 변화량·변화율, 면적당 효율, 그리고 동종 대비 분류(HH~LL)를 한눈에 정리한 표입니다.")
    usage_rows = []
    for p in present:
        cv_curr = _v(row, f"{p}_current")
        cv_prev = _v(row, f"{p}_previous")
        cv_chg  = _v(row, f"{p}_change")
        cv_pct  = _v(row, f"{p}_pct")
        cv_pm2  = _v(row, f"{p}_usage_per_m2")
        cv_ppy  = _v(row, f"{p}_usage_per_py")

        lo_c, hi_c, lo_p, hi_p = thresholds.get(p, (np.nan,) * 4)
        q_label = _QUAD_LABEL.get(_quad(cv_chg, cv_pct, lo_c, hi_c, lo_p, hi_p), "—")
        u = _UNIT.get(p, "")

        usage_rows.append({
            "항목":             _LABEL.get(p, p),
            f"이번달 ({u})":    _fmt(cv_curr),
            f"전달 ({u})":      _fmt(cv_prev),
            "변화량":           _fmt(cv_chg, sign=True),
            "변화율 (%)":       _fmt(cv_pct, sign=True),
            "m²당":             _fmt(cv_pm2, 4),
            "평당":             _fmt(cv_ppy, 4),
            "분류":             q_label,
        })
    if usage_rows:
        st.dataframe(pd.DataFrame(usage_rows), hide_index=True, use_container_width=True)
        # Usage summary narrative
        _hh_items = [r["항목"] for r in usage_rows if "HH" in r.get("분류", "")]
        _ll_items = [r["항목"] for r in usage_rows if "LL" in r.get("분류", "")]
        if _hh_items:
            st.warning(
                f"**{', '.join(_hh_items)}** 항목이 HH(변화·비율 모두 상위)로 분류되었습니다. "
                f"전월 대비 급증하면서 동종 대비 비율도 높아 즉시 원인 파악이 필요합니다."
            )
        if _ll_items:
            st.success(
                f"**{', '.join(_ll_items)}** 항목은 LL(변화·비율 모두 하위)로 "
                f"안정적인 사용 패턴을 보이고 있습니다."
            )

    # ── Radar chart: brand vs peers (percentile) ─────────────────────────────
    _radar_cats, _radar_brand, _radar_peer = [], [], []
    for p in present:
        col_c = f"{p}_current"
        if col_c not in cur_df.columns:
            continue
        s = to_numeric_series(cur_df[col_c]).dropna()
        bv = _v(row, col_c)
        if s.empty or pd.isna(bv):
            continue
        pctile = float((s < bv).sum()) / len(s) * 100
        _radar_cats.append(_LABEL.get(p, p))
        _radar_brand.append(round(pctile, 1))
        _radar_peer.append(50.0)
    if len(_radar_cats) >= 3:
        _rc1, _rc2 = st.columns([2, 1])
        with _rc1:
            _radar_cats_loop = _radar_cats + [_radar_cats[0]]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=_radar_brand + [_radar_brand[0]],
                theta=_radar_cats_loop,
                name=selected[:20],
                fill="toself",
                fillcolor="rgba(76,114,176,0.15)",
                line=dict(color=_C_CURR, width=2.5),
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=_radar_peer + [_radar_peer[0]],
                theta=_radar_cats_loop,
                name="중위 (50%)",
                line=dict(color="#999", dash="dot", width=1.5),
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100],
                                    ticktext=["25%", "50%", "75%", "100%"],
                                    gridcolor="rgba(0,0,0,0.08)"),
                    angularaxis=dict(gridcolor="rgba(0,0,0,0.08)"),
                ),
                title=dict(text="사용량 백분위 — 동종 대비 위치", font_size=13),
                height=320, margin=dict(t=50, b=20, l=60, r=60),
                legend=dict(orientation="h", y=-0.05),
                showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True, key="profile_radar")
        with _rc2:
            for cat, pctl in zip(_radar_cats, _radar_brand):
                _clr = "#C44E52" if pctl >= 80 else "#DD8A00" if pctl >= 60 else "#2ca02c"
                st.markdown(
                    f'<div style="margin:6px 0;padding:6px 10px;border-left:3px solid {_clr};'
                    f'border-radius:4px;font-size:0.85rem">'
                    f'<b>{cat}</b> <span style="color:{_clr};font-weight:700">'
                    f'상위 {100-pctl:.0f}%</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Peer ranking ──────────────────────────────────────────────────────────
    peer_lines = []
    for p in present:
        s = to_numeric_series(cur_df[f"{p}_change"]).dropna().sort_values()
        cv = _v(row, f"{p}_change")
        if s.empty or pd.isna(cv): continue
        rank = int((s < cv).sum()) + 1
        pctile = round(rank / len(s) * 100, 1)
        peer_lines.append(
            f"**{_LABEL.get(p,p)}** 변화량 {cv:+,.2f} {_UNIT.get(p,'')} — "
            f"전체 {len(s)}개 중 **{rank}위** (상위 {100-pctile:.1f}%)"
        )
    if peer_lines:
        st.caption("  \n".join(peer_lines))

    st.divider()
    _render_comparison_charts(row, cur_df, present, selected)
    st.divider()

    # ── 수도광열비: billing summary ───────────────────────────────────────────
    if billing_df is not None and not billing_df.empty:
        st.subheader("💰 수도광열비 부과 내역")
        st.caption("전용·공용 요금을 포함한 실제 부과 금액입니다. 동종 평균 대비 높은 항목은 계량기 오류 또는 공용 배분 기준을 확인하세요.")
        bdf = _filter_brand(billing_df, selected)
        if bdf.empty:
            st.caption("해당 브랜드의 수도광열비 내역을 찾을 수 없습니다.")
        else:
            _id_cols     = ["building", "floor", "unit", "size_m2"]
            _bill_detail = [
                "total", "total_excl", "total_comm",
                "water_excl", "water_comm", "water_total",
                "elect_excl", "elect_comm", "elect_total",
                "hotwater_excl", "hotwater_comm",
                "hvac_excl", "hvac_comm",
                "heat_total",
            ]
            _bill_cols = [c for c in _id_cols + _bill_detail if c in bdf.columns]
            _peer_chart(billing_df, selected, [
                ("total",       "총합계"),
                ("water_total", "상하수도"),
                ("elect_total", "전기"),
                ("heat_total",  "열"),
            ], "수도광열비 — 동종 평균 대비 (만원)", "만원")
            st.dataframe(st_safe(bdf[_bill_cols].reset_index(drop=True)),
                         hide_index=True, use_container_width=True)
        st.divider()

    # ── 수도: detailed water ──────────────────────────────────────────────────
    if water_df is not None and not water_df.empty:
        st.subheader("💧 수도 사용 내역")
        st.caption("전용 계량 사용량과 공용 배분 금액을 포함합니다. 사용량(m³)과 총액 모두 동종 평균 대비 확인하세요.")
        wdf = _filter_brand(water_df, selected)
        if wdf.empty:
            st.caption("해당 브랜드의 수도 내역을 찾을 수 없습니다.")
        else:
            _w_id   = ["building", "floor", "unit", "size_m2", "size_py"]
            _w_data = [
                "usage_m3", "pipe_fee_comm",
                "water_excl", "water_comm",
                "sewage_excl", "sewage_comm",
                "levy_excl", "levy_comm",
                "total_excl", "total_comm", "total",
                "avg_unit_price",
            ]
            _w_cols = [c for c in _w_id + _w_data if c in wdf.columns]
            _peer_chart(water_df, selected, [
                ("usage_m3", "사용량 (m³)"),
                ("total",    "총액"),
            ], "수도 — 동종 평균 대비")
            st.dataframe(st_safe(wdf[_w_cols].reset_index(drop=True)),
                         hide_index=True, use_container_width=True)
        st.divider()

    # ── 온수: detailed hot water ──────────────────────────────────────────────
    if hotwater_df is not None and not hotwater_df.empty:
        st.subheader("🌡️ 온수 사용 내역")
        st.caption("온수 사용량과 요금입니다. 음식점·미용실 등 업종에서 높은 사용량이 일반적이며, 그 외 업종에서의 급증은 누수 의심 사유가 됩니다.")
        hwdf = _filter_brand(hotwater_df, selected)
        if hwdf.empty:
            st.caption("해당 브랜드의 온수 내역을 찾을 수 없습니다.")
        else:
            _hw_id   = ["building", "floor", "unit", "size_m2", "size_py"]
            _hw_data = ["usage_m3", "fee_excl", "fee_comm", "total"]
            _hw_cols = [c for c in _hw_id + _hw_data if c in hwdf.columns]
            _peer_chart(hotwater_df, selected, [
                ("usage_m3", "사용량 (m³)"),
                ("total",    "총액"),
            ], "온수 — 동종 평균 대비")
            st.dataframe(st_safe(hwdf[_hw_cols].reset_index(drop=True)),
                         hide_index=True, use_container_width=True)
        st.divider()

    # ── 전기: detailed electricity ────────────────────────────────────────────
    if electricity_df is not None and not electricity_df.empty:
        st.subheader("⚡ 전기 사용 내역")
        st.caption("일반전기, EHP, FCU, 펌프 등 용도별 사용량과 전용·EHP·공용 요금 내역입니다.")
        edf = _filter_brand(electricity_df, selected)
        if edf.empty:
            st.caption("해당 브랜드의 전기 내역을 찾을 수 없습니다.")
        else:
            _e_id   = ["building", "floor", "unit", "size_m2", "size_py"]
            _e_kwh  = ["kwh_elec01", "kwh_elec02", "kwh_fcu", "kwh_ehp", "kwh_pump", "kwh_total"]
            _e_fee  = ["excl_total", "ehp_total", "comm_total", "grand_total"]
            _e_cols = [c for c in _e_id + _e_kwh + _e_fee if c in edf.columns]
            _peer_chart(electricity_df, selected, [
                ("kwh_total",   "총 사용량 (kWh)"),
                ("grand_total", "총 요금"),
            ], "전기 — 동종 평균 대비")
            st.dataframe(st_safe(edf[_e_cols].reset_index(drop=True)),
                         hide_index=True, use_container_width=True)
        st.divider()

    # ── Raw meter readings (pre-aggregation) ─────────────────────────────────
    st.subheader("📋 원본 검침 데이터 (층/호별)")
    raw_brand = _filter_brand(ref_df, selected) if ref_df is not None else pd.DataFrame()
    if raw_brand.empty:
        st.info("원본 데이터를 찾을 수 없습니다.")
    else:
        _id   = [c for c in ["building", "floor", "unit", "size_m2", "size_py"] if c in raw_brand.columns]
        _mtr  = [c for c in raw_brand.columns if "_meter_" in c]
        _usg  = [c for c in raw_brand.columns if "_current" in c or "_previous" in c]
        _chg  = [c for c in raw_brand.columns if "_change" in c or "_pct" in c]
        _rest = [c for c in raw_brand.columns
                 if c not in _id + _mtr + _usg + _chg + ["brand"]]
        ordered = [c for c in _id + _mtr + _usg + _chg + _rest if c in raw_brand.columns]
        raw_brand = raw_brand[ordered].reset_index(drop=True)
        raw_brand.insert(0, "No", range(1, len(raw_brand) + 1))
        st.dataframe(st_safe(raw_brand), hide_index=True, use_container_width=True)


# ── Brand Comparison ─────────────────────────────────────────────────────────

_COMPARE_COLORS = [
    "#4C72B0", "#DD8A00", "#C44E52", "#2CA02C", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


def _render_brand_comparison(
    cur_df, present, billing_period, prev_billing_period,
    billing_df, water_df, hotwater_df, electricity_df, brands,
):
    """Side-by-side comparison of 2+ brands."""
    from utils import esc as _esc
    selected = st.multiselect(
        "비교할 브랜드 선택 (2개 이상)", brands,
        default=brands[:2] if len(brands) >= 2 else brands,
        key="compare_brands",
    )
    if len(selected) < 2:
        st.info("비교하려면 2개 이상의 브랜드를 선택하세요.")
        return

    rows = cur_df[cur_df["brand"].astype(str).isin(selected)].copy()
    if rows.empty:
        st.warning("선택한 브랜드의 데이터를 찾을 수 없습니다.")
        return

    # ── 1. Overview table ─────────────────────────────────────────────────────
    st.subheader("📋 기본 정보")
    info_rows = []
    for b in selected:
        r = rows[rows["brand"].astype(str) == b]
        if r.empty:
            continue
        r = r.iloc[0]
        m2 = _v(r, "size_m2")
        info_rows.append({
            "브랜드": b,
            "건물": str(r.get("building", "—")),
            "층": str(r.get("floor", "—")),
            "면적(㎡)": round(m2, 1) if not pd.isna(m2) else None,
        })
    if info_rows:
        st.dataframe(pd.DataFrame(info_rows), hide_index=True, use_container_width=True)

    # ── 2. Usage comparison (검침내역) — table + charts per utility ────────────
    st.subheader("📊 사용량 비교 (검침내역)")
    st.caption("선택한 브랜드의 유틸리티별 사용량·변화량·면적당 효율을 나란히 비교합니다. 면적이 다른 브랜드는 m²당 수치로 비교하면 공정합니다.")

    # Collect all data first for summary chart
    _all_curr_data = {}   # {brand: {lbl: val}}
    _all_chg_data = {}    # {brand: {lbl: val}}
    _all_pm2_data = {}    # {brand: {lbl: val}}

    for p in present:
        lbl = _LABEL.get(p, p)
        unit = _UNIT.get(p, "")
        curr_col = f"{p}_current"
        chg_col = f"{p}_change"
        pct_col = f"{p}_pct"
        pm2_col = f"{p}_usage_per_m2"

        comp_rows = []
        for b in selected:
            r = rows[rows["brand"].astype(str) == b]
            if r.empty:
                continue
            r = r.iloc[0]
            _cv = _v(r, curr_col)
            _chg = _v(r, chg_col)
            _pct = _v(r, pct_col)
            _pm2 = _v(r, pm2_col)
            comp_rows.append({
                "브랜드": b,
                f"사용량 ({unit})": _cv,
                "변화량": _chg,
                "변화율(%)": _pct,
                "m²당": _pm2,
            })
            _all_curr_data.setdefault(b, {})[lbl] = float(_cv) if not pd.isna(_cv) else 0
            _all_chg_data.setdefault(b, {})[lbl] = float(_chg) if not pd.isna(_chg) else 0
            _all_pm2_data.setdefault(b, {})[lbl] = float(_pm2) if not pd.isna(_pm2) else 0
        if not comp_rows:
            continue
        cdf = pd.DataFrame(comp_rows)
        if len(comp_rows) == 2:
            _num_cols = [c for c in cdf.columns if c != "브랜드"]
            _diff_row = {"브랜드": f"△ ({comp_rows[0]['브랜드'][:8]} − {comp_rows[1]['브랜드'][:8]})"}
            for _nc in _num_cols:
                _a = comp_rows[0].get(_nc)
                _b = comp_rows[1].get(_nc)
                _diff_row[_nc] = (_a - _b) if not pd.isna(_a) and not pd.isna(_b) else np.nan
            cdf = pd.concat([cdf, pd.DataFrame([_diff_row])], ignore_index=True)

        st.markdown(f"**{lbl}**")
        st.dataframe(
            cdf, hide_index=True, use_container_width=True,
            column_config={
                f"사용량 ({unit})": st.column_config.NumberColumn(format="%,.2f"),
                "변화량": st.column_config.NumberColumn(format="%+,.2f"),
                "변화율(%)": st.column_config.NumberColumn(format="%+,.2f"),
                "m²당": st.column_config.NumberColumn(format="%,.4f"),
            },
        )

    # ── 사용량 비교 시각화 ─────────────────────────────────────────────────────
    _util_labels = [_LABEL.get(p, p) for p in present]
    if _all_curr_data and _util_labels:
        # 3-column layout: 사용량 | 변화량 | m²당
        _vc1, _vc2, _vc3 = st.columns(3)

        with _vc1:
            st.markdown("**사용량**")
            fig_curr = go.Figure()
            for bi, b in enumerate(selected):
                vals = [_all_curr_data.get(b, {}).get(l, 0) for l in _util_labels]
                fig_curr.add_trace(go.Bar(
                    name=b, x=_util_labels, y=vals,
                    marker_color=_COMPARE_COLORS[bi % len(_COMPARE_COLORS)],
                    text=[f"{v:,.1f}" for v in vals],
                    textposition="outside", textfont_size=9,
                ))
            fig_curr.update_layout(
                barmode="group", height=320,
                legend=dict(orientation="h", y=1.12, x=0, font_size=10),
                margin=dict(t=50, b=20, l=30, r=10),
            )
            fig_curr.update_yaxes(zeroline=True, rangemode="tozero")
            st.plotly_chart(fig_curr, use_container_width=True, key="cmp_curr_bar")

        with _vc2:
            st.markdown("**전월 대비 변화량**")
            fig_chg = go.Figure()
            for bi, b in enumerate(selected):
                vals = [_all_chg_data.get(b, {}).get(l, 0) for l in _util_labels]
                clrs = [("#C44E52" if v > 0 else "#2ca02c") for v in vals]
                fig_chg.add_trace(go.Bar(
                    name=b, x=_util_labels, y=vals,
                    marker_color=_COMPARE_COLORS[bi % len(_COMPARE_COLORS)],
                    text=[f"{v:+,.1f}" for v in vals],
                    textposition="outside", textfont_size=9,
                ))
            fig_chg.add_hline(y=0, line_color="#888", line_width=1)
            fig_chg.update_layout(
                barmode="group", height=320,
                legend=dict(orientation="h", y=1.12, x=0, font_size=10),
                margin=dict(t=50, b=20, l=30, r=10),
            )
            st.plotly_chart(fig_chg, use_container_width=True, key="cmp_chg_bar")

        with _vc3:
            st.markdown("**m²당 사용량**")
            fig_pm2 = go.Figure()
            for bi, b in enumerate(selected):
                vals = [_all_pm2_data.get(b, {}).get(l, 0) for l in _util_labels]
                fig_pm2.add_trace(go.Bar(
                    name=b, x=_util_labels, y=vals,
                    marker_color=_COMPARE_COLORS[bi % len(_COMPARE_COLORS)],
                    text=[f"{v:,.3f}" for v in vals],
                    textposition="outside", textfont_size=9,
                ))
            fig_pm2.update_layout(
                barmode="group", height=320,
                legend=dict(orientation="h", y=1.12, x=0, font_size=10),
                margin=dict(t=50, b=20, l=30, r=10),
            )
            fig_pm2.update_yaxes(zeroline=True, rangemode="tozero")
            st.plotly_chart(fig_pm2, use_container_width=True, key="cmp_pm2_bar")

        # ── Difference charts (2 brands only) — 3-column layout ──────────
        if len(selected) == 2:
            _da, _db = selected[0], selected[1]
            st.markdown(f"**△ 차이 ({_esc(_da[:10])} − {_esc(_db[:10])})**")
            st.caption(
                f"양수(빨간색)는 {_da[:10]}이 더 높음, "
                f"음수(초록색)는 {_db[:10]}이 더 높음을 의미합니다."
            )

            _dc1, _dc2, _dc3 = st.columns(3)

            # Usage difference
            with _dc1:
                st.markdown("**사용량 차이**")
                _diff_curr = [
                    _all_curr_data.get(_da, {}).get(l, 0)
                    - _all_curr_data.get(_db, {}).get(l, 0)
                    for l in _util_labels
                ]
                fig_dc = go.Figure(go.Bar(
                    x=_util_labels, y=_diff_curr,
                    marker_color=["#C44E52" if v > 0 else "#2ca02c" for v in _diff_curr],
                    text=[f"{v:+,.1f}" for v in _diff_curr],
                    textposition="outside", textfont_size=9,
                ))
                fig_dc.add_hline(y=0, line_color="#888", line_width=1)
                fig_dc.update_layout(
                    height=280, showlegend=False,
                    margin=dict(t=20, b=20, l=40, r=10),
                )
                st.plotly_chart(fig_dc, use_container_width=True, key="cmp_diff_curr")

            # Change difference
            with _dc2:
                st.markdown("**변화량 차이**")
                _diff_chg = [
                    _all_chg_data.get(_da, {}).get(l, 0)
                    - _all_chg_data.get(_db, {}).get(l, 0)
                    for l in _util_labels
                ]
                fig_dg = go.Figure(go.Bar(
                    x=_util_labels, y=_diff_chg,
                    marker_color=["#C44E52" if v > 0 else "#2ca02c" for v in _diff_chg],
                    text=[f"{v:+,.1f}" for v in _diff_chg],
                    textposition="outside", textfont_size=9,
                ))
                fig_dg.add_hline(y=0, line_color="#888", line_width=1)
                fig_dg.update_layout(
                    height=280, showlegend=False,
                    margin=dict(t=20, b=20, l=40, r=10),
                )
                st.plotly_chart(fig_dg, use_container_width=True, key="cmp_diff_chg")

            # Per-m² difference
            with _dc3:
                st.markdown("**m²당 차이**")
                _diff_pm2 = [
                    _all_pm2_data.get(_da, {}).get(l, 0)
                    - _all_pm2_data.get(_db, {}).get(l, 0)
                    for l in _util_labels
                ]
                fig_dp = go.Figure(go.Bar(
                    x=_util_labels, y=_diff_pm2,
                    marker_color=["#C44E52" if v > 0 else "#2ca02c" for v in _diff_pm2],
                    text=[f"{v:+,.4f}" for v in _diff_pm2],
                    textposition="outside", textfont_size=9,
                ))
                fig_dp.add_hline(y=0, line_color="#888", line_width=1)
                fig_dp.update_layout(
                    height=280, showlegend=False,
                    margin=dict(t=20, b=20, l=40, r=10),
                )
                st.plotly_chart(fig_dp, use_container_width=True, key="cmp_diff_pm2")

            # Narrative summary for difference
            _max_diff_idx = max(range(len(_diff_curr)), key=lambda i: abs(_diff_curr[i]))
            _max_lbl = _util_labels[_max_diff_idx]
            _max_val = _diff_curr[_max_diff_idx]
            _higher = _da[:10] if _max_val > 0 else _db[:10]
            st.info(
                f"💡 **{_max_lbl}** 항목에서 가장 큰 차이가 발생합니다. "
                f"**{_higher}**이(가) {abs(_max_val):,.1f} 더 많이 사용하고 있으며, "
                f"면적 효율(m²당) 차이까지 고려하면 "
                f"규모 차이와 실제 사용 효율의 차이를 구분할 수 있습니다."
            )

    # ── 5. Billing comparison ─────────────────────────────────────────────────
    if billing_df is not None and not billing_df.empty:
        st.subheader("💰 수도광열비 비교")
        _bill_metrics = [
            ("total", "총합계"), ("water_total", "상하수도"),
            ("elect_total", "전기"), ("heat_total", "열"),
        ]
        _avail = [(c, l) for c, l in _bill_metrics if c in billing_df.columns]
        if _avail:
            bill_rows = []
            for b in selected:
                bdf = _filter_brand(billing_df, b)
                if bdf.empty:
                    continue
                row_data = {"브랜드": b}
                for col, lbl in _avail:
                    row_data[lbl] = to_numeric_series(bdf[col]).sum()
                bill_rows.append(row_data)
            if bill_rows:
                bdf_cmp = pd.DataFrame(bill_rows)
                # Add difference row if exactly 2 brands
                if len(bill_rows) == 2:
                    _bdiff = {"브랜드": f"△ ({bill_rows[0]['브랜드'][:8]} − {bill_rows[1]['브랜드'][:8]})"}
                    for _, l in _avail:
                        _bdiff[l] = bill_rows[0].get(l, 0) - bill_rows[1].get(l, 0)
                    bdf_cmp = pd.concat([bdf_cmp, pd.DataFrame([_bdiff])], ignore_index=True)
                st.dataframe(bdf_cmp, hide_index=True, use_container_width=True,
                             column_config={l: st.column_config.NumberColumn(f"{l} (만원)", format="%,.2f")
                                            for _, l in _avail})

                fig_b = go.Figure()
                for bi, b in enumerate(selected):
                    br = next((r for r in bill_rows if r["브랜드"] == b), None)
                    if not br:
                        continue
                    vals = [br.get(l, 0) for _, l in _avail]
                    clr = _COMPARE_COLORS[bi % len(_COMPARE_COLORS)]
                    fig_b.add_trace(go.Bar(
                        name=b, x=[l for _, l in _avail], y=vals,
                        marker_color=clr,
                        text=[f"{v:,.1f}" for v in vals],
                        textposition="outside", cliponaxis=False,
                    ))
                fig_b.update_layout(
                    barmode="group", height=350,
                    yaxis=dict(title="만원", zeroline=True, rangemode="tozero"),
                    legend=dict(orientation="h", y=1.08, x=0),
                    margin=dict(t=40, b=20, l=40, r=10),
                )
                _ev = st.plotly_chart(fig_b, use_container_width=True, on_select="rerun",
                                      key="cmp_bill_bar")
                handle_chart_click(_ev, billing_df, brand_col="brand", field="x")

                # Billing difference chart (2 brands)
                if len(selected) == 2 and len(bill_rows) == 2:
                    st.markdown(f"**△ 비용 차이 ({_esc(selected[0][:10])} − {_esc(selected[1][:10])})**")
                    _b_diff_vals = [
                        bill_rows[0].get(l, 0) - bill_rows[1].get(l, 0) for _, l in _avail
                    ]
                    _b_diff_clrs = ["#C44E52" if v > 0 else "#2ca02c" for v in _b_diff_vals]
                    fig_bd = go.Figure(go.Bar(
                        x=[l for _, l in _avail], y=_b_diff_vals,
                        marker_color=_b_diff_clrs,
                        text=[f"{v:+,.1f}" for v in _b_diff_vals],
                        textposition="outside", textfont_size=10,
                    ))
                    fig_bd.add_hline(y=0, line_color="#888", line_width=1)
                    fig_bd.update_layout(
                        height=280, showlegend=False,
                        yaxis_title="만원",
                        margin=dict(t=20, b=20, l=40, r=10),
                    )
                    st.plotly_chart(fig_bd, use_container_width=True, key="cmp_bill_diff")

                    _max_bill_idx = max(range(len(_b_diff_vals)), key=lambda i: abs(_b_diff_vals[i]))
                    _mb_lbl = [l for _, l in _avail][_max_bill_idx]
                    _mb_val = _b_diff_vals[_max_bill_idx]
                    _mb_higher = selected[0][:10] if _mb_val > 0 else selected[1][:10]
                    st.caption(
                        f"**{_mb_lbl}** 항목에서 **{_mb_higher}**이(가) "
                        f"{abs(_mb_val):,.1f}만원 더 높은 비용이 부과되었습니다."
                    )

    # ── 6. Radar chart (normalized) ───────────────────────────────────────────
    st.subheader("🕸️ 레이더 비교")
    radar_cols = [(f"{p}_current", _LABEL.get(p, p)) for p in present
                  if f"{p}_current" in rows.columns]
    if len(radar_cols) >= 3:
        fig_r = go.Figure()
        cats = [lbl for _, lbl in radar_cols]
        # Normalize each metric 0–1 across selected brands
        raw_vals = {}
        for b in selected:
            r = rows[rows["brand"].astype(str) == b]
            if r.empty:
                continue
            raw_vals[b] = [float(_v(r.iloc[0], c)) if not pd.isna(_v(r.iloc[0], c)) else 0
                           for c, _ in radar_cols]
        if raw_vals:
            all_v = np.array(list(raw_vals.values()))
            maxes = all_v.max(axis=0)
            maxes[maxes == 0] = 1  # avoid div-by-zero
            for bi, (b, vals) in enumerate(raw_vals.items()):
                normed = [v / m for v, m in zip(vals, maxes)]
                clr = _COMPARE_COLORS[bi % len(_COMPARE_COLORS)]
                fig_r.add_trace(go.Scatterpolar(
                    r=normed + [normed[0]],  # close the polygon
                    theta=cats + [cats[0]],
                    name=b, fill="toself", opacity=0.3,
                    line=dict(color=clr, width=2),
                ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1.05])),
                height=420,
                legend=dict(orientation="h", y=-0.1, x=0),
                margin=dict(t=30, b=60, l=40, r=40),
            )
            st.plotly_chart(fig_r, use_container_width=True)
            st.caption("각 항목을 브랜드 간 최댓값 기준으로 0–1 정규화한 레이더 차트입니다.")
    elif radar_cols:
        st.caption("레이더 차트는 3개 이상의 유틸리티 항목이 필요합니다.")
